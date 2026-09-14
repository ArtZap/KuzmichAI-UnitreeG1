"""
stt_engine.py

Module for speech-to-text recognition for the voice robot.
Supports two backends (selected via env VOICE_ENGINE_STT_BACKEND):
  1. "whisper" (default) - faster-whisper with strict hallucination protection.
  2. "nemo" - NVIDIA NeMo (Parakeet/Canary models) for extremely noisy conditions.

Compatible with Python 3.8+.
"""

from __future__ import annotations

import logging
import ctypes
import os
import time
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# CTranslate2 Hack for local inference
# ---------------------------------------------------------------------------
_CT2_LIB_DIR = Path(os.environ.get("CTRANSLATE2_LIB_DIR", "/home/unitree/AgroBot/build_deps/ctranslate2_cuda_install/lib"))
_CT2_LIB = _CT2_LIB_DIR / "libctranslate2.so.4"
if _CT2_LIB.exists():
    try:
        ctypes.CDLL(str(_CT2_LIB), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass

logger = logging.getLogger(__name__)

class STTEngineError(RuntimeError):
    """Base exception for STTEngine errors."""

# ---------------------------------------------------------------------------
# Backend 1: faster-whisper (Original, but with hallucination protection)
# ---------------------------------------------------------------------------
class WhisperBackend:
    def __init__(self, model_size: str, device: str, compute_type: str, download_root: Optional[str]):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError("Package 'faster-whisper' is not installed.") from exc
            
        logger.info(f"[Whisper] Loading model '{model_size}' (device={device}, compute_type={compute_type})...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )

    def transcribe(self, audio_array: np.ndarray, language: Optional[str] = None) -> Tuple[str, str]:
        segments, info = self.model.transcribe(
            audio_array,
            language=language,
            beam_size=5,                       
            temperature=0.0,                   
            no_speech_threshold=0.5,           
            condition_on_previous_text=False,  
            vad_filter=False,                  
        )
        text = "".join(segment.text for segment in segments).strip()
        detected_lang = getattr(info, "language", None)
        lang_prob = getattr(info, "language_probability", 0.0)

        return text, detected_lang

# ---------------------------------------------------------------------------
# Backend 2: NVIDIA NeMo (Parakeet / Canary)
# ---------------------------------------------------------------------------
class NeMoBackend:
    def __init__(self, model_size: str, device: str):
        try:
            import nemo.collections.asr as nemo_asr
            import soundfile as sf
            self.sf = sf
        except ImportError as exc:
            raise ImportError(
                "NeMo packages are not installed. Install them.: "
                "pip install nemo_toolkit['asr'] soundfile"
            ) from exc

        if "whisper" in model_size.lower() or os.path.exists(model_size):
            nemo_model_name = os.environ.get("VOICE_ENGINE_NEMO_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
        else:
            nemo_model_name = model_size if "/" in model_size else "nvidia/parakeet-tdt-0.6b-v3"
        
        logger.info(f"[NeMo] Loading ASR model '{nemo_model_name}' (device={device})...")
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=nemo_model_name, map_location=device)
        self.model.eval()

        if "canary" in nemo_model_name.lower():
            self.is_canary = True
            self.model.task = "asr"
            self.model.source_lang = "ru"
            self.model.dest_lang = "ru"
        else:
            self.is_canary = False

    def transcribe(self, audio_array: np.ndarray, language: Optional[str] = None) -> Tuple[str, str]:
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            self.sf.write(tmp_path, audio_array, 16000)
            
            if self.is_canary and language:
                self.model.source_lang = language
                self.model.dest_lang = language

            try:
                texts = self.model.transcribe(audio=[tmp_path], batch_size=1)
            except TypeError:
                try:
                    texts = self.model.transcribe(paths2audio_files=[tmp_path], batch_size=1)
                except TypeError:
                    try:
                        texts = self.model.transcribe(audio_paths=[tmp_path], batch_size=1)
                    except TypeError:
                        texts = self.model.transcribe([tmp_path])
            
            if not texts or not texts[0]:
                recognized_text = ""
            else:
                res = texts[0]
                if isinstance(res, list):
                    res = res[0]
                
                if hasattr(res, 'text'):
                    recognized_text = res.text
                elif isinstance(res, dict):
                    recognized_text = res.get('text', '')
                elif isinstance(res, str):
                    recognized_text = res
                else:
                    recognized_text = str(res)
                    
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        recognized_text = recognized_text.strip()
        
        return recognized_text, language

# ---------------------------------------------------------------------------
# Backend 3: Hibrid (Whisper + GigaAM) (special for Russian language)
# ---------------------------------------------------------------------------

import onnx_asr
from faster_whisper import WhisperModel

class HybridBackend:
    def __init__(self, device: str, compute_type: str, gigaam_path: str):
        logger.info("[Hybrid] Loading Whisper large-v3-turbo...")
        self.whisper = WhisperModel(
            "large-v3-turbo", 
            device=device, 
            compute_type=compute_type
        )
        
        logger.info("[Hybrid] Loading GigaAM-v3-e2e-rnnt (INT8)...")
        self.gigaam = onnx_asr.Model(gigaam_path, provider="CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider")

    def transcribe(self, audio_array: np.ndarray, language: Optional[str] = None) -> Tuple[str, str]:
        _, info = self.whisper.detect_language(audio_array)
        detected_lang = info.language

        if detected_lang == "ru":
            logger.debug("[Hybrid] Detected 'ru', routing to GigaAM.")
            recognized_text = self.gigaam.transcribe(audio_array)
            return recognized_text.strip(), "ru"
        else:
            logger.debug(f"[Hybrid] Detected '{detected_lang}', routing to Whisper.")
            segments, _ = self.whisper.transcribe(
                audio_array, 
                language=detected_lang,
                beam_size=1,
                condition_on_previous_text=False
            )
            recognized_text = "".join(segment.text for segment in segments).strip()
            return recognized_text, detected_lang

# ---------------------------------------------------------------------------
# Main facade (Dispatcher)
# ---------------------------------------------------------------------------
class STTEngine:
    """
    Wrapper around STT. Loads the required backend based on VOICE_ENGINE_STT_BACKEND.
    """

    _WHISPER_TO_XTTS_LANG_MAP = {
        "en": "en", "ru": "ru", "es": "es", "fr": "fr", "de": "de",
        "zh": "zh-cn", "ar": "ar", "pt": "pt", "it": "it", "pl": "pl",
        "tr": "tr", "nl": "nl", "cs": "cs", "ja": "ja", "ko": "ko", "hu": "hu",
    }
    _FALLBACK_LANG = "ru"

    @classmethod
    def supported_tts_languages(cls) -> Tuple[str, ...]:
        return tuple(sorted(set(cls._WHISPER_TO_XTTS_LANG_MAP.values())))

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: Optional[str] = None,
    ) -> None:
        self.device = device
        self._is_warmed_up = False
        
        self.backend_type = os.environ.get("VOICE_ENGINE_STT_BACKEND", "whisper").strip().lower()
        
        load_start = time.perf_counter()
        if self.backend_type == "hybrid":
            default_gigaam = str(Path(__file__).resolve().parent / "models" / "gigaam_v3_e2e_rnnt_int8.onnx")
            gigaam_path = os.environ.get("VOICE_ENGINE_GIGAAM_MODEL", default_gigaam)
            
            self.backend = HybridBackend(
                device=device, 
                compute_type=compute_type,
                gigaam_path=gigaam_path
            )
        elif self.backend_type == "nemo":
            self.backend = NeMoBackend(model_size=model_size, device=device)
        else:
            self.backend_type = "whisper"
            self.backend = WhisperBackend(
                model_size=model_size, 
                device=device, 
                compute_type=compute_type, 
                download_root=download_root
            )
            
        load_time = time.perf_counter() - load_start
        logger.info("Backend '%s' loaded in %.2f seconds.", self.backend_type, load_time)

    def warmup(self, sample_rate: int = 16000, duration_sec: float = 1.0) -> None:
        """Warms up the model with a silence array for CUDA graph initialization."""
        logger.debug("Starting warmup() STTEngine...")
        silence = np.zeros(int(sample_rate * duration_sec), dtype=np.float32)

        warmup_start = time.perf_counter()
        try:
            self.backend.transcribe(silence)
        except Exception as exc:
            logger.exception("Error occurred during warmup().")
            raise STTEngineError(f"Error warming up STT model: {exc}") from exc

        warmup_time = time.perf_counter() - warmup_start
        self._is_warmed_up = True
        logger.info("Warmup STTEngine completed in %.3f seconds.", warmup_time)

    def _map_language_to_xtts(self, detected_lang: Optional[str]) -> str:
        if not detected_lang:
            return self._FALLBACK_LANG
        xtts_lang = self._WHISPER_TO_XTTS_LANG_MAP.get(detected_lang.lower())
        return xtts_lang if xtts_lang else self._FALLBACK_LANG

    def transcribe(
        self,
        audio_array: np.ndarray,
        language: Optional[str] = None,
        beam_size: int = 1,
    ) -> Tuple[str, str]:
        
        if not self._is_warmed_up:
            logger.warning("transcribe() called before warmup().")

        if audio_array is None or audio_array.size == 0:
            return "", self._FALLBACK_LANG

        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        start_time = time.perf_counter()
        try:
            recognized_text, detected_lang = self.backend.transcribe(audio_array, language=language)
        except Exception as exc:
            logger.exception("Error occurred during transcribe().")
            raise STTEngineError(f"Error transcribing: {exc}") from exc

        latency = time.perf_counter() - start_time
        xtts_lang_code = self._map_language_to_xtts(detected_lang)

        logger.debug(
            "STT latency=%.3f seconds | backend=%s | audio_len=%.2f seconds | lang=%s -> xtts_lang=%s | text_len=%d",
            latency,
            self.backend_type,
            audio_array.shape[0] / 16000.0,
            detected_lang,
            xtts_lang_code,
            len(recognized_text),
        )

        return recognized_text, xtts_lang_code


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # For test NeMo: export VOICE_ENGINE_STT_BACKEND=nemo
    engine = STTEngine()
    engine.warmup()
    dummy_audio = np.zeros(16000 * 2, dtype=np.float32)
    text, lang = engine.transcribe(dummy_audio)
    logger.info("Результат: text=%r, lang=%s", text, lang)
