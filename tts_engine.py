"""
tts_engine.py
--------------
Local multilingual TTS for Unitree G1 EDU.

Primary backend: Coqui XTTS v2 loaded from local files only.
Fallback backend: Piper CLI with one local ONNX voice per language.

Output contract for audio_io.AudioPlayer:
    WAV, mono, signed 16-bit PCM, 16000 Hz.

Python: 3.8+
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

logger = logging.getLogger("tts_engine")
logger.setLevel(logging.INFO)


class TTSEngineError(RuntimeError):
    """Base error for TTS failures."""


PROJECT_DIR = Path(__file__).resolve().parent
FALLBACK_LANG = "en"

# Must cover every value in STTEngine._WHISPER_TO_XTTS_LANG_MAP.
XTTS_LANGS: Set[str] = {
    "en",
    "ru",
    "es",
    "fr",
    "de",
    "zh-cn",
    "ar",
    "pt",
    "it",
    "pl",
    "tr",
    "nl",
    "cs",
    "ja",
    "ko",
    "hu",
}

PIPER_MODEL_FILES: Dict[str, str] = {
    "ru": "ru_RU-irina-medium.onnx",
    "en": "en_US-lessac-medium.onnx",
    "es": "es_ES-davefx-medium.onnx",
    "fr": "fr_FR-upmc-medium.onnx",
    "de": "de_DE-thorsten-medium.onnx",
    "zh-cn": "zh_CN-huayan-medium.onnx",
    "ar": "ar_JO-kareem-medium.onnx",
    "pt": "pt_BR-faber-medium.onnx",
    "it": "it_IT-riccardo-x_low.onnx",
    "pl": "pl_PL-darkman-medium.onnx",
    "tr": "tr_TR-dfki-medium.onnx",
    "nl": "nl_NL-mls-medium.onnx",
    "cs": "cs_CZ-jirka-medium.onnx",
    "ja": "ja_JA-hi_fi_captain-medium.onnx",
    "ko": "ko_KR-kss-medium.onnx",
    "hu": "hu_HU-anna-medium.onnx",
}

_WARMUP_PHRASES: Dict[str, str] = {
    "ru": "Привет.",
    "en": "Hello.",
    "es": "Hola.",
    "fr": "Bonjour.",
    "de": "Hallo.",
    "zh-cn": "你好。",
    "ar": "مرحبا.",
    "pt": "Olá.",
    "it": "Ciao.",
    "pl": "Cześć.",
    "tr": "Merhaba.",
    "nl": "Hallo.",
    "cs": "Ahoj.",
    "ja": "こんにちは。",
    "ko": "안녕하세요.",
    "hu": "Szia.",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _run_checked(command: Sequence[str], *, timeout: float, input_bytes: Optional[bytes] = None) -> None:
    result = subprocess.run(
        list(command),
        input=input_bytes,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        stdout = result.stdout.decode("utf-8", errors="replace").strip()
        raise TTSEngineError(
            "Command failed rc={}: {}\nstdout={}\nstderr={}".format(
                result.returncode,
                " ".join(command),
                stdout,
                stderr,
            )
        )


class TTSEngine:
    """
    TTS engine with stable public API for main.py.

    synthesize(text, lang_code, output_path) always writes a G1-ready WAV.
    """

    def __init__(
        self,
        models_dir: str = "~/.local/share/piper",
        piper_binary: str = "",
        g1_audio_binary: Optional[str] = None,
        network_interface: str = "eth0",
        warmup_on_init: bool = True,
        preload_langs: Optional[List[str]] = None,
        backend: Optional[str] = None,
        xtts_dir: Optional[str] = None,
        speaker_wav: Optional[str] = None,
    ) -> None:
        self.interface = network_interface
        self.backend = (backend or os.environ.get("VOICE_ENGINE_TTS", "piper")).strip().lower()
        if self.backend not in {"xtts", "piper", "espeak", "auto"}:
            logger.warning("Unknown VOICE_ENGINE_TTS=%r, using piper.", self.backend)
            self.backend = "piper"

        self.allow_piper_fallback = _env_bool("VOICE_ENGINE_TTS_FALLBACK_PIPER", True)
        self.speech_tempo = float(os.environ.get("VOICE_ENGINE_TTS_TEMPO", "1.18"))
        self.gain_db = float(os.environ.get("VOICE_ENGINE_TTS_GAIN_DB", "0"))
        self.xtts_split_sentences = _env_bool("VOICE_ENGINE_XTTS_SPLIT_SENTENCES", False)

        self.models_dir = Path(os.path.expanduser(models_dir))
        self.piper_bin = piper_binary or os.environ.get(
            "PIPER_BINARY",
            str(PROJECT_DIR / "piper" / "piper"),
        )
        self._piper_model_cache: Dict[str, str] = {}

        self.xtts_dir = Path(os.path.expanduser(xtts_dir or os.environ.get(
            "XTTS_MODEL_DIR",
            str(PROJECT_DIR / "models" / "xtts_v2"),
        )))
        self.xtts_model_path = self.xtts_dir / "model.pth"
        self.xtts_config_path = self.xtts_dir / "config.json"
        self.xtts_speakers_path = self.xtts_dir / "speakers_xtts.pth"
        self.speaker_wav = Path(os.path.expanduser(speaker_wav or os.environ.get(
            "XTTS_SPEAKER_WAV",
            str(PROJECT_DIR / "speaker.wav"),
        )))

        self._xtts_lock = threading.Lock()
        self._xtts: Optional[Any] = None
        self._xtts_device = "cpu"
        self._xtts_languages = self._load_xtts_language_set()

        self._ensure_system_tools()
        self._log_configuration()

        if warmup_on_init:
            self.warmup(preload_langs=preload_langs)

    # ------------------------------------------------------------------ #
    # Initialization and validation
    # ------------------------------------------------------------------ #

    def _ensure_system_tools(self) -> None:
        if subprocess.run(["which", "sox"], capture_output=True).returncode != 0:
            raise TTSEngineError("sox is not installed. Install: sudo apt install sox")
        if self.backend == "espeak":
            if subprocess.run(["which", "espeak"], capture_output=True).returncode != 0:
                raise TTSEngineError("espeak is not installed. Install: sudo apt install espeak")
        if self.backend in {"piper", "auto"} or self.allow_piper_fallback:
            if not Path(self.piper_bin).exists():
                raise TTSEngineError("Piper binary not found: {}".format(self.piper_bin))

    def _load_xtts_language_set(self) -> Set[str]:
        if not self.xtts_config_path.exists():
            return set(XTTS_LANGS)
        try:
            with self.xtts_config_path.open("r", encoding="utf-8") as config_file:
                data = json.load(config_file)
            langs = data.get("languages") or []
            return {str(lang).lower() for lang in langs} or set(XTTS_LANGS)
        except Exception as exc:
            logger.warning("Cannot read XTTS config languages from %s: %s", self.xtts_config_path, exc)
            return set(XTTS_LANGS)

    def _xtts_files_ready(self) -> bool:
        required = [
            self.xtts_model_path,
            self.xtts_config_path,
            self.xtts_dir / "vocab.json",
            self.xtts_speakers_path,
            self.speaker_wav,
        ]
        missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
        if missing:
            logger.warning("XTTS local files are missing: %s", missing)
            return False
        return True

    def _log_configuration(self) -> None:
        logger.info(
            "TTSEngine initialized | backend=%s | fallback_piper=%s | tempo=%.2f | gain=%.1f dB | xtts_dir=%s | speaker=%s",
            self.backend,
            self.allow_piper_fallback,
            self.speech_tempo,
            self.gain_db,
            self.xtts_dir,
            self.speaker_wav,
        )

    def _load_xtts(self) -> Any:
        if self._xtts is not None:
            return self._xtts

        if not self._xtts_files_ready():
            raise TTSEngineError("XTTS v2 is not available from local files.")

        try:
            import torch
            import torchaudio
            import soundfile as sf

            if not hasattr(torchaudio, "_patched_for_soundfile"):
                def _sf_load(filepath, **kwargs):
                    import soundfile as sf # на всякий случай
                    data, sr = sf.read(filepath, dtype='float32')
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                    tensor = torch.from_numpy(data.T).contiguous()
                    return tensor, sr
                
                torchaudio.load = _sf_load
                torchaudio._patched_for_soundfile = True

            if not hasattr(torch, "_patched_for_tts"):
                _original_load = torch.load
                def _patched_load(*args, **kwargs):
                    kwargs["weights_only"] = False
                    return _original_load(*args, **kwargs)
                torch.load = _patched_load
                torch._patched_for_tts = True
            
            from TTS.api import TTS
        except Exception as exc:
            raise TTSEngineError("Coqui TTS import failed: {}".format(exc)) from exc

        requested = os.environ.get("VOICE_ENGINE_XTTS_DEVICE", "auto").strip().lower()
        cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        if requested == "cuda" and not cuda_available:
            raise TTSEngineError("VOICE_ENGINE_XTTS_DEVICE=cuda requested, but torch.cuda.is_available() is false.")
        device = "cuda" if (requested == "cuda" or (requested == "auto" and cuda_available)) else "cpu"

        logger.info("Loading local XTTS v2 on %s from %s", device, self.xtts_dir)
        model = TTS(
            model_path=str(self.xtts_dir),
            config_path=str(self.xtts_config_path),
            progress_bar=False,
        )
        model.to(device)
        self._xtts = model
        self._xtts_device = device
        return model

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def validate_languages(self, languages: Sequence[str]) -> None:
        missing_xtts = sorted(set(languages) - self._xtts_languages)
        missing_piper = sorted(set(languages) - set(PIPER_MODEL_FILES))
        if missing_xtts:
            logger.warning("XTTS config does not list languages: %s", missing_xtts)
        if missing_piper:
            raise TTSEngineError("Piper fallback does not cover languages: {}".format(missing_piper))

    def synthesize(self, text: str, lang_code: str, output_path: str) -> str:
        if not text or not text.strip():
            raise TTSEngineError("Empty text is not allowed for synthesis.")

        lang = self._normalize_lang(lang_code)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if self.backend == "piper":
            return self._synthesize_piper(text, lang, str(output))

        if self.backend == "espeak":
            return self._synthesize_espeak(text, lang, str(output))

        try:
            if lang not in self._xtts_languages:
                raise TTSEngineError("Language {} is not supported by local XTTS config.".format(lang))
            return self._synthesize_xtts(text, lang, str(output))
        except Exception as exc:
            if not self.allow_piper_fallback:
                raise
            logger.warning("XTTS failed for lang=%s, using Piper fallback: %s", lang, exc)
            return self._synthesize_piper(text, lang, str(output))

    def synthesize_compat(self, text: str, lang_code: str, output_path: str) -> str:
        return self.synthesize(text, lang_code, output_path)

    def warmup(self, preload_langs: Optional[List[str]] = None) -> None:
        if preload_langs is not None:
            langs = preload_langs
        else:
            raw_langs = os.environ.get("VOICE_ENGINE_TTS_WARMUP_LANGS", "ru,en").strip()
            if raw_langs.lower() == "all":
                langs = sorted(PIPER_MODEL_FILES)
            else:
                langs = [lang.strip().lower() for lang in raw_langs.split(",") if lang.strip()]
                if not langs:
                    langs = ["ru", "en"]
        logger.info("TTSEngine warmup for languages: %s", langs)
        tmp = Path(tempfile.gettempdir()) / "tts_warmup.wav"
        for lang in langs:
            phrase = _WARMUP_PHRASES.get(lang, "Hello.")
            try:
                self.synthesize(phrase, lang, str(tmp))
                logger.info("  ok: %s", lang)
            except Exception as exc:
                logger.warning("  failed: %s: %s", lang, exc)
        tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # XTTS backend
    # ------------------------------------------------------------------ #

    def _synthesize_xtts(self, text: str, lang: str, output_path: str) -> str:
        raw_wav = output_path + ".xtts.raw.wav"
        with self._xtts_lock:
            model = self._load_xtts()
            self._disable_transformers_distributed_checks()
            model.tts_to_file(
                text=text,
                file_path=raw_wav,
                speaker_wav=str(self.speaker_wav),
                language=lang,
                split_sentences=self.xtts_split_sentences,
            )
        self._convert_to_g1_wav(raw_wav, output_path)
        Path(raw_wav).unlink(missing_ok=True)
        return output_path

    @staticmethod
    def _disable_transformers_distributed_checks() -> None:
        try:
            import transformers.generation.utils as generation_utils
            import transformers.integrations.deepspeed as deepspeed_utils
            import transformers.integrations.fsdp as fsdp_utils

            generation_utils.is_deepspeed_zero3_enabled = lambda: False
            generation_utils.is_fsdp_managed_module = lambda module: False
            deepspeed_utils.is_deepspeed_zero3_enabled = lambda: False
            fsdp_utils.is_fsdp_managed_module = lambda module: False
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Piper backend
    # ------------------------------------------------------------------ #

    def _resolve_piper_model(self, lang: str) -> str:
        if lang in self._piper_model_cache:
            return self._piper_model_cache[lang]
        filename = PIPER_MODEL_FILES.get(lang) or PIPER_MODEL_FILES[FALLBACK_LANG]
        model_path = self.models_dir / filename
        if not model_path.exists():
            model_path = self.models_dir / PIPER_MODEL_FILES[FALLBACK_LANG]
        if not model_path.exists():
            raise TTSEngineError("Piper model not found for {} or fallback {} in {}".format(lang, FALLBACK_LANG, self.models_dir))
        self._piper_model_cache[lang] = str(model_path)
        return str(model_path)

    def _synthesize_piper(self, text: str, lang: str, output_path: str) -> str:
        model_path = self._resolve_piper_model(lang)
        raw_wav = output_path + ".piper.raw.wav"
        command = [
            self.piper_bin,
            "--model",
            model_path,
            "--output_file",
            raw_wav,
            "--sentence_silence",
            "0.05",
        ]
        _run_checked(command, input_bytes=text.encode("utf-8"), timeout=30)
        self._convert_to_g1_wav(raw_wav, output_path)
        Path(raw_wav).unlink(missing_ok=True)
        return output_path

    # ------------------------------------------------------------------ #
    # eSpeak backend
    # ------------------------------------------------------------------ #
    def _synthesize_espeak(self, text: str, lang: str, output_path: str) -> str:
        raw_wav = output_path + ".espeak.raw.wav"
        
        # Настройки eSpeak:
        # -v <lang> : язык (например, ru или en)
        # -p 20     : pitch (высота тона). Значение 20 делает голос ниже и "суровее"
        # -s 140    : speed (скорость). 140 чуть медленнее дефолта, звучит более размеренно
        command = [
            "espeak",
            "-v", lang,
            "-p", "20",
            "-s", "140",
            "-w", raw_wav,
            text
        ]
        
        _run_checked(command, timeout=10)
        
        self._convert_to_g1_wav(raw_wav, output_path)
        
        Path(raw_wav).unlink(missing_ok=True)
        return output_path

    # ------------------------------------------------------------------ #
    # Audio post-processing
    # ------------------------------------------------------------------ #

    def _convert_to_g1_wav(self, input_path: str, output_path: str) -> None:
        command = ["sox", input_path, "-r", "16000", "-c", "1", "-b", "16", output_path, "norm", "-3"]

        if abs(self.speech_tempo - 1.0) > 1e-3:
            command.extend(["tempo", str(self.speech_tempo)])
        if abs(self.gain_db) > 1e-3:
            command.extend(["gain", "-l", str(self.gain_db)])
        _run_checked(command, timeout=20)

    @staticmethod
    def _normalize_lang(lang_code: str) -> str:
        lang = (lang_code or FALLBACK_LANG).strip().lower()
        if lang == "zh":
            return "zh-cn"
        return lang if lang in XTTS_LANGS or lang in PIPER_MODEL_FILES else FALLBACK_LANG

    # ------------------------------------------------------------------ #
    # Legacy helpers
    # ------------------------------------------------------------------ #

    def play_on_robot(self, wav_path: str) -> None:
        raise TTSEngineError("play_on_robot() is deprecated. Use audio_io.AudioPlayer.play().")

    def synthesize_and_play(self, text: str, lang_code: str) -> None:
        tmp = Path(tempfile.gettempdir()) / "tts_{}.wav".format(uuid.uuid4().hex)
        try:
            self.synthesize(text, lang_code, str(tmp))
            self.play_on_robot(str(tmp))
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = TTSEngine(warmup_on_init=False)
    for text, lang in [
        ("Привет, я говорю через локальный синтез.", "ru"),
        ("Hello, I speak locally.", "en"),
        ("Bonjour, je parle localement.", "fr"),
    ]:
        out = "/tmp/test_{}.wav".format(lang)
        engine.synthesize(text, lang, out)
        print("{} -> {}".format(lang, out))
