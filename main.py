"""
main.py — Final asynchronous pipeline for the voice assistant.

Architecture:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  AudioListener ──► STT ──► SemanticCache ──► AudioPlayer (Cache Hit)│
  │                                │                                    │
  │                           (Cache Miss)                              │
  │                                ▼                                    │
  │              ┌─────── StreamingPipeline ─────────┐                  │
  │              │  LLMEngine.generate_stream()      │                  │
  │              │       │ (Sentences)               │                  │
  │              │       ▼                           │                  │
  │              │  Worker-1: TTS → wav_path         │                  │
  │              │       │ put() in Queue            │                  │
  │              │       ▼                           │                  │
  │              │  Worker-2: get() → AudioPlayer    │                  │
  │              └───────────────────────────────────┘                  │
  │                                │                                    │
  │                    merge wavs → SemanticCache.put()                 │
  └─────────────────────────────────────────────────────────────────────┘

Compatibility: Python 3.8+
"""

import asyncio
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Import modules of the voice assistant
# ---------------------------------------------------------------------------
from audio_io import AudioListener, AudioPlayer, LocalAudioPlayer
from llm_engine import LLMEngine
from semantic_cache import SemanticCache
from stt_engine import STTEngine
from tts_engine import TTSEngine
from g1_greeting_gestures import G1GestureController
import re

GESTURE_REGEX = re.compile(
    r'\[*(?:жест|jest|gest|gesture)[\s:]*([^\]]+)\]', re.IGNORECASE
)

GESTURE_MAP = {
    'facechest': 'face_chest',
    'face_chest': 'face_chest',
    'dashakoza': 'dasha_koza',
    'dasha_koza': 'dasha_koza',
    'moetglaza': 'moet_glaza',
    'moet_glaza': 'moet_glaza',
    'sixtyseven': 'sixty_seven',
    'sixty_seven': 'sixty_seven',
    'xray': 'x-ray',
    'x-ray': 'x-ray',
    'shake_hands': 'shakehands',
    'shakehands': 'shakehands',
    'mouth_keeper': 'mouthkeeper',
    'mouthkeeper': 'mouthkeeper',
}


def normalize_gesture_name(raw_name: str) -> str:
  if not raw_name:
    return ''
  cleaned = raw_name.strip().lower()
  cleaned = cleaned.translate(str.maketrans('осаехр', 'osacxp'))
  cleaned = cleaned.replace('"', '').replace("'", '').strip()
  return GESTURE_MAP.get(cleaned, cleaned)

# ---------------------------------------------------------------------------
# Configuration of logging (Colored)
# ---------------------------------------------------------------------------
class ColorFormatter(logging.Formatter):
    """Custom formatter for coloring logs in the console."""
    
    grey = "\x1b[38;5;240m"
    blue = "\x1b[38;5;39m"
    green = "\x1b[32m"             
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        # logging.INFO: blue + format_str + reset,       
        logging.WARNING: yellow + format_str + reset,  
        logging.ERROR: red + format_str + reset,       
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.format_str)
        
        if record.levelno == logging.INFO and record.name == "VoiceAssistant":
            log_fmt = self.green + self.format_str + self.reset
            
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColorFormatter())

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler],
    force=True,
)

log = logging.getLogger("VoiceAssistant")

# ---------------------------------------------------------------------------
# Function for scene analysis
# ---------------------------------------------------------------------------

sys.path.append("/home/unitree/agrohub_cloud")
try:
    from analyze_for_tts import analyze_for_tts
except ImportError:
    analyze_for_tts = None
    log.warning("The analyze_for_tts module was not found. The 'Analyze' command will not work..")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

# Directory for storing final wav files of responses (for cache)
CACHE_WAV_DIR = PROJECT_DIR / "cache_audio"
CACHE_ANALYZ_WAV_DIR = PROJECT_DIR / "cache_analyz_audio"
# Directory for storing protected responses from the Knowledge Base
KB_CACHE_WAV_DIR = PROJECT_DIR / "kb_audio"

# Token-«poison» (sentinel), which Worker-1 puts in the queue,
# to notify Worker-2 about the end of the stream.
# Using a singleton object ensures strict comparison by identity (is).
_QUEUE_SENTINEL = object()

# Maximum number of threads in the pool (ML models are heavy, don't create too many)
EXECUTOR_MAX_WORKERS = 4

# Audio mode:
#   "g1"    (by default) — microphone and speakers of the real G1 robot
#            via UDP multicast and DDS/AudioClient. Requires the robot's network.
#   "local" — microphone and speakers of the local laptop via sounddevice. Fully
#            local, no network/connection to the robot needed.
# Switched by the environment variable, for example:
#   VOICE_ENGINE_AUDIO=local python main.py
AUDIO_MODE = os.environ.get("VOICE_ENGINE_AUDIO", "g1").strip().lower()
if AUDIO_MODE not in ("g1", "local"):
    log.warning("Unknown value for VOICE_ENGINE_AUDIO=%r, using 'g1'.", AUDIO_MODE)
    AUDIO_MODE = "g1"

ENABLE_STREAMING = os.environ.get("VOICE_ENGINE_ENABLE_STREAMING", "true").strip().lower() == "true"
ENABLE_PLAYBACK = os.environ.get("VOICE_ENGINE_ENABLE_PLAYBACK", "true").strip().lower() == "true"
ENABLE_CONTEXT = os.environ.get("VOICE_ENGINE_ENABLE_CONTEXT", "true").strip().lower() == "true"
MAX_HISTORY_TURNS = int(os.environ.get("VOICE_ENGINE_MAX_HISTORY_TURNS", "3"))

WAKEUP_WORD = os.environ.get("VOICE_ENGINE_WAKEUP_WORD", "привет").strip().lower()
STOP_WORD = os.environ.get("VOICE_ENGINE_STOP_WORD", "стоп").strip().lower()
ANALYSIS_WORD = os.environ.get("VOICE_ENGINE_ANALYSIS_WORD", "анализ").strip().lower()
USE_TRIGGERS = os.environ.get("VOICE_ENGINE_USE_TRIGGERS", "true").strip().lower() == "true"
NOISE_THRESHOLD = float(os.environ.get("VOICE_ENGINE_NOISE_THRESHOLD", "0.6"))
ENABLE_INTERRUPT = os.environ.get("VOICE_ENGINE_ENABLE_INTERRUPT", "true").strip().lower() == "true"
STT_LANGUAGE_ENV = os.environ.get("VOICE_ENGINE_STT_LANGUAGE", "auto").strip().lower()
ENABLE_GESTURES = os.environ.get("VOICE_ENGINE_ENABLE_GESTURES", "true").strip().lower() == "true"

# ===========================================================================
# Helper function: merging WAV files
# ===========================================================================

def merge_wav_files(input_paths: List[str], output_path: str) -> None:
    """
    Merges several .wav files into one using the standard `wave` module.

    All input files MUST have the same parameters:
      - number of channels (nchannels)
      - sample width (sampwidth)
      - sampling rate (framerate)

    Parameters
    ----------
    input_paths : list of paths to temporary .wav files in playback order
    output_path : path to the final .wav file
    """
    if not input_paths:
        raise ValueError("merge_wav_files: list of files is empty")

    with wave.open(input_paths[0], "rb") as first:
        params = first.getparams()  # namedtuple: nchannels, sampwidth, framerate, ...

    with wave.open(output_path, "wb") as out_wav:
        out_wav.setparams(params)
        for path in input_paths:
            with wave.open(path, "rb") as src:
                if (src.getnchannels() != params.nchannels or
                        src.getsampwidth() != params.sampwidth or
                        src.getframerate() != params.framerate):
                    log.warning(
                        "Parameters of file %s do not match the template — skipping",
                        path,
                    )
                    continue
                out_wav.writeframes(src.readframes(src.getnframes()))


# ===========================================================================
# Main class of the application
# ===========================================================================

class VoiceAssistant:
    """
    Orchestrates the entire lifecycle of the voice assistant:
      1. Initialization and warming up all engines.
      2. Main listening loop.
      3. Streaming pipeline LLM → TTS → Player with parallel workers.
      4. Interrupt handling (person speaks while the assistant is responding).
      5. Saving the result to the semantic cache.
    """

    def __init__(self) -> None:
        from audio_io import SharedRobotState, AudioConfig
        self.shared_state = SharedRobotState()

        cfg = AudioConfig(vad_threshold=NOISE_THRESHOLD)

        local_mode = (AUDIO_MODE == "local")
        self.listener = AudioListener(self.shared_state, config=cfg, local_mode=local_mode)
        self.player = LocalAudioPlayer(self.shared_state) if local_mode else AudioPlayer(self.shared_state)
        log.info("Audio mode: %s", "LOCAL (microphone/speakers of the laptop)" if local_mode else "G1 (robot network)")

        self.is_awake = not USE_TRIGGERS
        self.is_analiz = False
        self.uncensored_mode = False
        self.stt_lang = None if STT_LANGUAGE_ENV == "auto" else STT_LANGUAGE_ENV

        # STTEngine is designed for GPU (see its docstring: "device — strictly
        # cuda", "compute_type — strictly int8_float16"). 
        # We automatically detect CUDA availability, with a safe fallback
        # to CPU for debugging on a laptop without a GPU (VOICE_ENGINE_AUDIO=local).
        try:
            import torch
            stt_use_cuda = torch.cuda.is_available()
        except Exception:
            stt_use_cuda = False

        if stt_use_cuda:
            stt_device, stt_compute_type = "cuda", "int8_float16"
        else:
            stt_device, stt_compute_type = "cpu", "int8"
            log.warning(
                "CUDA unavailable — STTEngine runs on CPU (int8). "
                "On the real robot G1 this is noticeably slower than cuda/"
                "int8_float16, and does not meet the requirement for fast speech. "
                "Check the installation of the CUDA version of PyTorch/faster-whisper."
            )
        whisper_model = os.environ.get("VOICE_ENGINE_WHISPER_MODEL", "").strip()
        if not whisper_model:
            for candidate in (
                PROJECT_DIR / "models" / "whisper_small",
                Path("/home/unitree/AgroBot/models/whisper_small"),
                Path("/home/unitree/agrobot/AgroHub/models/whisper_small"),
            ):
                if (candidate / "model.bin").exists():
                    whisper_model = str(candidate)
                    break
        if not whisper_model:
            whisper_model = "small"
            log.warning(
                "Local faster-whisper model not found; model_size='small' "
                "may attempt to use internet/cache HuggingFace. For "
                "fully offline mode, set VOICE_ENGINE_WHISPER_MODEL."
            )

        self.stt = STTEngine(
            model_size=whisper_model,
            device=stt_device,
            compute_type=stt_compute_type,
            download_root=str(PROJECT_DIR / "models" / "faster_whisper"),
        )
        self.cache = SemanticCache(
            index_path=PROJECT_DIR / "semantic_cache_index.faiss",
            mapping_path=PROJECT_DIR / "semantic_cache_mapping.pkl",
            similarity_threshold=0.8,
        )
        self.kb_cache = SemanticCache(
            index_path=PROJECT_DIR / "kb_index.faiss",
            mapping_path=PROJECT_DIR / "kb_mapping.pkl",
            similarity_threshold=0.8,
        )
        
        llm_model_path = os.environ.get(
            "VOICE_ENGINE_LLM_MODEL",
            str(PROJECT_DIR / "models" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
        )
        self.llm = LLMEngine(
            model_path=llm_model_path,
            enable_context=ENABLE_CONTEXT,
            max_history_turns=MAX_HISTORY_TURNS,
            n_gpu_layers=-1,
        )
        self.tts = TTSEngine(
            warmup_on_init=False,   # warmup will be done in _warmup_all
            network_interface="eth0",
        )
        self.tts.validate_languages(self.stt.supported_tts_languages())
        self.gestures = G1GestureController(enabled=ENABLE_GESTURES, log=log)

        # --- Pool thead for blocking ML operations ---
        # All calls to model.transcribe(), model.generate(), model.synthesize()
        # are executed through loop.run_in_executor(self.executor, ...) —
        # this allows the event loop not to hang on heavy computations.
        self.executor = ThreadPoolExecutor(max_workers=EXECUTOR_MAX_WORKERS)

        # --- Interrupt Flag ---
        # Set to True when the voice detector detects user speech
        # during the robot's response. Workers check this flag
        # in each iteration of their loop.
        self._interrupted = threading.Event()
        self._running = False
        self.operator_queue = asyncio.Queue()

        CACHE_WAV_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_ANALYZ_WAV_DIR.mkdir(parents=True, exist_ok=True)
        KB_CACHE_WAV_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Initialization and Warmup
    # -----------------------------------------------------------------------

    async def _warmup_all(self) -> None:
        """
        Warmup of ML engines is performed sequentially.

        On Jetson, parallel warmup of CTranslate2/Whisper and TTS may
        hang on CUDA/runtime locks. Sequential warmup
        takes a few seconds longer at startup, but cannot
        hang the launch and does not affect SLA after the end of the phrase.
        """
        loop = asyncio.get_event_loop()

        log.info("--> Warmup of engines...")

        warmups = (
            ("STT", self.stt.warmup, 20.0),
            ("CACHE", self.cache.warmup, 10.0),
            ("TTS", self.tts.warmup, 15.0),
        )
        for name, fn, timeout_s in warmups:
            start = time.perf_counter()
            try:
                await asyncio.wait_for(loop.run_in_executor(self.executor, fn), timeout=timeout_s)
                log.info("--> Warmup %s completed in %.2f s", name, time.perf_counter() - start)
            except asyncio.TimeoutError:
                log.error("Warmup %s exceeded %.1f s — continuing startup without waiting", name, timeout_s)
            except Exception as exc:
                log.error("Warmup %s error: %s — continuing startup", name, exc)

        log.info("--> All engines are ready for work")

    # -----------------------------------------------------------------------
    # Helper async wrappers around blocking calls
    # -----------------------------------------------------------------------

    async def _run_stt(self, audio_data) -> "tuple":
        """
        Transcribes audio to text in a separate thread. 

        Returns a Tuple[str, str] = (text, language_code_for_xtts) — just like
        STTEngine.transcribe(). audio_data is a numpy.ndarray (float32, mono,
        16kHz)—the format returned by AudioListener.listen_for_phrase()—rather than bytes.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.stt.transcribe,
            audio_data,
            self.stt_lang,
        )

    async def _run_tts(self, sentence: str, lang_code: str = "ru") -> str:
        import uuid, tempfile
        loop = asyncio.get_event_loop()
        tmp_path = str(Path(tempfile.gettempdir()) / f"tts_{uuid.uuid4().hex}.wav")
        await loop.run_in_executor(
            self.executor,
            self.tts.synthesize,   # synthesize(text, lang_code, output_path)
            sentence,
            lang_code,
            tmp_path,
        )
        return tmp_path

    async def _run_cache_search(self, text: str, lang_code: str):
        """Searches for an answer in the semantic cache."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.cache.search,
            text,
            lang_code,
        )

    async def _run_cache_put(self, query: str, answer: str, wav_path: str, lang_code: str) -> None:
        """Saves a (query, answer, wav) pair to the cache."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self.cache.put,
            query,
            answer,
            wav_path,
            lang_code,
        )

    async def _play(self, wav_path: str) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self.player.play,
            wav_path,
        )

    # -----------------------------------------------------------------------
    # Worker-1: audio generation (LLM + TTS)
    # -----------------------------------------------------------------------

    async def _worker_tts(
        self,
        text: str,
        lang_code: str,
        wav_queue: asyncio.Queue,
        collected_sentences: List[str],
        custom_generator = None,
        detected_gestures: List[str] = None,
        uncensored: bool = False,
    ) -> None:
        """
        Worker-1: “Producer” in the Producer-Consumer scheme. 

        Logic: 
        1. Receives a stream of proposals from LLMEngine (sync generator, 
        launched in the executor so as not to block the event loop). 
        2. Each sentence is sent to TTS → receives wav_path. 
        3. Puts wav_path in asyncio.Queue (unlimited queue, 
        because worker player reads faster than TTS generates). 
        4. Upon completion, places a sentinel object (_QUEUE_SENTINEL), 
        so that Worker-2 knows that there will be no more files. 
        5. When the _interrupted flag is set, it stops working immediately 
        and signals Worker-2 via sentinel. 

        Options 
        --------- 
        text : transcribed user request 
        wav_queue : asyncio.Queue to pass wav_path → Worker-2 
        collected_sentences : list for collecting sentences (needed for 
        final merging wav and saving to cache)
        """
        loop = asyncio.get_event_loop()

        try:

            gen = custom_generator if custom_generator is not None else self.llm.generate_stream(text, lang_code, uncensored=uncensored)

            while True:
                # --- Interrupt check ---
                if self._interrupted.is_set():
                    log.info("⚡ Worker-TTS: прерывание обнаружено, останавливаемся")
                    break

                sentence = await loop.run_in_executor(
                    self.executor,
                    _safe_next, 
                    gen,
                )

                if sentence is None:
                    log.debug("Worker-TTS: генератор LLM завершён")
                    break

                sentence = sentence.strip()
                if not sentence:
                    continue

                # --- Artifact Filter for XTTS ---
                sentence = re.sub(r'\.{2,}', ',', sentence)
                sentence = re.sub(r'[*_~"«»]', '', sentence)

                if lang_code not in ("zh", "zh-cn", "ja", "ko"):
                    sentence = sentence.replace('。', '.').replace('！', '!').replace('？', '?').replace('，', ',')
                    sentence = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', '', sentence)

                # --- Intercept gesture tags ---
                gesture_match = GESTURE_REGEX.search(sentence)
                g_name_to_pass = None

                if gesture_match:
                  raw_g_name = gesture_match.group(1).strip()
                  g_name = normalize_gesture_name(raw_g_name)
                  g_name_to_pass = g_name

                  if detected_gestures is not None and not detected_gestures:
                    detected_gestures.append(g_name)

                sentence = GESTURE_REGEX.sub('', sentence).strip()
                if not sentence:
                    continue
                # ----------------------------------

                log.info("--> LLM → TTS: «%s»", sentence)
                collected_sentences.append(sentence)

                try:
                    wav_path = await self._run_tts(sentence, lang_code)
                except Exception as exc:
                    log.error("TTS ошибка для «%s»: %s", sentence, exc)
                    continue  
                
                await wav_queue.put((wav_path, g_name_to_pass)) 
                log.debug("Worker-TTS: wav помещён в очередь (%s)", wav_path)

        except asyncio.CancelledError:
            log.info("Worker-TTS: задача отменена")
            raise

        except Exception as exc:
            log.exception("Worker-TTS: необработанное исключение: %s", exc)

        finally:
            # --- Send the sentinel regardless ---
            # This is CRITICAL: Worker-2 is waiting for the sentinel to exit its loop. 
            # Without the sentinel, Worker-2 will hang on queue.get() forever.
            await wav_queue.put(_QUEUE_SENTINEL)
            log.debug("Worker-TTS: sentinel отправлен в очередь")

    # -----------------------------------------------------------------------
    # Worker-2: audio playback
    # -----------------------------------------------------------------------

    async def _worker_player(
        self,
        wav_queue: asyncio.Queue,
        played_wavs: List[str],
    ) -> None:
        """
        Worker-2: The "Consumer" in the Producer-Consumer pattern. 

        Logic:
        1. Continuously reads from the asyncio.Queue. 
        2. If it receives _QUEUE_SENTINEL, it exits the loop (thread/task complete). 
        3. If it receives a .wav path, it plays it immediately. 
        4. Stores the path in played_wavs for subsequent merging. 
        5. If the _interrupted flag is set, it stops playback
        and exits without waiting for the sentinel
        (the sentinel will arrive anyway from Worker-1's finally block). 

        Key characteristics:
        - Worker-2 runs in parallel with Worker-1 via asyncio.gather(). 
        - While Worker-1 is synthesizing the second .wav, Worker-2 is already playing the first. 
        - queue.get() is a coroutine that "parks" Worker-2 without blocking
        the event loop while the queue is empty. 

        Parameters
        ----------
        wav_queue   : asyncio.Queue from which wav_path or sentinel is read
        played_wavs : list for accumulating paths (for the final merge)
        """
        try:
            while True:
                # --- Interrupt check --- 
                if self._interrupted.is_set():
                    log.info("--> Worker-Player: interruption, stopping playback")
                    _drain_queue(wav_queue)
                    break

                try:
                    item = await asyncio.wait_for(wav_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                # --- Checking: is it sentinel or wav_path ---
                if item is _QUEUE_SENTINEL:
                    log.debug("Worker-Player: Sentinel received, terminating the loop.")
                    wav_queue.task_done()
                    break

                wav_path, g_name = item
                
                if g_name:
                    self.gestures.start(g_name) 
                
                log.info("--> Playing: %s", wav_path)
                played_wavs.append(wav_path)

                try:
                    await self._play(wav_path)
                except Exception as exc:
                    log.error("AudioPlayer ошибка: %s", exc)
                finally:
                    wav_queue.task_done()

        except asyncio.CancelledError:
            log.info("Worker-Player: задача отменена")
            raise

        except Exception as exc:
            log.exception("Worker-Player: необработанное исключение: %s", exc)

    # -----------------------------------------------------------------------
    # Interrupt detector (runs in parallel with the pipeline)
    # -----------------------------------------------------------------------

    async def _interruption_watchdog(
        self,
        stop_event: asyncio.Event,
    ) -> None:
        """
        A parallel "watchdog" coroutine that listens to the microphone while
        the robot is responding and sets `_interrupted` upon detecting a voice. 

        It uses `AudioListener.check_interrupt()`—a lightweight method
        that returns `True` as soon as a simple energy detector
        registers sustained speech, without fully recording the utterance. 

        IMPORTANT: Previously, `listen_for_phrase()` was called here—the same method
        used in the main listening loop. However, `listen_for_phrase()`
        specifically pauses capture while `self.state.is_speaking` is true
        (i.e., while the robot is speaking)—which is precisely when the
        watchdog needs to be active. Consequently, the interruption mechanism
        never triggered. `check_interrupt()` is a separate method that reads
        the microphone specifically during the robot's response (see
        `audio_io.py` for details and caveats). 

        The `stop_event` is set by the main coroutine after the pipeline
        completes, ensuring the watchdog does not run indefinitely.
        """
        loop = asyncio.get_event_loop()

        try:
            while not stop_event.is_set():
                detected = await loop.run_in_executor(
                    self.executor,
                    self.listener.check_interrupt,
                )

                if detected:
                    log.info("--> Прерывание: обнаружена речь пользователя")
                    self._interrupted.set()
                    break

                await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            pass

    # -----------------------------------------------------------------------
    # Operator server (Wizard of Oz)
    # -----------------------------------------------------------------------
    async def _operator_handler(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        import json
        
        try:
            data = await reader.readline()
            if not data:
                return
                
            payload = json.loads(data.decode('utf-8').strip())
            text = payload.get("text", "").strip()
            lang = payload.get("lang", "ru").strip()
        except json.JSONDecodeError as exc:
            log.error("Failed to parse operator JSON: %s", exc)
            text = ""
            lang = "ru"
        
        if text:
            if text.lower() == "/stop":
                log.info("--> OPERATOR: Forced stop of current speech")
                self._interrupted.set()
            else:
                log.info("--> OPERATOR: Phrase added to queue (%s): «%s»", lang, text)
                await self.operator_queue.put({"text": text, "lang": lang})
                
        writer.close()
        await writer.wait_closed()

    # -----------------------------------------------------------------------
    # Streaming pipeline (cache miss)
    # -----------------------------------------------------------------------

    async def _run_streaming_pipeline(self, query: str, lang_code: str = "ru", custom_generator = None, uncensored: bool = False) -> None:
        """
        Launches a two-worker streaming pipeline to generate the response. 

        Queue workflow
        --------------

        wav_queue (asyncio.Queue)
            │
        Worker-TTS ──put()──►──get()── Worker-Player
        (Producer)                      (Consumer)
            │                              │
        TTS.synthesize()            AudioPlayer.play()
            │                              │
        wav_path                     played_wavs[]

        Sentinel pattern:
        Worker-TTS puts _QUEUE_SENTINEL after the last wav
        (or upon interruption/error—from the finally block). 
        Worker-Player exits the loop upon receiving the sentinel. 
        This guarantees that Worker-Player ALWAYS terminates. 

        After both workers finish:
        - Check if an interruption occurred. 
        - If not, concatenate all wavs into a single final file. 
        - Save to SemanticCache.
        """
        self._interrupted.clear()

        collected_sentences: List[str] = [] 
        played_wavs: List[str] = []        
        detected_gestures: List[str] = []

        # asyncio.Queue — the primary communication channel between workers. 
        # maxsize=0 means unlimited size. This is safe because:
        #   a) TTS is slower than the player (items won't pile up)
        #   b) We want TTS not to wait for the player (full overlap)
        wav_queue: asyncio.Queue = asyncio.Queue()

        watchdog_stop = asyncio.Event()
 
        tts_task = asyncio.ensure_future(
            self._worker_tts(query, lang_code, wav_queue, collected_sentences, custom_generator, detected_gestures, uncensored)
        )
        
        watchdog_task = None
        if ENABLE_INTERRUPT:
            watchdog_task = asyncio.ensure_future(
                self._interruption_watchdog(watchdog_stop)
            )

        # If audio output is disabled, we launch dummy_player, which simply pulls WAVs from the queue;
        # if enabled, we launch the original player_task.
        if ENABLE_PLAYBACK:
            player_task = asyncio.ensure_future(self._worker_player(wav_queue, played_wavs))
        else:
            async def _dummy_player():
                while True:
                    item = await wav_queue.get()
                    if item is _QUEUE_SENTINEL:
                        wav_queue.task_done()
                        break
                    
                    wav_path, g_name = item
                    if g_name:
                        self.gestures.start(g_name)
                        
                    wav_queue.task_done()
            player_task = asyncio.ensure_future(_dummy_player())

        try:
            await asyncio.gather(tts_task, player_task)

        except asyncio.CancelledError:
            tts_task.cancel()
            player_task.cancel()
            await asyncio.gather(tts_task, player_task, return_exceptions=True)
            raise

        finally:
            if watchdog_task is not None:
                watchdog_stop.set()
                watchdog_task.cancel()
                await asyncio.gather(watchdog_task, return_exceptions=True)

        # --- Post-processing ---

        if self._interrupted.is_set():
            log.info("Конвейер прерван — пропускаем сохранение в кэш")
            _cleanup_temp_wavs(played_wavs)
            return

        if not played_wavs:
            log.warning("Конвейер завершён, но wav-файлов нет — ничего не сохраняем")
            return

        # --- Merging WAV files ---
        full_answer = " ".join(collected_sentences)
        
        cache_text = full_answer
        if detected_gestures:
            cache_text = f"[жест: {detected_gestures[0]}] {full_answer}"

        if self.is_analiz:
            final_wav_path = str(CACHE_ANALYZ_WAV_DIR / f"{uuid.uuid4().hex}.wav")
        else:
            final_wav_path = str(CACHE_WAV_DIR / f"{uuid.uuid4().hex}.wav")

        try:
            merge_wav_files(played_wavs, final_wav_path)
            log.info("Финальный wav сохранён: %s", final_wav_path)
        except Exception as exc:
            log.error("Ошибка слияния wav: %s", exc)
            _cleanup_temp_wavs(played_wavs)
            return

        # --- Caching ---
        if self.is_analiz:
            log.info("Анализ не сохраняем в кэш")
            self.is_analiz = False
        else:   
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    self.cache.put,
                    query,
                    cache_text,
                    final_wav_path,
                    lang_code,
                )
                log.info("--> Записано в кеш: «%s...»", cache_text[:60])
            except Exception as exc:
                log.error("Ошибка записи в кэш: %s", exc)

        _cleanup_temp_wavs(played_wavs)

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    async def run(self) -> None:
        """
        Voice assistant main loop. 

        Workflow for each iteration:
        1. listen()       — blocking recording of the utterance
        2. STT            — transcription
        3. cache.search() — semantic cache lookup
        4a. cache hit    → play(cached_wav)
        4b. cache miss   → _run_streaming_pipeline()
        """
        await self._warmup_all()

        self._running = True
        self.listener.start()
        log.info("--> Assistant started. Press Ctrl+C to exit..")

        loop = asyncio.get_event_loop()

        # =========================================================
        # --- Kuzmich's opening phrase ---
        # =========================================================
        startup_path = str(PROJECT_DIR / "startup.wav")
        if not os.path.exists(startup_path):
            log.info("Synthesizing the starting phrase...")
            startup_text = "Ну всё, лампы прогрелись, моторы смазаны. Чего стоим? Я готов, говорите, только четко и не бормочите."
            try:
                await loop.run_in_executor(
                    self.executor,
                    self.tts.synthesize,
                    startup_text,
                    "ru",
                    startup_path
                )
            except Exception as exc:
                log.error("Error synthesizing the starting phrase: %s", exc)

        if os.path.exists(startup_path):
            log.info("Playing the starting phrase...")
            try:
                await self._play(startup_path)
            except Exception as exc:
                log.error("Error playing the starting phrase: %s", exc)

        # =========================================================
        # --- Kuzmich's scanning phrase ---
        # =========================================================
        scanning_path = str(PROJECT_DIR / "scanning.wav")
        if not os.path.exists(scanning_path):
            log.info("Synthesizing the scanning phrase...")
            scanning_text = "Так, минуточку, навожу резкость на своих старых оптических датчиках... Сканирую обстановку!"
            try:
                await loop.run_in_executor(
                    self.executor,
                    self.tts.synthesize,
                    scanning_text,
                    "ru",
                    scanning_path
                )
            except Exception as exc:
                log.error("Error synthesizing the scanning phrase: %s", exc)
        # =========================================================

        operator_server = await asyncio.start_server(self._operator_handler, '127.0.0.1', 9999)
        asyncio.create_task(operator_server.serve_forever())
        log.info("--> The operator console server is running on port 9999")

        try:
            queue_task = asyncio.create_task(self.operator_queue.get())
            listen_task = None
            counter = 0

            while self._running:
                if counter == 10:
                    self.llm.clear_history()
                    counter = 0
                else:
                    counter += 1
                # ── Step 1: Competitive Waiting (Microphone OR Remote Control) ────────────
                if listen_task is None or listen_task.done():
                    log.info("--> Listening...")
                    listen_task = loop.run_in_executor(
                        self.executor, self.listener.listen_for_phrase
                    )

                if queue_task.done():
                    queue_task = asyncio.create_task(self.operator_queue.get())
                    
                done, pending = await asyncio.wait(
                    [listen_task, queue_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # ── Scenario A: Remote control button pressed ──
                if queue_task in done:
                    op_data = queue_task.result()
                    op_text, op_lang = op_data["text"], op_data["lang"]
                    
                    log.info("--> Operator phrase voiceover (%s): «%s»", op_lang, op_text)
                    # if ENABLE_CONTEXT:
                    #     self.llm.chat_history.append({"role": "assistant", "content": op_text})
                    
                    def op_gen(text=op_text): yield text
                    try:
                        self.is_analiz = True
                        await self._run_streaming_pipeline("operator", op_lang, custom_generator=op_gen())
                    except Exception as exc:
                        log.error("Error synthesizing the operator phrase: %s", exc)
                    
                    continue

                # ── Scenario B: Microphone triggered (someone said something) ──
                try:
                    audio_data = listen_task.result()
                except Exception as exc:
                    log.error("AudioListener error: %s", exc)
                    await asyncio.sleep(0.5)
                    continue

                if audio_data is None or len(audio_data) == 0:
                    log.debug("Audio is empty — skipping")
                    continue

                # ── Step 2: Speech-to-Text ────────────────────────────────
                try:
                    text, lang_code = await self._run_stt(audio_data)
                except Exception as exc:
                    log.error("STT error: %s", exc)
                    continue

                if not text or not text.strip():
                    log.debug("STT returned an empty string — skipping")
                    continue

                query = text.strip()

                # --- WHISPER ANTI-HALLUCINATIONS ---
                query_lower = query.lower()
                
                hallucinations = [
                    "субтитры", "dima", "torzok", "продолжение следует",
                    "amara", "редактор", "перевод", "озвучено", "смотреть до конца",
                    "obrigado", "gracias", 
                ]
                exact_hallucinations = [
                    "okay", "mm-hmm", "yeah", "you", "thank you.",
                    "thank you", "thanks", "окей", "да", "нет", "gracias.", 
                    "gracias", "obrigado.", "obrigado", "bye", "bye.", "i'm sorry", "i'm sorry.",
                ]
                
                query_lower = query.lower()
                valid_short_words = {"да", "нет", "ну", "ок", "ага", "yes", "no", "hi", "hey", "yo", "ok"} 

                if (any(h in query_lower for h in hallucinations) or 
                (query_lower in exact_hallucinations) or 
                (len(query) < 3 and query_lower not in valid_short_words)):
                    log.info("Picked up a Whisper hallucination: «%s» — ignoring", query)
                    continue
                # ---------------------------------

                log.info("--> Recognized (%s): «%s»", lang_code, query)

                 # ── Trigger processing ───────────────────

                clean_words = set(re.findall(r'\b\w+\b', query.lower()))

                if USE_TRIGGERS:
                    # ── "Aunt Vasya" Secret Protocol (only ru) ───────────────────
                    query_lower_full = query.lower()
                    
                    if "тётя вася" in query_lower_full or "тетя вася" in query_lower_full:
                        if not self.uncensored_mode:
                            log.warning("--> 🔓 'Auntie Vasya' Protocol activated permanently.!")
                            self.uncensored_mode = True
                            self.llm.clear_history()
                        
                        self.is_analiz = True
                        
                        query = re.sub(r'(?i)т[её]тя вася[, ]*', '', query).strip()
                        if not query:
                            query = "Скажи что-нибудь от души."
                            
                        try:
                            await self._run_streaming_pipeline(query, "ru", uncensored=True)
                        except Exception as exc:
                            log.error("Error in 'Aunt Vasya' mode: %s", exc)
                        
                        continue
                    # ── Deactivation of the "Aunt Vasya" secret protocol (only ru) ───────────────────
                    disable_vasya_triggers = ["дядя вася", "отбой протокола", "дядя вайсе"]
                    if self.uncensored_mode and any(h in query_lower_full for h in disable_vasya_triggers):
                        log.warning("--> 🔒 'Auntie Vasya' Protocol deactivated. Returning to normal mode.")
                        self.uncensored_mode = False
                        self.llm.clear_history()
                        self.is_analiz = True
                        
                        def back_to_normal_gen():
                            yield "[жест: facepalm] Ох, батюшки... Что это на меня нашло?"
                            yield "Простите, магнитные бури, видимо. Я снова с вами!"
                            
                        try:
                            await self._run_streaming_pipeline(query, "ru", custom_generator=back_to_normal_gen())
                        except Exception as exc:
                            log.error("Error when disabling mode: %s", exc)
                        continue
                    # ────────────────────────────────────────────────────────
                    only_ru_words = [
                        "балалайка", "достоевский", "лалалайка", "ла-ла-лайка",
                    ]
                    if any(h in clean_words for h in only_ru_words):
                        log.info("--> Command ONLY RU. Transitioning to only ru mode.")
                        self.llm.clear_history()
                        self.stt_lang = "ru"
                        continue
                    all_lang_words = [
                        "мультиязычность", "рахманинов",
                    ]
                    if any(h in clean_words for h in all_lang_words):
                        log.info("--> Command ALL LANG. Transitioning to all lang mode.")
                        self.llm.clear_history()
                        self.stt_lang = None
                        continue
                    stop_words = [
                        "stop", "top", "стоп", "топ",
                    ]
                    if any(h in clean_words for h in stop_words):
                        log.info("--> Command STOP. Transitioning to sleep mode.")
                        self.llm.clear_history()
                        self.is_awake = False
                        continue
                    hello_words = [
                        "hello", "привет", "hi", "what's up",
                    ]
                    if any(h in clean_words for h in hello_words):
                        log.info("--> Command HELLO. Transitioning to active mode.")
                        self.is_awake = True
                        self.is_analiz = True 

                        def greeting_gen():
                            if lang_code == "en":
                                yield "[жест: moet_glaza] Greetings, dear guest!"
                                yield "How is your day going? I am glad to see you here!"
                            else:
                                yield "[жест: moet_glaza] Здравствуйте, уважаемый гость!"
                                yield "Как проходит ваш день? Я рад видеть вас здесь!"

                        try:
                            await self._run_streaming_pipeline(query, lang_code, custom_generator=greeting_gen())
                        except Exception as exc:
                            log.error("Error occurred while playing the greeting: %s", exc)
                        continue

                    bye_words = [
                        "пока", "bye", "goodbye", "прощай", "до встречи",
                    ]
                    if any(h in clean_words for h in bye_words):
                        log.info("--> Command BYE. Starting greeting bye with gestures.")
                        self.is_analiz = True 

                        def greeting_gen():
                            if lang_code == "en":
                                yield "[жест: moet_glaza] Goodbye, dear guest!"
                                yield "I wish you a wonderful time here!"
                            else:
                                yield "[жест: moet_glaza] До встречи, уважаемый гость!"
                                yield "Желаю вам хорошего времяпрепровождения!"

                        try:
                            await self._run_streaming_pipeline(query, lang_code, custom_generator=greeting_gen())
                        except Exception as exc:
                            log.error("Error occurred while playing the greeting: %s", exc)
                        continue

                    heart_words = [
                        "сделаем фото", "сделаем селфи", "сделаем сердечко", "сердечко вместе", "сердечко один", "пол сердца", "половина сердца", "половина сердечка", "пол сердечка",
                        "make a photo", "make a selfie", "make a heart", "heart together", "heart one",
                    ]
                    if any(h in clean_words for h in heart_words):
                        log.info("--> Command HEART. Starting greeting heart with gestures.")
                        self.is_analiz = True 

                        def greeting_gen():
                            if lang_code == "en":
                                yield "[жест: rightheart] Let's take a photo, dear guest!"
                                yield "Please smile and complete my heart!"
                            else:
                                yield "[жест: rightheart] Давайте сделаем фото, уважаемый гость!"
                                yield "Пожалуйста, улыбнитесь и дополните мое сердечко!"

                        try:
                            await self._run_streaming_pipeline(query, lang_code, custom_generator=greeting_gen())
                        except Exception as exc:
                            log.error("Error occurred while playing the greeting: %s", exc)
                        continue

                    shakehands_words = [
                        "поздоровайся", "пожми руку", "handshake", "протяни руку", "рукопожатие", 
                        "shakehands", "shake", "shakehand", "hand-check", "вожми руку", 
                    ]
                    if any(h in clean_words for h in shakehands_words):
                        log.info("--> Command HANDSHAKE. Starting greeting handshake with gestures.")
                        self.is_analiz = True 

                        def greeting_gen():
                            if lang_code == "en":
                                yield "[жест: shakehands] Greetings, dear guest!"
                                yield "My name is Kuzmich, and what is yours?"
                            else:
                                yield "[жест: shakehands] Здравствуй, уважаемый гость!"
                                yield "Меня зовут Кузьмич, а тебя как?"

                        try:
                            await self._run_streaming_pipeline(query, lang_code, custom_generator=greeting_gen())
                        except Exception as exc:
                            log.error("Error occurred while playing the greeting: %s", exc)
                        continue
                    comision_words = [
                        "представься комиссии", "ставься комиссии", "поздоровайся с комиссией", "представься комиссии.", 
                        "перед тобой комиссия", "здоровайся с комиссией", "тобой комиссия",
                    ]
                    if any(h in clean_words for h in comision_words):
                        log.info("--> Command GREETING. Starting greeting speech with gestures.")
                        
                        self.is_analiz = True 
                        
                        def greeting_gen():
                            yield "[жест: tolk] Здравствуйте, уважаемые представители Министерства сельского хозяйства!"
                            yield "Меня зовут Кузьмич,и я — главный «цифровой сотрудник» команды студентов агрохакатона на Истринской сыроварне."
                            yield "Пока мои создатели изучают ремесленные традиции сыроделия, я занимаюсь тем, что умею лучше всего:"
                            yield "анализирую процессы, автоматизирую рутину, собираю данные и помогаю принимать точные технологические решения."
                            yield "Мы здесь, чтобы показать, как современные агротехнологии могут работать рука об руку с вековыми традициями."
                            yield "А я — живое доказательство того"
                            yield "что будущее сельского хозяйства — за умными машинами и людьми, которые умеют ими управлять."
                            yield "Спасибо, что вы с нами. Мы готовы к диалогу!"

                        try:
                            await self._run_streaming_pipeline(query, lang_code, custom_generator=greeting_gen())
                        except Exception as exc:
                            log.error("Error occurred while playing the greeting: %s", exc)
                            
                        continue                            

                if not self.is_awake:
                    log.debug("Sleep mode. Ignoring: %s", query)
                    continue

                analyze_words = [
                    "анализ", "analyze", "analysis", "analice",
                    "alice", "analize", "analyce", "аналис", 
                    "аналайс", "аналайз",
                ]
                if any(h in clean_words for h in analyze_words) and analyze_for_tts is not None:
                    self.is_analiz = True
                    log.info("--> Launching scene analysis...")

                    scanning_path = str(PROJECT_DIR / "scanning.wav")
                    if os.path.exists(scanning_path):
                        log.info("--> Playing scanning notification...")
                        if ENABLE_GESTURES:
                            self.gestures.start("self_prez_right")
                            
                        asyncio.create_task(self._play(scanning_path))

                    try:
                        out_path = await loop.run_in_executor(self.executor, analyze_for_tts)
                        with open(out_path, "r", encoding="utf-8") as f:
                            scene_desc = f.read().strip()
                        
                        query = f"Пользователь попросил анализ сцены. Данные с твоих камер: {scene_desc}. Кратко расскажи, что ты видишь, от своего лица."
                        log.info("--> Vision data received: %s", scene_desc)
                    except Exception as exc:
                        log.error("Error in VLM analysis: %s", exc)
                        query = "Пользователь попросил анализ, но твоя камера не отвечает. Пошути на тему сломанных советских датчиков."

                # ── Step 3: Search in semantic caches ───────────
                cache_result = None
                is_kb_hit = False

                if self.uncensored_mode:
                    log.info("'Aunt Vasya' mode: skip cache lookup.")
                    self.is_analiz = True
                else:
                    clean_query = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', query).strip()

                    try:
                        cache_result = await loop.run_in_executor(
                            self.executor, self.kb_cache.search, query, None
                        )
                        if not cache_result and clean_query != query:
                            cache_result = await loop.run_in_executor(
                                self.executor,
                                self.kb_cache.search,
                                clean_query,
                                None,
                            )
                        if cache_result:
                            is_kb_hit = True
                    except Exception as exc:
                        log.error('Error searching in kb_cache: %s', exc)

                    if not cache_result and (
                        not ENABLE_CONTEXT or not self.llm.chat_history
                    ):
                        try:
                            cache_result = await self._run_cache_search(
                                query, lang_code
                            )
                        except Exception as exc:
                            log.error('SemanticCache search error: %s', exc)

                    if cache_result is not None and not self.is_analiz:
                        cached_wav = (
                            cache_result.get('audio_path')
                            if isinstance(cache_result, dict)
                            else getattr(cache_result, 'audio_path', cache_result)
                        )
                        response_text = (
                            cache_result.get('response_text', '')
                            if isinstance(cache_result, dict)
                            else ''
                        )
                    
                        cached_lang = (
                            cache_result.get('language')
                            if isinstance(cache_result, dict)
                            else getattr(cache_result, 'language', None)
                        ) or "ru"

                        if cached_lang != lang_code:
                            log.info("Found an answer in the database for '%s', but the question was asked in '%s'. Translating....", cached_lang, lang_code)
                        
                            translation_prompt = (
                                f"Гость задал вопрос: «{query}». "
                                f"В нашей базе знаний есть точный ответ: «{response_text}». "
                                f"Переведи и озвучь этот ответ гостю СТРОГО на языке '{lang_code}'. "
                                f"Будь предельно вежлив, гостеприимен и обращайся к гостю уважительно. "
                                f"ОБЯЗАТЕЛЬНО начни свой ответ с тега жеста в строгом формате [жест: ИМЯ]."
                            )
                        
                            try:
                                await self._run_streaming_pipeline(translation_prompt, lang_code)
                            except Exception as exc:
                                log.error("Error generating translation from the database: %s", exc)
                            
                            self.llm.clear_history()
                            continue

                        if cached_wav:
                            if is_kb_hit:
                                import random
                                kb_gesture = random.choice([
                                    "self_prez_left", 
                                    "self_prez_right", 
                                    "speaker_left", 
                                    "speaker_right"
                                ])
                                if ENABLE_GESTURES:
                                    self.gestures.start(kb_gesture)
                                log.info('KB Cache hit! Selected random gesture: %s', kb_gesture)
                            else:
                                if response_text:
                                    gesture_match = GESTURE_REGEX.search(response_text)
                                    if gesture_match:
                                        raw_g_name = gesture_match.group(1).strip()
                                        g_name = normalize_gesture_name(raw_g_name)
                                        if g_name and ENABLE_GESTURES:
                                            self.gestures.start(g_name)

                        log.info('Cache hit! Playing: %s', cached_wav)
                        try:
                            await self._play(cached_wav)
                        except Exception as exc:
                            log.error('Error playing cache: %s', exc)

                        self.llm.clear_history()
                        continue

                # ── Step 4: Streaming pipeline (cache miss) ────────────
                log.info("Cache miss, launching LLM→TTS→Player pipeline")

                try:
                    await self._run_streaming_pipeline(query, ("ru" if self.uncensored_mode else lang_code), uncensored=self.uncensored_mode)
                except asyncio.CancelledError:
                    log.info("Pipeline cancelled")
                    raise
                except Exception as exc:
                    log.exception("Error in streaming pipeline: %s", exc)

        except (KeyboardInterrupt, asyncio.CancelledError):
            log.info("Signal received to stop")
        finally:
            self._running = False
            self.executor.shutdown(wait=False)
            self.listener.close()
            log.info("Assistant stopped")


# ===========================================================================
# Helper functions (module-level, not class methods)
# ===========================================================================

def _safe_next(generator):
    """
    Safely call next() on a synchronous generator. 

    Returns the next value, or None upon StopIteration. 
    Used in run_in_executor, as StopIteration cannot
    propagate out of the executor in Python 3.7+.
    """
    try:
        return next(generator)
    except StopIteration:
        return None


def _drain_queue(queue: asyncio.Queue) -> None:
    """
    Immediately drains the asyncio.Queue without waiting (non-blocking drain). 

    Called upon interruption to "drain" remaining WAV paths
    and the sentinel from the queue, preventing resource leaks.
    """
    while not queue.empty():
        try:
            queue.get_nowait()
            queue.task_done()
        except asyncio.QueueEmpty:
            break


def _cleanup_temp_wavs(wav_paths: List[str]) -> None:
    """
    Deletes temporary .wav files created by TTS for individual sentences. 
    The final concatenated file is NOT deleted (it is saved in CACHE_WAV_DIR).
    """
    for path in wav_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                log.debug("Deleted temporary file: %s", path)
        except OSError as exc:
            log.warning("Failed to delete %s: %s", path, exc)


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    """
    Entry point. Creates the event loop and starts the VoiceAssistant. 

    Python 3.8 compatibility:
    - asyncio.run() is available since 3.7; we use it. 
    - asyncio.to_thread() is 3.9+ only, so we use
    loop.run_in_executor() with an explicit ThreadPoolExecutor throughout. 
    - On Python 3.8, Windows defaults to ProactorEventLoop
    (required for subprocesses), while Unix uses SelectorEventLoop. 
    This usually doesn't matter for audio, but we do not explicitly set the policy
    to avoid breaking platform-specific defaults.
    """
    assistant = VoiceAssistant()

    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
