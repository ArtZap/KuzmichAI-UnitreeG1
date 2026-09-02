"""
main.py — Финальный асинхронный пайплайн голосового ассистента.

Архитектура:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  AudioListener ──► STT ──► SemanticCache ──► AudioPlayer (кэш-хит)  │
  │                                │                                    │
  │                           (кэш-промах)                              │
  │                                ▼                                    │
  │              ┌─────── StreamingPipeline ─────────┐                  │
  │              │  LLMEngine.generate_stream()      │                  │
  │              │       │ (предложения)             │                  │
  │              │       ▼                           │                  │
  │              │  Worker-1: TTS → wav_path         │                  │
  │              │       │ put() в Queue             │                  │
  │              │       ▼                           │                  │
  │              │  Worker-2: get() → AudioPlayer    │                  │
  │              └───────────────────────────────────┘                  │
  │                                │                                    │
  │                    merge wavs → SemanticCache.put()                 │
  └─────────────────────────────────────────────────────────────────────┘

Совместимость: Python 3.8+
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
# Импорт модулей проекта
# ---------------------------------------------------------------------------
from audio_io import AudioListener, AudioPlayer, LocalAudioPlayer
from llm_engine import LLMEngine
from semantic_cache import SemanticCache
from stt_engine import STTEngine
from tts_engine import TTSEngine
from g1_greeting_gestures import G1GestureController
import re

# ---------------------------------------------------------------------------
# Конфигурация логирования (Цветная)
# ---------------------------------------------------------------------------
class ColorFormatter(logging.Formatter):
    """Кастомный форматтер для раскраски логов в консоли."""
    
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
# Функция анализа сцены
# ---------------------------------------------------------------------------

sys.path.append("/home/unitree/agrohub_cloud")
try:
    from analyze_for_tts import analyze_for_tts
except ImportError:
    analyze_for_tts = None
    log.warning("Модуль analyze_for_tts не найден. Команда 'Анализ' не будет работать.")

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

# Директория для хранения финальных wav-файлов ответов (для кэша)
CACHE_WAV_DIR = PROJECT_DIR / "cache_audio"
CACHE_ANALYZ_WAV_DIR = PROJECT_DIR / "cache_analyz_audio"
# Директория для хранения защищенных ответов из Базы Знаний
KB_CACHE_WAV_DIR = PROJECT_DIR / "kb_audio"

# Токен-«яд» (sentinel), который Worker-1 кладёт в очередь,
# чтобы сообщить Worker-2 об окончании потока.
# Использование объекта-синглтона гарантирует строгое сравнение по идентичности (is).
_QUEUE_SENTINEL = object()

# Максимальное число потоков в пуле (ML-модели тяжёлые, не делаем слишком много)
EXECUTOR_MAX_WORKERS = 4

# Режим работы с аудио:
#   "g1"    (по умолчанию) — микрофон и динамики настоящего робота G1
#            через UDP-мультикаст и DDS/AudioClient. Требует сеть робота.
#   "local" — микрофон и динамики ноутбука через sounddevice. Полностью
#            локально, сеть/подключение к роботу не нужны вообще.
# Переключается переменной окружения, например:
#   VOICE_ENGINE_AUDIO=local python main.py
AUDIO_MODE = os.environ.get("VOICE_ENGINE_AUDIO", "g1").strip().lower()
if AUDIO_MODE not in ("g1", "local"):
    log.warning("Неизвестное значение VOICE_ENGINE_AUDIO=%r, использую 'g1'.", AUDIO_MODE)
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
STT_LANG = None if STT_LANGUAGE_ENV == "auto" else STT_LANGUAGE_ENV
ENABLE_GESTURES = os.environ.get("VOICE_ENGINE_ENABLE_GESTURES", "true").strip().lower() == "true"

# ===========================================================================
# Вспомогательная функция: слияние wav-файлов
# ===========================================================================

def merge_wav_files(input_paths: List[str], output_path: str) -> None:
    """
    Склеивает несколько .wav файлов в один при помощи стандартного модуля `wave`.

    Все входные файлы ДОЛЖНЫ иметь одинаковые параметры:
      - число каналов (nchannels)
      - ширину сэмпла (sampwidth)
      - частоту дискретизации (framerate)

    Параметры
    ---------
    input_paths : список путей к временным .wav файлам в порядке воспроизведения
    output_path : путь к итоговому .wav файлу
    """
    if not input_paths:
        raise ValueError("merge_wav_files: список файлов пуст")

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
                        "Параметры файла %s не совпадают с эталоном — пропускаем",
                        path,
                    )
                    continue
                out_wav.writeframes(src.readframes(src.getnframes()))


# ===========================================================================
# Основной класс приложения
# ===========================================================================

class VoiceAssistant:
    """
    Оркестрирует весь жизненный цикл голосового ассистента:
      1. Инициализация и прогрев всех движков.
      2. Главный цикл прослушивания.
      3. Стриминговый конвейер LLM → TTS → Player с параллельными воркерами.
      4. Обработка прерываний (человек заговорил во время ответа).
      5. Сохранение результата в семантический кэш.
    """

    def __init__(self) -> None:
        # --- Движки ---
        from audio_io import SharedRobotState, AudioConfig
        self.shared_state = SharedRobotState()

        cfg = AudioConfig(vad_threshold=NOISE_THRESHOLD)

        local_mode = (AUDIO_MODE == "local")
        self.listener = AudioListener(self.shared_state, config=cfg, local_mode=local_mode)
        self.player = LocalAudioPlayer(self.shared_state) if local_mode else AudioPlayer(self.shared_state)
        log.info("🔊 Аудио-режим: %s", "ЛОКАЛЬНЫЙ (микрофон/динамики ноутбука)" if local_mode else "G1 (сеть робота)")

        self.is_awake = not USE_TRIGGERS
        self.is_analiz = False

        # STTEngine рассчитан на GPU (см. его докстринг: "device — строго
        # cuda", "compute_type — строго int8_float16"). 
        # Определяем доступность CUDA автоматически, с безопасным откатом
        # на CPU для отладки на ноутбуке без GPU (VOICE_ENGINE_AUDIO=local).
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
                "CUDA недоступна — STTEngine запускается на CPU (int8). "
                "На реальном роботе G1 это заметно медленнее, чем cuda/"
                "int8_float16, и не соответствует требованию быстрой речи. "
                "Проверьте установку CUDA-версии PyTorch/faster-whisper."
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
                "Локальная faster-whisper модель не найдена; model_size='small' "
                "может попытаться использовать интернет/кэш HuggingFace. Для "
                "полностью офлайн-режима задайте VOICE_ENGINE_WHISPER_MODEL."
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
        )
        self.kb_cache = SemanticCache(
            index_path=PROJECT_DIR / "kb_index.faiss",
            mapping_path=PROJECT_DIR / "kb_mapping.pkl",
        )
        
        llm_model_path = os.environ.get(
            "VOICE_ENGINE_LLM_MODEL",
            str(PROJECT_DIR / "models" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
        )
        self.llm = LLMEngine(
            model_path=llm_model_path,
            enable_context=ENABLE_CONTEXT,
            max_history_turns=MAX_HISTORY_TURNS,
        )
        self.tts = TTSEngine(
            warmup_on_init=False,   # прогрев будет через _warmup_all
            network_interface="eth0",
        )
        self.tts.validate_languages(self.stt.supported_tts_languages())
        self.gestures = G1GestureController(enabled=ENABLE_GESTURES, log=log)

        # --- Пул потоков для блокирующих ML-операций ---
        # Все вызовы model.transcribe(), model.generate(), model.synthesize()
        # выполняются через loop.run_in_executor(self.executor, ...) —
        # это позволяет event loop не зависать на тяжёлых вычислениях.
        self.executor = ThreadPoolExecutor(max_workers=EXECUTOR_MAX_WORKERS)

        # --- Флаг прерывания ---
        # Устанавливается в True, когда детектор голоса фиксирует речь
        # пользователя ВО ВРЕМЯ ответа робота. Воркеры проверяют этот флаг
        # в каждой итерации своего цикла.
        self._interrupted = threading.Event()
        self._running = False

        CACHE_WAV_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_ANALYZ_WAV_DIR.mkdir(parents=True, exist_ok=True)
        KB_CACHE_WAV_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Инициализация и прогрев
    # -----------------------------------------------------------------------

    async def _warmup_all(self) -> None:
        """
        Прогрев ML-движков выполняется последовательно.

        На Jetson параллельный прогрев CTranslate2/Whisper и TTS может
        зависать на CUDA/runtime блокировках. Последовательный прогрев
        занимает на несколько секунд больше при старте, зато не может
        повесить запуск и не влияет на SLA после конца фразы.
        """
        loop = asyncio.get_event_loop()

        log.info("--> Прогрев движков...")

        warmups = (
            ("STT", self.stt.warmup, 20.0),
            ("CACHE", self.cache.warmup, 10.0),
            ("TTS", self.tts.warmup, 15.0),
        )
        for name, fn, timeout_s in warmups:
            start = time.perf_counter()
            try:
                await asyncio.wait_for(loop.run_in_executor(self.executor, fn), timeout=timeout_s)
                log.info("--> Warmup %s завершён за %.2f с", name, time.perf_counter() - start)
            except asyncio.TimeoutError:
                log.error("Warmup %s превысил %.1f с — продолжаю запуск без ожидания", name, timeout_s)
            except Exception as exc:
                log.error("Warmup %s ошибка: %s — продолжаю запуск", name, exc)

        log.info("--> Все движки готовы к работе")

    # -----------------------------------------------------------------------
    # Вспомогательные async-обёртки над блокирующими вызовами
    # -----------------------------------------------------------------------

    async def _run_stt(self, audio_data) -> "tuple":
        """
        Транскрибирует аудио в текст в отдельном потоке.

        Возвращает Tuple[str, str] = (текст, код_языка_для_xtts) — как
        STTEngine.transcribe(). audio_data — numpy.ndarray (float32, mono,
        16kHz), как отдаёт AudioListener.listen_for_phrase(), а не bytes.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.stt.transcribe,
            audio_data,
            STT_LANG,
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
        """Ищет ответ в семантическом кэше."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.cache.search,
            text,
            lang_code,
        )

    async def _run_cache_put(self, query: str, answer: str, wav_path: str, lang_code: str) -> None:
        """Сохраняет пару (запрос, ответ, wav) в кэш."""
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
    # Worker-1: генерация аудио (LLM + TTS)
    # -----------------------------------------------------------------------

    async def _worker_tts(
        self,
        text: str,
        lang_code: str,
        wav_queue: asyncio.Queue,
        collected_sentences: List[str],
        custom_generator = None,
        detected_gestures: List[str] = None,
    ) -> None:
        """
        Worker-1: «Производитель» (Producer) в схеме Producer-Consumer.

        Логика:
          1. Получает от LLMEngine поток предложений (sync-генератор,
             запускаемый в executor чтобы не блокировать event loop).
          2. Каждое предложение отправляет в TTS → получает wav_path.
          3. Кладёт wav_path в asyncio.Queue (неограниченная очередь,
             т.к. воркер-плеер читает быстрее, чем TTS генерирует).
          4. По завершении кладёт sentinel-объект (_QUEUE_SENTINEL),
             чтобы Worker-2 знал, что больше файлов не будет.
          5. При установке флага _interrupted немедленно прекращает работу
             и сигнализирует Worker-2 через sentinel.

        Параметры
        ---------
        text              : транскрибированный запрос пользователя
        wav_queue         : asyncio.Queue для передачи wav_path → Worker-2
        collected_sentences : список для накопления предложений (нужен для
                             финального слияния wav и сохранения в кэш)
        """
        loop = asyncio.get_event_loop()

        try:

            gen = custom_generator if custom_generator is not None else self.llm.generate_stream(text, lang_code)

            while True:
                # --- Проверка прерывания ---
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

                # --- ФИЛЬТР АРТЕФАКТОВ ДЛЯ XTTS ---
                sentence = re.sub(r'\.{2,}', ',', sentence)
                sentence = re.sub(r'[*_~"«»]', '', sentence)

                if lang_code not in ("zh", "zh-cn", "ja", "ko"):
                    sentence = sentence.replace('。', '.').replace('！', '!').replace('？', '?').replace('，', ',')
                    sentence = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', '', sentence)

                # Перехват тегов жестикуляции 
                gesture_match = re.search(r'\[жест:?\s*([^\]]+)\]', sentence, flags=re.IGNORECASE)
                g_name_to_pass = None
                
                if gesture_match:
                    raw_g_name = gesture_match.group(1).strip().lower()
                    g_name = raw_g_name.translate(str.maketrans("осаехр", "osacxp"))
                    
                    g_name_to_pass = g_name 
                    
                    if detected_gestures is not None and not detected_gestures:
                        detected_gestures.append(g_name) 
                        
                    sentence = sentence.replace(gesture_match.group(0), "")

                sentence = sentence.strip()
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
            # --- Отправляем sentinel в любом случае ---
            # Это КРИТИЧНО: Worker-2 ждёт sentinel, чтобы выйти из своего цикла.
            # Без sentinel Worker-2 зависнет на queue.get() навсегда.
            await wav_queue.put(_QUEUE_SENTINEL)
            log.debug("Worker-TTS: sentinel отправлен в очередь")

    # -----------------------------------------------------------------------
    # Worker-2: воспроизведение аудио
    # -----------------------------------------------------------------------

    async def _worker_player(
        self,
        wav_queue: asyncio.Queue,
        played_wavs: List[str],
    ) -> None:
        """
        Worker-2: «Потребитель» (Consumer) в схеме Producer-Consumer.

        Логика:
          1. Бесконечно читает из asyncio.Queue.
          2. Если получил _QUEUE_SENTINEL — выходит из цикла (поток завершён).
          3. Если получил путь к wav — немедленно воспроизводит его.
          4. Сохраняет путь в played_wavs для последующего слияния.
          5. При установке флага _interrupted останавливает воспроизведение
             и выходит, НЕ дожидаясь sentinel
             (sentinel всё равно придёт из finally Worker-1).

        Ключевые свойства:
          - Worker-2 запускается параллельно с Worker-1 через asyncio.gather().
          - Пока Worker-1 синтезирует второй wav, Worker-2 уже играет первый.
          - queue.get() — корутина, которая «паркует» Worker-2 без блокировки
            event loop, пока в очереди нет данных.

        Параметры
        ---------
        wav_queue   : asyncio.Queue, из которой читаем wav_path или sentinel
        played_wavs : список для накопления путей (для финального merge)
        """
        try:
            while True:
                # --- Проверка прерывания ДО чтения из очереди ---
                if self._interrupted.is_set():
                    log.info("--> Worker-Player: прерывание, прекращаем воспроизведение")
                    _drain_queue(wav_queue)
                    break

                try:
                    item = await asyncio.wait_for(wav_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                # --- Проверяем: это sentinel или wav_path? ---
                if item is _QUEUE_SENTINEL:
                    log.debug("Worker-Player: получен sentinel, завершаем цикл")
                    wav_queue.task_done()
                    break

                wav_path, g_name = item
                
                if g_name:
                    self.gestures.start(g_name) 
                
                log.info("--> Воспроизводим: %s", wav_path)
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
    # Детектор прерывания (запускается параллельно с конвейером)
    # -----------------------------------------------------------------------

    async def _interruption_watchdog(
        self,
        stop_event: asyncio.Event,
    ) -> None:
        """
        Параллельная корутина-«сторож», которая слушает микрофон во время
        ответа робота и устанавливает _interrupted при обнаружении голоса.

        Использует AudioListener.check_interrupt() — облегчённый метод,
        который возвращает True как только простой энергетический детектор
        фиксирует устойчивую речь, без полной записи utterance.

        ВАЖНО: раньше здесь вызывался listen_for_phrase() — тот же метод,
        что и в основном цикле прослушивания. Но listen_for_phrase()
        специально ставит захват на паузу, пока self.state.is_speaking
        (робот говорит) — то есть ИМЕННО в то время, когда должен работать
        watchdog. Из-за этого прерывание никогда не могло сработать.
        check_interrupt() — отдельный метод, который читает микрофон именно
        во время ответа робота (подробности и оговорки см. в audio_io.py).

        stop_event устанавливается основной корутиной после завершения
        конвейера, чтобы сторож не работал вечно.
        """
        loop = asyncio.get_event_loop()

        try:
            while not stop_event.is_set():
                # Проверяем наличие голоса (неблокирующий poll, ~50 мс)
                detected = await loop.run_in_executor(
                    self.executor,
                    self.listener.check_interrupt,
                )

                if detected:
                    log.info("--> Прерывание: обнаружена речь пользователя")
                    self._interrupted.set()
                    break

                # Небольшая пауза, чтобы не сжигать CPU
                await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            pass

    # -----------------------------------------------------------------------
    # Стриминговый конвейер (кэш-промах)
    # -----------------------------------------------------------------------

    async def _run_streaming_pipeline(self, query: str, lang_code: str = "ru", custom_generator = None) -> None:
        """
        Запускает двухворкерный стриминговый конвейер для генерации ответа.

        Схема работы очереди
        --------------------

                    wav_queue (asyncio.Queue)
                         │
        Worker-TTS ──put()──►──get()── Worker-Player
         (Producer)                      (Consumer)
              │                              │
          TTS.synthesize()            AudioPlayer.play()
              │                              │
          wav_path                     played_wavs[]

        Sentinel-паттерн:
          Worker-TTS кладёт _QUEUE_SENTINEL после последнего wav
          (или при прерывании/ошибке — из блока finally).
          Worker-Player при получении sentinel выходит из цикла.
          Это гарантирует, что Worker-Player ВСЕГДА завершится.

        После завершения обоих воркеров:
          - Проверяем, было ли прерывание.
          - Если нет — склеиваем все wav в один финальный файл.
          - Сохраняем в SemanticCache.
        """
        self._interrupted.clear()

        collected_sentences: List[str] = []  # Worker-TTS пишет
        played_wavs: List[str] = []          # Worker-Player пишет
        detected_gestures: List[str] = []

        # asyncio.Queue — основной канал связи между воркерами.
        # maxsize=0 означает неограниченный размер. Это безопасно, т.к.:
        #   а) TTS медленнее плеера (не накопится много элементов)
        #   б) Мы хотим, чтобы TTS не ждал плеера (полный оверлап)
        wav_queue: asyncio.Queue = asyncio.Queue()

        watchdog_stop = asyncio.Event()
 
        tts_task = asyncio.ensure_future(
            self._worker_tts(query, lang_code, wav_queue, collected_sentences, custom_generator, detected_gestures)
        )
        
        watchdog_task = None
        if ENABLE_INTERRUPT:
            watchdog_task = asyncio.ensure_future(
                self._interruption_watchdog(watchdog_stop)
            )

        # Если озвучка выключена - запускаем dummy_player, который просто забирает wav из очереди,
        # если включена - оригинальный player_task
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

        # --- Постобработка ---

        if self._interrupted.is_set():
            log.info("Конвейер прерван — пропускаем сохранение в кэш")
            _cleanup_temp_wavs(played_wavs)
            return

        if not played_wavs:
            log.warning("Конвейер завершён, но wav-файлов нет — ничего не сохраняем")
            return

        # --- Слияние wav-файлов ---
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

        # --- Сохранение в кэш ---
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
    # Главный цикл
    # -----------------------------------------------------------------------

    async def run(self) -> None:
        """
        Главный цикл голосового ассистента.

        Порядок работы в каждой итерации:
          1. listen()       — блокирующая запись utterance
          2. STT            — транскрипция
          3. cache.search() — поиск в семантическом кэше
          4a. кэш-хит  → play(cached_wav)
          4b. кэш-промах → _run_streaming_pipeline()
        """
        await self._warmup_all()

        self._running = True
        self.listener.start()
        log.info("--> Ассистент запущен. Нажмите Ctrl+C для выхода.")

        loop = asyncio.get_event_loop()

        # =========================================================
        # --- Стартовая фраза Кузьмича ---
        # =========================================================
        startup_path = str(PROJECT_DIR / "startup.wav")
        if not os.path.exists(startup_path):
            log.info("Синтезирую стартовую фразу...")
            startup_text = "Ну всё, лампы прогрелись, сервоприводы смазаны. Чего стоим? Я готов, говорите, только четко и не бормочите."
            try:
                await loop.run_in_executor(
                    self.executor,
                    self.tts.synthesize,
                    startup_text,
                    "ru",
                    startup_path
                )
            except Exception as exc:
                log.error("Ошибка синтеза стартовой фразы: %s", exc)

        if os.path.exists(startup_path):
            log.info("Проигрываю стартовую речь.")
            try:
                await self._play(startup_path)
            except Exception as exc:
                log.error("Ошибка воспроизведения стартовой фразы: %s", exc)
        # =========================================================

        try:
            while self._running:
                # ── Шаг 1: Запись аудио ──────────────────────────────────
                log.info("--> Слушаю...")

                try:
                    audio_data: Optional[bytes] = await loop.run_in_executor(
                        self.executor,
                        self.listener.listen_for_phrase,
                    )
                except Exception as exc:
                    log.error("AudioListener ошибка: %s", exc)
                    await asyncio.sleep(0.5)
                    continue

                if audio_data is None or len(audio_data) == 0:
                    log.debug("Аудио пустое — пропускаем")
                    continue

                # ── Шаг 2: Speech-to-Text ────────────────────────────────
                try:
                    text, lang_code = await self._run_stt(audio_data)
                except Exception as exc:
                    log.error("STT ошибка: %s", exc)
                    continue

                if not text or not text.strip():
                    log.debug("STT вернул пустую строку — пропускаем")
                    continue

                query = text.strip()

                # --- АНТИ-ГАЛЛЮЦИНАЦИИ WHISPER ---
                query_lower = query.lower()
                
                hallucinations = [
                    "субтитры", "dima", "torzok", "продолжение следует",
                    "amara", "редактор", "перевод", "озвучено", "смотреть до конца"
                ]
                exact_hallucinations = [
                    "okay", "mm-hmm", "yeah", "you", "thank you", "thanks", "окей", "да", "нет"
                ]
                
                if any(h in query_lower for h in hallucinations) or query_lower in exact_hallucinations or len(query) < 3:
                    log.info("Поймана галлюцинация Whisper: «%s» — игнорируем", query)
                    continue
                # ---------------------------------

                log.info("--> Распознано (%s): «%s»", lang_code, query)

                 # ── Обработка триггеров ───────────────────

                clean_words = set(re.findall(r'\b\w+\b', query.lower()))

                if USE_TRIGGERS:
                    stop_words = [
                        "stop", "top", "стоп", "топ",
                    ]
                    if any(h in clean_words for h in stop_words):
                        log.info("--> Команда СТОП. Переход в спящий режим.")
                        self.llm.clear_history()
                        self.is_awake = False
                        continue
                    if "привет" in clean_words:
                        log.info("--> Команда ПРИВЕТ. Переход в активный режим.")
                        self.is_awake = True
                    if "поздоровайся" in clean_words:
                        log.info("--> Команда ПОЗДОРОВАЙСЯ. Запуск приветственной речи с жестами.")
                        
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
                            log.error("Ошибка при воспроизведении приветствия: %s", exc)
                            
                        continue                            

                if not self.is_awake:
                    log.debug("Спящий режим. Игнорирую: %s", query)
                    continue

                analyze_words = [
                    "анализ", "analyze", "analysis", "analice",
                    "alice", "analize", "analyce", "аналис", 
                    "аналайс", "аналайз",
                ]
                if any(h in clean_words for h in analyze_words) and analyze_for_tts is not None:
                    self.is_analiz = True
                    log.info("--> Запуск анализа сцены...")
                    try:
                        out_path = await loop.run_in_executor(self.executor, analyze_for_tts)
                        with open(out_path, "r", encoding="utf-8") as f:
                            scene_desc = f.read().strip()
                        
                        query = f"Пользователь попросил анализ сцены. Данные с твоих камер: {scene_desc}. Кратко расскажи, что ты видишь, от своего лица."
                        log.info("--> Зрение получено: %s", scene_desc)
                    except Exception as exc:
                        log.error("Ошибка VLM-анализа: %s", exc)
                        query = "Пользователь попросил анализ, но твоя камера не отвечает. Пошути на тему сломанных советских датчиков."

                # ── Шаг 3: Поиск в семантических кэшах ───────────
                cache_result = None

                # 1. Базу знаний (факты) проверяем ВСЕГДА, независимо от контекста диалога
                try:
                    cache_result = await loop.run_in_executor(self.executor, self.kb_cache.search, query, lang_code)
                except Exception as exc:
                    log.error("Ошибка поиска в kb_cache: %s", exc)

                # 2. Обычный разговорный кэш проверяем, только если память пуста
                if not cache_result and (not ENABLE_CONTEXT or not self.llm.chat_history):
                    try:
                        cache_result = await self._run_cache_search(query, lang_code)
                    except Exception as exc:
                        log.error("SemanticCache ошибка поиска: %s", exc)

                if cache_result is not None and not self.is_analiz:
                    cached_wav = cache_result.get("audio_path") if isinstance(cache_result, dict) else getattr(cache_result, "audio_path", cache_result)
                    response_text = cache_result.get("response_text", "") if isinstance(cache_result, dict) else ""
                    
                    if cached_wav:
                        if response_text:
                            gesture_match = re.search(r'\[жест:?\s*([^\]]+)\]', response_text, flags=re.IGNORECASE)
                            if gesture_match:
                                raw_g_name = gesture_match.group(1).strip().lower()
                                g_name = raw_g_name.translate(str.maketrans("осаехр", "osacxp"))
                                self.gestures.start(g_name)
                                
                        log.info("Кэш-хит! Воспроизводим: %s", cached_wav)
                        try:
                            await self._play(cached_wav)
                        except Exception as exc:
                            log.error("Ошибка воспроизведения кэша: %s", exc)
                            
                        self.llm.clear_history()
                        continue

                # ── Шаг 4: Стриминговый конвейер (кэш-промах) ────────────
                log.info("Кэш-промах, запускаем конвейер LLM→TTS→Player")

                try:
                    await self._run_streaming_pipeline(query, lang_code)
                except asyncio.CancelledError:
                    log.info("Конвейер отменён")
                    raise
                except Exception as exc:
                    log.exception("Ошибка в стриминговом конвейере: %s", exc)

        except (KeyboardInterrupt, asyncio.CancelledError):
            log.info("Получен сигнал завершения")
        finally:
            self._running = False
            self.executor.shutdown(wait=False)
            self.listener.close()
            log.info("Ассистент остановлен")


# ===========================================================================
# Вспомогательные функции (модуль-уровень, не методы класса)
# ===========================================================================

def _safe_next(generator):
    """
    Безопасный вызов next() для синхронного генератора.

    Возвращает следующее значение или None при StopIteration.
    Используется в run_in_executor, т.к. StopIteration не может
    нормально выйти из executor в Python 3.7+ (PEP 479).
    """
    try:
        return next(generator)
    except StopIteration:
        return None


def _drain_queue(queue: asyncio.Queue) -> None:
    """
    Немедленно очищает asyncio.Queue без ожидания (non-blocking drain).

    Вызывается при прерывании, чтобы «дренировать» оставшиеся wav-пути
    и sentinel из очереди, не допустив утечки ресурсов.
    """
    while not queue.empty():
        try:
            queue.get_nowait()
            queue.task_done()
        except asyncio.QueueEmpty:
            break


def _cleanup_temp_wavs(wav_paths: List[str]) -> None:
    """
    Удаляет временные .wav файлы, созданные TTS для отдельных предложений.
    Финальный склеенный файл НЕ удаляется (он сохранён в CACHE_WAV_DIR).
    """
    for path in wav_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                log.debug("Удалён временный файл: %s", path)
        except OSError as exc:
            log.warning("Не удалось удалить %s: %s", path, exc)


# ===========================================================================
# Точка входа
# ===========================================================================

def main() -> None:
    """
    Точка входа. Создаёт event loop и запускает VoiceAssistant.

    Python 3.8 совместимость:
      - asyncio.run() доступен с 3.7, используем его.
      - asyncio.to_thread() — только 3.9+, поэтому везде используем
        loop.run_in_executor() с явным ThreadPoolExecutor.
      - В Python 3.8 на Windows по умолчанию используется
        ProactorEventLoop (нужно для subprocesses), на Unix — SelectorEventLoop.
        Для аудио это обычно не важно, но явно не задаём policy,
        чтобы не ломать платформенные значения по умолчанию.
    """
    assistant = VoiceAssistant()

    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        pass  # asyncio.run() сам обрабатывает Ctrl+C через CancelledError


if __name__ == "__main__":
    main()
