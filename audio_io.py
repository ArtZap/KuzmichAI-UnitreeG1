"""
audio_io.py

Модуль ввода/вывода аудио для автономного робота Unitree G1.

ОБНОВЛЕНИЕ v3 (ИСПРАВЛЕНИЕ "робот ничего не слышит"):
-------------------------------------------------------
Метод `AudioClient.GetAudioData()`, который использовался раньше для
захвата микрофона, НЕ является рабочим способом получить поток с
микрофона G1. Это подтверждается:

  * issue unitreerobotics/unitree_sdk2_python #80 — разработчики Unitree
    прямо пишут, что в Python-SDK отсутствует функциональность
    "Multicast microphone data recording", которая есть в C++ SDK;
  * issue unitreerobotics/unitree_sdk2_python #143 — человек ловит
    tcpdump'ом UDP-пакеты на мультикаст-группе 239.168.123.161:5555 и
    видит, что пакеты реально идут, но полезная нагрузка — нули, если
    читать её через RPC-клиент;
  * независимый проект SaxionMechatronics/unitree_converse (README,
    "Key Discovery: Audio Routing") прямо описывает архитектуру:
    микрофон G1 физически подключён к отдельному RockChip-контроллеру
    (239.168.123.161), который транслирует СЫРОЙ PCM (16-bit mono,
    16 kHz) через UDP MULTICAST на порт 5555 — в обход DDS/RPC.

Поэтому:
  * захват микрофона теперь делает `MulticastMicReceiver` — обычный
    UDP-сокет, подписанный на мультикаст-группу 239.168.123.161:5555;
  * `AudioClient` (обёрнутый в `G1AudioClientWrapper`) используется
    только там, где он реально работает: воспроизведение звука через
    `PlayStream()` (это подтверждается вашим же логом — TTS/warmup и
    проигрывание работали) и best-effort вызовы StartRecording /
    StopRecording (некоторые прошивки требуют явно "включить" поток
    с RockChip — если такого метода нет в вашей версии SDK, это не
    считается ошибкой).

Компоненты:
    * SharedRobotState      - потокобезопасный флаг is_speaking.
    * AudioConfig           - параметры захвата, VAD и мультикаста.
    * G1AudioClientWrapper  - синглтон-обёртка над SDK (DDS/RPC), только
                              для управления и воспроизведения.
    * MulticastMicReceiver  - фоновый поток, читающий сырой PCM с
                              микрофона G1 через UDP multicast.
    * LocalMicSource        - фоновый поток, читающий PCM с обычного
                              микрофона ноутбука (sounddevice). Не требует
                              никакой сети робота — для локальной отладки.
    * AudioListener         - разбор потока на фразы через Silero-VAD v4.
                              Источник PCM (мультикаст G1 или локальный
                              микрофон) выбирается параметром `source`.
    * AudioPlayer           - воспроизведение WAV через AudioClient.PlayStream().
    * LocalAudioPlayer      - воспроизведение WAV через колонки ноутбука
                              (sounddevice). Тоже не требует сети робота.

РЕЖИМ "ЛОКАЛЬНЫЙ НОУТБУК" (LOCAL_AUDIO_MODE):
-------------------------------------------------------
Если нужно гонять весь пайплайн (STT → LLM → TTS) прямо на ноутбуке для
разработки/отладки, не имея физического подключения к роботу G1 —
AudioListener и AudioPlayer можно создать с флагом `local_mode=True`
(или просто выставить переменную окружения VOICE_ENGINE_AUDIO=local,
это делает main.py). В этом случае:
  * не инициализируется DDS/AudioClient (не нужен sudo/подключение к G1);
  * не открывается UDP-сокет на мультикаст-группу робота;
  * микрофон и динамики — обычные устройства ноутбука через `sounddevice`.
Требуется пакет `sounddevice` (pip install sounddevice) и системная
библиотека PortAudio (на Ubuntu: sudo apt install libportaudio2).
"""

from __future__ import annotations

import logging
import os
import queue
import shlex
import socket
import struct
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

# Импорты официального SDK Unitree
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    print("ВНИМАНИЕ: unitree_sdk2_python не установлен! Будут использованы заглушки.")

# sounddevice нужен только для ЛОКАЛЬНОГО режима (микрофон/динамики ноутбука).
# Импортируем лениво/защищённо, чтобы работа с настоящим роботом G1 не
# ломалась, если пакет не установлен и локальный режим не используется.
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

logger = logging.getLogger("audio_io")

# ---------------------------------------------------------------------------
# Общее состояние робота
# ---------------------------------------------------------------------------
class SharedRobotState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._is_speaking = False

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._is_speaking

    @is_speaking.setter
    def is_speaking(self, value: bool) -> None:
        with self._lock:
            self._is_speaking = value


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    vad_frame_samples: int = 512
    vad_threshold: float = 0.5
    min_silence_duration_ms: int = 300
    speech_pad_ms: int = 200

    # --- Параметры UDP-мультикаста микрофона RockChip ---
    # Группа и порт фиксированы прошивкой G1 (см. официальные примеры
    # и community-проекты), обычно менять их не нужно.
    mic_multicast_group: str = "239.168.123.161"
    mic_multicast_port: int = 5555
    # IP локального сетевого интерфейса, которым машина смотрит в
    # внутреннюю сеть робота (обычно интерфейс до 192.168.123.161).
    # Если None — пытаемся определить автоматически, но лучше задать
    # явно (типичный адрес Jetson на борту G1 — "192.168.123.164").
    mic_local_ip: Optional[str] = field(default_factory=lambda: os.environ.get("VOICE_ENGINE_MIC_LOCAL_IP", "192.168.123.164"))
    mic_recv_buf_size: int = 65536
    mic_queue_maxsize: int = 400


# ---------------------------------------------------------------------------
# Инициализация DDS и Клиента Аудио (используется только для управления
# и воспроизведения — НЕ для захвата микрофона, см. шапку модуля)
# ---------------------------------------------------------------------------
_channel_initialized = False
_channel_lock = threading.Lock()

def init_unitree_channel(network_interface: Optional[str] = None) -> None:
    """Инициализирует DDS-сеть (делается ровно 1 раз на процесс)"""
    global _channel_initialized
    network_interface = network_interface or os.environ.get("VOICE_ENGINE_DDS_INTERFACE", "eth0")
    with _channel_lock:
        if not _channel_initialized and SDK_AVAILABLE:
            try:
                ChannelFactoryInitialize(0, network_interface)
                _channel_initialized = True
                logger.info("Unitree DDS ChannelFactory инициализирован (Интерфейс: %s)", network_interface)
            except Exception as e:
                logger.warning("Ошибка инициализации ChannelFactory: %s", e)


class G1AudioClientWrapper:
    """
    Синглтон-обёртка над AudioClient.

    ВАЖНО: используется только для воспроизведения (PlayStream) и
    best-effort управления записью (StartRecording/StopRecording).
    Сам поток микрофона идёт НЕ через этот клиент, а через
    MulticastMicReceiver (см. ниже).
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> "G1AudioClientWrapper":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        init_unitree_channel()

        self.client = None
        if SDK_AVAILABLE:
            try:
                client = AudioClient()
                client.SetTimeout(3.0)
                client.Init()
                self.client = client
                logger.info("Unitree G1 AudioClient успешно инициализирован.")
            except Exception as e:
                # НЕ роняем процесс: захват микрофона идёт через отдельный
                # UDP-мультикаст (MulticastMicReceiver) и от AudioClient
                # не зависит. Без рабочего AudioClient не будет работать
                # только PlayStream (озвучка ответа) — это лучше, чем
                # падение всего приложения при старте.
                #
                # Типичная причина именно этой ошибки —
                # DDS_RETCODE_PRECONDITION_NOT_MET на создании Topic:
                # в DDS-домене уже зарегистрирован топик аудио-сервиса
                # с другим типом (несовпадение версии SDK/прошивки).
                # Проверьте:
                #   1) прошивка G1 >= 1.3.0;
                #   2) версия пакета cyclonedds ТОЧНО 0.10.2
                #      (pip show cyclonedds), собранная как в
                #      https://github.com/unitreerobotics/unitree_ros2;
                #   3) нет ли другого процесса на этой же машине/сети,
                #      уже держащего DDS-участника с этим сервисом;
                #   4) подключение к роботу ПРОВОДНОЕ (eth0), а не по
                #      wlan0 — Wi-Fi даёт менее стабильный DDS discovery.
                logger.error(
                    "Не удалось инициализировать AudioClient (%s: %s). "
                    "Микрофон продолжит работать (UDP multicast), но "
                    "воспроизведение через PlayStream будет недоступно, "
                    "пока не будет устранена причина ошибки DDS.",
                    type(e).__name__, e,
                )
        else:
            logger.warning("Mock-клиент используется из-за отсутствия SDK.")

        self._initialized = True

    def start_recording(self) -> None:
        """
        Best-effort: на некоторых прошивках нужно явно 'включить' поток
        с RockChip через RPC. Если метода нет в установленной версии
        SDK — просто логируем и продолжаем: сам мультикаст-поток может
        идти и без этого вызова (см. SaxionMechatronics/unitree_converse,
        где мультикаст читается вообще без обращения к AudioClient).
        """
        if not self.client:
            return
        try:
            self.client.StartRecording()
            logger.info("DDS/RPC: StartRecording() отправлен (best-effort).")
        except AttributeError:
            logger.debug("AudioClient.StartRecording() отсутствует в этой версии SDK — пропускаем.")
        except Exception as e:
            logger.warning("StartRecording() вернул ошибку (не критично): %s", e)

    def stop_recording(self) -> None:
        if not self.client:
            return
        try:
            self.client.StopRecording()
            logger.info("DDS/RPC: StopRecording() отправлен (best-effort).")
        except AttributeError:
            logger.debug("AudioClient.StopRecording() отсутствует в этой версии SDK — пропускаем.")
        except Exception as e:
            logger.warning("StopRecording() вернул ошибку (не критично): %s", e)

    # Размер одного куска аудио, отправляемого за один вызов PlayStream().
    # ПОЧЕМУ: полный wav многосекундного ответа, переданный ОДНИМ RPC-вызовом
    # как list(audio_bytes), может превысить лимит размера сообщения
    # DDS/RPC (в частности, list() из сотен тысяч int превращает несколько
    # секунд PCM в очень тяжёлый объект для сериализации). Дробим на куски
    # ~1 сек (32000 байт = 16000 Гц * 2 байта/семпл), как это обычно делают
    # стриминговые PlayStream-реализации. Если ваша версия SDK ведёт себя
    # иначе (например, требует stream_id или другой протокол чанкинга) —
    # свяжите это значение с реальным лимитом вашей прошивки/SDK.
    _PLAYSTREAM_CHUNK_BYTES = 32000

    def play_stream(self, audio_bytes: bytes, sample_rate: int) -> bool:
        if not self.client:
            logger.warning(
                "AudioClient недоступен (см. ошибку инициализации выше) — "
                "%d байт звука НЕ отправлены на динамик робота.",
                len(audio_bytes),
            )
            return False

        if not hasattr(self.client, "PlayStream"):
            # PlayStream появился в unitree_sdk2_python сравнительно недавно
            # (см. PR unitreerobotics/unitree_sdk2_python#70 "Add playStream
            # in g1 audio client"). В более старых установленных версиях
            # пакета этого метода может не быть вообще — тогда AttributeError
            # рухнул бы прямо здесь без внятного сообщения.
            logger.error(
                "AudioClient.PlayStream() отсутствует в установленной версии "
                "unitree_sdk2_python. Обновите пакет: "
                "pip install --upgrade unitree_sdk2_python (нужна версия с "
                "PlayStream в g1_audio_client.py). Аудио НЕ отправлено на динамик."
            )
            return False

        total = len(audio_bytes)
        stream_name = os.environ.get("VOICE_ENGINE_PLAYSTREAM_NAME", "kuzmich")
        stream_id = str(int(time.time() * 1000))
        for offset in range(0, total, self._PLAYSTREAM_CHUNK_BYTES):
            piece = audio_bytes[offset: offset + self._PLAYSTREAM_CHUNK_BYTES]
            try:
                ret_code, _ = self.client.PlayStream(stream_name, stream_id, piece)
            except TypeError:
                ret_code, _ = self.client.PlayStream(stream_name, stream_id, list(piece))
            except Exception as e:
                logger.error(
                    "PlayStream() ошибка на смещении %d/%d байт: %s",
                    offset, total, e,
                )
                return False
            if ret_code != 0:
                logger.error("PlayStream() вернул код %s на смещении %d/%d байт.", ret_code, offset, total)
                return False
        return True


# ---------------------------------------------------------------------------
# Захват микрофона: UDP multicast (RockChip, 239.168.123.161:5555)
# ---------------------------------------------------------------------------
class MulticastMicReceiver:
    """
    Слушает сырой PCM-поток микрофона G1.

    Микрофон физически обслуживается отдельным RockChip-контроллером,
    который транслирует 16-bit mono PCM (по умолчанию 16 kHz) через
    UDP multicast на 239.168.123.161:5555. Это НЕ DDS-топик и не RPC —
    обычный multicast-сокет, который может слушать любой процесс в
    той же сети.
    """

    def __init__(
        self,
        multicast_group: str = "239.168.123.161",
        port: int = 5555,
        local_ip: Optional[str] = None,
        recv_buf_size: int = 8192,
        queue_maxsize: int = 400,
    ) -> None:
        self.multicast_group = multicast_group
        self.port = port
        self.local_ip = local_ip or self._detect_local_ip(multicast_group)
        self.recv_buf_size = recv_buf_size

        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._queue: "queue.Queue[bytes]" = queue.Queue(maxsize=queue_maxsize)

    @staticmethod
    def _detect_local_ip(remote_hint_ip: str) -> str:
        """
        Пытается определить, каким локальным интерфейсом машина смотрит
        во внутреннюю сеть робота. Это UDP-connect без реальной отправки
        пакетов (соединение не устанавливается), просто способ спросить
        у ОС "какой у меня адрес в сторону этого хоста".

        Если определить не удалось — используем типичный адрес Jetson
        на борту G1 (192.168.123.164). ЛУЧШЕ ВСЕГО передать local_ip
        явно через AudioConfig.mic_local_ip, автоопределение может
        ошибиться при нескольких сетевых интерфейсах.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((remote_hint_ip, 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = "192.168.123.164"
        finally:
            s.close()
        return ip

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        sock.settimeout(0.5)
        sock.bind(("", self.port))

        try:
            mreq = struct.pack(
                "4s4s",
                socket.inet_aton(self.multicast_group),
                socket.inet_aton(self.local_ip),
            )
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except OSError as e:
            sock.close()
            raise RuntimeError(
                f"Не удалось подписаться на мультикаст-группу "
                f"{self.multicast_group}:{self.port} с локальным IP "
                f"{self.local_ip}. Проверьте, что этот IP реально "
                f"принадлежит интерфейсу в сети робота (192.168.123.x). "
                f"Исходная ошибка: {e}"
            ) from e

        self._sock = sock
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="MicMulticastRecv"
        )
        self._thread.start()
        logger.info(
            "Микрофон G1: подписка на UDP-мультикаст %s:%d (local_ip=%s) активна.",
            self.multicast_group, self.port, self.local_ip,
        )

    def _recv_loop(self) -> None:
        assert self._sock is not None
        while not self._stop_event.is_set():
            try:
                data, _addr = self._sock.recvfrom(self.recv_buf_size)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                continue
            try:
                self._queue.put_nowait(data)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(data)
                except queue.Full:
                    pass

    def get_chunk(self, timeout: float = 0.05) -> bytes:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return b""

    def drain(self) -> None:
        """Сбрасывает всё, что накопилось в очереди (например, пока is_speaking=True)."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        logger.info("Микрофон G1: приём UDP-мультикаста остановлен.")


# ---------------------------------------------------------------------------
# Захват микрофона: локальное аудиоустройство ноутбука (без сети робота)
# ---------------------------------------------------------------------------
class LocalMicSource:
    """
    Источник PCM с обычного микрофона ноутбука через `sounddevice`.

    Реализует тот же интерфейс, что и MulticastMicReceiver
    (start/stop/get_chunk/drain), поэтому AudioListener может работать
    с любым из двух источников без изменения логики VAD/нарезки на фразы.
    Никакой сети/DDS/UDP-мультикаста робота здесь не используется вообще.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        blocksize: int = 512,
        device: Optional[Union[int, str]] = None,
        queue_maxsize: int = 400,
    ) -> None:
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError(
                "Пакет 'sounddevice' не установлен, а он нужен для локального "
                "режима работы с микрофоном ноутбука. Установите: "
                "pip install sounddevice (и системно: libportaudio2)."
            )

        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.device = device

        self._stream: Optional["sd.InputStream"] = None
        self._stop_event = threading.Event()
        self._queue: "queue.Queue[bytes]" = queue.Queue(maxsize=queue_maxsize)

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            logger.debug("LocalMicSource: sounddevice status=%s", status)
        # indata приходит как int16 numpy-массив (см. dtype в InputStream ниже),
        # приводим к сырым байтам — в том же формате, что и MulticastMicReceiver.
        data = bytes(indata)
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(data)
            except queue.Full:
                pass

    def start(self) -> None:
        if self._stream is not None:
            return

        self._stop_event.clear()
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=self.blocksize,
                device=self.device,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            self._stream = None
            raise RuntimeError(
                f"Не удалось открыть локальный микрофон ноутбука "
                f"(device={self.device!r}): {e}. Проверьте вывод "
                f"`python -c \"import sounddevice as sd; print(sd.query_devices())\"` "
                f"и доступ к микрофону у процесса."
            ) from e

        logger.info(
            "Микрофон ноутбука (LocalMicSource) запущен: device=%s, sr=%d, ch=%d",
            self.device, self.sample_rate, self.channels,
        )

    def get_chunk(self, timeout: float = 0.05) -> bytes:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return b""

    def drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        self._stop_event.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        logger.info("Микрофон ноутбука (LocalMicSource) остановлен.")


# ---------------------------------------------------------------------------
# Захват + VAD
# ---------------------------------------------------------------------------
class AudioListener:
    def __init__(
        self,
        shared_state: SharedRobotState,
        config: Optional[AudioConfig] = None,
        local_mode: bool = False,
        mic_device: Optional[Union[int, str]] = None,
    ) -> None:
        """
        :param local_mode: если True — захват идёт с обычного микрофона
            ноутбука (LocalMicSource) и полностью пропускается
            инициализация DDS/AudioClient робота. Используйте это для
            разработки/отладки без физического подключения к G1.
        :param mic_device: (только для local_mode) имя/индекс устройства
            для sounddevice; None = микрофон по умолчанию в ОС.
        """
        self.state = shared_state
        self.cfg = config or AudioConfig()
        self.local_mode = local_mode

        self._stop_event = threading.Event()
        self._vad_model, self._vad_iterator = self._load_silero_vad()

        self._speech_buffer: List[np.ndarray] = []
        self._is_recording_phrase: bool = False

        self._interrupt_voiced_run: int = 0

        if local_mode:
            self.g1_audio = None
            self.mic = LocalMicSource(
                sample_rate=self.cfg.sample_rate,
                channels=self.cfg.channels,
                blocksize=self.cfg.vad_frame_samples,
                device=mic_device,
                queue_maxsize=self.cfg.mic_queue_maxsize,
            )
            logger.info("AudioListener инициализирован в ЛОКАЛЬНОМ режиме (микрофон ноутбука).")
        else:
            self.g1_audio = G1AudioClientWrapper()

            self.mic = MulticastMicReceiver(
                multicast_group=self.cfg.mic_multicast_group,
                port=self.cfg.mic_multicast_port,
                local_ip=self.cfg.mic_local_ip,
                recv_buf_size=self.cfg.mic_recv_buf_size,
                queue_maxsize=self.cfg.mic_queue_maxsize,
            )
            logger.info("AudioListener инициализирован в режиме РОБОТА (UDP multicast).")

        self._byte_buffer = bytearray()
        self._samples_per_frame = self.cfg.vad_frame_samples
        self._bytes_per_frame = self._samples_per_frame * 2  

    def _load_silero_vad(self) -> Tuple[Any, Any]:
        logger.info("Загрузка модели Silero-VAD v4...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        model.eval()
        vad_iterator_cls = utils[3]
        vad_iterator = vad_iterator_cls(
            model,
            threshold=self.cfg.vad_threshold,
            sampling_rate=self.cfg.sample_rate,
            min_silence_duration_ms=self.cfg.min_silence_duration_ms,
            speech_pad_ms=self.cfg.speech_pad_ms,
        )
        return model, vad_iterator

    def start(self) -> None:
        self._stop_event.clear()
        if self.g1_audio is not None:
            self.g1_audio.start_recording()
        self.mic.start()
        logger.info("Захват аудио запущен (local_mode=%s).", self.local_mode)

    def stop(self) -> None:
        self._stop_event.set()
        self.mic.stop()
        if self.g1_audio is not None:
            self.g1_audio.stop_recording()
        logger.info("Захват аудио остановлен.")

    def close(self) -> None:
        self.stop()

    def _reset_phrase_state(self) -> None:
        self._speech_buffer = []
        self._is_recording_phrase = False
        self._byte_buffer.clear()
        self._vad_iterator.reset_states()

    def listen_for_phrase(self, poll_timeout: float = 0.05) -> Optional[np.ndarray]:
        self.mic.drain()  # Жестко сбрасываем всё накопленное эхо перед прослушиванием

        while not self._stop_event.is_set():
            # --- Пауза: робот говорит сам ---
            if self.state.is_speaking:
                self.mic.drain()
                if self._is_recording_phrase:
                    logger.info("is_speaking=True: сбрасываем незавершённую запись.")
                    self._reset_phrase_state()
                time.sleep(poll_timeout)
                continue

            chunk = self.mic.get_chunk(timeout=poll_timeout)
            if not chunk:
                continue

            self._byte_buffer.extend(chunk)

            while len(self._byte_buffer) >= self._bytes_per_frame:
                frame_bytes = self._byte_buffer[:self._bytes_per_frame]
                del self._byte_buffer[:self._bytes_per_frame]

                audio_int16 = np.frombuffer(frame_bytes, dtype=np.int16)
                frame = audio_int16.astype(np.float32) / 32768.0

                frame_tensor = torch.from_numpy(frame)

                try:
                    vad_event = self._vad_iterator(frame_tensor, return_seconds=False)
                except Exception:
                    logger.exception("Ошибка в Silero-VAD")
                    continue

                if vad_event is not None and "start" in vad_event:
                    logger.info("VAD: начало речи зафиксировано.")
                    self._is_recording_phrase = True

                if self._is_recording_phrase:
                    self._speech_buffer.append(frame)

                if vad_event is not None and "end" in vad_event and self._is_recording_phrase:
                    phrase_len = len(self._speech_buffer) * self.cfg.vad_frame_samples / float(self.cfg.sample_rate)
                    logger.info("VAD: конец речи. Фраза собрана, длина %.2f с.", phrase_len)

                    phrase = np.concatenate(self._speech_buffer).astype(np.float32)
                    self._reset_phrase_state()
                    return phrase

        return None

    def check_interrupt(
        self,
        energy_threshold: float = 0.06,
        min_voiced_frames: int = 3,
        poll_timeout: float = 0.05,
    ) -> bool:
        """
        Быстрая, лёгкая проверка "не заговорил ли человек поверх ответа робота".

        ПОЧЕМУ ЭТО ОТДЕЛЬНЫЙ МЕТОД, А НЕ listen_for_phrase():
        listen_for_phrase() специально ставит захват на паузу и дренирует
        очередь, когда self.state.is_speaking == True (чтобы основной цикл
        прослушивания не записывал голос самого робота как фразу
        пользователя). Но именно ЭТО время — единственное, когда нужно
        детектировать прерывание. Если использовать listen_for_phrase() для
        watchdog'а во время ответа робота, он всегда будет
        уходить в ветку "is_speaking → drain → continue" и НИКОГДА не дойдёт
        до VAD — прерывание было бы физически невозможно обнаружить.
        check_interrupt() читает из того же mic-источника, но не проверяет
        is_speaking и не использует состояние основного Silero-VAD/буфера
        фразы (чтобы не конфликтовать с listen_for_phrase(), который в
        этот момент не вызывается — конвейер ответа и обычное прослушивание
        строго последовательны в main.py, гонки по _speech_buffer нет).

        ОГРАНИЧЕНИЕ (важно понимать перед использованием на реальном роботе):
        микрофон и динамик G1 в этом пайплайне НЕ имеют аппаратного подавления
        эха (AEC). Полноценный Silero-VAD в этих условиях часто распознаёт
        собственную речь робота, доносящуюся до микрофона, как речь
        пользователя — ложные "прерывания" каждым сгенерированным
        предложением. Поэтому здесь сознательно используется простой
        энергетический (RMS) детектор с порогом и требованием НЕСКОЛЬКИХ
        подряд идущих "громких" фреймов — это грубее полноценного VAD,
        но заметно устойчивее к самоэху без AEC. energy_threshold нужно
        откалибровать под конкретное помещение/громкость динамика G1.
        Если ложные срабатывания/пропуски всё ещё мешают — рассмотрите
        физическую кнопку как более надёжный триггер прерывания (см.
        F1/F3 в SaxionMechatronics/unitree_converse) вместо чисто голосового
        barge-in.

        :return: True, если обнаружено устойчивое голосовое прерывание.
        """
        chunk = self.mic.get_chunk(timeout=poll_timeout)
        if not chunk:
            return False

        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        if audio_int16.size == 0:
            return False

        samples = audio_int16.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples ** 2)))

        if rms < energy_threshold:
            self._interrupt_voiced_run = 0
            return False

        self._interrupt_voiced_run += 1
        return self._interrupt_voiced_run >= min_voiced_frames


# ---------------------------------------------------------------------------
# Воспроизведение
# ---------------------------------------------------------------------------
class AudioPlayer:
    def __init__(self, shared_state: SharedRobotState) -> None:
        self.state = shared_state
        self.g1_audio = G1AudioClientWrapper()
        self.fallback_command = os.environ.get("VOICE_ENGINE_PLAYBACK_COMMAND", self._default_playback_command()).strip()

    @staticmethod
    def _default_playback_command() -> str:
        g1_player = Path("/home/unitree/g1_audio_play")
        if g1_player.exists():
            iface = os.environ.get("VOICE_ENGINE_DDS_INTERFACE", "eth0")
            volume = os.environ.get("VOICE_ENGINE_G1_VOLUME", "90")
            return "env -u LD_LIBRARY_PATH -u CYCLONEDDS_URI {} --iface {} --volume {} --file".format(
                str(g1_player),
                iface,
                volume,
            )
        return "aplay -q"

    @staticmethod
    def _get_wav_duration(wav_file_path: str) -> float:
        with wave.open(wav_file_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
        return frames / float(rate)

    @staticmethod
    def _read_wav_bytes(wav_file_path: str) -> Tuple[bytes, int]:
        with wave.open(wav_file_path, "rb") as wf:
            audio_bytes = wf.readframes(wf.getnframes())
            sample_rate = wf.getframerate()
        return audio_bytes, sample_rate

    def play(self, wav_file_path: str) -> None:
        path = Path(wav_file_path)
        if not path.exists():
            raise FileNotFoundError(f"WAV-файл не найден: {wav_file_path}")

        duration_sec = self._get_wav_duration(str(path))
        logger.info("Аудио: начало воспроизведения '%s' (%.2f с).", path.name, duration_sec)

        self.state.is_speaking = True
        try:
            played = False
            if self.g1_audio.client:
                audio_bytes, sample_rate = self._read_wav_bytes(str(path))
                played = self.g1_audio.play_stream(audio_bytes, sample_rate)
                if played:
                    time.sleep(duration_sec)
            if not played:
                self._play_with_command(str(path), duration_sec)
        finally:
            self.state.is_speaking = False
            logger.info("Аудио: воспроизведение завершено.")

    def _play_with_command(self, wav_file_path: str, duration_sec: float) -> None:
        if not self.fallback_command:
            logger.warning("VOICE_ENGINE_PLAYBACK_COMMAND пустой — WAV не воспроизведен.")
            return
        command = shlex.split(self.fallback_command) + [wav_file_path]
        logger.info("Audio fallback: %s", " ".join(command))
        timeout_s = max(
            duration_sec + float(os.environ.get("VOICE_ENGINE_PLAYBACK_TIMEOUT_PAD", "8.0")),
            float(os.environ.get("VOICE_ENGINE_PLAYBACK_TIMEOUT_MIN", "10.0")),
        )
        try:
            result = subprocess.run(command, check=False, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            logger.error("Audio fallback timeout after %.1f s: %s", timeout_s, " ".join(command))
            return
        if result.returncode != 0:
            logger.error("Audio fallback failed rc=%s: %s", result.returncode, " ".join(command))


class LocalAudioPlayer:
    """
    Воспроизведение WAV через динамики ноутбука (sounddevice), без DDS
    и без всякой зависимости от сети/подключения к роботу G1.

    Имеет тот же публичный интерфейс, что и AudioPlayer (`.play(path)`),
    поэтому VoiceAssistant в main.py использует его без изменений в
    остальной логике — меняется только то, ЧТО создаётся при local_mode.
    """

    def __init__(self, shared_state: SharedRobotState) -> None:
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError(
                "Пакет 'sounddevice' не установлен, а он нужен для локального "
                "режима воспроизведения через динамики ноутбука. Установите: "
                "pip install sounddevice (и системно: libportaudio2)."
            )
        self.state = shared_state

    def play(self, wav_file_path: str) -> None:
        path = Path(wav_file_path)
        if not path.exists():
            raise FileNotFoundError(f"WAV-файл не найден: {wav_file_path}")

        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        dtype_map = {1: "int8", 2: "int16", 4: "int32"}
        dtype = dtype_map.get(sample_width, "int16")
        audio_array = np.frombuffer(frames, dtype=dtype)
        if n_channels > 1:
            audio_array = audio_array.reshape(-1, n_channels)

        duration_sec = len(audio_array) / float(framerate)
        logger.info("Локально: воспроизведение '%s' (%.2f с) через динамики ноутбука.", path.name, duration_sec)

        self.state.is_speaking = True
        try:
            sd.play(audio_array, samplerate=framerate)
            sd.wait()
        except Exception as e:
            logger.error("Ошибка локального воспроизведения: %s", e)
            raise
        finally:
            self.state.is_speaking = False
            logger.info("Локально: воспроизведение завершено.")
