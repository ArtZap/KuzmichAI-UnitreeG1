"""
audio_io.py

Audio I/O module for the autonomous Unitree G1 robot.
Captures microphone via UDP multicast and handles audio playback.
Supports a local mode for debugging on a laptop without the robot.
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

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    print("WARNING: unitree_sdk2_python is not installed! Stubs will be used.")

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

logger = logging.getLogger("audio_io")

# ---------------------------------------------------------------------------
# Shared Robot State
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

    # UDP multicast parameters for RockChip microphone
    mic_multicast_group: str = "239.168.123.161"
    mic_multicast_port: int = 5555
    # Local network IP facing the robot (default for Jetson is 192.168.123.164)
    mic_local_ip: Optional[str] = field(default_factory=lambda: os.environ.get("VOICE_ENGINE_MIC_LOCAL_IP", "192.168.123.164"))
    mic_recv_buf_size: int = 65536
    mic_queue_maxsize: int = 400


# ---------------------------------------------------------------------------
# DDS and Audio Client Initialization
# ---------------------------------------------------------------------------
_channel_initialized = False
_channel_lock = threading.Lock()

def init_unitree_channel(network_interface: Optional[str] = None) -> None:
    """Initializes the DDS network (done exactly once per process)."""
    global _channel_initialized
    network_interface = network_interface or os.environ.get("VOICE_ENGINE_DDS_INTERFACE", "eth0")
    with _channel_lock:
        if not _channel_initialized and SDK_AVAILABLE:
            try:
                ChannelFactoryInitialize(0, network_interface)
                _channel_initialized = True
                logger.info("Unitree DDS ChannelFactory initialized (Interface: %s)", network_interface)
            except Exception as e:
                logger.warning("Error initializing ChannelFactory: %s", e)


class G1AudioClientWrapper:
    """
    Singleton wrapper for AudioClient.
    Used only for playback (PlayStream) and best-effort recording management.
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
                logger.info("Unitree G1 AudioClient successfully initialized.")
            except Exception as e:
                logger.error(
                    "Failed to initialize AudioClient (%s: %s). "
                    "Microphone will continue to work via UDP multicast, but "
                    "PlayStream playback will be unavailable.",
                    type(e).__name__, e,
                )
        else:
            logger.warning("Mock client used due to missing SDK.")

        self._initialized = True

    def start_recording(self) -> None:
        if not self.client:
            return
        try:
            self.client.StartRecording()
            logger.info("DDS/RPC: StartRecording() sent (best-effort).")
        except AttributeError:
            logger.debug("AudioClient.StartRecording() missing in this SDK version - skipping.")
        except Exception as e:
            logger.warning("StartRecording() returned an error (non-critical): %s", e)

    def stop_recording(self) -> None:
        if not self.client:
            return
        try:
            self.client.StopRecording()
            logger.info("DDS/RPC: StopRecording() sent (best-effort).")
        except AttributeError:
            logger.debug("AudioClient.StopRecording() missing in this SDK version - skipping.")
        except Exception as e:
            logger.warning("StopRecording() returned an error (non-critical): %s", e)

    # Chunk size sent per PlayStream() call (32000 bytes = 1 sec at 16kHz)
    _PLAYSTREAM_CHUNK_BYTES = 32000

    def play_stream(self, audio_bytes: bytes, sample_rate: int) -> bool:
        if not self.client:
            logger.warning(
                "AudioClient unavailable - %d bytes of audio NOT sent to robot speaker.",
                len(audio_bytes),
            )
            return False

        if not hasattr(self.client, "PlayStream"):
            logger.error(
                "AudioClient.PlayStream() missing in the installed unitree_sdk2_python version. "
                "Audio NOT sent to speaker."
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
                    "PlayStream() error at offset %d/%d bytes: %s",
                    offset, total, e,
                )
                return False
            if ret_code != 0:
                logger.error("PlayStream() returned code %s at offset %d/%d bytes.", ret_code, offset, total)
                return False
        return True


# ---------------------------------------------------------------------------
# Microphone Capture: UDP Multicast
# ---------------------------------------------------------------------------
class MulticastMicReceiver:
    """Listens to the raw PCM stream from the G1 microphone via UDP multicast."""

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
                f"Failed to subscribe to multicast group "
                f"{self.multicast_group}:{self.port} with local IP "
                f"{self.local_ip}. Error: {e}"
            ) from e

        self._sock = sock
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="MicMulticastRecv"
        )
        self._thread.start()
        logger.info(
            "G1 Microphone: UDP multicast subscription %s:%d (local_ip=%s) active.",
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
        """Drains the accumulated queue (e.g., while is_speaking=True)."""
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
        logger.info("G1 Microphone: UDP multicast reception stopped.")


# ---------------------------------------------------------------------------
# Microphone Capture: Local Laptop Audio
# ---------------------------------------------------------------------------
class LocalMicSource:
    """PCM source from standard laptop microphone via sounddevice."""

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
                "The 'sounddevice' package is not installed. Required for local microphone mode."
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
                f"Failed to open local laptop microphone (device={self.device!r}): {e}"
            ) from e

        logger.info(
            "Laptop microphone (LocalMicSource) started: device=%s, sr=%d, ch=%d",
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
        logger.info("Laptop microphone (LocalMicSource) stopped.")


# ---------------------------------------------------------------------------
# Capture + VAD
# ---------------------------------------------------------------------------
class AudioListener:
    def __init__(
        self,
        shared_state: SharedRobotState,
        config: Optional[AudioConfig] = None,
        local_mode: bool = False,
        mic_device: Optional[Union[int, str]] = None,
    ) -> None:
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
            logger.info("AudioListener initialized in LOCAL mode (laptop microphone).")
        else:
            self.g1_audio = G1AudioClientWrapper()

            self.mic = MulticastMicReceiver(
                multicast_group=self.cfg.mic_multicast_group,
                port=self.cfg.mic_multicast_port,
                local_ip=self.cfg.mic_local_ip,
                recv_buf_size=self.cfg.mic_recv_buf_size,
                queue_maxsize=self.cfg.mic_queue_maxsize,
            )
            logger.info("AudioListener initialized in ROBOT mode (UDP multicast).")

        self._byte_buffer = bytearray()
        self._samples_per_frame = self.cfg.vad_frame_samples
        self._bytes_per_frame = self._samples_per_frame * 2  

    def _load_silero_vad(self) -> Tuple[Any, Any]:
        logger.info("Loading Silero-VAD v4 model...")
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
        logger.info("Audio capture started (local_mode=%s).", self.local_mode)

    def stop(self) -> None:
        self._stop_event.set()
        self.mic.stop()
        if self.g1_audio is not None:
            self.g1_audio.stop_recording()
        logger.info("Audio capture stopped.")

    def close(self) -> None:
        self.stop()

    def _reset_phrase_state(self) -> None:
        self._speech_buffer = []
        self._is_recording_phrase = False
        self._byte_buffer.clear()
        self._vad_iterator.reset_states()

    def listen_for_phrase(self, poll_timeout: float = 0.05) -> Optional[np.ndarray]:
        self.mic.drain()

        while not self._stop_event.is_set():
            if self.state.is_speaking:
                self.mic.drain()
                if self._is_recording_phrase:
                    logger.info("is_speaking=True: dropping incomplete recording.")
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
                    logger.exception("Error in Silero-VAD")
                    continue

                if vad_event is not None and "start" in vad_event:
                    logger.info("VAD: speech start detected.")
                    self._is_recording_phrase = True

                if self._is_recording_phrase:
                    self._speech_buffer.append(frame)

                if vad_event is not None and "end" in vad_event and self._is_recording_phrase:
                    phrase_len = len(self._speech_buffer) * self.cfg.vad_frame_samples / float(self.cfg.sample_rate)
                    logger.info("VAD: speech end. Phrase assembled, length %.2f s.", phrase_len)

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
        Fast, lightweight RMS energy check for voice interruption over robot's speech.
        Returns True if a sustained voice interruption is detected.
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
# Playback
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
            raise FileNotFoundError(f"WAV file not found: {wav_file_path}")

        duration_sec = self._get_wav_duration(str(path))
        logger.info("Audio: playback start '%s' (%.2f s).", path.name, duration_sec)

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
            logger.info("Audio: playback finished.")

    def _play_with_command(self, wav_file_path: str, duration_sec: float) -> None:
        if not self.fallback_command:
            logger.warning("VOICE_ENGINE_PLAYBACK_COMMAND is empty — WAV not played.")
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
    """Playback of WAV via laptop speakers (sounddevice) without robot network dependency."""

    def __init__(self, shared_state: SharedRobotState) -> None:
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError(
                "The 'sounddevice' package is not installed. Required for local speaker playback."
            )
        self.state = shared_state

    def play(self, wav_file_path: str) -> None:
        path = Path(wav_file_path)
        if not path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_file_path}")

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
        logger.info("Local: playback start '%s' (%.2f s) via laptop speakers.", path.name, duration_sec)

        self.state.is_speaking = True
        try:
            sd.play(audio_array, samplerate=framerate)
            sd.wait()
        except Exception as e:
            logger.error("Local playback error: %s", e)
            raise
        finally:
            self.state.is_speaking = False
            logger.info("Local: playback finished.")