"""
ext_mic.py

Laptop side of the "g1_extmic" audio mode (VOICE_ENGINE_AUDIO=g1_extmic).

An external microphone (e.g. a wired lavalier on a USB sound card) is plugged
into the robot's Jetson. `jetson_mic_bridge.py` runs there and:
  * streams the microphone as raw PCM (16 kHz, mono, int16 LE) over UDP, the
    same format the RockChip multicast uses, to the laptop that subscribed;
  * accepts WAV files over TCP and plays them on the robot speaker.

ExternalMicReceiver subscribes to that stream and exposes the same interface
as MulticastMicReceiver / LocalMicSource (start, get_chunk, drain, stop), so
AudioListener (VAD) works with it unchanged. BridgeAudioPlayer has the same
`play(wav_path)` contract as AudioPlayer / LocalAudioPlayer.

Every message is authenticated with a pre-shared key (HMAC-SHA256, truncated
to 16 bytes); deploy_jetson_bridge.sh creates it and copies it to the robot.
No timestamps are used, so the laptop and Jetson clocks need not agree.

Wire protocol (keep in sync with jetson_mic_bridge.py):
  UDP subscribe, laptop -> bridge, every SUBSCRIBE_INTERVAL_S:
      SUB_MAGIC | nonce(16) | mac(SUB_MAGIC | nonce)
  UDP audio, bridge -> laptop:
      nonce(16) | session(8) | seq(8, BE) | pcm | mac(all previous fields)
      nonce = one of the laptop's recent subscribe nonces, so packets captured
      earlier cannot be replayed later; seq grows within a bridge session.
  TCP play:
      bridge -> laptop  PLAY_MAGIC | challenge(16)
      laptop -> bridge  PLAY_MAGIC | length(8, BE) | mac(b"play" | challenge | length)
                        | WAV bytes | mac(b"data" | challenge | sha256(WAV))
      bridge -> laptop  b"OK\\n" after playback, or b"ERR <reason>\\n"
"""

from __future__ import annotations

import collections
import hashlib
import hmac
import logging
import os
import queue
import socket
import struct
import threading
import time
import wave
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ext_mic")

SUB_MAGIC = b"KZM_MIC_SUB2"
PLAY_MAGIC = b"KZMPLAY2"
MAC_LEN = 16
NONCE_LEN = 16
AUDIO_HEADER_LEN = NONCE_LEN + 8 + 8
SUBSCRIBE_INTERVAL_S = 1.0
RECENT_NONCES = 5

DEFAULT_MIC_PORT = 5556
DEFAULT_PLAY_PORT = 5557
DEFAULT_TOKEN_FILE = "~/.config/kuzmich/bridge_token"


def mac(key: bytes, *parts: bytes) -> bytes:
    return hmac.new(key, b"".join(parts), hashlib.sha256).digest()[:MAC_LEN]


def bridge_host_from_env() -> str:
    """Bridge address: VOICE_ENGINE_BRIDGE_HOST, else the robot IP used elsewhere."""
    return (
        os.environ.get("VOICE_ENGINE_BRIDGE_HOST", "").strip()
        or os.environ.get("VOICE_ENGINE_ROBOT_IP", "192.168.1.103").strip()
    )


def load_bridge_key() -> bytes:
    """Key from VOICE_ENGINE_BRIDGE_TOKEN or the token file written by deploy_jetson_bridge.sh."""
    token = os.environ.get("VOICE_ENGINE_BRIDGE_TOKEN", "").strip()
    path = Path(os.path.expanduser(
        os.environ.get("VOICE_ENGINE_BRIDGE_TOKEN_FILE", "").strip() or DEFAULT_TOKEN_FILE))
    if not token and path.exists():
        token = path.read_text().strip()
    if not token:
        raise RuntimeError(
            f"Mic bridge key not found ({path}). Run ./deploy_jetson_bridge.sh once: "
            "it creates the key and installs it on the robot."
        )
    return token.encode()


class ExternalMicReceiver:
    """PCM source fed by jetson_mic_bridge.py over UDP."""

    def __init__(
        self,
        bridge_host: str,
        bridge_port: int = DEFAULT_MIC_PORT,
        key: Optional[bytes] = None,
        recv_buf_size: int = 65536,
        queue_maxsize: int = 400,
        no_audio_warn_s: float = 5.0,
    ) -> None:
        self.bridge_host = bridge_host
        self.bridge_port = bridge_port
        self.key = key
        self.recv_buf_size = recv_buf_size
        self.no_audio_warn_s = no_audio_warn_s

        self._bridge_ip: Optional[str] = None
        self._sock: Optional[socket.socket] = None
        self._threads: list = []
        self._stop_event = threading.Event()
        self._queue: "queue.Queue[bytes]" = queue.Queue(maxsize=queue_maxsize)
        self._last_audio_at = 0.0
        self._nonces: "collections.deque[bytes]" = collections.deque(maxlen=RECENT_NONCES)
        self._nonce_lock = threading.Lock()
        self._session: Optional[bytes] = None
        self._last_seq = -1

    def start(self) -> None:
        if self._sock is not None:
            return
        if self.key is None:
            self.key = load_bridge_key()
        try:
            self._bridge_ip = socket.gethostbyname(self.bridge_host)
        except OSError as e:
            raise RuntimeError(f"Cannot resolve mic bridge host {self.bridge_host!r}: {e}") from e

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind(("", 0))
        sock.settimeout(0.5)
        self._sock = sock
        self._stop_event.clear()
        self._last_audio_at = 0.0

        self._threads = [
            threading.Thread(target=self._recv_loop, args=(sock,), daemon=True, name="ExtMicRecv"),
            threading.Thread(target=self._subscribe_loop, args=(sock,), daemon=True, name="ExtMicSubscribe"),
        ]
        for t in self._threads:
            t.start()
        logger.info(
            "External mic: subscribing to bridge %s:%d (local port %d).",
            self._bridge_ip, self.bridge_port, sock.getsockname()[1],
        )

    def _subscribe_loop(self, sock: socket.socket) -> None:
        warned = False
        started_at = time.monotonic()
        while not self._stop_event.is_set():
            nonce = os.urandom(NONCE_LEN)
            with self._nonce_lock:
                self._nonces.append(nonce)
            try:
                sock.sendto(SUB_MAGIC + nonce + mac(self.key, SUB_MAGIC, nonce),
                            (self._bridge_ip, self.bridge_port))
            except OSError as e:
                logger.debug("External mic: subscribe send failed: %s", e)

            since = time.monotonic() - (self._last_audio_at or started_at)
            if since > self.no_audio_warn_s and not warned:
                logger.warning(
                    "External mic: no audio from bridge %s:%d for %.0f s. "
                    "Is jetson_mic_bridge.py running on the robot (./deploy_jetson_bridge.sh)?",
                    self._bridge_ip, self.bridge_port, since,
                )
                warned = True
            elif since <= self.no_audio_warn_s and warned:
                logger.info("External mic: audio stream restored.")
                warned = False
            self._stop_event.wait(SUBSCRIBE_INTERVAL_S)

    def _accept(self, data: bytes) -> Optional[bytes]:
        """Returns the PCM payload of an authentic, fresh audio packet, else None."""
        if len(data) < AUDIO_HEADER_LEN + 2 + MAC_LEN:
            return None
        body, tag = data[:-MAC_LEN], data[-MAC_LEN:]
        if not hmac.compare_digest(mac(self.key, body), tag):
            return None
        nonce = body[:NONCE_LEN]
        session = body[NONCE_LEN:NONCE_LEN + 8]
        (seq,) = struct.unpack(">Q", body[NONCE_LEN + 8:AUDIO_HEADER_LEN])
        pcm = body[AUDIO_HEADER_LEN:]
        with self._nonce_lock:
            fresh = nonce in self._nonces
        if not fresh or len(pcm) % 2:
            return None
        if session != self._session:
            self._session, self._last_seq = session, -1  # bridge (re)started
        if seq <= self._last_seq:
            return None
        self._last_seq = seq
        return pcm

    def _recv_loop(self, sock: socket.socket) -> None:
        first = True
        rejected = 0
        while not self._stop_event.is_set():
            try:
                data, addr = sock.recvfrom(self.recv_buf_size)
            except socket.timeout:
                continue
            except OSError:
                break
            pcm = self._accept(data) if addr[0] == self._bridge_ip else None
            if pcm is None:
                rejected += 1
                if rejected in (1, 100, 1000):
                    logger.warning(
                        "External mic: dropped %d unauthenticated/stale packets (from %s). "
                        "If this persists, re-run ./deploy_jetson_bridge.sh to sync the key.",
                        rejected, addr[0],
                    )
                continue
            self._last_audio_at = time.monotonic()
            if first:
                logger.info("External mic: receiving audio from %s:%d.", addr[0], addr[1])
                first = False
            try:
                self._queue.put_nowait(pcm)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(pcm)
                except queue.Full:
                    pass

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
        for t in self._threads:
            t.join(timeout=2.0)
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        logger.info("External mic: stopped.")


class BridgeAudioPlayer:
    """Plays WAV files on the robot speaker through jetson_mic_bridge.py."""

    def __init__(
        self,
        shared_state,
        bridge_host: str,
        bridge_port: int = DEFAULT_PLAY_PORT,
        key: Optional[bytes] = None,
        connect_timeout_s: float = 5.0,
    ) -> None:
        self.state = shared_state
        self.bridge_host = bridge_host
        self.bridge_port = bridge_port
        self.key = key
        self.connect_timeout_s = connect_timeout_s
        self.timeout_pad_s = float(os.environ.get("VOICE_ENGINE_PLAYBACK_TIMEOUT_PAD", "8.0"))

    def play(self, wav_file_path: str) -> None:
        path = Path(wav_file_path)
        if not path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_file_path}")
        if self.key is None:
            self.key = load_bridge_key()

        with wave.open(str(path), "rb") as wf:
            duration_sec = wf.getnframes() / float(wf.getframerate())
        payload = path.read_bytes()
        logger.info("Bridge: playback start '%s' (%.2f s) on robot speaker.", path.name, duration_sec)

        # is_speaking pauses the microphone, so it must cover the whole remote playback
        self.state.is_speaking = True
        try:
            with socket.create_connection(
                (self.bridge_host, self.bridge_port), timeout=self.connect_timeout_s
            ) as sock:
                reader = sock.makefile("rb")
                hello = reader.read(len(PLAY_MAGIC) + NONCE_LEN)
                if len(hello) != len(PLAY_MAGIC) + NONCE_LEN or not hello.startswith(PLAY_MAGIC):
                    raise ConnectionError("unexpected greeting from bridge")
                challenge = hello[len(PLAY_MAGIC):]
                length = struct.pack(">Q", len(payload))
                # Upload and playback both happen before the reply, so one budget covers them
                sock.settimeout(duration_sec + self.timeout_pad_s + len(payload) / 50_000)
                sock.sendall(PLAY_MAGIC + length + mac(self.key, b"play", challenge, length))
                sock.sendall(payload)
                sock.sendall(mac(self.key, b"data", challenge, hashlib.sha256(payload).digest()))
                reply = reader.readline(256).strip()
            if reply != b"OK":
                logger.error("Bridge: playback failed: %s", reply.decode("utf-8", "replace") or "no reply")
        except OSError as e:
            logger.error(
                "Bridge: cannot play on %s:%d (%s). Is jetson_mic_bridge.py running?",
                self.bridge_host, self.bridge_port, e,
            )
        finally:
            self.state.is_speaking = False
            logger.info("Bridge: playback finished.")
