#!/usr/bin/env python3
"""
jetson_mic_bridge.py

Runs on the robot's Jetson for the "g1_extmic" audio mode (see G1_EXTMIC.md).
Standard library only (Python 3.6+), plus `arecord` from alsa-utils.

  * Captures an external microphone (USB sound card) with arecord as raw PCM
    16 kHz mono int16 LE - the RockChip multicast format - and sends it over
    UDP to the laptop that subscribed (the laptop renews a challenge-based,
    signed subscription every second, so the bridge never needs its IP).
  * Accepts WAV files over TCP from the laptop and plays them on the robot
    speaker with g1_audio_play (or aplay when that is not present).

Everything is authenticated with the pre-shared key in --token-file
(deploy_jetson_bridge.sh creates and installs it). Protocol: see ext_mic.py.

Usage:
    python3 jetson_mic_bridge.py                 # auto-detect the USB sound card
    python3 jetson_mic_bridge.py --list-devices
    python3 jetson_mic_bridge.py --device plughw:2,0
"""

import argparse
import collections
import hashlib
import hmac
import logging
import os
import re
import shlex
import socket
import struct
import subprocess
import tempfile
import threading
import time
import wave

# Protocol constants, shared with ext_mic.py on the laptop
HELLO_MAGIC = b"KZM_MIC_HELO"
CHAL_MAGIC = b"KZM_MIC_CHAL"
SUB_MAGIC = b"KZM_MIC_SUB3"
PLAY_MAGIC = b"KZMPLAY2"
MAC_LEN = 16
NONCE_LEN = 16

SAMPLE_RATE = 16000
CHUNK_BYTES = 1024            # 512 samples = 32 ms, one Silero-VAD frame
SUBSCRIBER_TTL_S = 5.0        # stop streaming when the laptop goes quiet
CHALLENGE_TTL_S = 5.0         # a subscribe must answer a challenge issued this recently
MAX_PENDING_CHALLENGES = 64
MAX_WAV_BYTES = 50 * 1024 * 1024
MAX_PLAY_S = 300.0
MAX_CLIENTS = 4
PRE_AUTH_TIMEOUT_S = 5.0

G1_PLAYER = "/home/unitree/g1_audio_play"
DEFAULT_TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "token")

log = logging.getLogger("mic_bridge")


def mac(key, *parts):
    return hmac.new(key, b"".join(parts), hashlib.sha256).digest()[:MAC_LEN]


def list_capture_devices():
    """Returns [("plughw:CARD,DEV", description)] from `arecord -l`."""
    try:
        out = subprocess.run(["arecord", "-l"], check=False, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, universal_newlines=True).stdout
    except FileNotFoundError:
        raise SystemExit("arecord not found: sudo apt install alsa-utils")
    devices = []
    for line in out.splitlines():
        m = re.match(r"card (\d+): (.*), device (\d+):", line)
        if m:
            # plughw resamples/remixes to 16 kHz mono in ALSA
            devices.append(("plughw:{},{}".format(m.group(1), m.group(3)), line.strip()))
    return devices


def pick_device(explicit):
    if explicit:
        return explicit
    for dev, desc in list_capture_devices():
        if "usb" in desc.lower():
            log.info("Using USB capture device %s: %s", dev, desc)
            return dev
    log.warning("No USB capture card found, falling back to the ALSA 'default' device. "
                "Use --list-devices / --device to choose one.")
    return "default"


def default_play_cmd(iface, volume):
    if os.path.exists(G1_PLAYER):
        return "env -u LD_LIBRARY_PATH -u CYCLONEDDS_URI {} --iface {} --volume {} --file".format(
            G1_PLAYER, iface, volume)
    return "aplay -q"


class MicStreamer:
    """arecord -> UDP to the current (authenticated) subscriber."""

    def __init__(self, sock, capture_cmd, key):
        self.sock = sock
        self.capture_cmd = capture_cmd
        self.key = key
        self.session = os.urandom(8)
        self.seq = 0
        self.subscriber = None          # (ip, port)
        self.nonce = None               # latest subscribe nonce, echoed in audio packets
        self.subscriber_seen = 0.0
        self._lock = threading.Lock()
        # Single-use challenges: server nonce -> (address it was issued to, issue time)
        self._challenges = collections.OrderedDict()

    def _issue_challenge(self, addr):
        challenge = os.urandom(NONCE_LEN)
        self._challenges[challenge] = (addr, time.monotonic())
        if len(self._challenges) > MAX_PENDING_CHALLENGES:
            # A HELLO flood must not evict the connected laptop's challenge, or its
            # renewal fails and the mic stream stops; drop the oldest stranger's instead.
            victim = next((c for c, (a, _) in self._challenges.items() if a != self.subscriber),
                          next(iter(self._challenges)))
            del self._challenges[victim]
        try:
            self.sock.sendto(CHAL_MAGIC + challenge, addr)
        except OSError as e:
            log.debug("challenge send failed: %s", e)

    def _redeem_challenge(self, challenge, addr):
        """True once for a fresh challenge issued to addr; replays fail."""
        issued = self._challenges.pop(challenge, None)
        return (issued is not None and issued[0] == addr
                and time.monotonic() - issued[1] < CHALLENGE_TTL_S)

    def subscription_loop(self):
        bad = 0
        expected = len(SUB_MAGIC) + 2 * NONCE_LEN + MAC_LEN
        while True:
            try:
                data, addr = self.sock.recvfrom(256)
            except OSError:
                return
            if data == HELLO_MAGIC:
                self._issue_challenge(addr)
                continue
            body, tag = data[:-MAC_LEN], data[-MAC_LEN:]
            challenge = body[len(SUB_MAGIC):len(SUB_MAGIC) + NONCE_LEN]
            # Check the MAC before redeeming, so junk packets cannot burn real challenges
            if (len(data) != expected or not data.startswith(SUB_MAGIC)
                    or not hmac.compare_digest(mac(self.key, body), tag)
                    or not self._redeem_challenge(challenge, addr)):
                bad += 1
                if bad in (1, 100, 1000):
                    log.warning("Rejected %d unauthenticated or replayed subscribe packets "
                                "(last from %s). Wrong key? Re-run deploy_jetson_bridge.sh.",
                                bad, addr[0])
                continue
            with self._lock:
                if addr != self.subscriber:
                    if self.subscriber and time.monotonic() - self.subscriber_seen < SUBSCRIBER_TTL_S:
                        log.warning("Subscriber switched %s:%d -> %s:%d (two assistants running?)",
                                    self.subscriber[0], self.subscriber[1], addr[0], addr[1])
                    else:
                        log.info("Laptop subscribed: %s:%d", addr[0], addr[1])
                self.subscriber = addr
                self.nonce = body[len(SUB_MAGIC) + NONCE_LEN:]
                self.subscriber_seen = time.monotonic()

    def _target(self):
        with self._lock:
            if self.subscriber and time.monotonic() - self.subscriber_seen < SUBSCRIBER_TTL_S:
                return self.subscriber, self.nonce
            if self.subscriber:
                log.info("Laptop %s:%d went quiet, pausing stream.", *self.subscriber)
                self.subscriber = None
            return None, None

    @staticmethod
    def _drain_stderr(pipe, tail):
        # arecord blocks once its stderr pipe is full, which would silently stop the audio
        for line in iter(pipe.readline, b""):
            tail.append(line.decode("utf-8", "replace").rstrip())

    def capture_loop(self):
        while True:
            log.info("Starting capture: %s", " ".join(self.capture_cmd))
            tail = collections.deque(maxlen=5)
            try:
                proc = subprocess.Popen(self.capture_cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
            except OSError as e:
                log.error("Cannot start capture: %s. Retrying in 2 s...", e)
                time.sleep(2)
                continue
            threading.Thread(target=self._drain_stderr, args=(proc.stderr, tail), daemon=True).start()
            buf = b""
            sent = 0
            try:
                while True:
                    data = proc.stdout.read1(CHUNK_BYTES)
                    if not data:
                        break
                    buf += data
                    while len(buf) >= CHUNK_BYTES:
                        chunk, buf = buf[:CHUNK_BYTES], buf[CHUNK_BYTES:]
                        target, nonce = self._target()
                        if target is None:
                            continue
                        self.seq += 1
                        body = nonce + self.session + struct.pack(">Q", self.seq) + chunk
                        try:
                            self.sock.sendto(body + mac(self.key, body), target)
                            sent += 1
                            if sent == 1:
                                log.info("Streaming microphone to %s:%d", *target)
                        except OSError as e:
                            log.debug("send failed: %s", e)
            finally:
                proc.kill()
                proc.wait()
            log.error("Capture stopped (rc=%s). %s Restarting in 2 s...",
                      proc.returncode, " | ".join(tail))
            time.sleep(2)


class PlaybackServer:
    """TCP: authenticate, receive a WAV, play it on the robot speaker, reply OK/ERR."""

    def __init__(self, port, play_cmd, key, save_dir=None):
        self.port = port
        self.play_cmd = shlex.split(play_cmd)
        self.key = key
        self.save_dir = save_dir
        self._play_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(MAX_CLIENTS)

    def serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", self.port))
        srv.listen(MAX_CLIENTS)
        log.info("Playback server on TCP :%d, player: %s", self.port, " ".join(self.play_cmd))
        while True:
            conn, addr = srv.accept()
            if not self._slots.acquire(blocking=False):
                log.warning("Too many playback connections, dropping %s", addr[0])
                conn.close()
                continue
            threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()

    @staticmethod
    def _recv_exact(conn, n):
        buf = bytearray()
        while len(buf) < n:
            part = conn.recv(min(65536, n - len(buf)))
            if not part:
                raise ConnectionError("connection closed after {} of {} bytes".format(len(buf), n))
            buf += part
        return bytes(buf)

    def _handle(self, conn, addr):
        try:
            with conn:
                try:
                    conn.settimeout(PRE_AUTH_TIMEOUT_S)
                    challenge = os.urandom(NONCE_LEN)
                    conn.sendall(PLAY_MAGIC + challenge)
                    header = self._recv_exact(conn, len(PLAY_MAGIC) + 8 + MAC_LEN)
                    length = header[len(PLAY_MAGIC):len(PLAY_MAGIC) + 8]
                    if (not header.startswith(PLAY_MAGIC) or not hmac.compare_digest(
                            mac(self.key, b"play", challenge, length), header[-MAC_LEN:])):
                        raise PermissionError("authentication failed")
                    (size,) = struct.unpack(">Q", length)
                    if not 44 <= size <= MAX_WAV_BYTES:
                        raise ValueError("bad size {}".format(size))
                    conn.settimeout(30 + size / 50000.0)
                    payload = self._recv_exact(conn, size)
                    tag = self._recv_exact(conn, MAC_LEN)
                    if not hmac.compare_digest(
                            mac(self.key, b"data", challenge, hashlib.sha256(payload).digest()), tag):
                        raise PermissionError("payload authentication failed")
                    conn.settimeout(None)
                    self._play(payload, addr)
                    conn.sendall(b"OK\n")
                except Exception as e:  # report every failure to the laptop instead of dying
                    log.error("Playback request from %s failed: %s", addr[0], e)
                    try:
                        conn.sendall("ERR {}\n".format(e).encode("utf-8", "replace"))
                    except OSError:
                        pass
        finally:
            self._slots.release()

    def _play(self, payload, addr):
        fd, path = tempfile.mkstemp(prefix="kuzmich_bridge_", suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        try:
            with wave.open(path, "rb") as wf:
                frame_bytes = wf.getnchannels() * wf.getsampwidth()
                rate = wf.getframerate()
            # Duration from the real data size, not the (untrusted) frame count in the header
            duration = min((len(payload) - 44) / float(frame_bytes * rate), MAX_PLAY_S)
            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                with open(os.path.join(self.save_dir, "play_{}.wav".format(int(time.time() * 1000))), "wb") as f:
                    f.write(payload)
            with self._play_lock:
                log.info("Playing %.2f s from %s", duration, addr[0])
                started = time.monotonic()
                res = subprocess.run(self.play_cmd + [path], check=False, timeout=duration + 15,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if res.returncode != 0:
                    raise RuntimeError("player rc={}: {}".format(
                        res.returncode, res.stderr.decode("utf-8", "replace").strip()[-200:]))
                # Some players return before the sound ends; keep the laptop's
                # "speaking" window (its mic is ignored meanwhile) aligned with real audio.
                left = duration - (time.monotonic() - started)
                if left > 0:
                    time.sleep(left)
        finally:
            os.unlink(path)


def load_key(path):
    try:
        with open(path) as f:
            token = f.read().strip()
    except OSError as e:
        raise SystemExit("Cannot read the bridge key {}: {}. Install it with "
                         "deploy_jetson_bridge.sh from the laptop.".format(path, e))
    if not token:
        raise SystemExit("Bridge key file {} is empty.".format(path))
    return token.encode()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mic-port", type=int, default=5556, help="UDP port for the mic stream (default 5556)")
    p.add_argument("--play-port", type=int, default=5557, help="TCP port for playback (default 5557)")
    p.add_argument("--device", help="ALSA capture device, e.g. plughw:2,0 (default: first USB card)")
    p.add_argument("--list-devices", action="store_true", help="list capture devices and exit")
    p.add_argument("--token-file", default=DEFAULT_TOKEN_FILE,
                   help="pre-shared key file (default: 'token' next to this script)")
    p.add_argument("--iface", default="eth0", help="network interface for g1_audio_play (default eth0)")
    p.add_argument("--volume", default="90", help="g1_audio_play volume (default 90)")
    p.add_argument("--play-cmd", help="player command; the WAV path is appended "
                                      "(default: g1_audio_play if present, else aplay)")
    p.add_argument("--capture-cmd", help="raw PCM s16le 16k mono source command instead of arecord "
                                         "(for tests)")
    p.add_argument("--save-dir", help="also keep every played WAV here (debugging)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.list_devices:
        for dev, desc in list_capture_devices():
            print("{}  ->  {}".format(dev, desc))
        return

    key = load_key(args.token_file)

    if args.capture_cmd:
        capture_cmd = shlex.split(args.capture_cmd)
    else:
        capture_cmd = ["arecord", "-q", "-D", pick_device(args.device), "-f", "S16_LE",
                       "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"]

    mic_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    mic_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    mic_sock.bind(("0.0.0.0", args.mic_port))
    log.info("Mic stream on UDP :%d, waiting for the laptop to subscribe...", args.mic_port)

    mic = MicStreamer(mic_sock, capture_cmd, key)
    player = PlaybackServer(args.play_port, args.play_cmd or default_play_cmd(args.iface, args.volume),
                            key, args.save_dir)

    threading.Thread(target=mic.subscription_loop, daemon=True).start()
    threading.Thread(target=mic.capture_loop, daemon=True).start()
    try:
        player.serve()
    except KeyboardInterrupt:
        log.info("Stopped.")


if __name__ == "__main__":
    main()
