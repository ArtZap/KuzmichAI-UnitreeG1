#!/usr/bin/env python3
"""Runtime validation for KuzmichAI on Unitree G1 EDU."""

from __future__ import annotations

import json
import ctypes
import os
import shutil
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def preload_ctranslate2_runtime() -> None:
    lib_dir = Path(os.environ.get("CTRANSLATE2_LIB_DIR", "/home/unitree/AgroBot/build_deps/ctranslate2_cuda_install/lib"))
    lib_path = lib_dir / "libctranslate2.so.4"
    if lib_path.exists():
        try:
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


def ok(message: str) -> None:
    print("[OK] " + message)


def warn(message: str) -> None:
    print("[WARN] " + message)


def fail(message: str) -> None:
    print("[FAIL] " + message)
    raise SystemExit(1)


def require_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        fail("{} missing: {}".format(label, path))
    ok("{}: {} ({:.1f} MB)".format(label, path, path.stat().st_size / 1024 / 1024))


def main() -> None:
    if sys.version_info[:2] != (3, 8):
        warn("Expected Python 3.8 on robot, got {}".format(sys.version.split()[0]))
    else:
        ok("Python {}".format(sys.version.split()[0]))

    require_file(PROJECT_DIR / "models" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "LLM GGUF")
    require_file(PROJECT_DIR / "speaker.wav", "XTTS speaker reference")

    xtts_dir = Path(os.environ.get("XTTS_MODEL_DIR", PROJECT_DIR / "models" / "xtts_v2"))
    require_file(xtts_dir / "model.pth", "XTTS model")
    require_file(xtts_dir / "config.json", "XTTS config")
    require_file(xtts_dir / "vocab.json", "XTTS vocab")
    require_file(xtts_dir / "speakers_xtts.pth", "XTTS speakers")

    config = json.loads((xtts_dir / "config.json").read_text(encoding="utf-8"))
    xtts_langs = set(config.get("languages") or [])

    from stt_engine import STTEngine
    required_langs = set(STTEngine.supported_tts_languages())
    missing_xtts = sorted(required_langs - xtts_langs)
    if missing_xtts:
        warn("XTTS config does not list required languages: {}".format(missing_xtts))
    else:
        ok("XTTS covers all STT mapped languages: {}".format(sorted(required_langs)))

    piper_bin = Path(os.environ.get("PIPER_BINARY", PROJECT_DIR / "piper" / "piper"))
    require_file(piper_bin, "Piper binary")

    from tts_engine import PIPER_MODEL_FILES
    piper_dir = Path(os.path.expanduser(os.environ.get("PIPER_MODELS_DIR", "~/.local/share/piper")))
    missing_piper = []
    for name in PIPER_MODEL_FILES.values():
        model_path = piper_dir / name
        config_path = piper_dir / "{}.json".format(name)
        if not model_path.exists() or model_path.stat().st_size == 0:
            missing_piper.append(str(model_path))
        if not config_path.exists() or config_path.stat().st_size == 0:
            missing_piper.append(str(config_path))
    if missing_piper:
        warn("Missing Piper fallback voices: {}".format(missing_piper))
    else:
        ok("Piper fallback voices exist for all mapped languages")

    whisper_candidates = [
        Path(os.environ.get("VOICE_ENGINE_WHISPER_MODEL", "")),
        PROJECT_DIR / "models" / "whisper_small",
        Path("/home/unitree/AgroBot/models/whisper_small"),
        Path("/home/unitree/agrobot/AgroHub/models/whisper_small"),
    ]
    whisper_local = next((p for p in whisper_candidates if p and (p / "model.bin").exists()), None)
    if whisper_local:
        ok("Local faster-whisper model: {}".format(whisper_local))
    else:
        warn("No local faster-whisper model found; STT may rely on external cache unless VOICE_ENGINE_WHISPER_MODEL is set")

    embedder_candidates = [
        Path(os.environ.get("VOICE_ENGINE_EMBEDDER_MODEL", "")),
        PROJECT_DIR / "models" / "sentence_transformer",
        Path("/home/unitree/AgroBot-G1-Unified/models/sentence_transformer"),
        Path("/home/unitree/agrobot/AgroHub/models/sentence_transformer"),
    ]
    embedder_local = next((p for p in embedder_candidates if p and (p / "modules.json").exists()), None)
    if embedder_local:
        ok("Local sentence-transformer: {}".format(embedder_local))
    else:
        warn("No local sentence-transformer found; semantic cache may try online model resolution")

    if shutil.which("sox"):
        ok("sox found")
    else:
        fail("sox missing")

    try:
        import torch
        ok("torch {} cuda={} cuda_version={}".format(torch.__version__, torch.cuda.is_available(), getattr(torch.version, "cuda", None)))
        if not torch.cuda.is_available():
            warn("XTTS will run on CPU unless a Jetson CUDA PyTorch build is installed")
    except Exception as exc:
        warn("torch import failed: {}".format(exc))

    try:
        preload_ctranslate2_runtime()
        import ctranslate2
        count = ctranslate2.get_cuda_device_count()
        ok("ctranslate2 {} cuda_devices={}".format(ctranslate2.__version__, count))
        if count < 1:
            warn("faster-whisper will not use CUDA with current CTranslate2 package")
    except Exception as exc:
        warn("ctranslate2 import failed: {}".format(exc))

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: F401
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient  # noqa: F401
        ok("unitree_sdk2py AudioClient imports")
    except Exception as exc:
        warn("unitree_sdk2py audio imports failed: {}".format(exc))

    print("Validation complete.")


if __name__ == "__main__":
    main()
