# KuzmichAI v2 — Offline Voice Assistant for Unitree G1

An entirely offline, multilingual voice assistant tailored for the Unitree G1 EDU humanoid robot. "Kuzmich" features a unique persona of a grumbly Soviet agricultural robot, designed with strict safety guardrails and a highly responsive architecture.

## Key Technical Features
* **Asynchronous Pipeline:** A dual-worker streaming architecture minimizing latency between LLM generation and TTS synthesis.
* **LLM Engine:** Streaming token generation via `llama-cpp-python`, dynamically slicing tokens into complete sentences for instant playback.
* **STT Engine:** Integration of `faster-whisper` and NVIDIA NeMo (Parakeet) for robust speech recognition in high-noise environments.
* **TTS Engine:** High-fidelity local synthesis using Coqui XTTS v2, with a lightweight Piper fallback mechanism.
* **Hardware Audio Routing:** Direct interception of UDP multicast (239.168.123.161) from the robot's RockChip microphone, bypassing standard DDS overhead.
* **Semantic Caching:** Zero-latency responses for repetitive queries utilizing `FAISS` and `sentence-transformers`.
