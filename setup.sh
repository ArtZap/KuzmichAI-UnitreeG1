#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}   Initial Setup of Voice Engine 'Kuzmich'          ${NC}"
echo -e "${BLUE}====================================================${NC}"

echo -e "\n${YELLOW}[1/8] Acquiring administrator privileges...${NC}"
sudo -v

echo -e "\n${YELLOW}[2/8] Adding PPA repository for Python 3.11...${NC}"
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update

echo -e "\n${YELLOW}[2.1/8] Installing system packages (Audio, Build Tools, Python)...${NC}"
sudo apt-get install -y     sox     libsox-fmt-all     libportaudio2     python3.11     python3.11-venv     python3.11-dev     build-essential     cmake     gcc     g++     ffmpeg     espeak

echo -e "\n${YELLOW}[3/8] Creating and activating virtual environment (venv)...${NC}"
PYTHON_BIN="python3.11"

if [ ! -d "venv" ]; then
    $PYTHON_BIN -m venv venv
    echo -e "${GREEN}[+] Virtual environment created.${NC}"
else
    echo -e "${GREEN}[+] Virtual environment already exists.${NC}"
fi

source venv/bin/activate
pip install --upgrade pip setuptools wheel

echo -e "\n${YELLOW}[4/8] Installing project dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    pip install onnx-asr huggingface_hub
else
    echo -e "${RED}[!] requirements.txt not found. Installing minimum set...${NC}"
    pip install faster-whisper TTS faiss-cpu sentence-transformers sounddevice soundfile numpy torch torchaudio onnx-asr huggingface_hub
fi

echo -e "\n${YELLOW}[5/8] Building llama-cpp-python with hardware acceleration (CUDA)...${NC}"
export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE=1
pip install --upgrade --no-cache-dir llama-cpp-python

echo -e "\n${YELLOW}[6/8] Creating folder structure for cache and models...${NC}"
mkdir -p cache_audio
mkdir -p cache_analyz_audio
mkdir -p models/xtts_v2
mkdir -p models/faster_whisper
mkdir -p models/sentence_transformer
mkdir -p models/gigaam
echo -e "${GREEN}[+] Folders created.${NC}"

echo -e "\n${YELLOW}[6.1/8] Installing Piper binary (TTS)...${NC}"
if [ ! -d "piper" ] || [ ! -f "piper/piper" ]; then
    wget -q --show-progress https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz
    tar -xzf piper_linux_x86_64.tar.gz
    rm piper_linux_x86_64.tar.gz
    echo -e "${GREEN}[+] Piper executable downloaded successfully.${NC}"
else
    echo -e "${GREEN}[+] Piper executable already exists.${NC}"
fi

echo -e "\n${YELLOW}[6.2/8] Downloading models for hybrid STT (GigaAM + Whisper Turbo)...${NC}"
cat << 'PYEOF' > download_hybrid.py
import os
from huggingface_hub import hf_hub_download, snapshot_download

print("Downloading GigaAM-v3-e2e-rnnt (INT8)...")
try:
    hf_hub_download(
        repo_id="salute-developers/GigaAM",
        filename="gigaam_v3_e2e_rnnt_int8.onnx",
        local_dir="models/gigaam"
    )
except Exception as e:
    print(f"Warning: Failed to automatically download GigaAM: {e}")
    print("Please place gigaam_v3_e2e_rnnt_int8.onnx in models/gigaam/ manually.")

print("Downloading Whisper large-v3-turbo...")
try:
    snapshot_download(
        repo_id="Systran/faster-whisper-large-v3-turbo",
        local_dir="models/faster_whisper/large-v3-turbo"
    )
except Exception as e:
    print(f"Warning: Failed to automatically download Whisper Turbo: {e}")
    print("The model will be downloaded on the first run of STTEngine.")
    
print("Hybrid STT models successfully prepared!")
PYEOF
python download_hybrid.py
rm download_hybrid.py

echo -e "\n${YELLOW}[7/8] Configuring access rights...${NC}"
chmod +x start.sh
chmod +x setup.sh

echo -e "\n${BLUE}====================================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "Before starting, make sure you have placed the model files:"
echo -e "  1. LLM file (.gguf) -> in the ${YELLOW}models/${NC} folder"
echo -e "  2. XTTS files (model.pth, config.json, vocab.json) -> in the ${YELLOW}models/xtts_v2/${NC} folder"
echo -e "  3. Voice file (speaker.wav) -> in the ${YELLOW}project root${NC}"
echo -e "  4. GigaAM model -> in ${YELLOW}models/gigaam/${NC}"
echo -e "\nTo start the bot, run: ${GREEN}./start.sh${NC}"
echo -e "${BLUE}====================================================${NC}"