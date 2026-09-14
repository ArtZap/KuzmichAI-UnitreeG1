#!/bin/bash

# Unitree G1 Voice Engine - Startup & Supervisor Script

cd "$(dirname "$(readlink -f "$0")")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ==============================================================================
# CONFIGURATION BLOCK
# ==============================================================================

# Main Operating Modes
export VOICE_ENGINE_ENABLE_STREAMING="false"  # true: streaming generation, false: wait for complete response
export VOICE_ENGINE_ENABLE_PLAYBACK="true"    # true: robot speaks the response, false: console output only
export VOICE_ENGINE_AUDIO="local"
export VOICE_ENGINE_ENABLE_CONTEXT="true"     # true: robot remembers last phrases, false: isolated questions
export VOICE_ENGINE_MAX_HISTORY_TURNS=5       # number of QA pairs remembered
export VOICE_ENGINE_ENABLE_GESTURES="true"   # true: robot gestures, false: stands still

# STT Settings (Speech Recognition)
export VOICE_ENGINE_STT_BACKEND="whisper"
export VOICE_ENGINE_WHISPER_MODEL="large-v3-turbo"
export VOICE_ENGINE_NEMO_MODEL="nvidia/parakeet-tdt-0.6b-v3"
export VOICE_ENGINE_STT_LANGUAGE="auto"

# TTS Settings (Text-to-Speech)
export VOICE_ENGINE_TTS="xtts"                # "espeak", "piper", "xtts" or "auto"
export VOICE_ENGINE_TTS_FALLBACK_PIPER="1"    # 1: use Piper if XTTS fails
export VOICE_ENGINE_TTS_TEMPO="0.9"           # Speech speed
export VOICE_ENGINE_TTS_GAIN_DB="16"          # Sound amplification in dB
export VOICE_ENGINE_TTS_WARMUP_LANGS="all"
export VOICE_ENGINE_XTTS_SPLIT_SENTENCES="true"

# Network and G1 Settings
export VOICE_ENGINE_DDS_INTERFACE="eth0"
export VOICE_ENGINE_MIC_LOCAL_IP="192.168.123.164"
export VOICE_ENGINE_G1_VOLUME="90"
export VOICE_ENGINE_ROBOT_IP="192.168.1.103"

# System Settings and Python
export CUDA_VISIBLE_DEVICES="0"
export JACK_NO_AUDIO_RESERVATION="1"
export AUDIODEV="null"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export HF_ENDPOINT="https://huggingface.co"

export TOKENIZERS_PARALLELISM="false"
export NO_PROXY="127.0.0.1,localhost,192.168.1.103,192.168.123.161,192.168.123.164"
PYTHON_BIN="python3.11"
export VOICE_ENGINE_LLM_MODEL="models/model-q4_K_M.gguf"
# export VOICE_ENGINE_LLM_MODEL="models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

# Noise Cancellation and Trigger Word Settings
export VOICE_ENGINE_NOISE_THRESHOLD="0.75"     # RMS filter threshold
export VOICE_ENGINE_USE_TRIGGERS="true"       # true: respond to trigger words
export VOICE_ENGINE_SERVER_URL="http://192.168.1.103:8002" # Perception server URL
export VOICE_ENGINE_ENABLE_INTERRUPT="false"

export VOICE_ENGINE_PLAYBACK_COMMAND="env -u LD_LIBRARY_PATH -u CYCLONEDDS_URI /home/unitree/g1_audio_play --iface ${VOICE_ENGINE_DDS_INTERFACE} --volume ${VOICE_ENGINE_G1_VOLUME} --file"
export VOICE_ENGINE_PLAYBACK_TIMEOUT_PAD="8.0"
export VOICE_ENGINE_PLAYBACK_TIMEOUT_MIN="10.0"
export LD_LIBRARY_PATH="$(pwd)/venv/lib/python3.11/site-packages/nvidia/cublas/lib:$(pwd)/venv/lib/python3.11/site-packages/nvidia/cudnn/lib:$(pwd)/piper:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# ==============================================================================

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}  Unitree G1 Voice Engine - Auto-Start Supervisor   ${NC}"
echo -e "${BLUE}====================================================${NC}"

echo -e "${GREEN}[+] Checking permissions for Jetson hardware acceleration...${NC}"
sudo -v || { echo -e "${RED}[-] Error: failed to get sudo privileges.${NC}"; exit 1; }

LOG_FILE="g1_voice_engine.log"
echo -e "${GREEN}[+] Configuring logging to ${LOG_FILE}...${NC}"
exec > >(tee -a "$LOG_FILE") 2>&1

PYTHON_PID=""
SHUTTING_DOWN=false

cleanup() {
    echo -e "\\n${YELLOW}[!] Received SIGINT/SIGTERM signal. Shutting down...${NC}"
    SHUTTING_DOWN=true
    
    if [ -n "$PYTHON_PID" ]; then
        echo -e "${YELLOW}[!] Stopping Python process (PID: $PYTHON_PID)...${NC}"
        kill -TERM "$PYTHON_PID" 2>/dev/null
        wait "$PYTHON_PID" 2>/dev/null
    fi
    
    echo -e "${YELLOW}[!] Running stop.sh...${NC}"
    if [ -f "./stop.sh" ]; then
        bash ./stop.sh
    else
        echo -e "${RED}[-] stop.sh script not found in the current directory.${NC}"
    fi

    echo -e "${GREEN}[+] Engine stopped successfully. Goodbye!${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

echo -e "${GREEN}[+] Switching hardware to MAXN mode...${NC}"
sudo nvpmodel -m 0 2>/dev/null || true
sudo jetson_clocks 2>/dev/null || true

echo -e "${GREEN}[+] Removing open file limits (ulimit -n 65535)...${NC}"
ulimit -n 65535

VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}[!] Virtual environment not found. Creating with ${PYTHON_BIN}...${NC}"
    $PYTHON_BIN -m venv "$VENV_DIR"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}[-] Error creating venv. Make sure ${PYTHON_BIN}-venv package is installed.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}[+] Activating virtual environment...${NC}"
    source "$VENV_DIR/bin/activate"
    
    echo -e "${GREEN}[+] Installing dependencies from requirements.txt...${NC}"
    if [ -f "requirements.txt" ]; then
        pip install --upgrade pip
        pip install -r requirements.txt
    else
        echo -e "${RED}[-] requirements.txt not found! Skipping library installation.${NC}"
    fi
else
    echo -e "${GREEN}[+] Activating existing virtual environment...${NC}"
    source "$VENV_DIR/bin/activate"
fi

echo -e "${GREEN}[+] Starting main pipeline (main.py)...${NC}"

while true; do
    if [ "$SHUTTING_DOWN" = true ]; then
        break
    fi

    echo -e "${BLUE}[*] Starting Python process...${NC}"
    
    python main.py &
    PYTHON_PID=$!
    
    wait $PYTHON_PID
    EXIT_CODE=$?

    if [ "$SHUTTING_DOWN" = true ]; then
        break
    fi

    if [ $EXIT_CODE -ne 0 ]; then
        echo -e "${RED}[-] Process crashed with exit code: $EXIT_CODE.${NC}"
    else
        echo -e "${YELLOW}[!] Process exited normally, but continuous operation was expected.${NC}"
    fi

    echo -e "${YELLOW}[!] Auto-restart in 5 seconds... (Press Ctrl+C to cancel)${NC}"
    sleep 5
done
