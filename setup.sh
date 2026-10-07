#!/bin/bash
# One-shot installer: system packages, venv, Python deps and every model the
# pipeline needs at runtime. Safe to re-run: finished steps are skipped.

set -e

cd "$(dirname "$(readlink -f "$0")")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}   Initial Setup of Voice Engine 'Kuzmich'          ${NC}"
echo -e "${BLUE}====================================================${NC}"

PYTHON_BIN="python3.11"
APT_PACKAGES="sox libsox-fmt-all libportaudio2 python3.11 python3.11-venv python3.11-dev build-essential cmake gcc g++ ffmpeg espeak wget"

echo -e "\n${YELLOW}[1/8] Checking system packages...${NC}"
MISSING=""
for pkg in $APT_PACKAGES; do
    dpkg -s "$pkg" >/dev/null 2>&1 || MISSING="$MISSING $pkg"
done

# sudo is only needed when something is missing, so a re-run works without a password
if [ -n "$MISSING" ]; then
    echo -e "${YELLOW}[!] Missing:${MISSING}${NC}"
    sudo -v
    sudo apt-get update
    if ! apt-cache show python3.11 >/dev/null 2>&1; then
        echo -e "\n${YELLOW}[2/8] Adding PPA repository for Python 3.11...${NC}"
        sudo apt-get install -y software-properties-common
        sudo add-apt-repository ppa:deadsnakes/ppa -y
        sudo apt-get update
    fi
    sudo apt-get install -y $MISSING
else
    echo -e "${GREEN}[+] All system packages are installed.${NC}"
fi

echo -e "\n${YELLOW}[3/8] Creating and activating virtual environment (venv)...${NC}"
if [ ! -d "venv" ]; then
    $PYTHON_BIN -m venv venv
    echo -e "${GREEN}[+] Virtual environment created.${NC}"
else
    echo -e "${GREEN}[+] Virtual environment already exists.${NC}"
fi

source venv/bin/activate
pip install --upgrade pip setuptools wheel

SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
CUBLAS_DIR="$SITE_PACKAGES/nvidia/cublas/lib"
# Same library path start.sh uses, so the GPU checks below see what runtime sees
export LD_LIBRARY_PATH="$CUBLAS_DIR:$SITE_PACKAGES/nvidia/cudnn/lib:$(pwd)/piper:${LD_LIBRARY_PATH:-}"

# Highest CUDA version the installed NVIDIA driver supports (major only), 0 without a GPU
DRIVER_CUDA_MAJOR=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    DRIVER_CUDA_MAJOR="$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' || echo 0)"
fi

echo -e "\n${YELLOW}[4/8] Installing PyTorch...${NC}"
# PyPI torch is built for CUDA 13 and needs driver >= 580; older drivers get the CUDA 12.6 build
if [ "$DRIVER_CUDA_MAJOR" -ge 13 ]; then
    echo -e "${GREEN}[+] NVIDIA driver supports CUDA ${DRIVER_CUDA_MAJOR}, using the default (CUDA 13) build.${NC}"
    pip install torch==2.14.1 torchaudio==2.11.0
elif [ "$DRIVER_CUDA_MAJOR" -ge 12 ]; then
    echo -e "${YELLOW}[!] NVIDIA driver supports CUDA ${DRIVER_CUDA_MAJOR}, using the CUDA 12.6 build.${NC}"
    pip install --index-url https://download.pytorch.org/whl/cu126 \
        --extra-index-url https://pypi.org/simple torch==2.14.1 torchaudio==2.11.0
else
    echo -e "${RED}[!] No NVIDIA GPU / driver older than CUDA 12 found. Installing CPU-only PyTorch;${NC}"
    echo -e "${RED}    Whisper, XTTS and the LLM will be very slow. Install the NVIDIA driver and re-run.${NC}"
    pip install --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple torch==2.14.1 torchaudio==2.11.0
fi

llama_has_gpu() {
    python -c "import llama_cpp, sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)" 2>/dev/null
}

# Installed before requirements.txt, otherwise pip builds a CPU-only llama-cpp-python first
echo -e "\n${YELLOW}[4.1/8] Installing llama-cpp-python...${NC}"
if llama_has_gpu; then
    echo -e "${GREEN}[+] llama-cpp-python with GPU offload is already installed.${NC}"
elif [ "$DRIVER_CUDA_MAJOR" -lt 12 ]; then
    echo -e "${YELLOW}[!] No usable GPU, building CPU-only llama-cpp-python.${NC}"
    pip install "llama-cpp-python==0.3.19"
elif command -v nvcc >/dev/null 2>&1; then
    echo -e "${GREEN}[+] nvcc found, building from source with CUDA.${NC}"
    CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
        pip install --force-reinstall --no-deps --no-cache-dir "llama-cpp-python==0.3.19"
    pip install "llama-cpp-python==0.3.19"
else
    # No CUDA toolkit: use the prebuilt CUDA 12.4 wheel (needs only the NVIDIA driver)
    echo -e "${YELLOW}[!] nvcc not found, installing prebuilt CUDA 12.4 wheel (~1.4 GB).${NC}"
    pip install --force-reinstall --no-deps --only-binary=:all: \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
        "llama-cpp-python==0.3.19"
    pip install "llama-cpp-python==0.3.19"
fi

echo -e "\n${YELLOW}[5/8] Installing project dependencies...${NC}"
# The original Coqui `TTS` and `coqpit` share package folders with their forks
# (`coqui-tts`, `coqpit-config`); remove the old ones first, then reinstall the forks
# so their files are not left half-deleted.
# nemo_toolkit is optional (see requirements.txt); an old one pins a broken `datasets`.
pip uninstall -y TTS coqpit nemo_toolkit 2>/dev/null || true
pip install -r requirements.txt
pip install --force-reinstall --no-deps coqui-tts==0.27.5 coqpit-config==0.2.5

# start.sh puts only nvidia/cublas/lib and nvidia/cudnn/lib on LD_LIBRARY_PATH;
# link the CUDA 12 runtime there so ctranslate2 and llama.cpp can find it.
CUDART_DIR="$SITE_PACKAGES/nvidia/cuda_runtime/lib"
if [ -d "$CUDART_DIR" ] && [ -d "$CUBLAS_DIR" ]; then
    for lib in "$CUDART_DIR"/libcudart.so*; do
        ln -sf "$lib" "$CUBLAS_DIR/$(basename "$lib")"
    done
fi

if llama_has_gpu; then
    echo -e "${GREEN}[+] LLM will run on the GPU.${NC}"
else
    echo -e "${RED}[!] llama-cpp-python has no GPU offload, the LLM will run on the CPU (slow).${NC}"
fi
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo -e "${GREEN}[+] PyTorch sees the GPU: Whisper/XTTS/Silero will use CUDA.${NC}"
else
    echo -e "${RED}[!] PyTorch does not see a GPU, speech models will run on the CPU.${NC}"
fi

echo -e "\n${YELLOW}[6/8] Creating folder structure for cache and models...${NC}"
mkdir -p cache_audio cache_analyz_audio models/xtts_v2 models/faster_whisper models/sentence_transformer
echo -e "${GREEN}[+] Folders created.${NC}"

echo -e "\n${YELLOW}[6.1/8] Installing Piper binary (TTS)...${NC}"
if [ ! -f "piper/piper" ]; then
    wget -q --show-progress https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz
    tar -xzf piper_linux_x86_64.tar.gz
    rm piper_linux_x86_64.tar.gz
    echo -e "${GREEN}[+] Piper executable downloaded successfully.${NC}"
else
    echo -e "${GREEN}[+] Piper executable already exists.${NC}"
fi

# Downloads into $2 unless it is already there; never leaves an empty file behind
fetch() {
    local url="$1" target="$2"
    [ -s "$target" ] && return 0
    echo "Downloading $(basename "$target")..."
    if ! wget -q --show-progress -O "$target.part" "$url"; then
        rm -f "$target.part"
        echo -e "${RED}[!] Failed to download $url${NC}"
        return 1
    fi
    mv "$target.part" "$target"
}

echo -e "\n${YELLOW}[7/8] Downloading models...${NC}"

echo -e "\n${YELLOW}[7.1] LLM (Saiga Llama 3 8B, Q4_K_M)...${NC}"
if [ -s "models/model-q4_K_M.gguf" ]; then
    echo -e "${GREEN}[+] LLM already present.${NC}"
else
    python - <<'PYEOF'
import glob, os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="IlyaGusev/saiga_llama3_8b_gguf",
    allow_patterns=["*q4_K_M*.gguf", "*q4_k_m*.gguf", "*Q4_K_M*.gguf", "*q4_K*.gguf"],
    local_dir="models",
)
found = [f for f in glob.glob("models/*q4_*.gguf") + glob.glob("models/*Q4_*.gguf")
         if not f.endswith("model-q4_K_M.gguf")]
if not found:
    raise SystemExit("LLM file not found in the repository")
os.rename(found[0], "models/model-q4_K_M.gguf")
PYEOF
fi

echo -e "\n${YELLOW}[7.2] XTTS v2...${NC}"
XTTS_BASE="https://huggingface.co/coqui/XTTS-v2/resolve/main"
for file in config.json vocab.json model.pth speakers_xtts.pth; do
    fetch "$XTTS_BASE/$file" "models/xtts_v2/$file"
done
fetch "$XTTS_BASE/samples/ru_sample.wav" "speaker.wav"

echo -e "\n${YELLOW}[7.3] Piper voices...${NC}"
PIPER_DIR="$HOME/.local/share/piper"
PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"
mkdir -p "$PIPER_DIR"
# Key: file name tts_engine.py looks for. Value: voice in rhasspy/piper-voices.
# nl/ja/ko/hu voices named in tts_engine.py do not exist upstream, so the closest
# available voice is saved under the expected name.
declare -A PIPER_VOICES=(
    ["ru_RU-dmitri-medium"]="ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium"
    ["en_US-ryan-medium"]="en/en_US/ryan/medium/en_US-ryan-medium"
    ["es_ES-davefx-medium"]="es/es_ES/davefx/medium/es_ES-davefx-medium"
    ["fr_FR-tom-medium"]="fr/fr_FR/tom/medium/fr_FR-tom-medium"
    ["de_DE-thorsten-medium"]="de/de_DE/thorsten/medium/de_DE-thorsten-medium"
    ["zh_CN-huayan-medium"]="zh/zh_CN/huayan/medium/zh_CN-huayan-medium"
    ["ar_JO-kareem-medium"]="ar/ar_JO/kareem/medium/ar_JO-kareem-medium"
    ["pt_BR-faber-medium"]="pt/pt_BR/faber/medium/pt_BR-faber-medium"
    ["it_IT-riccardo-x_low"]="it/it_IT/riccardo/x_low/it_IT-riccardo-x_low"
    ["pl_PL-darkman-medium"]="pl/pl_PL/darkman/medium/pl_PL-darkman-medium"
    ["tr_TR-dfki-medium"]="tr/tr_TR/dfki/medium/tr_TR-dfki-medium"
    ["cs_CZ-jirka-medium"]="cs/cs_CZ/jirka/medium/cs_CZ-jirka-medium"
    ["nl_NL-rdh-medium"]="nl/nl_NL/pim/medium/nl_NL-pim-medium"
    ["ja_JP-kenichi-medium"]="ja/ja_JP/hi_fi_captain/medium/ja_JP-hi_fi_captain-medium"
    ["ko_KR-chunom-medium"]="ko/ko_KR/kss/medium/ko_KR-kss-medium"
    ["hu_HU-mate-medium"]="hu/hu_HU/imre/medium/hu_HU-imre-medium"
)
for name in "${!PIPER_VOICES[@]}"; do
    src="${PIPER_VOICES[$name]}"
    fetch "$PIPER_BASE/$src.onnx" "$PIPER_DIR/$name.onnx" || true
    fetch "$PIPER_BASE/$src.onnx.json" "$PIPER_DIR/$name.onnx.json" || true
done

# start.sh runs with HF_HUB_OFFLINE=1, so everything fetched lazily must be cached now
echo -e "\n${YELLOW}[7.4] Whisper, embeddings, ruaccent, Silero VAD/TTS...${NC}"
python - <<'PYEOF'
import torch
from faster_whisper import download_model
from sentence_transformers import SentenceTransformer
from ruaccent import RUAccent

print("-> faster-whisper large-v3-turbo")
download_model("large-v3-turbo", cache_dir="models/faster_whisper")

print("-> paraphrase-multilingual-MiniLM-L12-v2")
st_dir = "models/sentence_transformer"
import os
if not os.path.exists(os.path.join(st_dir, "modules.json")):
    SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu").save(st_dir)

print("-> ruaccent")
RUAccent().load(omograph_model_size="turbo")

print("-> Silero VAD and Silero TTS")
torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
torch.hub.load("snakers4/silero-models", "silero_tts", language="ru", speaker="v4_ru", trust_repo=True)
PYEOF

# main.py synthesizes these phrases only when the files are missing and then replays
# them forever; move the copies shipped in the repo aside once so they are re-voiced
# with the current Silero voice on first start.
for phrase in startup.wav scanning.wav; do
    if [ -f "$phrase" ] && [ ! -f "$phrase.bak" ]; then
        mv "$phrase" "$phrase.bak"
        echo -e "${GREEN}[+] $phrase will be re-synthesized on first start (old one kept as $phrase.bak).${NC}"
    fi
done

echo -e "\n${YELLOW}[8/8] Configuring access rights...${NC}"
chmod +x start.sh setup.sh

echo -e "\n${BLUE}====================================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "To use your own voice for XTTS, replace ${YELLOW}speaker.wav${NC} in the project root."
echo -e "\nTo start the bot, run: ${GREEN}./start.sh${NC}"
echo -e "${BLUE}====================================================${NC}"
