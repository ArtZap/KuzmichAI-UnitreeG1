#!/bin/bash
set -e

export HF_HOME="$(pwd)/hf_cache"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}     Загрузка ML-моделей для Кузьмича (Offline)     ${NC}"
echo -e "${BLUE}====================================================${NC}"

mkdir -p models/xtts_v2
mkdir -p models/faster_whisper
mkdir -p models/sentence_transformer
mkdir -p models/gigaam
source venv/bin/activate

# 1. LLM (Saiga Llama 3 8B)
echo -e "\n${YELLOW}[1/5] Загрузка LLM через Python API (сплит-модели)...${NC}"

cat << 'PYEOF' > download_llm.py
import os
import glob
from huggingface_hub import snapshot_download

# Перенаправляем системный кэш в папку проекта
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")

print("Скачивание всех частей модели Q4_K_M...")
snapshot_download(
    repo_id="IlyaGusev/saiga_llama3_8b_gguf", 
    allow_patterns=["*q4_K_M*.gguf", "*q4_k_m*.gguf", "*Q4_K_M*.gguf", "*q4_K*.gguf"], 
    local_dir="models"
)

# Находим скачанный файл и переименовываем, чтобы start.sh точно его нашел
found_files = glob.glob("models/*q4_*.gguf") + glob.glob("models/*Q4_*.gguf")
if found_files:
    target = "models/model-q4_K_M.gguf"
    if found_files[0] != target:
        os.rename(found_files[0], target)
    print("Модель успешно загружена!")
else:
    print("Ошибка: файл не найден в репозитории.")
PYEOF

python download_llm.py
rm download_llm.py

# 2. TTS (XTTS v2)
echo -e "\n${YELLOW}[2/5] Загрузка файлов синтеза речи (XTTS v2)...${NC}"
XTTS_BASE="https://huggingface.co/coqui/XTTS-v2/resolve/main"
for file in config.json vocab.json model.pth speakers_xtts.pth; do
    if [ ! -f "models/xtts_v2/$file" ]; then
        echo "Скачивание $file..."
        wget -q --show-progress -O "models/xtts_v2/$file" "$XTTS_BASE/$file"
    fi
done

if [ ! -f "speaker.wav" ]; then
    echo "Скачивание референсного голоса (speaker.wav)..."
    wget -qO speaker.wav "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/ru_sample.wav"
fi

# 3. Резервный TTS (Piper - Суровые мужские голоса)
echo -e "\n${YELLOW}[3/5] Загрузка Piper-моделей (басовитые/мужские голоса)...${NC}"
PIPER_DIR="$HOME/.local/share/piper"
mkdir -p "$PIPER_DIR"

declare -A MALE_VOICES=(
    ["ru"]="ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium"
    ["en"]="en/en_US/ryan/medium/en_US-ryan-medium"
    ["es"]="es/es_ES/davefx/medium/es_ES-davefx-medium"
    ["fr"]="fr/fr_FR/tom/medium/fr_FR-tom-medium"
    ["de"]="de/de_DE/thorsten/medium/de_DE-thorsten-medium"
    ["zh-cn"]="zh/zh_CN/huayan/medium/zh_CN-huayan-medium"
    ["ar"]="ar/ar_JO/kareem/medium/ar_JO-kareem-medium"
    ["pt"]="pt/pt_BR/faber/medium/pt_BR-faber-medium"
    ["it"]="it/it_IT/riccardo/x_low/it_IT-riccardo-x_low"
    ["pl"]="pl/pl_PL/darkman/medium/pl_PL-darkman-medium"
    ["tr"]="tr/tr_TR/dfki/medium/tr_TR-dfki-medium"
    ["nl"]="nl/nl_NL/rdh/medium/nl_NL-rdh-medium"
    ["cs"]="cs/cs_CZ/jirka/medium/cs_CZ-jirka-medium"
    ["ja"]="ja/ja_JP/kenichi/medium/ja_JP-kenichi-medium"
    ["ko"]="ko/ko_KR/chunom/medium/ko_KR-chunom-medium"
    ["hu"]="hu/hu_HU/mate/medium/hu_HU-mate-medium"
)

BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main"

for lang in "${!MALE_VOICES[@]}"; do
    repo_path="${MALE_VOICES[$lang]}"
    filename=$(basename "$repo_path")
    
    echo "Скачиваю мужской голос для $lang ($filename)..."
    wget -q --show-progress -O "$PIPER_DIR/${filename}.onnx" "$BASE_URL/${repo_path}.onnx" || true
    wget -q -O "$PIPER_DIR/${filename}.onnx.json" "$BASE_URL/${repo_path}.onnx.json" || true
done

# 4. STT (Whisper & GigaAM)
echo -e "\n${YELLOW}[4/5] Предзагрузка Whisper, GigaAM и SentenceTransformers...${NC}"
cat << 'PYEOF' > preload.py
import os
import warnings
warnings.filterwarnings("ignore")

# Фиксируем кэш внутри папки проекта
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")

from huggingface_hub import snapshot_download, hf_hub_download
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer

print("-> Загрузка faster-whisper (small)...")
WhisperModel("small", device="cpu", compute_type="int8", download_root="models/faster_whisper")

print("-> Загрузка faster-whisper (large-v3-turbo)...")
snapshot_download(repo_id="deepdml/faster-whisper-large-v3-turbo-ct2", local_dir="models/faster_whisper/large-v3-turbo")

print("-> Загрузка GigaAM-v3-e2e-rnnt (INT8)...")
hf_hub_download(repo_id="istupakov/gigaam-v3-onnx", filename="v3_e2e_ctc.int8.onnx", local_dir="models/gigaam")
if os.path.exists("models/gigaam/v3_e2e_ctc.int8.onnx"):
    os.rename("models/gigaam/v3_e2e_ctc.int8.onnx", "models/gigaam/gigaam_v3_e2e_rnnt_int8.onnx")

print("-> Загрузка SentenceTransformer для семантической памяти...")
SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", cache_folder="models/sentence_transformer")
PYEOF

python preload.py
rm preload.py

# 5. Очистка 
echo -e "\n${YELLOW}[5/5] Очистка временных файлов...${NC}"
rm -rf hf_cache

echo -e "\n${GREEN}Все необходимые модели загружены! Кузьмич готов к автономной работе.${NC}"
echo "Запустите пайплайн командой: ./start.sh"