#!/bin/bash

# Останавливать выполнение при любой ошибке
set -e

# --- Настройки Цветов ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}   Первичная настройка Voice Engine 'Кузьмич'       ${NC}"
echo -e "${BLUE}====================================================${NC}"

# 1. Проверка прав sudo
echo -e "\n${YELLOW}[1/7] Получение прав администратора...${NC}"
sudo -v

# 2. Установка системных зависимостей
echo -e "\n${YELLOW}[2/7] Добавление PPA-репозитория для Python 3.11...${NC}"
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update

echo -e "\n${YELLOW}[2.1/7] Установка системных пакетов (Audio, Build Tools, Python)...${NC}"
sudo apt-get install -y \
    sox \
    libsox-fmt-all \
    libportaudio2 \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    cmake \
    gcc \
    g++ \
    ffmpeg

# 3. Настройка виртуального окружения
echo -e "\n${YELLOW}[3/7] Создание и активация виртуального окружения (venv)...${NC}"
PYTHON_BIN="python3.11"

if [ ! -d "venv" ]; then
    $PYTHON_BIN -m venv venv
    echo -e "${GREEN}[+] Виртуальное окружение создано.${NC}"
else
    echo -e "${GREEN}[+] Виртуальное окружение уже существует.${NC}"
fi

source venv/bin/activate
pip install --upgrade pip setuptools wheel

# 4. Установка базовых Python-зависимостей
echo -e "\n${YELLOW}[4/7] Установка зависимостей проекта...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo -e "${RED}[!] Файл requirements.txt не найден. Устанавливаю минимальный набор...${NC}"
    pip install faster-whisper TTS faiss-cpu sentence-transformers sounddevice soundfile numpy torch torchaudio
fi

# 5. Сборка llama-cpp-python с поддержкой CUDA
# Это критически важно для быстрой работы LLM на GPU
echo -e "\n${YELLOW}[5/7] Сборка llama-cpp-python с аппаратным ускорением (CUDA)...${NC}"
export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE=1
pip install --upgrade --no-cache-dir llama-cpp-python

# 6. Создание структуры директорий
echo -e "\n${YELLOW}[6/7] Создание структуры папок для кэша и моделей...${NC}"
mkdir -p cache_audio
mkdir -p cache_analyz_audio
mkdir -p models/xtts_v2
mkdir -p models/faster_whisper
mkdir -p models/sentence_transformer
echo -e "${GREEN}[+] Папки созданы.${NC}"

# 7. Выдача прав на выполнение
echo -e "\n${YELLOW}[7/7] Настройка прав доступа...${NC}"
chmod +x start.sh
chmod +x setup.sh

echo -e "\n${BLUE}====================================================${NC}"
echo -e "${GREEN}Настройка успешно завершена!${NC}"
echo -e "Перед запуском убедитесь, что вы поместили файлы моделей:"
echo -e "  1. Файл LLM (.gguf) -> в папку ${YELLOW}models/${NC}"
echo -e "  2. Файлы XTTS (model.pth, config.json, vocab.json) -> в папку ${YELLOW}models/xtts_v2/${NC}"
echo -e "  3. Файл голоса (speaker.wav) -> в ${YELLOW}корень проекта${NC}"
echo -e "\nДля старта бота выполните: ${GREEN}./start.sh${NC}"
echo -e "${BLUE}====================================================${NC}"
