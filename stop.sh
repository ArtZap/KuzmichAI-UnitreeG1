#!/bin/bash

# ==============================================================================
# Unitree G1 Voice Engine - Stop Script
# ==============================================================================

# --- Настройки Цветов ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}  Unitree G1 Voice Engine - Shutting Down...        ${NC}"
echo -e "${BLUE}====================================================${NC}"

# 1. Поиск и остановка супервизора (родительского bash-скрипта) и Python процесса
# Используем pgrep, чтобы найти процесс python main.py
PYTHON_PIDS=$(pgrep -f "python main.py")

if [ -n "$PYTHON_PIDS" ]; then
    for PID in $PYTHON_PIDS; do
        # Ищем родительский процесс (наш bash-скрипт с циклом while true)
        SUPERVISOR_PID=$(ps -o ppid= -p "$PID" | tr -d ' ')
        
        if [ -n "$SUPERVISOR_PID" ] && [ "$SUPERVISOR_PID" -ne 1 ]; then
            echo -e "${YELLOW}[!] Отправка сигнала SIGTERM скрипту-супервизору (PID: $SUPERVISOR_PID)...${NC}"
            kill -TERM "$SUPERVISOR_PID" 2>/dev/null
        fi
        
        echo -e "${YELLOW}[!] Отправка сигнала SIGTERM процессу Python (main.py) (PID: $PID)...${NC}"
        kill -TERM "$PID" 2>/dev/null
    done
    
    # Ждем пару секунд для корректного завершения (Graceful Shutdown)
    sleep 2
    
    # Если процессы зависли, добиваем их через SIGKILL
    for PID in $PYTHON_PIDS; do
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "${RED}[-] Процесс Python (PID: $PID) не остановился. Принудительное завершение (SIGKILL)...${NC}"
            kill -9 "$PID" 2>/dev/null
        fi
    done
else
    echo -e "${GREEN}[+] Процессы Python (main.py) не найдены.${NC}"
fi

# 2. Остановка процесса воспроизведения звука (если он завис)
AUDIO_PLAY_PIDS=$(pgrep -f "g1_audio_play")
if [ -n "$AUDIO_PLAY_PIDS" ]; then
    echo -e "${YELLOW}[!] Принудительная остановка фоновых процессов g1_audio_play...${NC}"
    pkill -9 -f "g1_audio_play"
fi

echo -e "${GREEN}[+] Все процессы голосового модуля успешно остановлены!${NC}"
