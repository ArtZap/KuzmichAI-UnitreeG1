#!/bin/bash

# Unitree G1 Voice Engine - Stop Script

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}  Unitree G1 Voice Engine - Shutting Down...        ${NC}"
echo -e "${BLUE}====================================================${NC}"

PYTHON_PIDS=$(pgrep -f "python main.py")

if [ -n "$PYTHON_PIDS" ]; then
    for PID in $PYTHON_PIDS; do
        SUPERVISOR_PID=$(ps -o ppid= -p "$PID" | tr -d ' ')
        
        if [ -n "$SUPERVISOR_PID" ] && [ "$SUPERVISOR_PID" -ne 1 ]; then
            echo -e "${YELLOW}[!] Sending SIGTERM signal to supervisor script (PID: $SUPERVISOR_PID)...${NC}"
            kill -TERM "$SUPERVISOR_PID" 2>/dev/null
        fi
        
        echo -e "${YELLOW}[!] Sending SIGTERM signal to Python process (main.py) (PID: $PID)...${NC}"
        kill -TERM "$PID" 2>/dev/null
    done
    
    sleep 2
    
    for PID in $PYTHON_PIDS; do
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "${RED}[-] Python process (PID: $PID) did not stop. Forcing termination (SIGKILL)...${NC}"
            kill -9 "$PID" 2>/dev/null
        fi
    done
else
    echo -e "${GREEN}[+] Python processes (main.py) not found.${NC}"
fi

AUDIO_PLAY_PIDS=$(pgrep -f "g1_audio_play")
if [ -n "$AUDIO_PLAY_PIDS" ]; then
    echo -e "${YELLOW}[!] Forcing shutdown of background g1_audio_play processes...${NC}"
    pkill -9 -f "g1_audio_play"
fi

echo -e "${GREEN}[+] All voice engine processes successfully stopped!${NC}"