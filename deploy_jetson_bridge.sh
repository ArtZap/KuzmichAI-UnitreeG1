#!/bin/bash
# Copies jetson_mic_bridge.py and the shared key to the robot's Jetson and
# (re)starts the bridge in the background.
#
#   ./deploy_jetson_bridge.sh                      # unitree@$VOICE_ENGINE_ROBOT_IP (192.168.1.103)
#   ./deploy_jetson_bridge.sh unitree@192.168.123.164
#   ./deploy_jetson_bridge.sh unitree@<ip> --device plughw:2,0   # extra args go to the bridge
#   ./deploy_jetson_bridge.sh unitree@<ip> --stop
#
# The SSH password is asked once (one shared connection is used for all steps).
# The key lives in ~/.config/kuzmich/bridge_token on this laptop (created on first run);
# ext_mic.py reads it from there, so nothing has to be configured by hand.

set -e
cd "$(dirname "$(readlink -f "$0")")"

TARGET="${1:-unitree@${VOICE_ENGINE_ROBOT_IP:-192.168.1.103}}"
shift || true
REMOTE_DIR="kuzmich_bridge"
LOG="/tmp/kuzmich_mic_bridge.log"
TOKEN_FILE="${VOICE_ENGINE_BRIDGE_TOKEN_FILE:-$HOME/.config/kuzmich/bridge_token}"
TOKEN_FILE="${TOKEN_FILE/#\~/$HOME}"

# Quote the bridge arguments for the remote shell
BRIDGE_ARGS=""
for arg in "$@"; do
    BRIDGE_ARGS+=" $(printf '%q' "$arg")"
done

CTL_DIR="$(mktemp -d)"
trap 'ssh -o "ControlPath=$CTL_DIR/cm" -O exit "$TARGET" 2>/dev/null; rm -rf "$CTL_DIR"' EXIT
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$CTL_DIR/cm" -o ControlPersist=60 -o ConnectTimeout=10)

echo "[*] Connecting to $TARGET ..."
ssh "${SSH_OPTS[@]}" "$TARGET" true

STOP_CMD="pkill -f '[j]etson_mic_bridge.py' && echo '[+] Old bridge stopped.' || true"

if [ "$*" = "--stop" ]; then
    ssh "${SSH_OPTS[@]}" "$TARGET" "$STOP_CMD"
    exit 0
fi

if [ ! -s "$TOKEN_FILE" ]; then
    mkdir -p "$(dirname "$TOKEN_FILE")"
    (umask 077; python3 -c 'import secrets; print(secrets.token_hex(32))' > "$TOKEN_FILE")
    echo "[+] New bridge key created: $TOKEN_FILE"
fi
chmod 600 "$TOKEN_FILE"

ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p ~/$REMOTE_DIR && chmod 700 ~/$REMOTE_DIR"
scp "${SSH_OPTS[@]}" -q jetson_mic_bridge.py "$TARGET:$REMOTE_DIR/"
scp "${SSH_OPTS[@]}" -q "$TOKEN_FILE" "$TARGET:$REMOTE_DIR/token"
ssh "${SSH_OPTS[@]}" "$TARGET" "chmod 600 ~/$REMOTE_DIR/token"
echo "[+] jetson_mic_bridge.py and the key copied to ~/$REMOTE_DIR"

ssh "${SSH_OPTS[@]}" "$TARGET" bash -s <<EOF
set -e
command -v arecord >/dev/null || { echo "[-] arecord missing: sudo apt install alsa-utils"; exit 1; }
echo "[*] Capture devices on the robot:"
python3 ~/$REMOTE_DIR/jetson_mic_bridge.py --list-devices || true
$STOP_CMD
sleep 1
nohup python3 ~/$REMOTE_DIR/jetson_mic_bridge.py$BRIDGE_ARGS > $LOG 2>&1 < /dev/null &
sleep 3
echo "[*] Bridge log ($LOG):"
cat $LOG
pgrep -f '[j]etson_mic_bridge.py' >/dev/null && echo "[+] Bridge is running." || { echo "[-] Bridge exited, see the log above."; exit 1; }
EOF

echo
echo "[+] Done. Now start the assistant on this laptop:"
echo "    VOICE_ENGINE_AUDIO=g1_extmic VOICE_ENGINE_ROBOT_IP=${TARGET#*@} ./start.sh"
