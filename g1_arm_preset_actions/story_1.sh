#!/usr/bin/env bash
set -euo pipefail

cd /home/unitree/ARMS/g1_arm_preset_actions

EXECUTE=0
PAUSE_S="${PAUSE_S:-0.8}"

for arg in "$@"; do
  case "$arg" in
    --execute)
      EXECUTE=1
      ;;
    --help|-h)
      echo "Usage:"
      echo "  ./story_1.sh             # dry run, no robot movement"
      echo "  ./story_1.sh --execute   # play story gestures"
      echo
      echo "Optional:"
      echo "  PAUSE_S=1.5 ./story_1.sh --execute"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Run: ./story_1.sh --help"
      exit 2
      ;;
  esac
done

run_motion() {
  local title="$1"
  local motion="$2"

  echo
  echo "=== $title: $motion ==="
  if [[ "$EXECUTE" == "1" ]]; then
    ./run_pose_tuner_fixed_dds.sh motion-play "$motion" --execute
    sleep "$PAUSE_S"
  else
    ./run_pose_tuner_fixed_dds.sh motion-play "$motion"
  fi
}

run_clap() {
  echo
  echo "=== Finale: official clap ==="
  if [[ "$EXECUTE" == "1" ]]; then
    ./g1_preset_actions_test.py --action clap -y
  else
    ./g1_preset_actions_test.py --dry-run --action clap -y
  fi
}

echo "Story 1 gesture sequence"
if [[ "$EXECUTE" == "1" ]]; then
  echo "MODE: execute"
else
  echo "MODE: dry run"
fi

run_motion "Здравствуйте / рокенрол" "dasha_koza"
run_motion "Меня зовут / отдать честь" "face_chest"
run_motion "Я занимаюсь / две руки у рта" "mouthkeeper"
run_motion "Современные агротехнологии / сердечко" "heart"
run_motion "Рука об руку / рукопожатие" "shakehands"
run_motion "Живое доказательство / сикссевен" "sixty_seven"
run_motion "Спасибо / поцелуй" "kiss"
run_clap

echo
echo "Story 1 done."
