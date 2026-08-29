#!/usr/bin/env bash
set -euo pipefail

cd /home/unitree/ARMS/g1_arm_preset_actions

exec env -i \
  HOME="/home/unitree" \
  USER="unitree" \
  LOGNAME="unitree" \
  PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  /home/unitree/cyclonedds_py_0_10_fix/run_with_fixed_dds.sh \
  python3 -B ./g1_pose_tuner.py "$@"
