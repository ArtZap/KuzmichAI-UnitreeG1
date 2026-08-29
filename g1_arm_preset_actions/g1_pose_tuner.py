#!/usr/bin/env python3
"""
Capture and replay physical Unitree G1 arm poses.

Default behavior is conservative:
  - capture/show/list are read-only
  - move/nudge print a dry run unless --execute is provided
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SDK_PATHS = [
    "/home/unitree/KuzmichAI/unitree_sdk2_python",
    "/home/unitree/KuzmichAI_experiment/unitree_sdk2_python",
]

WAIST_JOINTS = (12, 13, 14)
LEFT_ARM_JOINTS = (15, 16, 17, 18, 19, 20, 21)
RIGHT_ARM_JOINTS = (22, 23, 24, 25, 26, 27, 28)
K_NOT_USED_JOINT = 29

ARM_JOINT_NAMES = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)

DEFAULT_POSE_DIR = "poses"
DEFAULT_MOTION_DIR = "motions"
HOME_HAND_PRESET = "fist"
ArmSnapshot = Dict[str, List[float]]

DEX3_JOINT_NAMES = (
    "thumb_abduction",
    "thumb_flexion_0",
    "thumb_flexion_1",
    "index_flexion_0",
    "index_flexion_1",
    "middle_flexion_0",
    "middle_flexion_1",
)

LEFT_DEX3_CLOSE_Q = (1.05, -0.724, 1.75, -1.57, -1.75, -1.57, -1.75)
RIGHT_DEX3_CLOSE_Q = (1.05, 0.742, -1.75, 1.57, 1.75, 1.57, 1.75)

HAND_PRESETS = {
    "open": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "flat": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "thumb_squeeze": [0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    "thumb_closed_two_straight": [0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "pinch": [0.55, 0.58, 0.48, 0.62, 0.50, 0.10, 0.08],
    "wrap": [0.0, -1.0, 1.0, 0.88, 0.85, 0.88, 0.85],
    "side": [0.80, 0.45, 0.30, 0.42, 0.28, 0.42, 0.28],
    "fist": [0.0, -1.0, 1.0, 0.95, 0.95, 0.95, 0.95],
    "point": [0.25, 0.15, 0.0, 0.0, 0.0, 0.95, 0.95],
    "shake_soft": [0.0, -0.35, 0.25, 0.0, 0.0, 0.0, 0.0],
    "shake_strong": [0.0, -0.85, 0.8, 0.0, 0.0, 0.0, 0.0],
}


def add_sdk_paths() -> None:
    for sdk_path in SDK_PATHS:
        if os.path.isdir(sdk_path) and sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune and replay saved G1 arm poses.")
    parser.add_argument("--pose-dir", default=DEFAULT_POSE_DIR)
    parser.add_argument("--motion-dir", default=DEFAULT_MOTION_DIR)
    parser.add_argument("--network-interface", default="")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--kp", type=float, default=60.0)
    parser.add_argument("--kd", type=float, default=1.5)
    parser.add_argument("--prepose", default="home1", help="Safe transit pose before moving to a target. Default: home1.")
    parser.add_argument("--prepose-duration", type=float, default=2.0)
    parser.add_argument("--no-prepose", action="store_true", help="Skip the safe transit pose.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="Read current arm joints and save a pose.")
    capture.add_argument("name", help="Pose name, for example home or heart_v1.")
    capture.add_argument("--overwrite", action="store_true")

    subparsers.add_parser("list", help="List saved poses.")
    subparsers.add_parser("catalog", help="Show saved poses grouped by name.")

    show = subparsers.add_parser("show", help="Show one saved pose.")
    show.add_argument("name")

    move = subparsers.add_parser("move", help="Move both arms to a saved pose.")
    move.add_argument("name")
    move.add_argument("--execute", action="store_true", help="Actually move the robot.")
    move.add_argument("--include-waist", action="store_true", help="Also command saved waist joints.")

    nudge = subparsers.add_parser("nudge", help="Move one current arm joint by a small delta.")
    nudge.add_argument("hand", choices=("left", "right"))
    nudge.add_argument("joint", type=int, choices=range(1, 8), metavar="1..7")
    nudge.add_argument("delta", type=float, help="Delta in radians, for example 0.03 or -0.03.")
    nudge.add_argument("--max-delta", type=float, default=0.15)
    nudge.add_argument("--execute", action="store_true", help="Actually move the robot.")
    nudge.add_argument("--save-as", default="", help="Capture the resulting pose under this name.")
    nudge.add_argument("--overwrite", action="store_true")

    hand = subparsers.add_parser("hand", help="Send a Dex3 finger preset.")
    hand.add_argument("--left", choices=sorted(HAND_PRESETS), default="")
    hand.add_argument("--right", choices=sorted(HAND_PRESETS), default="")
    hand.add_argument("--execute", action="store_true", help="Actually move fingers.")

    motion_create = subparsers.add_parser("motion-create", help="Create a motion from saved poses.")
    motion_create.add_argument("name")
    motion_create.add_argument(
        "steps",
        nargs="+",
        help="Step format: pose, pose:duration, or pose:duration:pause",
    )
    motion_create.add_argument("--overwrite", action="store_true")

    motion_show = subparsers.add_parser("motion-show", help="Show one saved motion.")
    motion_show.add_argument("name")

    motion_play = subparsers.add_parser("motion-play", help="Play a saved motion.")
    motion_play.add_argument("name")
    motion_play.add_argument("--execute", action="store_true", help="Actually move arms/fingers.")
    motion_play.add_argument("--include-waist", action="store_true")

    subparsers.add_parser("motion-init", help="Create starter motions from known pose names.")

    return parser


def pose_path(pose_dir: str, name: str) -> Path:
    safe_name = name.replace("/", "_").replace("\\", "_")
    return Path(pose_dir) / f"{safe_name}.json"


def motion_path(motion_dir: str, name: str) -> Path:
    safe_name = name.replace("/", "_").replace("\\", "_")
    return Path(motion_dir) / f"{safe_name}.json"


def load_pose(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Pose not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_pose(path: Path, name: str, snapshot: ArmSnapshot, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Pose already exists: {path}. Use --overwrite to replace it.")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "joint_names": {
            "waist": ["waist_yaw", "waist_roll", "waist_pitch"],
            "left": list(ARM_JOINT_NAMES),
            "right": list(ARM_JOINT_NAMES),
        },
        "joint_ids": {
            "waist": list(WAIST_JOINTS),
            "left": list(LEFT_ARM_JOINTS),
            "right": list(RIGHT_ARM_JOINTS),
        },
        "snapshot": snapshot,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def validate_snapshot(snapshot: ArmSnapshot) -> ArmSnapshot:
    for key, expected_len in (("left", 7), ("right", 7)):
        values = snapshot.get(key)
        if not isinstance(values, list) or len(values) != expected_len:
            raise ValueError(f"Bad pose: snapshot.{key} must contain {expected_len} values")
    waist = snapshot.get("waist", [])
    if waist and len(waist) != 3:
        raise ValueError("Bad pose: snapshot.waist must contain 3 values")
    return {
        "waist": [float(v) for v in waist],
        "left": [float(v) for v in snapshot["left"]],
        "right": [float(v) for v in snapshot["right"]],
    }


def load_saved_snapshot(pose_dir: str, name: str) -> ArmSnapshot:
    payload = load_pose(pose_path(pose_dir, name))
    return validate_snapshot(payload["snapshot"])


def initialize_sdk(network_interface: str) -> None:
    add_sdk_paths()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    if network_interface:
        ChannelFactoryInitialize(0, network_interface)
    else:
        ChannelFactoryInitialize(0)


def read_current_snapshot(network_interface: str, timeout_s: float = 3.0) -> ArmSnapshot:
    initialize_sdk(network_interface)
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    holder = {"msg": None}

    def callback(msg) -> None:
        holder["msg"] = msg

    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(callback, 10)

    start_s = time.monotonic()
    while holder["msg"] is None:
        if time.monotonic() - start_s > timeout_s:
            raise TimeoutError("No rt/lowstate received from SDK2")
        time.sleep(0.01)

    msg = holder["msg"]
    return {
        "waist": [float(msg.motor_state[joint_id].q) for joint_id in WAIST_JOINTS],
        "left": [float(msg.motor_state[joint_id].q) for joint_id in LEFT_ARM_JOINTS],
        "right": [float(msg.motor_state[joint_id].q) for joint_id in RIGHT_ARM_JOINTS],
    }


def publish_pose(
    network_interface: str,
    start: ArmSnapshot,
    target: ArmSnapshot,
    duration: float,
    dt: float,
    kp: float,
    kd: float,
) -> None:
    initialize_sdk(network_interface)
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
    publisher.Init()
    crc = CRC()
    steps = max(1, int(max(0.2, duration) / max(0.005, dt)))

    for step in range(steps + 1):
        ratio = step / steps
        pose = {
            "waist": blend(start["waist"], target["waist"], ratio),
            "left": blend(start["left"], target["left"], ratio),
            "right": blend(start["right"], target["right"], ratio),
        }
        cmd = unitree_hg_msg_dds__LowCmd_()
        fill_command(cmd, pose, kp, kd)
        cmd.crc = crc.Crc(cmd)
        publisher.Write(cmd)
        time.sleep(dt)


def fill_command(cmd, pose: ArmSnapshot, kp: float, kd: float) -> None:
    for section, joint_ids in (
        ("waist", WAIST_JOINTS),
        ("left", LEFT_ARM_JOINTS),
        ("right", RIGHT_ARM_JOINTS),
    ):
        for index, joint_id in enumerate(joint_ids):
            motor = cmd.motor_cmd[joint_id]
            motor.q = float(pose[section][index])
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = float(kp)
            motor.kd = float(kd)
    cmd.motor_cmd[K_NOT_USED_JOINT].q = 1.0


def normalize_hand_pose(value) -> List[float]:
    if isinstance(value, str):
        if value not in HAND_PRESETS:
            raise ValueError(f"Unknown hand preset: {value}")
        return list(HAND_PRESETS[value])

    if isinstance(value, list) and len(value) == 7:
        result = [float(item) for item in value]
    elif isinstance(value, dict):
        result = [float(value.get(name, 0.0)) for name in DEX3_JOINT_NAMES]
    else:
        raise ValueError("Hand pose must be a preset name, 7-value list, or dict")

    for index, item in enumerate(result):
        joint_name = DEX3_JOINT_NAMES[index]
        min_value = -1.0 if joint_name == "thumb_flexion_0" else 0.0
        if not min_value <= item <= 1.0:
            raise ValueError(f"Hand joint {joint_name}={item:.3f} outside range")
    return result


def dex3_to_sdk_q(hand: str, normalized: List[float]) -> List[float]:
    close_q = LEFT_DEX3_CLOSE_Q if hand == "left" else RIGHT_DEX3_CLOSE_Q
    return [float(value) * close_q[index] for index, value in enumerate(normalized)]


def send_hand_presets(network_interface: str, hands: Dict[str, object], execute: bool) -> None:
    if not hands:
        return

    normalized_by_hand = {}
    q_by_hand = {}
    for hand, pose in hands.items():
        if hand not in ("left", "right"):
            raise ValueError(f"Unknown hand side: {hand}")
        normalized = normalize_hand_pose(pose)
        normalized_by_hand[hand] = normalized
        q_by_hand[hand] = dex3_to_sdk_q(hand, normalized)

    print("Fingers:")
    for hand, normalized in sorted(normalized_by_hand.items()):
        print(f"  {hand}: normalized={round_values(normalized)} sdk_q={round_values(q_by_hand[hand])}")

    if not execute:
        print("DRY RUN: fingers were not moved. Add --execute to actually move.")
        return

    initialize_sdk(network_interface)
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_

    publishers = {
        "left": ChannelPublisher("rt/dex3/left/cmd", HandCmd_),
        "right": ChannelPublisher("rt/dex3/right/cmd", HandCmd_),
    }
    for publisher in publishers.values():
        publisher.Init()

    messages = {}
    for hand, q_values in q_by_hand.items():
        msg = unitree_hg_msg_dds__HandCmd_()
        for motor_id, q in enumerate(q_values):
            motor = msg.motor_cmd[motor_id]
            motor.mode = (motor_id & 0x0F) | (1 << 4)
            motor.q = float(q)
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = 1.5
            motor.kd = 0.1
        messages[hand] = msg

    for _ in range(25):
        for hand, msg in messages.items():
            publishers[hand].Write(msg)
        time.sleep(0.02)


def blend(start: List[float], target: List[float], ratio: float) -> List[float]:
    return [s * (1.0 - ratio) + t * ratio for s, t in zip(start, target)]


def round_values(values: List[float]) -> List[float]:
    return [round(value, 4) for value in values]


def print_snapshot(snapshot: ArmSnapshot) -> None:
    print("waist:", round_values(snapshot.get("waist", [])))
    for hand in ("left", "right"):
        print(f"{hand}:")
        for index, value in enumerate(snapshot[hand], start=1):
            motor_id = (LEFT_ARM_JOINTS if hand == "left" else RIGHT_ARM_JOINTS)[index - 1]
            print(f"  {index}. {ARM_JOINT_NAMES[index - 1]:<15} motor={motor_id:<2d} q={value:.4f}")


def command_capture(args: argparse.Namespace) -> int:
    snapshot = read_current_snapshot(args.network_interface)
    path = pose_path(args.pose_dir, args.name)
    save_pose(path, args.name, snapshot, args.overwrite)
    print(f"Saved pose: {path}")
    print_snapshot(snapshot)
    return 0


def command_list(args: argparse.Namespace) -> int:
    pose_dir = Path(args.pose_dir)
    if not pose_dir.exists():
        print(f"No poses directory yet: {pose_dir}")
        return 0

    poses = sorted(pose_dir.glob("*.json"))
    if not poses:
        print(f"No saved poses in: {pose_dir}")
        return 0

    for path in poses:
        print(path.stem)
    return 0


def command_catalog(args: argparse.Namespace) -> int:
    pose_dir = Path(args.pose_dir)
    poses = sorted(pose_dir.glob("*.json")) if pose_dir.exists() else []
    if not poses:
        print(f"No saved poses in: {pose_dir}")
        return 0

    print(f"Saved poses ({len(poses)}):")
    groups: Dict[str, List[str]] = {}
    for path in poses:
        name = path.stem
        prefix = name.split("_", 1)[0]
        groups.setdefault(prefix, []).append(name)
        payload = load_pose(path)
        created_at = payload.get("created_at", "unknown")
        print(f"  {name:<18} created={created_at}")

    print("\nGroups:")
    for prefix, names in sorted(groups.items()):
        print(f"  {prefix:<12} {', '.join(names)}")

    print("\nHand presets:")
    print("  " + ", ".join(sorted(HAND_PRESETS)))
    return 0


def command_show(args: argparse.Namespace) -> int:
    payload = load_pose(pose_path(args.pose_dir, args.name))
    snapshot = validate_snapshot(payload["snapshot"])
    print(f"Pose: {payload.get('name', args.name)}")
    print(f"Created: {payload.get('created_at', 'unknown')}")
    print_snapshot(snapshot)
    return 0


def command_move(args: argparse.Namespace) -> int:
    saved = load_saved_snapshot(args.pose_dir, args.name)

    if not args.execute:
        print(f"DRY RUN: would move both arms to pose '{args.name}'.")
        if should_use_prepose(args, args.name):
            print(f"DRY RUN: would first move through safe prepose '{args.prepose}'.")
        if args.name == "home":
            print(f"DRY RUN: would set both hands to '{HOME_HAND_PRESET}' after reaching home.")
        if args.include_waist:
            print("DRY RUN: would also command saved waist joints.")
        print_snapshot(saved)
        print("Add --execute to actually move.")
        return 0

    start = read_current_snapshot(args.network_interface)
    start = move_through_prepose(args, start, args.name)
    target = {
        "waist": saved["waist"] if args.include_waist else start["waist"],
        "left": saved["left"],
        "right": saved["right"],
    }
    publish_pose(args.network_interface, start, target, args.duration, args.dt, args.kp, args.kd)
    if args.name == "home":
        send_hand_presets(
            args.network_interface,
            {"left": HOME_HAND_PRESET, "right": HOME_HAND_PRESET},
            execute=True,
        )
    print(f"Moved to pose: {args.name}")
    return 0


def should_use_prepose(args: argparse.Namespace, target_pose_name: str) -> bool:
    if args.no_prepose:
        return False
    if not args.prepose:
        return False
    if target_pose_name == args.prepose:
        return False
    return pose_path(args.pose_dir, args.prepose).exists()


def move_through_prepose(
    args: argparse.Namespace,
    current: ArmSnapshot,
    target_pose_name: str,
) -> ArmSnapshot:
    if not should_use_prepose(args, target_pose_name):
        return current

    prepose = load_saved_snapshot(args.pose_dir, args.prepose)
    target = {
        "waist": prepose["waist"] if args.include_waist else current["waist"],
        "left": prepose["left"],
        "right": prepose["right"],
    }
    print(f"Safe transit: moving through {args.prepose}")
    publish_pose(args.network_interface, current, target, args.prepose_duration, args.dt, args.kp, args.kd)
    return target


def parse_motion_step(spec: str, default_duration: float) -> dict:
    parts = spec.split(":")
    if len(parts) > 3:
        raise ValueError(f"Bad step: {spec}")

    step = {"pose": parts[0], "duration": default_duration, "pause": 0.0}
    if len(parts) >= 2 and parts[1]:
        step["duration"] = float(parts[1])
    if len(parts) == 3 and parts[2]:
        step["pause"] = float(parts[2])
    return step


def load_motion(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Motion not found: {path}")
    motion = json.loads(path.read_text(encoding="utf-8"))
    if "steps" not in motion or not isinstance(motion["steps"], list):
        raise ValueError(f"Bad motion file: {path}")
    return motion


def save_motion(path: Path, motion: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Motion already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(motion, indent=2) + "\n", encoding="utf-8")


def command_motion_create(args: argparse.Namespace) -> int:
    steps = [parse_motion_step(spec, args.duration) for spec in args.steps]
    for step in steps:
        pose_path(args.pose_dir, step["pose"]).resolve(strict=True)

    motion = {
        "name": args.name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": steps,
    }
    path = motion_path(args.motion_dir, args.name)
    save_motion(path, motion, args.overwrite)
    print(f"Saved motion: {path}")
    print_motion(motion)
    return 0


def print_motion(motion: dict) -> None:
    print(f"Motion: {motion.get('name', 'unnamed')}")
    for index, step in enumerate(motion["steps"], start=1):
        hands = step.get("hands", {})
        hand_text = ""
        if hands:
            hand_text = " hands=" + json.dumps(hands, sort_keys=True)
        print(
            f"  {index:2d}. pose={step['pose']:<18} "
            f"duration={float(step.get('duration', 3.0)):.2f}s "
            f"pause={float(step.get('pause', 0.0)):.2f}s{hand_text}"
        )


def command_motion_show(args: argparse.Namespace) -> int:
    motion = load_motion(motion_path(args.motion_dir, args.name))
    print_motion(motion)
    return 0


def command_motion_play(args: argparse.Namespace) -> int:
    motion = load_motion(motion_path(args.motion_dir, args.name))
    print_motion(motion)

    if not args.execute:
        first_pose_name = motion["steps"][0]["pose"] if motion["steps"] else ""
        if first_pose_name and should_use_prepose(args, first_pose_name):
            print(f"DRY RUN: would first move through safe prepose '{args.prepose}'.")
        print("DRY RUN: motion was not played. Add --execute to actually move.")
        return 0

    current = read_current_snapshot(args.network_interface)
    if motion["steps"]:
        current = move_through_prepose(args, current, motion["steps"][0]["pose"])
    for index, step in enumerate(motion["steps"], start=1):
        pose_name = step["pose"]
        duration = float(step.get("duration", args.duration))
        pause = float(step.get("pause", 0.0))
        saved = load_saved_snapshot(args.pose_dir, pose_name)
        target = {
            "waist": saved["waist"] if args.include_waist else current["waist"],
            "left": saved["left"],
            "right": saved["right"],
        }

        print(f"Step {index}: moving to {pose_name}")
        publish_pose(args.network_interface, current, target, duration, args.dt, args.kp, args.kd)
        current = target

        hands = step.get("hands", {})
        if hands:
            send_hand_presets(args.network_interface, hands, execute=True)

        if pause > 0:
            time.sleep(pause)

    print(f"Played motion: {args.name}")
    return 0


def command_motion_init(args: argparse.Namespace) -> int:
    candidates = {
        "heart": ["home1:2:0.1", "hart_part1:2:0.2", "heart_part2:2:0.6", "hart_part1:2:0.2", "home:2.5:0.0"],
        "crying": ["home1:2:0.1", "crying:1:0.1", "crying_2_down:1:0.1", "crying:1:0.1", "crying_2_down:1:0.1", "crying:1:0.1", "crying_2_down:1:0.1", "crying:1:0.1", "crying_2_down:1:0.1", "crying:1:0.1", "crying_2_down:1:0.2", "home:2:0.0"],
        "sixty_seven": ["home1:2:0.1", "67_1:1:0.1", "67_2:1:0.1", "67_1:1:0.1", "67_2:1:0.1", "67_1:1:0.1", "67_2:1:0.1", "67_1:1:0.1", "67_2:1:0.2", "home:2:0.0"],
        "dasha_koza": ["home1:2:0.1", "dasha_koza_1:0.5:0.05", "dasha_koza_2:0.5:0.05", "dasha_koza_1:0.5:0.05", "dasha_koza_2:0.5:0.05", "dasha_koza_1:0.5:0.05", "dasha_koza_2:0.5:0.05", "dasha_koza_1:0.5:0.05", "dasha_koza_2:0.5:0.1", "home:2:0.0"],
        "face_chest": ["home1:2:0.1", "faceprepare:1.5:0.2", "chest:1.5:0.4", "home:2:0.0"],
        "facepalm": ["home1:2:0.1", "faceprepare:1.5:0.2", "facepalm:1.5:0.6", "home:2:0.0"],
        "kiss": ["home1:2:0.1", "kiss_1:1.5:0.2", "kiss_2:1.5:0.6", "home:2:0.0"],
        "mouthkeeper": ["home1:2:0.1", "mouthkeeper:1.5:0.2", "mouthkeeper_2:1.5:0.4", "home:2:0.0"],
        "shakehands": ["home1:2:0.1", "shakehands_1:0.8:0.1", "shakehands_2:0.8:0.1", "shakehands_1:0.8:0.2", "shakehands_2:0.8:0.1", "shakehands_1:0.8:0.2", "home:2:0.0"],
    }

    created = 0
    skipped = 0
    for name, specs in sorted(candidates.items()):
        steps = [parse_motion_step(spec, args.duration) for spec in specs]
        missing = [step["pose"] for step in steps if not pose_path(args.pose_dir, step["pose"]).exists()]
        if missing:
            print(f"Skip {name}: missing poses: {', '.join(missing)}")
            skipped += 1
            continue
        motion = {
            "name": name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "steps": steps,
        }
        path = motion_path(args.motion_dir, name)
        save_motion(path, motion, overwrite=True)
        print(f"Saved motion: {path}")
        created += 1

    print(f"Motion init done: created={created} skipped={skipped}")
    print("You can edit motion JSON files to add fingers per step, for example:")
    print('  "hands": {"right": "pinch", "left": "open"}')
    return 0


def command_hand(args: argparse.Namespace) -> int:
    hands = {}
    if args.left:
        hands["left"] = args.left
    if args.right:
        hands["right"] = args.right
    if not hands:
        print("Choose --left PRESET and/or --right PRESET.")
        print("Available presets:", ", ".join(sorted(HAND_PRESETS)))
        return 2

    send_hand_presets(args.network_interface, hands, args.execute)
    return 0


def command_nudge(args: argparse.Namespace) -> int:
    if abs(args.delta) > abs(args.max_delta):
        raise ValueError(f"delta {args.delta:.4f} exceeds max delta {args.max_delta:.4f}")

    start = read_current_snapshot(args.network_interface)
    target = {
        "waist": list(start["waist"]),
        "left": list(start["left"]),
        "right": list(start["right"]),
    }
    target[args.hand][args.joint - 1] += float(args.delta)

    motor_id = (LEFT_ARM_JOINTS if args.hand == "left" else RIGHT_ARM_JOINTS)[args.joint - 1]
    joint_name = ARM_JOINT_NAMES[args.joint - 1]

    if not args.execute:
        print(
            f"DRY RUN: would nudge {args.hand} joint {args.joint} "
            f"({joint_name}, motor={motor_id}) by {args.delta:.4f} rad."
        )
        print("Current:")
        print_snapshot(start)
        print("Target:")
        print_snapshot(target)
        print("Add --execute to actually move.")
        return 0

    publish_pose(args.network_interface, start, target, args.duration, args.dt, args.kp, args.kd)
    print(
        f"Nudged {args.hand} joint {args.joint} "
        f"({joint_name}, motor={motor_id}) by {args.delta:.4f} rad."
    )

    if args.save_as:
        final_snapshot = read_current_snapshot(args.network_interface)
        path = pose_path(args.pose_dir, args.save_as)
        save_pose(path, args.save_as, final_snapshot, args.overwrite)
        print(f"Saved pose: {path}")

    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "capture":
            return command_capture(args)
        if args.command == "list":
            return command_list(args)
        if args.command == "catalog":
            return command_catalog(args)
        if args.command == "show":
            return command_show(args)
        if args.command == "move":
            return command_move(args)
        if args.command == "nudge":
            return command_nudge(args)
        if args.command == "hand":
            return command_hand(args)
        if args.command == "motion-create":
            return command_motion_create(args)
        if args.command == "motion-show":
            return command_motion_show(args)
        if args.command == "motion-play":
            return command_motion_play(args)
        if args.command == "motion-init":
            return command_motion_init(args)
        raise ValueError(f"Unknown command: {args.command}")
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
