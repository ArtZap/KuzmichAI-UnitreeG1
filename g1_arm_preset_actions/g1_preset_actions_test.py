#!/usr/bin/env python3
"""
Interactive tester for Unitree G1 built-in arm actions.

This script is a safe menu wrapper around Unitree's official compiled example:
    /home/unitree/unitree_sdk2/build/bin/g1_arm_action_example

The Python G1ArmActionClient can conflict with the robot's current DDS/ROS2
topic setup on some images, while the official binary is already built for this
robot and can list actions successfully.
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, Optional, Tuple


DEFAULT_BINARY = "/home/unitree/unitree_sdk2/build/bin/g1_arm_action_example"

PRESET_ACTIONS = [
    ("two-hand kiss", 11, "blow_kiss_with_both_hands"),
    ("left kiss", 12, "blow_kiss_with_left_hand"),
    ("right kiss", 13, "blow_kiss_with_right_hand"),
    ("hands up", 15, "both_hands_up"),
    ("clap", 17, "clamp"),
    ("high five", 18, "high_five"),
    ("hug", 19, "hug"),
    ("heart", 20, "make_heart_with_both_hands"),
    ("right heart", 21, "make_heart_with_right_hand"),
    ("reject", 22, "refuse"),
    ("right hand up", 23, "right_hand_up"),
    ("x-ray", 24, "ultraman_ray"),
    ("face wave", 25, "wave_under_head"),
    ("high wave", 26, "wave_above_head"),
    ("shake hand", 27, "shake_hand"),
]

RELEASE_ACTION = ("release arm", 99, "release_arm")
ACTION_BY_NAME: Dict[str, Tuple[str, int, str]] = {
    item[0]: item for item in PRESET_ACTIONS + [RELEASE_ACTION]
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test Unitree G1 built-in arm actions using the official Unitree binary."
    )
    parser.add_argument(
        "network_interface",
        nargs="?",
        help="Optional DDS network interface. Usually leave empty when running on this robot.",
    )
    parser.add_argument(
        "-a",
        "--action",
        help="Action number, action name, or official action id to run once.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 15 preset actions in order.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Seconds to wait after each action before releasing arms. Default: 3.0",
    )
    parser.add_argument(
        "--no-release",
        action="store_true",
        help="Do not send release arm after each action.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run, but do not move arms.",
    )
    parser.add_argument(
        "--official-list",
        action="store_true",
        help="Print the official Unitree action list and exit.",
    )
    parser.add_argument(
        "--binary",
        default=os.environ.get("G1_ARM_ACTION_BIN", DEFAULT_BINARY),
        help=f"Path to Unitree official binary. Default: {DEFAULT_BINARY}",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the safety confirmation prompt.",
    )
    return parser


def print_action_list() -> None:
    print("\nBuilt-in G1 arm actions:")
    for index, (name, action_id, official_name) in enumerate(PRESET_ACTIONS, start=1):
        print(f"  {index:2d}. {name:<14} id={action_id:<2d} official={official_name}")
    print(f"\n  r.  {RELEASE_ACTION[0]:<14} id={RELEASE_ACTION[1]} official={RELEASE_ACTION[2]}")
    print("  list. show this menu")
    print("  q.    quit")


def resolve_action(user_value: str) -> Tuple[str, int, str]:
    value = user_value.strip().lower().replace("_", " ")

    if value in {"r", "release", "release arm"}:
        return RELEASE_ACTION

    if value.isdigit():
        number = int(value)
        if 1 <= number <= len(PRESET_ACTIONS):
            return PRESET_ACTIONS[number - 1]

        for item in PRESET_ACTIONS + [RELEASE_ACTION]:
            if item[1] == number:
                return item

    exact_matches = [
        item for item in PRESET_ACTIONS + [RELEASE_ACTION]
        if item[0].lower() == value or item[2].lower().replace("_", " ") == value
    ]
    if exact_matches:
        return exact_matches[0]

    partial_matches = [
        item for item in PRESET_ACTIONS + [RELEASE_ACTION]
        if value in item[0].lower() or value in item[2].lower().replace("_", " ")
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]

    if len(partial_matches) > 1:
        names = ", ".join(item[0] for item in partial_matches)
        raise ValueError(f"Ambiguous action '{user_value}': {names}")

    raise ValueError(f"Unknown action '{user_value}'")


def command_for_action(
    binary: str,
    action_id: int,
    network_interface: Optional[str],
) -> list:
    command = [binary]
    if network_interface:
        command.extend(["--network", network_interface])
    command.extend(["--id", str(action_id)])
    return command


def run_command(command: list, dry_run: bool) -> int:
    printable = " ".join(command)
    if dry_run:
        print(f"[DRY RUN] {printable}")
        return 0

    print(printable)
    return subprocess.call(command)


def execute_action(
    binary: str,
    network_interface: Optional[str],
    action: Tuple[str, int, str],
    delay: float,
    release_after: bool,
    dry_run: bool,
) -> None:
    name, action_id, official_name = action
    print(f"\nRunning: {name} (id={action_id}, official={official_name})")
    code = run_command(command_for_action(binary, action_id, network_interface), dry_run)
    print(f"Action returned code: {code}")

    if release_after and action_id != RELEASE_ACTION[1]:
        if dry_run:
            print(f"[DRY RUN] Would wait {delay:.1f}s")
        else:
            time.sleep(delay)

        release_name, release_id, release_official_name = RELEASE_ACTION
        print(f"Releasing arms: {release_name} (id={release_id}, official={release_official_name})")
        release_code = run_command(
            command_for_action(binary, release_id, network_interface),
            dry_run,
        )
        print(f"Release returned code: {release_code}")


def validate_binary(binary: str) -> bool:
    if not os.path.isfile(binary):
        print(f"Error: Unitree binary was not found: {binary}")
        return False

    if not os.access(binary, os.X_OK):
        print(f"Error: Unitree binary is not executable: {binary}")
        return False

    return True


def print_official_list(binary: str, network_interface: Optional[str], dry_run: bool) -> int:
    command = [binary]
    if network_interface:
        command.extend(["--network", network_interface])
    command.append("--list")
    return run_command(command, dry_run)


def main() -> int:
    args = build_parser().parse_args()

    if not validate_binary(args.binary):
        return 1

    if args.official_list:
        return print_official_list(args.binary, args.network_interface, args.dry_run)

    print("WARNING: Make sure the robot is stable and there are no obstacles nearby.")
    print("These are upper-body preset actions; some are large gestures.")
    if args.network_interface:
        print(f"DDS network interface: {args.network_interface}")
    else:
        print("DDS network interface: default robot configuration")
    if args.dry_run:
        print("DRY RUN: commands will be printed, not executed.")

    if not args.yes:
        input("Press Enter to continue, or Ctrl+C to cancel...")

    release_after = not args.no_release

    if args.all:
        for action in PRESET_ACTIONS:
            execute_action(
                args.binary,
                args.network_interface,
                action,
                delay=args.delay,
                release_after=release_after,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                time.sleep(1.0)
        return 0

    if args.action:
        action = resolve_action(args.action)
        execute_action(
            args.binary,
            args.network_interface,
            action,
            delay=args.delay,
            release_after=release_after,
            dry_run=args.dry_run,
        )
        return 0

    print_action_list()
    while True:
        user_value = input("\nChoose action number/name/id: ").strip()

        if user_value.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            return 0

        if user_value.lower() in {"list", "menu"}:
            print_action_list()
            continue

        try:
            action = resolve_action(user_value)
            execute_action(
                args.binary,
                args.network_interface,
                action,
                delay=args.delay,
                release_after=release_after,
                dry_run=args.dry_run,
            )
        except ValueError as exc:
            print(exc)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
