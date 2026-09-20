from __future__ import annotations

import logging
import os
import requests
from typing import Optional


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class G1GestureController:
    """
    Gesture controller using the G1 robot's local HTTP API.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self.enabled = _env_bool("VOICE_ENGINE_ENABLE_GESTURES", True) if enabled is None else enabled
        self.log = log or logging.getLogger("G1GestureController")
        
        self.robot_ip = os.environ.get("VOICE_ENGINE_ROBOT_IP", "192.168.1.103")
        self.api_url = f"http://{self.robot_ip}:8091"

    def start(self, gesture_name: str) -> None:
        """Sends an HTTP request to start a gesture if the robot is not busy."""
        if not self.enabled:
            self.log.debug("Gestures disabled, skip: %s", gesture_name)
            return

        self.log.info("--> Starting gesture: %s (via HTTP API)", gesture_name)

        try:
            status_response = requests.get(f"{self.api_url}/api/status", timeout=1.0)
            status_response.raise_for_status()
            
            if status_response.json().get("busy"):
                self.log.warning("Robot is busy with another gesture. Skipping: %s", gesture_name)
                return

            play_response = requests.post(
                f"{self.api_url}/api/motion",
                json={"name": gesture_name},
                timeout=1.0
            )
            play_response.raise_for_status()
            self.log.debug("Gesture successfully sent to the robot.")
            
        except requests.exceptions.RequestException as exc:
            self.log.error("Failed to launch gesture %s via the API: %s", gesture_name, exc)