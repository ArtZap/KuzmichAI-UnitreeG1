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
    Контроллер жестов с использованием локального HTTP API робота G1.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self.enabled = _env_bool("VOICE_ENGINE_ENABLE_GESTURES", True) if enabled is None else enabled
        self.log = log or logging.getLogger("G1GestureController")
        
        # IP робота берем из переменных окружения
        self.robot_ip = os.environ.get("VOICE_ENGINE_ROBOT_IP", "192.168.1.103")
        # Формируем URL для обращения к API на 8091 порту
        self.api_url = f"http://{self.robot_ip}:8091"

    def start(self, gesture_name: str) -> None:
        """Отправляет HTTP-запрос на запуск жеста, если робот не занят."""
        if not self.enabled:
            self.log.debug("Жесты отключены, пропуск: %s", gesture_name)
            return

        self.log.info("--> Запуск жеста: %s (через HTTP API)", gesture_name)

        try:
            # 1. Проверяем статус: не занят ли робот другим движением
            status_response = requests.get(f"{self.api_url}/api/status", timeout=1.0)
            status_response.raise_for_status()
            
            # Если "busy": true, пропускаем новый жест, чтобы не было конфликтов
            if status_response.json().get("busy"):
                self.log.warning("Робот занят другим жестом. Пропуск: %s", gesture_name)
                return

            # 2. Отправляем команду на старт нового жеста
            play_response = requests.post(
                f"{self.api_url}/api/motion",
                json={"name": gesture_name},
                timeout=1.0
            )
            play_response.raise_for_status()
            self.log.debug("Жест успешно передан на робота.")
            
        except requests.exceptions.RequestException as exc:
            self.log.error("Не удалось запустить жест %s через API: %s", gesture_name, exc)