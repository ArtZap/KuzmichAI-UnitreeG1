#!/usr/bin/env python3
"""Помощник голосовому модулю: функция — запросить анализ сцены на сервере и
получить путь к файлу с текстом для TTS.

    from analyze_for_tts import analyze_for_tts
    path = analyze_for_tts()
    tts_say(open(path, encoding="utf-8").read())   # озвучить

Функция делает POST /analyze к perception-серверу (192.168.10.179:8002), берёт
описание сцены (поле summary) и пишет его в файл (атомарно, write+rename — чтобы
TTS не прочитал наполовину). Возвращает путь к файлу. При ошибке в файл кладётся
fallback-текст, чтобы робот не молчал.

Только стандартная библиотека (urllib/json/os) — системный python3, venv НЕ нужен.
Подходит для робота и ПК.

Параметры (опциональны, дефолты из env):
  out     путь выходного файла   — env ANALYZE_TTS_FILE (дефолт /tmp/agrohab_analysis.txt)
  url     URL сервера            — env SERVER_URL       (дефолт http://192.168.10.179:8002)
  timeout таймаут HTTP, сек
"""
import json
import os
import urllib.request

SERVER_URL = os.environ.get("SERVER_URL", "http://192.168.10.179:8002")
OUT_FILE = os.environ.get("ANALYZE_TTS_FILE", "/tmp/agrohab_analysis.txt")
TIMEOUT = 90  # /analyze ~3-9с (cloud + reasoning M3) — с запасом
FALLBACK = "Не удалось выполнить анализ. Попробуйте ещё раз."


def analyze_for_tts(out=OUT_FILE, url=SERVER_URL, timeout=TIMEOUT):
    """POST /analyze -> описание сцены -> текст в файл -> вернуть путь к файлу.

    Возвращает `out` (путь к файлу с текстом для TTS).
    """
    try:
        req = urllib.request.Request(
            url.rstrip("/") + "/analyze",
            data=json.dumps({"command": "анализ"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
        text = (resp.get("summary") or "").strip()
        if not text:
            raise RuntimeError(resp.get("err") or "пустой ответ VLM")
    except Exception:
        text = FALLBACK  # чтобы TTS было что озвучить
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    os.replace(tmp, out)
    return out


if __name__ == "__main__":
    # Ручной тест: python3 analyze_for_tts.py  -> печатает путь.
    print(analyze_for_tts())