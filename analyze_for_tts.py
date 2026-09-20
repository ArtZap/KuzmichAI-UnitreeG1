#!/usr/bin/env python3
"""Helper for the voice module: a function to request a scene analysis from the server
and retrieve the path to a text file for TTS.

from analyze_for_tts import analyze_for_tts
path = analyze_for_tts()
tts_say(open(path, encoding="utf-8").read())   # speak the text

The function sends a POST /analyze request to the perception server (192.168.10.179:8002),
takes the scene description (the 'summary' field), and writes it to a file (atomically,
using write+rename, to prevent TTS from reading a partial file). It returns the file path.
In case of an error, fallback text is written to the file so the robot does not remain silent.

Uses only the standard library (urllib/json/os)—system Python 3; no venv required.
Suitable for both the robot and a PC.

Parameters (optional; defaults from environment variables):
out     output file path       — env ANALYZE_TTS_FILE (default /tmp/agrohab_analysis.txt)
url     server URL             — env SERVER_URL       (default http://192.168.10.179:8002)
timeout HTTP timeout, seconds
"""
import json
import os
import urllib.request

SERVER_URL = os.environ.get("SERVER_URL", "http://192.168.1.103:8002")
OUT_FILE = os.environ.get("ANALYZE_TTS_FILE", "/tmp/agrohab_analysis.txt")
TIMEOUT = 90
FALLBACK = "Analysis failed. Please try again."


def analyze_for_tts(out=OUT_FILE, url=SERVER_URL, timeout=TIMEOUT):
    """
    POST /analyze -> scene description -> text to file -> return file path. 

    Returns `out` (path to the file containing text for TTS).
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
            raise RuntimeError(resp.get("err") or "empty VLM response")
    except Exception:
        text = FALLBACK
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    os.replace(tmp, out)
    return out


if __name__ == "__main__":
    print(analyze_for_tts())