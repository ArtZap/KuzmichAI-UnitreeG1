import socket
import sys
import json

print("=========================================")
print("  KUZMICH'S OPERATOR CONSOLE (MULTILINGUAL)")
print("=========================================")
print("Enter text, and the robot will speak it aloud.")
print("  - Default voice output is in RUSSIAN.")
print("  - For a different language, use a prefix: /en, /zh-cn, /es, etc.")
print("    Example: /en Hello, dear guest!")
print("  - The /stop command will interrupt the current speech.")
print("  - Gestures work too: /en [жест|jest|gest|gesture: shakehands] Nice to meet you!")
print("Press Ctrl+C to exit\n")

while True:
    try:
        raw_text = input("Say: ").strip()
        if not raw_text:
            continue
        
        lang = "ru"
        text = raw_text

        if raw_text.startswith("/") and " " in raw_text and not raw_text.startswith("/stop"):
            parts = raw_text.split(" ", 1)
            potential_lang = parts[0][1:] 
            
            if 1 < len(potential_lang) <= 5: 
                lang = potential_lang
                text = parts[1].strip()

        payload = {
            "text": text,
            "lang": lang
        }

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(('127.0.0.1', 9999))
            s.sendall((json.dumps(payload) + '\n').encode('utf-8'))
            
    except KeyboardInterrupt:
        print("\nExiting the operator console.")
        sys.exit(0)
    except ConnectionRefusedError:
        print("[Error] Kuzmich is not running or port 9999 is not accessible.")
    except Exception as e:
        print(f"[Error] {e}")