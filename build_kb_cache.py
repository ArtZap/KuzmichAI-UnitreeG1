import json
import os
import re
import uuid
from pathlib import Path

os.environ["VOICE_ENGINE_TTS"] = "xtts"
os.environ["VOICE_ENGINE_TTS_TEMPO"] = "0.9"
os.environ["VOICE_ENGINE_TTS_GAIN_DB"] = "16"
os.environ["VOICE_ENGINE_XTTS_SPLIT_SENTENCES"] = "true"

from semantic_cache import SemanticCache
from tts_engine import TTSEngine

PROJECT_DIR = Path(__file__).resolve().parent
KB_CACHE_WAV_DIR = PROJECT_DIR / "kb_audio"
KB_CACHE_WAV_DIR.mkdir(parents=True, exist_ok=True)

from main import GESTURE_REGEX, normalize_gesture_name


def clean_text_for_tts(text: str) -> str:
  text = GESTURE_REGEX.sub('', text)
  text = re.sub(r'\.{2,}', ',', text)
  text = re.sub(r'[*_~"«»]', '', text)
  text = (
      text.replace('。', '.')
      .replace('！', '!')
      .replace('？', '?')
      .replace('，', ',')
  )
  text = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', '', text)
  return text.strip()

def main():
    dump_file = PROJECT_DIR / "kb_dump.json"
    if not dump_file.exists():
        print(f"[-] File {dump_file} not found. Save the database.!")
        return

    with open(dump_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    qa_items = data.get("qa", [])
    print(f"[+] Loaded {len(qa_items)} questions and answers. Initializing TTS engine...")

    kb_cache = SemanticCache(
        index_path=PROJECT_DIR / "kb_index.faiss",
        mapping_path=PROJECT_DIR / "kb_mapping.pkl"
    )
    
    tts = TTSEngine(warmup_on_init=False)

    print("[+] Starting audio file generation and knowledge base filling...\n")
    
    for item in qa_items:
        query = item.get('query', '').strip()
        raw_answer = item.get('answer', '').strip()

        if not query or not raw_answer:
            continue

        gesture_match = GESTURE_REGEX.search(raw_answer)
        g_name = (
            normalize_gesture_name(gesture_match.group(1)) if gesture_match else None
        )

        clean_answer = clean_text_for_tts(raw_answer)
        stored_response = (
            f'[gesture: {g_name}] {clean_answer}' if g_name else clean_answer
        )

        wav_path = str(KB_CACHE_WAV_DIR / f'{uuid.uuid4().hex}.wav')
        try:
            tts.synthesize(clean_answer, 'ru', wav_path)
            kb_cache.put(query, stored_response, wav_path, 'ru')
            print('-> Successfully added to knowledge base.')
        except Exception as e:
            print(f'-> Synthesis error: {e}')
        
    print("\n[+] All question-answer pairs successfully synthesized and saved in cache!")

if __name__ == "__main__":
    main()