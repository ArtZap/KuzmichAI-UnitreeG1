import json
import os
import re
import uuid
from pathlib import Path

os.environ["VOICE_ENGINE_TTS"] = "xtts"
os.environ["VOICE_ENGINE_TTS_TEMPO"] = "1.0"
os.environ["VOICE_ENGINE_TTS_GAIN_DB"] = "16"
os.environ["VOICE_ENGINE_XTTS_SPLIT_SENTENCES"] = "true"

from semantic_cache import SemanticCache
from llm_engine import LLMEngine
from tts_engine import TTSEngine

PROJECT_DIR = Path(__file__).resolve().parent
KB_CACHE_WAV_DIR = PROJECT_DIR / "kb_audio"
KB_CACHE_WAV_DIR.mkdir(parents=True, exist_ok=True)

def main():
    dump_file = PROJECT_DIR / "kb_dump.json"
    if not dump_file.exists():
        print(f"[-] Файл {dump_file} не найден. Сохраните базу!")
        return

    with open(dump_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    facts = data.get("facts", [])
    print(f"[+] Загружено {len(facts)} фактов. Инициализация движков...")

    kb_cache = SemanticCache(
        index_path=PROJECT_DIR / "kb_index.faiss",
        mapping_path=PROJECT_DIR / "kb_mapping.pkl"
    )
    
    llm = LLMEngine(
        model_path=str(PROJECT_DIR / "models" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
        enable_context=False
    )
    tts = TTSEngine(warmup_on_init=False)

    print("[+] Начинаем генерацию базы знаний...\n")
    
    for fact in facts:
        query = fact.get("title", "")
        raw_text = fact.get("body", "")
        
        if not query or not raw_text:
            continue
            
        print(f"Тема: {query}")
        
        styled_prompt = f"Пользователь интересуется темой: '{query}'. Ответь в своем фирменном стиле (коротко, 1-2 предложения), опираясь ИСКЛЮЧИТЕЛЬНО на этот факт из базы: '{raw_text}'"
        
        sentences = []
        for sentence in llm.generate_stream(styled_prompt, lang_code="ru"):
            sentence = re.sub(r'\.{2,}', ',', sentence)
            sentence = re.sub(r'[*_~"«»]', '', sentence)
            sentence = sentence.replace('。', '.').replace('！', '!').replace('？', '?').replace('，', ',')
            sentence = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', '', sentence)
            sentence = re.sub(r'\[жест:\s*([a-zA-Z0-9_]+)\]', '', sentence, flags=re.IGNORECASE)
            
            if sentence.strip():
                sentences.append(sentence.strip())
        
        full_answer = " ".join(sentences)
        print(f"Ответ Кузьмича: {full_answer}")
        
        wav_path = str(KB_CACHE_WAV_DIR / f"{uuid.uuid4().hex}.wav")
        try:
            tts.synthesize(full_answer, "ru", wav_path)
            kb_cache.put(query, full_answer, wav_path, "ru")
            print("-> Успешно добавлено в несгораемый кэш.")
        except Exception as e:
            print(f"-> Ошибка синтеза: {e}")
            
        print("-" * 50)
        
    print("\n[+] Все факты успешно стилизованы и сохранены в несгораемый кэш!")

if __name__ == "__main__":
    main()