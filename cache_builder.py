"""
Скрипт массового заполнения семантического кэша Кузьмича.
Генерирует ответы через LLM, синтезирует через XTTS и сохраняет в FAISS.
Не воспроизводит звук, работает в фоновом режиме.
"""

import os
import re
import uuid
import wave
import time
import tempfile
from pathlib import Path

# --- МАГИЧЕСКАЯ СТРОКА ---
# Импортируем PyTorch первым, чтобы он загрузил драйверы CUDA в память
import torch 

# Импорты движков проекта
from semantic_cache import SemanticCache
from llm_engine import LLMEngine
from tts_engine import TTSEngine

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_WAV_DIR = PROJECT_DIR / "cache_audio"

# Базовый список фраз (можно расширить до 1000+ в файле phrases.txt)
DEFAULT_PHRASES = [
    # Приветствия и статусы
    "Привет Кузьмич!", "Доброе утро.", "Как дела?", "Что делаешь?", 
    "Просыпайся.", "Как спалось?", "Как заряд батареи?", "Не скрипи сервоприводами.",
    
    # Сельское хозяйство
    "Как там урожай?", "Что думаешь про современный бизнес?", "Будет ли сегодня дождь?",
    "Пора собирать кукурузу.", "Трактор сломался.", "Олег Александрович зовёт.",
    "Как тебе новые удобрения?", "Коровы опять забор сломали.",
    
    # Оскорбления и проверки (для отработки защиты)
    "Ты тупая железка.", "Пошел ты.", "Неси какую-то дичь.", "Скучный ты.",
    
    # Бытовые вопросы
    "Расскажи о себе.", "Откуда ты взялся?", "Что ты умеешь?", "Спой песню.",
    "Расскажи анекдот.", "Пошути.", "Сколько тебе лет?", "Пора на металлолом."
]

def merge_wav_files(input_paths, output_path):
    if not input_paths:
        return
    with wave.open(input_paths[0], "rb") as first:
        params = first.getparams()
    with wave.open(output_path, "wb") as out_wav:
        out_wav.setparams(params)
        for path in input_paths:
            with wave.open(path, "rb") as src:
                if (src.getnchannels() == params.nchannels and
                    src.getsampwidth() == params.sampwidth and
                    src.getframerate() == params.framerate):
                    out_wav.writeframes(src.readframes(src.getnframes()))

def load_phrases(file_path="phrases.txt"):
    """Загружает фразы из файла, если он существует, иначе берет базовый список."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            phrases = [line.strip() for line in f if line.strip()]
        print(f"📂 Загружено {len(phrases)} фраз из {file_path}")
        return phrases
    else:
        print(f"📂 Файл {file_path} не найден. Использую базовый список ({len(DEFAULT_PHRASES)} фраз).")
        # Создаем файл для будущего использования
        with open(file_path, "w", encoding="utf-8") as f:
            for p in DEFAULT_PHRASES:
                f.write(p + "\n")
        return DEFAULT_PHRASES

def main():
    CACHE_WAV_DIR.mkdir(parents=True, exist_ok=True)
    phrases = load_phrases("phrases.txt")

    print("\n⚙️  Инициализация движков (потребуется пара минут)...")
    
    # Инициализируем LLM с расширенным контекстом
    llm_path = os.environ.get("VOICE_ENGINE_LLM_MODEL", str(PROJECT_DIR / "models" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"))
    llm = LLMEngine(model_path=llm_path, n_ctx=2048, n_gpu_layers=-1, inter_token_timeout=30.0)
    
    # Инициализируем XTTS без прогрева динамиков
    tts = TTSEngine(backend="xtts", warmup_on_init=False)
    
    # Подключаем кэш
    cache = SemanticCache(
        index_path=PROJECT_DIR / "semantic_cache_index.faiss",
        mapping_path=PROJECT_DIR / "semantic_cache_mapping.pkl"
    )

    print("\n🚀 Начинаем генерацию кэша!\n")
    
    success_count = 0
    start_time = time.time()

    for i, phrase in enumerate(phrases, 1):
        # 1. Проверяем, есть ли уже эта фраза в кэше (защита от двойной работы при рестарте)
        hit = cache.search(phrase, language="ru")
        # Если сходство выше 0.98, считаем, что ответ на этот вопрос уже есть
        if hit and hit.get("similarity", 0) > 0.98:
            print(f"[{i}/{len(phrases)}] ⏭️ ПРОПУСК (уже в кэше): «{phrase}»")
            continue

        print(f"[{i}/{len(phrases)}] 🧠 ГЕНЕРАЦИЯ: «{phrase}»")
        
        sentences = []
        temp_wav_paths = []
        
        try:
            # 2. Сначала ПОЛНОСТЬЮ генерируем текст через LLM
            for chunk in llm.generate_stream(phrase, "ru"):
                sentence = chunk.strip()
                
                # Санитайзер текста (убираем многоточия, чтобы XTTS не заикался)
                sentence = re.sub(r'\.{2,}', ',', sentence)
                sentence = re.sub(r'[*_~"«»]', '', sentence)
                sentence = sentence.strip()
                
                if not sentence:
                    continue
                
                sentences.append(sentence)
                print(f"   -> Текст: {sentence}")
            
            if not sentences:
                print("   ❌ Ошибка: LLM выдал пустой ответ.")
                continue
                
            # 3. Затем синтезируем аудио во временные файлы
            for sentence in sentences:
                tmp_wav = str(Path(tempfile.gettempdir()) / f"cache_tts_{uuid.uuid4().hex}.wav")
                tts.synthesize(sentence, "ru", tmp_wav)
                temp_wav_paths.append(tmp_wav)
                
            # 4. Склеиваем аудио
            full_text = " ".join(sentences)
            final_wav = str(CACHE_WAV_DIR / f"precache_{uuid.uuid4().hex}.wav")
            merge_wav_files(temp_wav_paths, final_wav)
            
            # 5. Сохраняем в семантический кэш
            cache.put(phrase, full_text, final_wav, "ru")
            success_count += 1
            print(f"   ✅ СОХРАНЕНО. (Ответ: {full_text[:40]}...)")
        
        except Exception as e:
            print(f"   ❌ ОШИБКА при обработке «{phrase}»: {e}")
        finally:
            # Убираем временные файлы
            for p in temp_wav_paths:
                if os.path.exists(p):
                    os.remove(p)

    elapsed = round(time.time() - start_time, 2)
    print(f"\n🎉 Завершено! Добавлено новых записей: {success_count}. Затрачено времени: {elapsed} сек.")

if __name__ == "__main__":
    main()
