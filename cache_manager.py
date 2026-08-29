import os
import glob
import pickle
import asyncio
import numpy as np
import faiss #
from pathlib import Path

# Импорты локальных модулей KuzmichAI
from semantic_cache import SemanticCache
from llm_engine import LLMEngine
from tts_engine import TTSEngine

# Пути из технического описания проекта
CACHE_DIR = Path("/home/unitree/KuzmichAI/cache_audio") #[cite: 1]
MAPPING_PATH = "/home/unitree/KuzmichAI/semantic_cache_mapping.pkl" #[cite: 1]
FAISS_PATH = "/home/unitree/KuzmichAI/semantic_cache_index.faiss" #[cite: 1]

# 16 поддерживаемых языков[cite: 1]
SUPPORTED_LANGS = [
    "ar", "cs", "de", "en", "es", "fr", "hu", "it", 
    "ja", "ko", "nl", "pl", "pt", "ru", "tr", "zh-cn"
] #[cite: 1]

EXHIBITION_QUESTIONS = [
    # Знакомство (10)
    "Кто ты такой?", "Как тебя зовут?", "Откуда ты?", "Как твои дела?", "Рад познакомиться.", 
    "Ты меня понимаешь?", "Сколько тебе лет?", "Какой у тебя характер?", "Как к тебе обращаться?", "Ты живой?",
    # Железо и Unitree G1 (20)
    "Кто тебя сделал?", "Где твой компьютер?", "Сколько ты весишь?", "Из чего ты сделан?", "Сколько у тебя моторов?", 
    "У тебя есть камеры?", "Как ты меня слышишь?", "Какая у тебя батарея?", "На сколько хватает зарядки?", "У тебя есть видеокарта?", 
    "Ты подключен к интернету?", "Как работает твой микрофон?", "Зачем тебе лидар?", "Ты можешь упасть?", "Сколько у тебя степеней свободы?", 
    "Какой у тебя процессор?", "Ты водонепроницаемый?", "Сколько электричества ты потребляешь?", "У тебя есть датчики касания?", "Что у тебя в голове?",
    # Выставка и АгроХаб (20)
    "Что ты делаешь на выставке?", "Что такое АгроХаб?", "Как роботы помогают в сельском хозяйстве?", "Ты умеешь сажать картошку?", "Ты умеешь доить коров?", 
    "Что ты знаешь про тракторы?", "Можешь собрать урожай?", "Зачем фермерам роботы?", "Какое будущее у агропромышленности?", "Ты можешь работать в поле?", 
    "Как нейросети помогают выращивать пшеницу?", "Ты умеешь полоть грядки?", "Какие датчики нужны для теплицы?", "Ты можешь управлять комбайном?", "Какая сегодня погода для посева?", 
    "Заменит ли ИИ агрономов?", "Что ты знаешь про гидропонику?", "Умеешь собирать яблоки?", "Где применяются дроны в сельском хозяйстве?", "Ты знаешь виды удобрений?",
    # Возможности и команды (20)
    "Что ты умеешь делать?", "Ты умеешь ходить?", "Можешь потанцевать?", "Дай пять!", "Сделай сальто.", 
    "Пройдись вперед.", "Подними руку.", "Поверни голову.", "Сядь.", "Встань.", 
    "Можешь заварить кофе?", "Умеешь играть на гитаре?", "Ты умеешь прыгать?", "Как быстро ты бегаешь?", "Ты можешь принести мне воды?", 
    "Спой песню.", "Прочитай стих.", "Ты умеешь плавать?", "Сможешь поднять тяжесть?", "Можешь погладить собаку?",
    # ИИ, LLM, TTS (20)
    "Где твой мозг?", "Какая у тебя нейросеть?", "Ты используешь ChatGPT?", "Как ты генерируешь голос?", "Какая у тебя языковая модель?", 
    "Ты знаешь все языки?", "Как быстро ты думаешь?", "У тебя есть сознание?", "Что такое Qwen?", "Как работает твой распознаватель речи?", 
    "Ты можешь учиться новому?", "Ты запоминаешь людей?", "О чем ты мечтаешь?", "Ты боишься выключения?", "Ты умнее человека?", 
    "Что будет, если отключить свет?", "Зачем тебе кэширование?", "Ты можешь ошибаться?", "Ты соблюдаешь законы робототехники?", "Ты терминатор?",
    # Навигация и разное (10)
    "Расскажи анекдот.", "Скажи что-нибудь смешное.", "Ты хочешь захватить мир?", "Где здесь туалет?", "Где главный стенд?", 
    "Можно с тобой сфотографироваться?", "Сколько ты стоишь?", "Где тебя можно купить?", "Кто твой лучший друг?", "В чем смысл жизни?"
]

def clean_database():
    """Очистка базы данных от мусорных вопросов и legacy-записей[cite: 1]."""
    print("Начинаем очистку базы от мусора...")
    
    if not os.path.exists(MAPPING_PATH) or not os.path.exists(FAISS_PATH):
        print("База пуста или не найдена.")
        return

    with open(MAPPING_PATH, "rb") as f:
        mapping = pickle.load(f)
        
    index = faiss.read_index(FAISS_PATH) #[cite: 1]
    valid_mapping = {}
    ids_to_remove = []
    
    for uid, data in mapping.items():
        wav_path = data.get("audio_path")
        lang = data.get("language")
        
        # Записи без языка (legacy) или с отсутствующим файлом считаем мусором[cite: 1]
        if wav_path and os.path.exists(wav_path) and lang in SUPPORTED_LANGS:
            valid_mapping[uid] = data
        else:
            ids_to_remove.append(uid)
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception as e:
                    print(f"Не удалось удалить файл {wav_path}: {e}")
                    
    all_wavs = glob.glob(str(CACHE_DIR / "*.wav"))
    valid_wavs = set(data.get("audio_path") for data in valid_mapping.values())
    
    for wav in all_wavs:
        if wav not in valid_wavs:
            os.remove(wav)

    if ids_to_remove:
        print(f"Удаляем {len(ids_to_remove)} мусорных записей из FAISS индекса...")
        index.remove_ids(np.array(ids_to_remove, dtype=np.int64)) #[cite: 1]
        
        with open(MAPPING_PATH, "wb") as f:
            pickle.dump(valid_mapping, f)
        faiss.write_index(index, FAISS_PATH)
        print("Очистка успешно завершена.")
    else:
        print("Мусор не найден, база в порядке.")

async def boost_database():
    """Искусственный буст базы на 100 вопросов для 16 языков."""
    print("Инициализация моделей. Это может занять время на Jetson Orin...[cite: 1]")
    
    llm = LLMEngine() #[cite: 1]
    tts = TTSEngine(warmup_on_init=False) #[cite: 1]
    cache = SemanticCache() #[cite: 1]
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    for i, base_q in enumerate(EXHIBITION_QUESTIONS, 1):
        for lang in SUPPORTED_LANGS:
            print(f"[{i}/100] Обработка: '{base_q}' -> Язык: {lang}")
            
            prompt = (
                f"Translate this question to language code '{lang}': '{base_q}'. "
                f"Then write a very short answer (1-2 sentences) in '{lang}' acting as KuzmichAI, "
                f"a robot at the AgroHub exhibition. "
                f"Format strictly as: TRANSLATED_QUESTION|ANSWER"
            )
            
            try:
                # В зависимости от реализации LLMEngine может потребоваться await[cite: 1]
                response = llm.generate(prompt) 
                
                if "|" not in response:
                    continue
                    
                translated_q, answer = response.split("|", 1)
                translated_q = translated_q.strip()
                answer = answer.strip()
                
                wav_filename = f"boost_{hash(translated_q)}_{lang}.wav"
                wav_path = str(CACHE_DIR / wav_filename)
                
                tts.synthesize(answer, lang, wav_path) #[cite: 1]
                cache.add(translated_q, wav_path, language=lang) #[cite: 1]
                
            except Exception as e:
                print(f"Ошибка при обработке {lang}: {e}")
                
    print("Буст базы завершен! Данные сохранены.")

if __name__ == "__main__":
    clean_database()
    # Запускаем генерацию кэша
    asyncio.run(boost_database())
