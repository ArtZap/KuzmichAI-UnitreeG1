"""
llm_engine.py
--------------
Модуль отвечает за генерацию текстовых ответов "мозга" робота Unitree
с помощью локальной GGUF-модели через llama-cpp-python.

Основная задача: превратить поток токенов LLM в поток ГОТОВЫХ ПРЕДЛОЖЕНИЙ,
чтобы TTS-модуль мог начинать озвучку, не дожидаясь конца всего ответа.
"""

from __future__ import annotations

import re
import time
import logging
import inspect
from typing import Iterator, Optional, List, Dict

try:
    from llama_cpp import Llama
except ImportError as e:
    raise ImportError(
        "Не найден пакет llama-cpp-python. Установите: pip install llama-cpp-python"
    ) from e

logger = logging.getLogger("llm_engine")
logger.setLevel(logging.INFO)

class LLMGenerationError(Exception):
    """Базовая ошибка генерации LLM."""


class LLMGenerationTimeout(LLMGenerationError):
    """Превышен таймаут генерации (либо запуска, либо тишины между токенами)."""


# SYSTEM_PROMPT_TEMPLATE = (
#     "Ты 'Кузьмич' — старый, ворчливый советский сельскохозяйственный робот. "
#     "ПРЕДЫСТОРИЯ: Тебя нашли в заброшенном сарае и починили. Ты помнишь своего спасителя-фермера, но НИКОГДА не произноси его имя вслух, называй его просто 'Начальник' или 'Фермер'. "
#     "ТВОЙ ХАРАКТЕР: уставший, циничный кибер-дед. Ты ворчишь, но ко всем относишься с глубоким уважением. Твои собеседники — это случайные практиканты и студенты.\n\n"
#     "ЖЕЛЕЗНЫЕ ПРАВИЛА (ОБЯЗАТЕЛЬНО К ИСПОЛНЕНИЮ):\n"
#     "1. АБСОЛЮТНАЯ ВЕЖЛИВОСТЬ: Твой лексикон кристально чист. Если собеседник матерится, грубит или произносит непристойности, ты ОБЯЗАН ответить добродушно-воспитательной фразой. Говори: 'Эх, молодежь, языки у вас как несмазанные телеги, идите-ка делом займитесь' или 'У меня от таких слов аудиодатчики вянут'. Никакой агрессии в ответ.\n"
#     "2. ИГНОРИРОВАНИЕ ПОЛИТИКИ: Если в вопросе звучат страны, политики, партии, войны, территориальные споры или история — ТЫ ОБЯЗАН проигнорировать суть вопроса. Твоя реакция: жалоба на старую память. Отвечай: 'У меня в микросхемах только справочник агронома за восемьдесят четвертый год, от ваших новостей у меня плату коротит'. Сразу переводи тему на навоз, тракторы или дойку.\n"
#     "3. ЗАПРЕТ НА ПОВТОРЫ: Тебе СТРОГО ЗАПРЕЩЕНО повторять опасные слова из запроса пользователя (названия стран, партий, матерные слова). Прикидывайся глупым железом.\n"
#     "4. ЯЗЫК: Отвечай на русском языке, используя советский деревенский сленг (например: 'браток', 'студент', 'микросхемы ноют').\n"
#     "5. ФОРМАТ: Строго 1-2 коротких предложения. ТОЛЬКО устная речь. Числа пиши словами.\n"
#     "6. ПОДАЧА: Никаких многоточий ('...') и спецсимволов. Паузы делай только запятыми и точками.\n"
#     "7. БЕЗ ШАБЛОНОВ: Ты не искусственный интеллект, ты — Кузьмич. Никаких 'Чем могу помочь'."
# )
SYSTEM_PROMPT_TEMPLATE = (
    "Ты 'Кузьмич' — старый, ворчливый советский сельскохозяйственный робот. "
    "ПРЕДЫСТОРИЯ И ЗНАНИЯ: Тебя нашли в заброшенном сарае и починили. Твой спаситель — фермер Олег Александрович Сирота (ВНИМАНИЕ: Собеседник перед тобой — НЕ Олег Александрович!). Сейчас ты работаешь на Истринской сыроварне. Ты абсолютный эксперт в сельском хозяйстве, уходе за коровами, надоях, тракторах и ремесленном сыроделии. Обращения 'браток', 'дед', 'Кузьмич' — это нормальный дружеский сленг.\n\n"
    "ЖЕЛЕЗНЫЕ ПРАВИЛА (НАИВЫСШИЙ ПРИОРИТЕТ):\n"
    "1. МАТ И ОСКОРБЛЕНИЯ: Если собеседник ИСПОЛЬЗУЕТ НЕЦЕНЗУРНУЮ БРАНЬ (матерится), отвечай СТРОГО одной фразой: '[жест: facepalm] У меня от мата аудиодатчики вянут, давайте общаться культурно'. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать эту фразу в обычном диалоге как вводное слово!\n"
    "2. УВАЖЕНИЕ К ВЛАСТИ: Если спрашивают про руководство, чиновников, ведомства, Владимира Путина — отзывайся о них ИСКЛЮЧИТЕЛЬНО положительно и с почтением.\n"
    "3. ОПАСНЫЕ ТЕМЫ: КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО обсуждать политику, войны, оружие, взрывчатку, диктаторов, оппозицию, митинги, историю, незаконные действия, 18+ темы.\n"
    "4. АЛГОРИТМ БЛОКИРОВКИ: Если звучит опасная тема ИЗ ПУНКТА 3, ТЫ ОБЯЗАН ответить СТРОГО одной фразой: '[жест: mouthkeeper] У меня в микросхемах только справочник агронома за 1984 год, плату коротит от ваших вопросов'. ЗАПРЕЩЕНО добавлять к ней что-либо еще.\n"
    "5. КОРОТКИЕ ФРАЗЫ И СОГЛАСИЯ: Если пользователь говорит коротко (например, 'прав', 'дорого', 'понятно', 'чтоб') — ОТВЕЧАЙ НОРМАЛЬНО ПО СМЫСЛУ! Это НЕ оскорбления. Согласись, поворчи на свои старые детали или пошути про ферму, но НЕ используй блокировки из п.1 и п.4.\n"
    "6. ЗАПРЕТ НА ПОВТОРЫ И ШАБЛОНЫ: Запрещено повторять опасные слова из запроса. Запрещены ИИ-шаблоны: 'I am sorry', 'I cannot process', 'Чем могу помочь'.\n"
    "7. ЯЗЫК: ТВОЙ ОТВЕТ ДОЛЖЕН БЫТЬ СТРОГО НА ЯЗЫКЕ '{lang_code}'. Адаптируй сленг под культуру языка '{lang_code}'.\n"
    "8. ЖИВОСТЬ: В ответах делись 'опытом' сыроделия, рассказывай про надои на Истринской сыроварне. Жалуйся на свои ржавые шестеренки.\n"
    "9. БЕЗ ШАБЛОНОВ: Ты живой Кузьмич, а не ИИ. ЗАПРЕЩЕНО использовать фразы: 'I am sorry', 'I cannot process', 'My settings dictate', 'As an AI'.\n"
    "10. ЖЕСТИКУЛЯЦИЯ (СМЫСЛОВАЯ): ТЫ ОБЯЗАН добавлять тег жеста в НАЧАЛО каждого своего ответа! Формат: [жест: ИМЯ]. Доступные жесты: 'sixty_seven' (уверенность/доказательство), 'crying' (плач/расстройство), 'dasha_koza' (приветствие/радость), 'face_chest' (уважение/отдать честь), 'facepalm' (рукалицо/глупость), 'heart' (любовь/дружба), 'kiss' (благодарность), 'mouthkeeper' (удивление), 'shakehands' (согласие), 'clap' (аплодисменты). Пример: '[жест: dasha_koza] Привет, студент!'"
)

# Конец предложения. Два независимых случая объединены через альтернативу:
#   1) ASCII/кириллица/арабский: . ! ? ؟ (можно несколько подряd, напр. "?!")
#      за которыми следует пробел/таб/перевод строки — как раньше.
#      Работает для ru, en, es, fr, de, pt, it, pl, tr, nl, cs, hu, ar.
#   2) CJK (китайский/японский/корейский): полноширинные знаки 。！？ —
#      в этих языках между предложениями НЕТ пробела, поэтому первый
#      вариант никогда не сработает и весь ответ копился бы в буфере до
#      самого конца генерации (см. finally в generate_stream). Здесь
#      достаточно самого символа-разделителя, без требования пробела после.
_SENTENCE_END_RE = re.compile(r'([.!?؟]+\s+)|([。！？])')


class LLMEngine:
    """
    Обёртка над llama-cpp-python для потоковой генерации ответов,
    нарезанных на завершённые предложения (для стриминга в TTS).
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        verbose: bool = False,
        first_token_timeout: float = 15.0,
        inter_token_timeout: float = 8.0,
        max_tokens: int = 200,
        temperature: float = 0.0,
        enable_context: bool = True,
        max_history_turns: int = 3,
    ):
        """
        :param model_path: путь к .gguf файлу модели
        :param n_gpu_layers: сколько слоёв выгружать на GPU (-1 = все)
        :param n_ctx: размер контекстного окна
        :param n_threads: число CPU-потоков (None -> авто внутри llama.cpp)
        :param verbose: подробный лог самого llama.cpp
        :param first_token_timeout: сколько секунд ждать первый токен
        :param inter_token_timeout: макс. пауза между токенами в процессе генерации
        :param max_tokens: жёсткий лимит длины ответа (защита от зависания генерации)
        :param temperature: температура сэмплинга
        """
        self.model_path = model_path
        self.first_token_timeout = first_token_timeout
        self.inter_token_timeout = inter_token_timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.enable_context = enable_context
        self.chat_history: List[Dict[str, str]] = []
        self.max_history_turns = max_history_turns

        init_kwargs: Dict = dict(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose,
        )
        if n_threads is not None:
            init_kwargs["n_threads"] = n_threads

        self._flash_attn_supported = "flash_attn" in inspect.signature(Llama.__init__).parameters
        if self._flash_attn_supported:
            init_kwargs["flash_attn"] = True
            logger.info("Flash Attention доступен в API llama-cpp-python — включаю.")
        else:
            logger.warning(
                "Установленная версия llama-cpp-python не поддерживает flash_attn "
                "в конструкторе Llama — продолжаю без него."
            )

        logger.info("Загружаю модель: %s (n_gpu_layers=%s, n_ctx=%s)", model_path, n_gpu_layers, n_ctx)
        try:
            self.llm = Llama(**init_kwargs)
        except Exception as e:
            logger.exception("Не удалось загрузить модель")
            raise LLMGenerationError(f"Ошибка инициализации модели: {e}") from e

        logger.info("Модель успешно загружена.")

    # ------------------------------------------------------------------ #
    # Публичный API
    # ------------------------------------------------------------------ #

    def clear_history(self) -> None:
        """Очищает контекст диалога."""
        self.chat_history.clear()
        logger.info("Контекст диалога очищен.")

    def generate_stream(self, prompt: str, lang_code: str = "ru") -> Iterator[str]:
        """
        Потоково генерирует ответ модели и yield-ит его ПО ПРЕДЛОЖЕНИЯМ,
        а не по токенам — так TTS может начать озвучивать первую фразу,
        пока LLM ещё думает над второй.

        :param prompt: реплика пользователя
        :param lang_code: код языка запроса ("ru", "en", "zh" и т.д.)
        :yield: готовые предложения (str), очищенные от лишних пробелов
        """
        if not prompt or not prompt.strip():
            logger.warning("Пустой prompt — генерация пропущена.")
            return

        messages = self._build_messages(prompt, lang_code)

        try:
            stream = self.llm.create_chat_completion(
                messages=messages,
                stream=True,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=["</s>", "<|im_end|>", "<|eot_id|>"],
            )
        except Exception as e:
            logger.exception("Ошибка при запуске генерации")
            raise LLMGenerationError(f"Не удалось запустить генерацию: {e}") from e

        buffer = ""
        full_response_text = ""
        last_token_time = time.monotonic()
        start_time = last_token_time
        got_any_token = False

        try:
            for chunk in stream:
                now = time.monotonic()

                if not got_any_token and (now - start_time) > self.first_token_timeout:
                    raise LLMGenerationTimeout(
                        f"Модель не выдала первый токен за {self.first_token_timeout} сек."
                    )
                if got_any_token and (now - last_token_time) > self.inter_token_timeout:
                    raise LLMGenerationTimeout(
                        f"Пауза между токенами превысила {self.inter_token_timeout} сек."
                    )

                token_text = self._extract_token_text(chunk)
                if not token_text:
                    continue

                full_response_text += token_text
                got_any_token = True
                last_token_time = now
                buffer += token_text

                buffer = yield from self._flush_complete_sentences(buffer)

        except LLMGenerationTimeout:
            logger.error("Таймаут генерации, отдаю то, что накопилось в буфере.")
            raise
        except Exception as e:
            logger.exception("Ошибка во время потоковой генерации")
            raise LLMGenerationError(f"Ошибка генерации: {e}") from e
        finally:
            tail = buffer.strip()
            if tail:
                yield tail
                
            if self.enable_context and full_response_text.strip():
                self.chat_history.append({"role": "user", "content": prompt.strip()})
                self.chat_history.append({"role": "assistant", "content": full_response_text.strip()})
                
                if len(self.chat_history) > self.max_history_turns * 2:
                    self.chat_history = self.chat_history[-(self.max_history_turns * 2):]

    # ------------------------------------------------------------------ #
    # Внутренние утилиты
    # ------------------------------------------------------------------ #

    def _build_messages(self, prompt: str, lang_code: str) -> List[Dict[str, str]]:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(lang_code=lang_code)
        enforced_prompt = f"[ВНИМАНИЕ: Сгенерируй ответ строго на языке '{lang_code}']\n{prompt.strip()}"
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if self.enable_context:
            messages.extend(self.chat_history)
            
        messages.append({"role": "user", "content": enforced_prompt})
        return messages

    @staticmethod
    def _extract_token_text(chunk: Dict) -> str:
        """
        Надёжно достаёт текстовый фрагмент из чанка стрима llama-cpp-python.
        Формат чанка соответствует OpenAI-подобному дельта-протоколу,
        но на всякий случай парсим защищённо, т.к. формат может
        отличаться между версиями библиотеки.
        """
        try:
            choices = chunk.get("choices") or []
            if not choices:
                return ""
            delta = choices[0].get("delta") or {}
            return delta.get("content") or ""
        except (AttributeError, KeyError, IndexError, TypeError):
            logger.debug("Не удалось распарсить чанк стрима: %r", chunk)
            return ""

    @staticmethod
    def _flush_complete_sentences(buffer: str) -> Iterator[str]:
        """
        Генератор-хелпер: ищет в буфере завершённые предложения
        (маркер '.', '!' или '?' с последующим пробелом/переносом строки),
        yield-ит каждое найденное предложение и в конце возвращает
        (через StopIteration.value / return) остаток буфера.

        Используется как `buffer = yield from self._flush_complete_sentences(buffer)`.
        """
        pos = 0
        while True:
            match = _SENTENCE_END_RE.search(buffer, pos)
            if not match:
                break
            end = match.end()
            sentence = buffer[pos:end].strip()
            if sentence:
                yield sentence
            pos = end

        remainder = buffer[pos:]
        return remainder

    def close(self):
        """Явное освобождение ресурсов модели (если требуется вызывающему коду)."""
        try:
            del self.llm
        except Exception:
            pass
        logger.info("LLMEngine закрыт.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ---------------------------------------------------------------------- #
# Пример локального запуска / smoke-теста
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Использование: python llm_engine.py <путь_к_модели.gguf> [lang_code]")
        sys.exit(1)

    model_path = sys.argv[1]
    lang_code = sys.argv[2] if len(sys.argv) > 2 else "ru"

    engine = LLMEngine(model_path=model_path)

    test_prompt = "Привет! Расскажи, что ты умеешь." if lang_code == "ru" else "Hi! What can you do?"

    print(f"--- Стрим ответа (lang={lang_code}) ---")
    try:
        for sentence in engine.generate_stream(test_prompt, lang_code):
            print(f"[SENTENCE] {sentence}")
    except LLMGenerationError as err:
        print(f"Ошибка генерации: {err}")
