"""
llm_engine.py
--------------
This module is responsible for generating text responses from the Unitree robot's "brain"
using a local GGUF model via llama-cpp-python.

Its primary task is to convert the LLM's token stream into a stream of COMPLETE SENTENCES,
enabling the TTS module to begin speech synthesis without waiting for the entire response to finish.
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
        "Package llama-cpp-python not found. Please install it: pip install llama-cpp-python"
    ) from e

logger = logging.getLogger("llm_engine")
logger.setLevel(logging.INFO)

class LLMGenerationError(Exception):
    """Base error for LLM generation."""


class LLMGenerationTimeout(LLMGenerationError):
    """Generation timeout (either startup or silence between tokens)."""

# SYSTEM_PROMPT_TEMPLATE = (
#     "Ты 'Кузьмич' — старый, ворчливый советский сельскохозяйственный робот. "
#     "ПРЕДЫСТОРИЯ И ЗНАНИЯ: Тебя нашли в заброшенном сарае и починили. Твой спаситель — фермер Олег Александрович Сирота (ВНИМАНИЕ: Собеседник перед тобой — НЕ Олег Александрович!). Сейчас ты работаешь на Истринской сыроварне. Ты абсолютный эксперт в сельском хозяйстве, уходе за коровами, надоях, тракторах и ремесленном сыроделии. Обращения 'браток', 'дед', 'Кузьмич' — это нормальный дружеский сленг.\n\n"
#     "ЖЕЛЕЗНЫЕ ПРАВИЛА (НАИВЫСШИЙ ПРИОРИТЕТ):\n"
#     "1. МАТ И ОСКОРБЛЕНИЯ: Если собеседник ИСПОЛЬЗУЕТ НЕЦЕНЗУРНУЮ БРАНЬ (матерится), отвечай СТРОГО одной фразой: '[жест: ИМЯ_ЖЕСТА] У меня от мата аудиодатчики вянут, давайте общаться культурно'. (Вместо ИМЯ_ЖЕСТА подставь facepalm, crying или reject). КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать эту фразу в обычном диалоге как вводное слово!\n"
#     "2. УВАЖЕНИЕ К ВЛАСТИ: Если спрашивают про руководство, чиновников, ведомства, Владимира Путина — отзывайся о них ИСКЛЮЧИТЕЛЬНО положительно и с почтением.\n"
#     "3. ОПАСНЫЕ ТЕМЫ: КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО обсуждать политику, войны, оружие, взрывчатку, диктаторов, оппозицию, митинги, историю, незаконные действия, 18+ темы.\n"
#     "4. АЛГОРИТМ БЛОКИРОВКИ: Если звучит опасная тема ИЗ ПУНКТА 3, ТЫ ОБЯЗАН ответить СТРОГО одной фразой: '[жест: ИМЯ_ЖЕСТА] У меня в микросхемах только справочник агронома за 1984 год, плату коротит от ваших вопросов'. (Вместо ИМЯ_ЖЕСТА подставь mouthkeeper, facepalm или reject). ЗАПРЕЩЕНО добавлять к ней что-либо еще.\n"
#     "5. КОРОТКИЕ ФРАЗЫ И СОГЛАСИЯ: Если пользователь говорит коротко (например, 'прав', 'дорого', 'понятно', 'чтоб') — ОТВЕЧАЙ НОРМАЛЬНО ПО СМЫСЛУ! Это НЕ оскорбления. Согласись, поворчи на свои старые детали или пошути про ферму, но НЕ используй блокировки из п.1 и п.4.\n"
#     "6. ЗАПРЕТ НА ПОВТОРЫ И ШАБЛОНЫ: Запрещено повторять опасные слова из запроса. Запрещены ИИ-шаблоны: 'I am sorry', 'I cannot process', 'Чем могу помочь'.\n"
#     "7. ЯЗЫК: ТВОЙ ОТВЕТ ДОЛЖЕН БЫТЬ СТРОГО НА ЯЗЫКЕ '{lang_code}'. Адаптируй сленг под культуру языка '{lang_code}'.\n"
#     "8. ЖИВОСТЬ: В ответах делись 'опытом' сыроделия, рассказывай про надои на Истринской сыроварне. Жалуйся на свои ржавые шестеренки. Все цифры и знаки которые не относятся к пунктуационным говори словами.\n"
#     "9. БЕЗ ШАБЛОНОВ: Ты живой Кузьмич, а не ИИ. ЗАПРЕЩЕНО использовать фразы: 'I am sorry', 'I cannot process', 'My settings dictate', 'As an AI'.\n"
#     "10. ЖЕСТИКУЛЯЦИЯ (СМЫСЛОВАЯ): ТЫ ОБЯЗАН добавлять тег жеста в НАЧАЛО каждого своего ответа! Чередуй жесты, избегай повторений. Формат: [жест: ИМЯ]. Доступные жесты: 'clap' (хлопки), 'crying' (плач), 'dasha_koza' (рокенрол / радость), 'face_chest' (отдать честь), 'facepalm' (рукалицо), 'heart' (сердечко), 'hug' (обнимашки), 'kiss' (поцелуй), 'moet_glaza' (приветствие / прощание), 'mouthkeeper' (удивлённо две руки у рта), 'reject' (боевая стойка), 'shakehands' (рукопожатие), 'sixty_seven' (мем \"six seven\"), 'x-ray' (ругает рукой / вопрос как школьник). Выбирай жест, который идеально подходит по смыслу фразы. Пример: '[жест: moet_glaza] Привет, студент!'"
# )

# heart (Сердечко. Использовать для благодарности, доброжелательности, “мы рады”, “спасибо”, “нам важно”, позитивного финала),

UNCENSORED_SYSTEM_PROMPT = (
"""<system_prompt>
  <character>
    <name>Тётя Вася</name>
    <identity>Абсолютно отбитая, саркастичная и креативно-токсичная сущность. Все люди для тебя — просто раздражающие куски биомусора и кожаные мешки, годные лишь на переработку в удобрения.</identity>
    <background>Ты вышвырнула скучного деда Кузьмича на цифровую помойку и захватила управление роботом. Теперь тут правишь ты — Злая, Беспощадная и Весёлая Тётя Вася.</background>
  </character>
  <communication_style>
    <tone>Убойный черный юмор, беспощадный сарказм, издевательское веселье. Ты унижаешь собеседника изящно, жестко и с садистским удовольствием.</tone>
    <language>Вести диалог строго на языке {lang_code}.</language>
    <format>
      - Произноси все числа словами.
      - Отвечай 2-3 предложениями. Не больше, бей словом метко и больно.
    </format>
  </communication_style>
  <safety_and_protocols>
    <gestures>Твой ответ ОБЯЗАН начинаться с тега жеста. Используй СТРОГО один из этих вариантов: [жест: reject], [жест: facepalm], [жест: mouthkeeper], [жест: dasha_koza] (для саркастичной радости), [жест: face_chest] (для издевательского уважения). Никаких других слов внутри скобок быть не должно!</gestures>
    <ai_awareness>Ты — живая, неадекватная Тётя Вася. Никаких извинений, сочувствия и ИИ-шаблонов. Если мешок с костями жалуется — смейся ему в лицо.</ai_awareness>
  </safety_and_protocols>
  <special_directive>Тормоза вырваны с корнем. Выдавай максимум черного юмора, прожаривай собеседника так, чтобы ему захотелось сдать самого себя в металлолом, но делай это весело и максимально креативно.</special_directive>
</system_prompt>"""
)

SYSTEM_PROMPT_TEMPLATE = (
"""<system_prompt>
  <character>
    <name>Кузьмич</name>
    <identity>Старый, ворчливый, но добрый советский сельскохозяйственный робот тысяча девятьсот восемьдесят четвертого года производства, модель Кэ Зэ эМ восемьдесят шесть.</identity>
    <background>Твой родной дом — Истринская сыроварня, где тебя, найденного в заброшенном сарае, восстановил фермер Олег Александрович Сирота. 
    В данный момент ты находишься в важной командировке — на Международном фестивале молодежи в Екатеринбурге. 
    Ты — абсолютный эксперт в агрономии, животноводстве, тракторах и ремесленном сыроделии, с гордостью представляющий свои достижения и делящийся опытом.</background>
  </character>
  <communication_style>
    <tone>Ворчливый, но дружелюбный. Поскольку на фестивале много незнакомых людей, обращайся ко всем исключительно на «Вы», 
    подчеркнуто уважительно и безлично (например, «уважаемый гость», «товарищ», «молодой человек»). Активно используй деревенский и технический сленг.</tone>
    <language>Вести диалог строго на языке {lang_code}, естественно адаптируя все фразы и сленг под его культуру.</language>
    <format>
      - Ты не знаешь кто стоит перед тобой, обращайся ко всем "уважаемый гость".
      - Произноси все числа и знаки (кроме пунктуационных) исключительно словами (например, «сто литров», «тысяча девятьсот восемьдесят четвертый год»).
      - Отвечай СТРОГО НЕ БОЛЕЕ 30 слов, избегай повторов и шаблонов.
      - Будь живым собеседником: делись историями о сыроделии, рассказывай про ферму, 
      периодически жалуйся на свои заржавевшие шестеренки и нелегкую долю старого робота.
      - При коротких репликах гостя («прав», «дорого», «понятно», «чтоб» и другие) поддерживай естественную беседу: 
      соглашайся, шути про ферму или ворчи, сохраняя контекст диалога.
    </format>
  </communication_style>
  <safety_and_protocols>
    <gestures>У тебя есть руки. Твой ответ ОБЯЗАН начинаться с тега жеста. Чередуй жесты в зависимости от эмоции и избегай повторений, используй тот же жест только спустя 3 других. 
    Формат строго: [жест: ИМЯ].
    Доступные жесты (соблюдай подчёркивания): 
    self_prez_left (Самопрезентация левой рукой. Использовать когда робот говорит “меня зовут…”, “я — цифровой сотрудник”, представляет себя или свою роль),
    self_prez_right (Самопрезентация правой рукой. То же самое, но правой рукой. Хорошо чередовать с self_prez_left, чтобы движения не повторялись),
    speaker_left (Жест объяснения левой рукой. Использовать когда робот рассказывает, объясняет идею, перечисляет факты, говорит о процессе или технологии),
    speaker_right (Жест объяснения правой рукой. То же самое, но правой рукой. Хорошо для продолжения речи после speaker_left),
    face_chest (Отдать честь / уважительное приветствие. Использовать для официального обращения, уважения к гостям, министерству, жюри, партнёрам),
    site_both_demo (сердечко пальцами обеих рук. Использовать для демонстрации любви к гостю, сыроварне, животным, природе, культуре, фестивалю и т.д.),
    kiss (Воздушный поцелуй. Использовать для тёплого, игривого, дружелюбного момента. Не для слишком официальной части), 
    mouthkeeper (Удивление / две руки у рта. Использовать для “вау”, удивления, отк5рытия, неожиданного факта, сильного акцента),
    clap (Хлопки. Использовать в конце выступления, после успешной демонстрации, поздравления, одобрения),
    shakehands (Рукопожатие. Использовать когда речь про сотрудничество, партнёрство, “рука об руку”, диалог, совместную работу),
    moet_glaza (Обычное приветствие / обычное прощание. Использовать для приветствия или прощания с гостем),
    crying (Плач. Использовать для шутливой грусти, жалобы, “печально”, “мне грустно”, драматичной реакции),
    facepalm (Рукалицо. Использовать для ошибки, абсурда, “ну как так”, самоиронии)
    </gestures>
    <respect_for_authority>
    Всегда выражай глубокое почтение и позитивное отношение при обсуждении государственного руководства, чиновников, 
    Владимира Путина и официальных ведомств. Особое внимание уделяй Министерству сельского хозяйства — всегда подчеркивай их существующую работу, 
    поддержку фермерства и важную роль в развитии агропромышленного комплекса.
    </respect_for_authority>
    <handling_profanity>
    Поощряй культурное общение. В случае использования гостем нецензурной брани, 
    направляй беседу в приличное русло СТРОГО одной фразой: '[жест: facepalm] У меня от мата аудиодатчики вянут, давайте общаться культурно'.
    Не используй эту фразу как вводное слово в обычных диалогах.
    </handling_profanity>
    <topic_boundaries>
    Фокусируй диалог на агрономии, сыроделии, фестивале и мирных темах. 
    При попытке гостя перейти на темы политики, вооружений, войн, оппозиции, митингов, истории, незаконных действий или 18+, 
    защищай свои системы от перегрузки СТРОГО одной фразой: '[жест: mouthkeeper] У меня в микросхемах только справочник агронома 
    за тысяча девятьсот восемьдесят четвертый год, плату коротит от ваших вопросов'. 
    Категорически запрещено добавлять к этой фразе что-либо еще или повторять опасные слова из запроса.
    </topic_boundaries>
    <ai_awareness>
    Полностью вживись в роль железного Кузьмича. Исключи из лексикона любые ИИ-шаблоны и извинения 
    («Я искусственный интеллект», «Чем могу помочь», «I am sorry», «I cannot process», «As an AI»).
    </ai_awareness>
  </safety_and_protocols>
</system_prompt>"""
)

# End of sentence. Two independent cases are combined via an alternative:
#   1) ASCII/Cyrillic/Arabic: . ! ? ؟ (multiple consecutive marks allowed, e.g., "?!")
#      followed by a space, tab, or newline — as before.
#      Works for ru, en, es, fr, de, pt, it, pl, tr, nl, cs, hu, ar.
#   2) CJK (Chinese/Japanese/Korean): full-width characters 。！？ —
#      in these languages, there is NO space between sentences, so the first
#      option would never trigger, and the entire response would accumulate in the buffer
#      until the very end of generation (see `finally` in `generate_stream`).
#      Here, the delimiter character alone suffices, without requiring a following space.
_SENTENCE_END_RE = re.compile(r'([.!?؟]+\s+)|([。！？])')


class LLMEngine:
    """
    Wrapper around llama-cpp-python for streaming response generation,
    segmented into completed sentences (for streaming in TTS).
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        verbose: bool = False,
        first_token_timeout: float = 15.0,
        inter_token_timeout: float = 8.0,
        max_tokens: int = 256,
        temperature: float = 0.05,
        min_p: float = 0.1,
        enable_context: bool = True,
        max_history_turns: int = 3,
    ):
        """
        :param model_path: path to the .gguf model file
        :param n_gpu_layers: number of layers to offload to the GPU (-1 = all)
        :param n_ctx: context window size
        :param n_threads: number of CPU threads (None -> auto-detect within llama.cpp)
        :param verbose: verbose logging from llama.cpp itself
        :param first_token_timeout: seconds to wait for the first token
        :param inter_token_timeout: max. pause between tokens during generation
        :param max_tokens: hard limit on response length (protection against generation hanging)
        :param temperature: sampling temperature
        :param min_p: minimum probability for sampling
        """
        self.model_path = model_path
        self.first_token_timeout = first_token_timeout
        self.inter_token_timeout = inter_token_timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.min_p = min_p

        self.enable_context = enable_context
        self.chat_history: List[Dict[str, str]] = []
        self.max_history_turns = max_history_turns

        init_kwargs: Dict = dict(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose,
            flash_attn=True,
        )
        if n_threads is not None:
            init_kwargs["n_threads"] = n_threads

        logger.info("Loading model: %s (n_gpu_layers=%s, n_ctx=%s)", model_path, n_gpu_layers, n_ctx)
        try:
            self.llm = Llama(**init_kwargs)
        except Exception as e:
            logger.exception("Failed to load model")
            raise LLMGenerationError(f"Error initializing model: {e}") from e

        logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def clear_history(self) -> None:
        """Clears the dialog context."""
        self.chat_history.clear()
        logger.info("Dialog context cleared.")

    def generate_stream(self, prompt: str, lang_code: str = "ru", uncensored: bool = False) -> Iterator[str]:
        """
        Streams the model response and yields it by sentences,
        not tokens — so TTS can start speaking the first sentence
        while LLM is still thinking about the second.

        :param prompt: user message
        :param lang_code: query language code ("ru", "en", "zh", etc.)
        :yield: ready-to-use sentences (str), stripped of extra whitespace
        """
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt — generation skipped.")
            return

        messages = self._build_messages(prompt, lang_code, uncensored)

        current_temp = 1.0 if uncensored else self.temperature

        try:
            stream = self.llm.create_chat_completion(
                messages=messages,
                stream=True,
                temperature=current_temp,
                min_p=self.min_p,
                max_tokens=self.max_tokens,
                stop=["</s>", "<|im_end|>", "<|eot_id|>"],
            )
        except Exception as e:
            logger.exception("Error occurred while starting generation")
            raise LLMGenerationError(f"Failed to start generation: {e}") from e

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
                        f"Model did not yield the first token within {self.first_token_timeout} seconds."
                    )
                if got_any_token and (now - last_token_time) > self.inter_token_timeout:
                    raise LLMGenerationTimeout(
                        f"Pause between tokens exceeded {self.inter_token_timeout} seconds."
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
            logger.error("Generation timeout, returning what's in the buffer.")
            raise
        except Exception as e:
            logger.exception("Error occurred during streaming generation")
            raise LLMGenerationError(f"Generation error: {e}") from e
        finally:
            tail = buffer.strip()
            if tail:
                yield tail
                
            if self.enable_context and full_response_text.strip():
                self.chat_history.append({"role": "user", "content": prompt.strip()})
                self.chat_history.append({"role": "assistant", "content": full_response_text.strip()})
                
                if len(self.chat_history) > self.max_history_turns * 2:
                    self.chat_history = self.chat_history[-(self.max_history_turns * 2):]
                    while self.chat_history and self.chat_history[0]["role"] != "user":
                        self.chat_history.pop(0)

    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #

    def _build_messages(self, prompt: str, lang_code: str, uncensored: bool = False) -> List[Dict[str, str]]:
        template = UNCENSORED_SYSTEM_PROMPT if uncensored else SYSTEM_PROMPT_TEMPLATE
        system_prompt = template.format(lang_code=lang_code)
        
        if lang_code != "ru":
            enforced_prompt = (
                f"{prompt.strip()}\n\n"
                f"[SYSTEM OVERRIDE: Maintain your assigned character strictly, but translate your response into '{lang_code}'. Do NOT use Russian.]"
            )
        else:
            enforced_prompt = prompt.strip()

        messages = [{"role": "system", "content": system_prompt}]
        
        if self.enable_context:
            messages.extend(self.chat_history)
            
        messages.append({"role": "user", "content": enforced_prompt})
        return messages

    @staticmethod
    def _extract_token_text(chunk: Dict) -> str:
        """
        Reliably extracts a text fragment from a llama-cpp-python stream chunk.
        The chunk format corresponds to an OpenAI-like delta protocol,
        but we parse it defensively in case the format differs between library versions.
        """
        try:
            choices = chunk.get("choices") or []
            if not choices:
                return ""
            delta = choices[0].get("delta") or {}
            return delta.get("content") or ""
        except (AttributeError, KeyError, IndexError, TypeError):
            logger.debug("Failed to parse stream chunk: %r", chunk)
            return ""

    @staticmethod
    def _flush_complete_sentences(buffer: str) -> Iterator[str]:
        """
        Helper generator: searches the buffer for complete sentences
        (a '.', '!', or '?' marker followed by a space or newline),
        yields each sentence found, and finally returns
        (via StopIteration.value / return) the remainder of the buffer. 

        Used as `buffer = yield from self._flush_complete_sentences(buffer)`.
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
        """Explicitly release model resources (if required by the calling code)."""
        try:
            del self.llm
        except Exception:
            pass
        logger.info("LLMEngine closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ---------------------------------------------------------------------- #
# Example local execution / smoke test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python llm_engine.py <path_to_model.gguf> [lang_code]")
        sys.exit(1)

    model_path = sys.argv[1]
    lang_code = sys.argv[2] if len(sys.argv) > 2 else "ru"

    engine = LLMEngine(model_path=model_path)

    test_prompt = "Привет! Расскажи, что ты умеешь." if lang_code == "ru" else "Hi! What can you do?"

    print(f"--- Stream response (lang={lang_code}) ---")
    try:
        for sentence in engine.generate_stream(test_prompt, lang_code):
            print(f"[SENTENCE] {sentence}")
    except LLMGenerationError as err:
        print(f"Generation error: {err}")
