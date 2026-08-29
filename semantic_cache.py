"""
semantic_cache.py

Модуль семантического кэша для мгновенных ответов на повторяющиеся вопросы
(Zero-latency responses).

Использует:
- sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) для эмбеддингов
- faiss (CPU, IndexFlatIP) для поиска по косинусному сходству

ВАЖНО: для IndexFlatIP косинусное сходство корректно считается только если
все векторы (и в индексе, и запросные) L2-нормализованы. Поэтому нормализация
выполняется на каждом шаге: при добавлении и при поиске.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SemanticCache:
    """
    Семантический кэш "вопрос -> готовый ответ (текст + аудио)".

    Хранилище состоит из двух частей:
      1. FAISS-индекс (IndexFlatIP) с L2-нормализованными эмбеддингами.
      2. Маппинг vector_id -> {'query_text', 'response_text', 'audio_path'}
         (pickle или json на выбор).

    Пример:
        cache = SemanticCache(
            index_path="cache/index.faiss",
            mapping_path="cache/mapping.pkl",
        )
        cache.add("Как дела?", "У меня всё хорошо!", "audio/ok.wav")

        hit = cache.search("как у тебя дела")
        if hit:
            print(hit["response_text"], hit["audio_path"])
    """

    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    DEFAULT_THRESHOLD = 0.85
    DEFAULT_DUPLICATE_THRESHOLD = 0.995

    def __init__(
        self,
        index_path: Union[str, Path] = "semantic_cache_index.faiss",
        mapping_path: Union[str, Path] = "semantic_cache_mapping.pkl",
        model_name: str = MODEL_NAME,
        similarity_threshold: float = DEFAULT_THRESHOLD,
        duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
        device: Optional[str] = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.mapping_path = Path(mapping_path)
        self.similarity_threshold = similarity_threshold
        self.duplicate_threshold = duplicate_threshold

        self._lock = threading.Lock()

        model_ref = self._resolve_model_ref(model_name)
        
        if not hasattr(SemanticCache, "_shared_model"):
            logger.info("Загрузка модели эмбеддингов: %s", model_ref)
            SemanticCache._shared_model = SentenceTransformer(model_ref, device=device)
        else:
            logger.info("Использую уже загруженную модель эмбеддингов из кэша")
            
        self.model = SemanticCache._shared_model
        
        if hasattr(self.model, "get_embedding_dimension"):
            self.dim = self.model.get_embedding_dimension()
        else:
            self.dim = self.model.get_sentence_embedding_dimension()

        self.index: faiss.Index = self._load_or_create_index()
        self.mapping: Dict[int, Dict] = self._load_or_create_mapping()

        self._next_id = (max(self.mapping.keys()) + 1) if self.mapping else 0

    @staticmethod
    def _resolve_model_ref(model_name: str) -> str:
        env_model = os.environ.get("VOICE_ENGINE_EMBEDDER_MODEL", "").strip()
        if env_model:
            return env_model
        for candidate in (
            Path(__file__).resolve().parent / "models" / "sentence_transformer",
            Path("/home/unitree/AgroBot-G1-Unified/models/sentence_transformer"),
            Path("/home/unitree/agrobot/AgroHub/models/sentence_transformer"),
        ):
            if (candidate / "modules.json").exists():
                return str(candidate)
        return model_name

    # ------------------------------------------------------------------ #
    # Загрузка / создание индекса и маппинга
    # ------------------------------------------------------------------ #

    def _load_or_create_index(self) -> faiss.Index:
        if self.index_path.exists():
            logger.info("Загружаю FAISS индекс из %s", self.index_path)
            index = faiss.read_index(str(self.index_path))
            if index.d != self.dim:
                raise ValueError(
                    f"Размерность индекса ({index.d}) не совпадает с "
                    f"размерностью модели ({self.dim}). "
                    f"Проверьте, что индекс создан той же моделью."
                )
            return index

        logger.info("Индекс не найден, создаю новый IndexFlatIP(dim=%d)", self.dim)
        flat = faiss.IndexFlatIP(self.dim)
        # Оборачиваем в IndexIDMap2, чтобы можно было явно задавать id
        # и удалять записи по id (обычный IndexFlatIP этого не умеет).
        return faiss.IndexIDMap2(flat)

    def _load_or_create_mapping(self) -> Dict[int, Dict]:
        if self.mapping_path.exists():
            logger.info("Загружаю маппинг из %s", self.mapping_path)
            suffix = self.mapping_path.suffix.lower()
            if suffix == ".json":
                with open(self.mapping_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # JSON хранит ключи как строки -> конвертируем обратно в int
                return {int(k): v for k, v in raw.items()}
            else:
                with open(self.mapping_path, "rb") as f:
                    return pickle.load(f)

        logger.info("Маппинг не найден, создаю пустой")
        return {}

    def _save(self) -> None:
        """Сохраняет индекс и маппинг на диск."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.mapping_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(self.index_path))

        suffix = self.mapping_path.suffix.lower()
        if suffix == ".json":
            with open(self.mapping_path, "w", encoding="utf-8") as f:
                json.dump(
                    {str(k): v for k, v in self.mapping.items()},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        else:
            with open(self.mapping_path, "wb") as f:
                pickle.dump(self.mapping, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.debug("Кэш сохранён: %s, %s", self.index_path, self.mapping_path)

    # ------------------------------------------------------------------ #
    # Вспомогательное: эмбеддинг + L2-нормализация
    # ------------------------------------------------------------------ #

    def _embed(self, text: str) -> np.ndarray:
        """
        Считает эмбеддинг и L2-нормализует его.

        L2-нормализация ОБЯЗАТЕЛЬНА: IndexFlatIP считает просто скалярное
        произведение (inner product). Оно эквивалентно косинусному сходству
        только если ||a|| = ||b|| = 1. Без нормализации значения будут
        зависеть от длины вектора и порог 0.92 потеряет смысл.
        """
        vec = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=False,  # нормализуем сами, явно
        ).astype("float32")

        vec = vec.reshape(1, -1)
        faiss.normalize_L2(vec)  # in-place L2-нормализация (норма = 1)
        return vec

    # ------------------------------------------------------------------ #
    # Публичный API
    # ------------------------------------------------------------------ #

    def warmup(self) -> None:
        """Прогревает sentence-transformer, чтобы первый search не тормозил."""
        start = time.perf_counter()
        self._embed("привет")
        logger.info("Warmup SemanticCache завершён за %.3f сек.", time.perf_counter() - start)

    def search(self, query_text: str, language: Optional[str] = None) -> Optional[Dict]:
        """
        Ищет семантически близкий вопрос в кэше.

        Возвращает:
            {'response_text': str, 'audio_path': str, 'similarity': float,
             'matched_query': str, 'language': str}
            если найдено совпадение с cosine similarity > threshold,
            иначе None.
        """
        if self.index.ntotal == 0:
            return None

        if not query_text or not query_text.strip():
            return None

        query_vec = self._embed(query_text)

        search_k = min(8, self.index.ntotal)
        with self._lock:
            similarities, ids = self.index.search(query_vec, k=search_k)

        requested_language = (language or "").strip().lower() or None
        for raw_sim, raw_id in zip(similarities[0], ids[0]):
            best_sim = float(raw_sim)
            best_id = int(raw_id)

            # faiss возвращает -1, если совпадений не найдено вообще
            if best_id == -1 or best_sim <= self.similarity_threshold:
                continue

            entry = self.mapping.get(best_id)
            if entry is None:
                # Индекс и маппинг рассинхронизировались — защищаемся от KeyError
                logger.warning(
                    "id %d есть в FAISS, но отсутствует в mapping", best_id
                )
                continue

            entry_language = str(entry.get("language") or "").strip().lower()
            if requested_language:
                if entry_language:
                    if entry_language != requested_language:
                        continue
                elif requested_language not in {"ru", "en"}:
                    # Старые записи без языка могли быть русскими/английскими.
                    # Не отдаём их на испанский, японский и другие языки.
                    continue

            logger.debug(
                "search: query=%r best_id=%d sim=%.4f lang=%s",
                query_text,
                best_id,
                best_sim,
                entry_language or "legacy",
            )

            return {
                "response_text": entry["response_text"],
                "audio_path": entry["audio_path"],
                "similarity": best_sim,
                "matched_query": entry.get("query_text"),
                "language": entry_language or None,
            }

        return None

    def add(
        self,
        query_text: str,
        response_text: str,
        audio_path: str,
        language: Optional[str] = None,
        force: bool = False,
    ) -> Optional[int]:
        """
        Добавляет новую пару "вопрос -> ответ" в кэш.

        Перед добавлением проверяет, нет ли уже почти идентичного вектора
        в индексе (по умолчанию порог дубля = 0.995 — строже, чем порог
        поиска в search(), т.к. это разные семантические пороги:
        search() ищет "достаточно похожий вопрос", а дубль-проверка —
        "практически тот же самый вопрос").

        Args:
            force: если True, добавляет запись, даже если найден дубль.

        Returns:
            vector_id новой записи, либо id существующего дубля (если
            запись не была добавлена), либо None при ошибке.
        """
        if not query_text or not query_text.strip():
            raise ValueError("query_text не может быть пустым")
        if not response_text or not response_text.strip():
            raise ValueError("response_text не может быть пустым")

        vec = self._embed(query_text)

        normalized_language = (language or "").strip().lower() or None

        with self._lock:
            if not force and self.index.ntotal > 0:
                search_k = min(8, self.index.ntotal)
                similarities, ids = self.index.search(vec, k=search_k)
                for raw_sim, raw_id in zip(similarities[0], ids[0]):
                    sim = float(raw_sim)
                    existing_id = int(raw_id)
                    if existing_id == -1 or sim < self.duplicate_threshold:
                        continue
                    existing_language = str(self.mapping.get(existing_id, {}).get("language") or "").strip().lower()
                    if normalized_language and existing_language and existing_language != normalized_language:
                        continue
                    logger.info(
                        "Дубль обнаружен (id=%d, sim=%.4f >= %.4f), "
                        "добавление пропущено. Существующий вопрос: %r",
                        existing_id,
                        sim,
                        self.duplicate_threshold,
                        self.mapping.get(existing_id, {}).get("query_text"),
                    )
                    return existing_id

            new_id = self._next_id
            self._next_id += 1

            ids_array = np.array([new_id], dtype="int64")
            self.index.add_with_ids(vec, ids_array)

            self.mapping[new_id] = {
                "query_text": query_text,
                "response_text": response_text,
                "audio_path": audio_path,
                "language": normalized_language,
            }

            self._save()

        logger.info("Добавлена новая запись id=%d: %r", new_id, query_text)
        return new_id

    def put(
        self,
        query_text: str,
        response_text: str,
        audio_path: str,
        language: Optional[str] = None,
        force: bool = False,
    ) -> Optional[int]:
        """
        Совместимый алиас для add().
        """
        return self.add(query_text, response_text, audio_path, language=language, force=force)

    def remove(self, vector_id: int) -> bool:
        """Удаляет запись из кэша по id"""
        with self._lock:
            if vector_id not in self.mapping:
                return False

            selector = faiss.IDSelectorArray(np.array([vector_id], dtype="int64"))
            self.index.remove_ids(selector)
            del self.mapping[vector_id]
            self._save()

        logger.info("Запись id=%d удалена", vector_id)
        return True

    def __len__(self) -> int:
        return self.index.ntotal

    def stats(self) -> Dict:
        return {
            "total_entries": self.index.ntotal,
            "dim": self.dim,
            "index_path": str(self.index_path),
            "mapping_path": str(self.mapping_path),
            "similarity_threshold": self.similarity_threshold,
            "duplicate_threshold": self.duplicate_threshold,
        }


# ---------------------------------------------------------------------- #
# Пример использования
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    cache = SemanticCache(
        index_path="cache/semantic_index.faiss",
        mapping_path="cache/semantic_mapping.pkl",
    )

    cache.add(
        query_text="Как оформить возврат товара?",
        response_text="Чтобы оформить возврат, перейдите в раздел «Мои заказы»...",
        audio_path="audio/return_policy.wav",
    )

    # Дубль — не должен добавиться повторно
    cache.add(
        query_text="Как оформить возврат товара?",
        response_text="Другой ответ, но вопрос тот же",
        audio_path="audio/return_policy_v2.wav",
    )

    # Похожий, но не идентичный вопрос — должен найтись через search()
    result = cache.search("как вернуть товар обратно")
    if result:
        print(f"[HIT] sim={result['similarity']:.3f} -> {result['response_text']}")
        print(f"audio: {result['audio_path']}")
    else:
        print("[MISS] релевантного ответа не найдено")

    print(cache.stats())
