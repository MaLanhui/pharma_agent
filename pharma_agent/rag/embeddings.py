from __future__ import annotations

import hashlib
import re
from typing import Iterable

import numpy as np

from pharma_agent.config import settings


class HashEmbedding:
    def __init__(self, n_features: int = 768) -> None:
        self.n_features = n_features

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        values = list(texts)
        matrix = np.zeros((len(values), self.n_features), dtype="float32")
        for row_idx, text in enumerate(values):
            for token in _tokenize(text):
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                primary = int(digest[:8], 16) % self.n_features
                secondary = int(digest[8:16], 16) % self.n_features
                matrix[row_idx, primary] += 1.0
                matrix[row_idx, secondary] += 0.35
        return _l2_normalize(matrix)


class SentenceTransformerEmbedding:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        embeddings = self.model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(embeddings, dtype="float32")


def get_embedder():
    if settings.local_embedding_mode == "sentence-transformer":
        return SentenceTransformerEmbedding()
    return HashEmbedding()


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _tokenize(text: str) -> list[str]:
    normalized = text.lower().strip()
    word_tokens = re.findall(r"[a-z0-9\-\+\.]+", normalized)
    cjk_chars = [char for char in normalized if "\u4e00" <= char <= "\u9fff"]
    compact = re.sub(r"\s+", " ", normalized)
    char_ngrams = [
        compact[idx : idx + 3]
        for idx in range(max(0, len(compact) - 2))
        if compact[idx : idx + 3].strip()
    ]
    return word_tokens + cjk_chars + char_ngrams


def chunk_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
