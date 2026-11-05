"""Text embeddings. A real local model by default; a deterministic hashing embedder for tests and offline use."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from .config import Settings

HASH_DIM = 512
_TOKEN = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|\d+")


class Embedder(Protocol):
    model_id: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedEmbedder:
    """ONNX model run locally by fastembed: no API key, no network after the first download."""

    def __init__(self, model_name: str):
        self.model_id = f"fastembed:{model_name}"
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding  # slow import; only pay for it when embedding

            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return _normalise(np.array(list(self._load().passage_embed(texts, batch_size=32)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(np.array(list(self._load().query_embed(text)), dtype=np.float32))[0]


class HashEmbedder:
    """Bag of words hashed into a fixed vector. Weak semantically, but fast, offline and deterministic."""

    model_id = f"hash:{HASH_DIM}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.stack([self._vector(t) for t in texts])) if texts else np.zeros((0, HASH_DIM), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(self._vector(text)[None, :])[0]

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        vector = np.zeros(HASH_DIM, dtype=np.float32)
        for token in _TOKEN.findall(text):
            digest = hashlib.blake2b(token.lower().encode(), digest_size=4).digest()
            vector[int.from_bytes(digest, "little") % HASH_DIM] += 1.0
        return vector


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "hash":
        return HashEmbedder()
    if settings.embedder == "fastembed":
        return FastEmbedEmbedder(settings.embed_model)
    raise ValueError(f"Unknown PROJECT_BRAIN_EMBEDDER {settings.embedder!r}; use 'fastembed' or 'hash'.")


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)
