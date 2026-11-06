"""Embedding models.

The store depends on the :class:`Embedder` protocol rather than on
sentence-transformers directly, so tests and CI can run the whole retrieval
path against a deterministic stub without downloading a model.
"""

from __future__ import annotations

import zlib
from typing import Protocol

import numpy as np

from specrag.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL


class Embedder(Protocol):
    """Anything that turns text into unit-norm vectors."""

    @property
    def dimension(self) -> int: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local Hugging Face embedding model, CPU-friendly.

    Loaded lazily: importing this module must not pull torch into memory,
    which matters when the process only needs to read an existing index.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, batch_size: int = EMBEDDING_BATCH_SIZE):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    @property
    def dimension(self) -> int:
        # Renamed in sentence-transformers 5; keep working on both.
        getter = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        return int(getter())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder:
    """Deterministic stand-in used by tests and CI.

    Not a serious retrieval model: it hashes character trigrams into a fixed
    number of buckets. It exists so the index, retriever and evaluation code
    can be exercised without a 90MB download, and so a CI failure points at
    our code rather than at a model version.

    Uses crc32 rather than the builtin hash(), which is salted per process
    unless PYTHONHASHSEED is pinned and would make vectors differ between an
    index built in one run and a query issued in the next.
    """

    def __init__(self, dimension: int = 64):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            lowered = text.lower()
            for i in range(max(len(lowered) - 2, 0)):
                trigram = lowered[i : i + 3].encode("utf-8")
                out[row, zlib.crc32(trigram) % self._dimension] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms
