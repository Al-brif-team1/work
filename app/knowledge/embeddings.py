"""Embedding client abstraction for RAG pipelines."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    """Provider-independent interface for text embeddings."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return embeddings for a batch of documents."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Return an embedding for a single query string."""
        ...


class HashingEmbeddingClient:
    """Deterministic local embedding client based on hashed word counts."""

    def __init__(self, dimension: int = 256) -> None:
        """Create a hashing embedder with the given vector dimension."""
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")

        self._dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return embeddings for multiple texts."""
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a normalized hashed embedding for a single text."""
        vector = [0.0] * self._dimension
        tokens = re.findall(r"\w+", text.lower())

        for token in tokens:
            index = self._token_index(token)
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector

        return [value / norm for value in vector]

    def _token_index(self, token: str) -> int:
        """Map a token to a stable vector index."""
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self._dimension
