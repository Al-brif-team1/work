"""Модуль базы знаний. Он готовит, индексирует и ищет справочные материалы, чтобы ИИ-этапы опирались не только на бриф, но и на контекст проекта."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    """Класс «EmbeddingClient» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Выполняет шаг «embed documents». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Выполняет шаг «embed query». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        ...


class HashingEmbeddingClient:
    """Класс «HashingEmbeddingClient» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(self, dimension: int = 256) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")

        self._dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Выполняет шаг «embed documents». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Выполняет шаг «embed query». Документация описывает назначение метода, а сама логика остается в коде ниже."""
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
        """Выполняет шаг «token index». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self._dimension
