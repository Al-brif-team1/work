"""Модуль базы знаний. Он готовит, индексирует и ищет справочные материалы, чтобы ИИ-этапы опирались не только на бриф, но и на контекст проекта."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from app.schemas import Document, SearchResult


@runtime_checkable
class VectorStore(Protocol):
    """Класс «VectorStore» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Выполняет шаг «add documents». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        ...

    def delete_documents(self, document_ids: Sequence[str]) -> None:
        """Выполняет шаг «delete documents». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        ...

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Выполняет шаг «search». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        ...
