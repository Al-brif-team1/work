"""Модуль базы знаний. Он готовит, индексирует и ищет справочные материалы, чтобы ИИ-этапы опирались не только на бриф, но и на контекст проекта."""

from __future__ import annotations

from collections.abc import Mapping

from app.knowledge.embeddings import EmbeddingClient
from app.knowledge.vector_store import VectorStore
from app.schemas import SearchResult


class Retriever:
    """Класс «Retriever» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        default_top_k: int = 5,
    ) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        if default_top_k <= 0:
            raise ValueError("default_top_k must be greater than zero")

        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._default_top_k = default_top_k

    @property
    def default_top_k(self) -> int:
        """Выполняет шаг «default top k». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self._default_top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> list[SearchResult]:
        """Выполняет шаг «retrieve». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        effective_top_k = top_k if top_k is not None else self._default_top_k
        if effective_top_k <= 0:
            return []

        query_embedding = self._embedding_client.embed_query(query)
        filters = dict(metadata_filters) if metadata_filters is not None else None
        return self._vector_store.search(
            query_embedding=query_embedding,
            top_k=effective_top_k,
            metadata_filters=filters,
        )
