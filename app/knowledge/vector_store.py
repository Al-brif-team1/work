"""Vector store abstraction for knowledge retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from app.schemas import Document, SearchResult


@runtime_checkable
class VectorStore(Protocol):
    """Provider-independent interface for vector similarity search."""

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Store documents together with their embeddings."""
        ...

    def delete_documents(self, document_ids: Sequence[str]) -> None:
        """Delete documents by their identifiers."""
        ...

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return the top-k most similar documents for a query embedding."""
        ...
