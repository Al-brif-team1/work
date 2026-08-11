"""ChromaDB-backed vector store implementation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.knowledge.vector_store import VectorStore
from app.schemas import Document, DocumentMetadata, SearchResult

MetadataDict = dict[str, Any]


class VectorStoreError(RuntimeError):
    """Raised when the vector store cannot be created or used."""


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed implementation of the vector store protocol."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str | Path | None = None,
    ) -> None:
        """Create a Chroma-backed store for the given collection."""
        chromadb = self._import_chromadb()

        self._collection_name = collection_name
        if persist_directory is None:
            client = chromadb.EphemeralClient()
        else:
            client = chromadb.PersistentClient(path=str(persist_directory))

        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Store documents with matching embeddings in Chroma."""
        self._validate_alignment(documents, embeddings)

        self._collection.upsert(
            ids=[document.id for document in documents],
            documents=[document.text for document in documents],
            metadatas=[
                self._metadata_to_dict(document.metadata)
                for document in documents
            ],
            embeddings=[list(embedding) for embedding in embeddings],
        )

    def delete_documents(self, document_ids: Sequence[str]) -> None:
        """Delete documents by identifier."""
        if not document_ids:
            return None

        self._collection.delete(ids=list(document_ids))

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return the top-k documents most similar to the query embedding."""
        if top_k <= 0:
            return []

        where = self._normalize_filters(metadata_filters)
        response = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        results: list[SearchResult] = []
        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        for index, document_id in enumerate(ids):
            document = Document(
                id=document_id,
                text=documents[index] if index < len(documents) else "",
                metadata=self._metadata_from_dict(
                    metadatas[index] if index < len(metadatas) else {}
                ),
            )
            distance = (
                float(distances[index])
                if index < len(distances) and distances[index] is not None
                else None
            )
            score = 1.0 - distance if distance is not None else 0.0
            results.append(
                SearchResult(
                    document=document,
                    score=score,
                    rank=index + 1,
                    distance=distance,
                    metadata={
                        "collection": self._collection_name,
                    },
                )
            )

        return results

    @staticmethod
    def _validate_alignment(
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Ensure documents and embeddings line up one-to-one."""
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")

    @staticmethod
    def _metadata_to_dict(metadata: DocumentMetadata) -> MetadataDict:
        """Convert metadata to a Chroma-friendly dictionary."""
        raw = metadata.model_dump(exclude_none=True)
        return {
            key: ChromaVectorStore._normalize_value(value)
            for key, value in raw.items()
        }

    @staticmethod
    def _metadata_from_dict(metadata: MetadataDict) -> DocumentMetadata:
        """Convert raw metadata back to the document metadata model."""
        return DocumentMetadata.model_validate(metadata)

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """Normalize values for storage in Chroma metadata."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def _normalize_filters(
        metadata_filters: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Normalize metadata filters for Chroma exact-match queries."""
        if metadata_filters is None:
            return None

        clauses = [
            {key: {"$eq": ChromaVectorStore._normalize_value(value)}}
            for key, value in metadata_filters.items()
        ]

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]

        return {"$and": clauses}

    @staticmethod
    def _import_chromadb() -> Any:
        """Import chromadb lazily and raise a readable error if unavailable."""
        try:
            import chromadb
        except ImportError as exc:
            raise VectorStoreError(
                "chromadb is required for ChromaVectorStore. "
                "Add chromadb to requirements and install dependencies."
            ) from exc

        return chromadb
