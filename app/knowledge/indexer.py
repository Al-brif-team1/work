"""Модуль базы знаний. Он готовит, индексирует и ищет справочные материалы, чтобы ИИ-этапы опирались не только на бриф, но и на контекст проекта."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.knowledge.chunker import TextChunker
from app.knowledge.embeddings import EmbeddingClient
from app.knowledge.loader import DocumentLoader
from app.knowledge.vector_store import VectorStore
from app.schemas import Document, DocumentMetadata


@dataclass(frozen=True)
class IndexedChunk:
    """Класс «IndexedChunk» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    document: Document
    source_document_id: str
    chunk_index: int


class KnowledgeIndexer:
    """Класс «KnowledgeIndexer» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: TextChunker,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
    ) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        self._loader = loader
        self._chunker = chunker
        self._embedding_client = embedding_client
        self._vector_store = vector_store

    def index_directory(self, directory: str | Path | None = None) -> int:
        """Выполняет шаг «index directory». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        documents = self._loader.load_directory(directory)
        return self.index_documents(documents)

    def index_documents(self, documents: Sequence[Document]) -> int:
        """Выполняет шаг «index documents». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        indexed_chunks: list[Document] = []
        for document in documents:
            chunks = self._chunk_document(document)
            indexed_chunks.extend(chunk.document for chunk in chunks)

        if not indexed_chunks:
            return 0

        embeddings = self._embedding_client.embed_documents(
            [chunk.text for chunk in indexed_chunks]
        )
        self._vector_store.add_documents(indexed_chunks, embeddings)
        return len(indexed_chunks)

    def _chunk_document(self, document: Document) -> list[IndexedChunk]:
        """Выполняет шаг «chunk document». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        chunk_texts = self._chunker.chunk(document.text)
        chunks: list[IndexedChunk] = []

        for index, chunk_text in enumerate(chunk_texts):
            metadata = self._chunk_metadata(document.metadata, index, len(chunk_texts))
            chunk_document = Document(
                id=self._chunk_id(document.id, index),
                text=chunk_text,
                metadata=metadata,
            )
            chunks.append(
                IndexedChunk(
                    document=chunk_document,
                    source_document_id=document.id,
                    chunk_index=index,
                )
            )

        return chunks

    def _chunk_metadata(
        self,
        metadata: DocumentMetadata,
        chunk_index: int,
        chunk_count: int,
    ) -> DocumentMetadata:
        """Выполняет шаг «chunk metadata». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return DocumentMetadata.model_validate(
            {
                **metadata.model_dump(exclude_none=True),
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "chunk_size": self._chunker.chunk_size,
                "chunk_overlap": self._chunker.overlap,
            }
        )

    @staticmethod
    def _chunk_id(document_id: str, chunk_index: int) -> str:
        """Выполняет шаг «chunk id». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return f"{document_id}::chunk::{chunk_index:04d}"
