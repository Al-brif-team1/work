"""Knowledge indexing orchestration."""

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
    """A chunk produced during knowledge indexing."""

    document: Document
    source_document_id: str
    chunk_index: int


class KnowledgeIndexer:
    """Load, chunk, embed, and store knowledge documents."""

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: TextChunker,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
    ) -> None:
        """Create an indexer from the required infrastructure components."""
        self._loader = loader
        self._chunker = chunker
        self._embedding_client = embedding_client
        self._vector_store = vector_store

    def index_directory(self, directory: str | Path | None = None) -> int:
        """Load a directory and index all supported documents."""
        documents = self._loader.load_directory(directory)
        return self.index_documents(documents)

    def index_documents(self, documents: Sequence[Document]) -> int:
        """Chunk and store a set of already loaded documents."""
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
        """Split a document into indexed chunks."""
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
        """Attach chunk metadata to a document metadata object."""
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
        """Build a stable chunk identifier."""
        return f"{document_id}::chunk::{chunk_index:04d}"
