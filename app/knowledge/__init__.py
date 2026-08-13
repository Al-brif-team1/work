"""Модуль базы знаний. Он готовит, индексирует и ищет справочные материалы, чтобы ИИ-этапы опирались не только на бриф, но и на контекст проекта."""

from app.knowledge.chunker import Chunk, TextChunker
from app.knowledge.chroma_store import ChromaVectorStore, VectorStoreError
from app.knowledge.embeddings import EmbeddingClient, HashingEmbeddingClient
from app.knowledge.indexer import IndexedChunk, KnowledgeIndexer
from app.knowledge.loader import DocumentLoader, DocumentLoaderError
from app.knowledge.retriever import Retriever
from app.knowledge.vector_store import VectorStore

__all__ = [
    "Chunk",
    "ChromaVectorStore",
    "DocumentLoader",
    "DocumentLoaderError",
    "EmbeddingClient",
    "HashingEmbeddingClient",
    "IndexedChunk",
    "KnowledgeIndexer",
    "Retriever",
    "TextChunker",
    "VectorStore",
    "VectorStoreError",
]
