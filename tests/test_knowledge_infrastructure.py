import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from app.knowledge import (
    ChromaVectorStore,
    DocumentLoader,
    DocumentLoaderError,
    EmbeddingClient,
    HashingEmbeddingClient,
    KnowledgeIndexer,
    Retriever,
    TextChunker,
    VectorStore,
)
from app.schemas import Document, DocumentMetadata, SearchResult


class FakeEmbeddingClient:
    """Класс «FakeEmbeddingClient» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        total = float(len(text))
        vowels = float(sum(1 for char in text.lower() if char in "aeiou"))
        return [total, vowels]


class FakeVectorStore:
    """Класс «FakeVectorStore» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(self) -> None:
        self.added_documents: list[Document] = []
        self.added_embeddings: list[list[float]] = []
        self.deleted_document_ids: list[str] = []
        self.search_calls: list[dict[str, object]] = []
        self.search_results: list[SearchResult] = []

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        self.added_documents.extend(documents)
        self.added_embeddings.extend(embeddings)

    def delete_documents(self, document_ids: list[str]) -> None:
        self.deleted_document_ids.extend(document_ids)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filters: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        self.search_calls.append(
            {
                "query_embedding": query_embedding,
                "top_k": top_k,
                "metadata_filters": metadata_filters,
            }
        )
        return list(self.search_results)


class TestKnowledgeModels(unittest.TestCase):
    """Класс «TestKnowledgeModels» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_document_metadata_accepts_extra_fields(self) -> None:
        metadata = DocumentMetadata(
            source="brief",
            title="Project Brief",
            document_type="brief",
            category="analysis",
            version="v1",
            date="2026-07-29",
            owner="team-a",
        )

        self.assertEqual(metadata.source, "brief")
        self.assertEqual(metadata.model_extra["owner"], "team-a")

    def test_document_and_search_result_models_validate(self) -> None:
        document = Document(
            id="doc-1",
            text="Project brief text",
            metadata=DocumentMetadata(title="Brief"),
        )
        result = SearchResult(document=document, score=0.93, rank=1)

        self.assertEqual(result.document.id, "doc-1")
        self.assertEqual(result.score, 0.93)
        self.assertEqual(result.rank, 1)


class TestKnowledgeProtocols(unittest.TestCase):
    """Класс «TestKnowledgeProtocols» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_embedding_client_protocol_accepts_compatible_object(self) -> None:
        client = FakeEmbeddingClient()

        self.assertIsInstance(client, EmbeddingClient)

    def test_vector_store_protocol_accepts_compatible_object(self) -> None:
        store = FakeVectorStore()

        self.assertIsInstance(store, VectorStore)


class TestDocumentLoader(unittest.TestCase):
    """Класс «TestDocumentLoader» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_loader_defaults_to_knowledge_directory(self) -> None:
        loader = DocumentLoader()

        self.assertEqual(loader.base_directory, Path("knowledge"))

    def test_load_directory_reads_supported_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "guide.md").write_text("# Guide", encoding="utf-8")
            (root / "notes.txt").write_text("Plain text notes", encoding="utf-8")
            (root / "ignore.pdf").write_text("binary", encoding="utf-8")

            loader = DocumentLoader(base_directory=root)
            documents = loader.load_directory()

        self.assertEqual(len(documents), 2)
        self.assertEqual([document.id for document in documents], ["guide.md", "notes.txt"])
        self.assertEqual(documents[0].metadata.title, "guide")
        self.assertEqual(documents[0].metadata.document_type, "md")
        self.assertEqual(documents[0].metadata.category, "root")
        self.assertEqual(documents[0].metadata.source, "guide.md")

    def test_load_file_rejects_unsupported_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "image.png"
            file_path.write_text("not a real image", encoding="utf-8")
            loader = DocumentLoader(base_directory=tmp_dir)

            with self.assertRaisesRegex(
                DocumentLoaderError,
                "Unsupported document type",
            ):
                loader.load_file(file_path)


class TestTextChunker(unittest.TestCase):
    """Класс «TestTextChunker» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_chunk_splits_text_with_overlap(self) -> None:
        chunker = TextChunker(chunk_size=4, overlap=1)

        self.assertEqual(
            chunker.chunk("abcdefghijklmno"),
            ["abcd", "defg", "ghij", "jklm", "mno"],
        )

    def test_chunk_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap must be smaller"):
            TextChunker(chunk_size=10, overlap=10)


class TestKnowledgeIndexer(unittest.TestCase):
    """Класс «TestKnowledgeIndexer» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_index_documents_loads_chunks_and_embeddings(self) -> None:
        loader = Mock(spec=DocumentLoader)
        chunker = TextChunker(chunk_size=4, overlap=1)
        embedder = FakeEmbeddingClient()
        store = FakeVectorStore()
        indexer = KnowledgeIndexer(
            loader=loader,
            chunker=chunker,
            embedding_client=embedder,
            vector_store=store,
        )
        document = Document(
            id="guide.md",
            text="abcdefghijklmno",
            metadata=DocumentMetadata(source="guide.md", title="Guide"),
        )

        indexed_count = indexer.index_documents([document])

        self.assertEqual(indexed_count, 5)
        self.assertEqual([doc.id for doc in store.added_documents], [
            "guide.md::chunk::0000",
            "guide.md::chunk::0001",
            "guide.md::chunk::0002",
            "guide.md::chunk::0003",
            "guide.md::chunk::0004",
        ])
        self.assertEqual(store.added_embeddings, embedder.embed_documents([
            "abcd",
            "defg",
            "ghij",
            "jklm",
            "mno",
        ]))
        self.assertEqual(
            store.added_documents[0].metadata.model_extra["chunk_index"],
            0,
        )
        self.assertEqual(
            store.added_documents[0].metadata.model_extra["chunk_count"],
            5,
        )

    def test_index_directory_uses_loader(self) -> None:
        loader = Mock(spec=DocumentLoader)
        loader.load_directory.return_value = [
            Document(id="doc-1", text="hello", metadata=DocumentMetadata())
        ]
        chunker = TextChunker(chunk_size=10, overlap=1)
        embedder = FakeEmbeddingClient()
        store = FakeVectorStore()
        indexer = KnowledgeIndexer(loader, chunker, embedder, store)

        indexed_count = indexer.index_directory()

        self.assertEqual(indexed_count, 1)
        loader.load_directory.assert_called_once_with(None)


class TestRetriever(unittest.TestCase):
    """Класс «TestRetriever» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_retrieve_uses_embedding_and_metadata_filters(self) -> None:
        embedder = FakeEmbeddingClient()
        store = FakeVectorStore()
        document = Document(
            id="doc-1",
            text="hello world",
            metadata=DocumentMetadata(source="knowledge/doc.md"),
        )
        store.search_results = [
            SearchResult(document=document, score=0.88, rank=1)
        ]
        retriever = Retriever(embedder, store, default_top_k=3)

        results = retriever.retrieve(
            "sample query",
            metadata_filters={"category": "brief", "source": "knowledge/doc.md"},
        )

        self.assertEqual(results, store.search_results)
        self.assertEqual(store.search_calls[0]["top_k"], 3)
        self.assertEqual(
            store.search_calls[0]["metadata_filters"],
            {"category": "brief", "source": "knowledge/doc.md"},
        )
        self.assertEqual(store.search_calls[0]["query_embedding"], [12.0, 4.0])

    def test_retrieve_allows_top_k_override(self) -> None:
        embedder = FakeEmbeddingClient()
        store = FakeVectorStore()
        retriever = Retriever(embedder, store, default_top_k=3)

        retriever.retrieve("query", top_k=2)

        self.assertEqual(store.search_calls[0]["top_k"], 2)


class TestKnowledgePipelineIntegration(unittest.TestCase):
    """Класс «TestKnowledgePipelineIntegration» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_documents_loader_indexer_and_retriever_work_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "project").mkdir()
            (root / "project" / "alpha.md").write_text(
                "Alpha project brief with unique alpha keyword.",
                encoding="utf-8",
            )
            (root / "project" / "beta.txt").write_text(
                "Beta project brief with different scope.",
                encoding="utf-8",
            )

            loader = DocumentLoader(base_directory=root)
            chunker = TextChunker(chunk_size=200, overlap=20)
            embedder = HashingEmbeddingClient(dimension=64)
            store = ChromaVectorStore(
                collection_name="integration_knowledge_pipeline",
            )
            indexer = KnowledgeIndexer(loader, chunker, embedder, store)
            retriever = Retriever(embedder, store, default_top_k=3)

            indexed_count = indexer.index_directory()
            results = retriever.retrieve(
                "alpha keyword",
                metadata_filters={"category": "project", "document_type": "md"},
            )

        self.assertEqual(indexed_count, 2)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].document.metadata.title, "alpha")
        self.assertEqual(results[0].document.metadata.category, "project")
        self.assertIn("alpha keyword", results[0].document.text)
        self.assertLessEqual(len(results), 3)


@unittest.skipUnless(importlib.util.find_spec("chromadb"), "chromadb is not installed")
class TestChromaVectorStore(unittest.TestCase):
    """Класс «TestChromaVectorStore» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_add_search_filter_and_delete_documents(self) -> None:
        store = ChromaVectorStore(collection_name="test_documents")
        documents = [
            Document(
                id="doc-1",
                text="Alpha document",
                metadata=DocumentMetadata(
                    source="source-a",
                    title="Alpha",
                    category="alpha",
                ),
            ),
            Document(
                id="doc-2",
                text="Beta document",
                metadata=DocumentMetadata(
                    source="source-b",
                    title="Beta",
                    category="beta",
                ),
            ),
        ]
        embeddings = [[1.0, 0.0], [0.0, 1.0]]

        store.add_documents(documents, embeddings)

        results = store.search([1.0, 0.0], top_k=2)
        filtered = store.search(
            [1.0, 0.0],
            top_k=2,
            metadata_filters={"category": "alpha"},
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].document.id, "doc-1")
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].document.id, "doc-1")

        store.delete_documents(["doc-1"])
        remaining = store.search([1.0, 0.0], top_k=2)

        self.assertTrue(all(item.document.id != "doc-1" for item in remaining))

    def test_add_documents_requires_matching_lengths(self) -> None:
        store = ChromaVectorStore(collection_name="test_documents_mismatch")

        with self.assertRaisesRegex(
            ValueError,
            "documents and embeddings must have the same length",
        ):
            store.add_documents(
                [Document(id="doc-1", text="Alpha")],
                [],
            )


class TestHashingEmbeddingClient(unittest.TestCase):
    """Класс «TestHashingEmbeddingClient» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_embedder_returns_normalized_vectors(self) -> None:
        embedder = HashingEmbeddingClient(dimension=8)

        query_embedding = embedder.embed_query("alpha alpha beta")
        document_embeddings = embedder.embed_documents(["alpha alpha beta"])

        self.assertEqual(len(query_embedding), 8)
        self.assertEqual(len(document_embeddings), 1)
        self.assertAlmostEqual(sum(value * value for value in query_embedding), 1.0)


if __name__ == "__main__":
    unittest.main()
