"""
Tests for app/rag_engine.py
============================
Tests the RAG pipeline components with mocked dependencies.

WHY mock everything?
- Embedding model download takes ~2min (too slow for CI)
- LLM needs an API key (can't put secrets in CI easily)
- ChromaDB writes to disk (need cleanup)
- Tests should be FAST and ISOLATED
"""

import pytest
from unittest.mock import MagicMock, patch


class MockConfig:
    """Fake config for testing — no real API key needed."""
    GOOGLE_API_KEY = "fake-test-key-12345"
    CHUNK_SIZE = 100
    CHUNK_OVERLAP = 20
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "gemini-2.0-flash"
    CHROMA_PERSIST_DIR = "./test_chroma_db"


# We patch all 3 external dependencies so nothing real is created
@pytest.fixture
def engine():
    """Create a RAGEngine with all external dependencies mocked."""
    with patch("app.rag_engine.HuggingFaceEmbeddings") as mock_emb, \
         patch("app.rag_engine.Chroma") as mock_chroma, \
         patch("app.rag_engine.ChatGoogleGenerativeAI") as mock_llm:
        
        from app.rag_engine import RAGEngine
        eng = RAGEngine(config=MockConfig)
        yield eng


class TestEngineInit:
    def test_engine_creates_successfully(self, engine):
        """RAGEngine should initialize without errors."""
        assert engine is not None

    def test_engine_has_text_splitter(self, engine):
        """Engine should have a configured text splitter."""
        assert engine.text_splitter is not None
        assert engine.text_splitter._chunk_size == 100
        assert engine.text_splitter._chunk_overlap == 20


class TestIngestDocument:
    def test_ingest_returns_correct_filename(self, engine):
        result = engine.ingest_document("Hello world test content.", "test.txt")
        assert result["filename"] == "test.txt"

    def test_ingest_returns_character_count(self, engine):
        text = "Hello world test content."
        result = engine.ingest_document(text, "test.txt")
        assert result["total_characters"] == len(text)

    def test_ingest_creates_at_least_one_chunk(self, engine):
        result = engine.ingest_document("Short text.", "test.txt")
        assert result["chunks_created"] >= 1

    def test_ingest_splits_long_text(self, engine):
        """Text longer than chunk_size should create multiple chunks."""
        long_text = "A" * 250  # 250 chars, chunk_size is 100
        result = engine.ingest_document(long_text, "big.txt")
        assert result["chunks_created"] > 1

    def test_ingest_calls_vectorstore_add(self, engine):
        """Verify chunks are actually sent to the vector store."""
        engine.ingest_document("Some test content here.", "test.txt")
        engine.vectorstore.add_texts.assert_called_once()


class TestListDocuments:
    def test_list_with_documents(self, engine):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["1", "2", "3"],
            "metadatas": [
                {"source": "doc1.txt"},
                {"source": "doc1.txt"},
                {"source": "doc2.txt"},
            ],
        }
        engine.vectorstore._collection = mock_collection

        result = engine.list_documents()
        assert "doc1.txt" in result["documents"]
        assert "doc2.txt" in result["documents"]
        assert result["total_chunks"] == 3

    def test_list_empty(self, engine):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        engine.vectorstore._collection = mock_collection

        result = engine.list_documents()
        assert result["documents"] == []
        assert result["total_chunks"] == 0


class TestClearDocuments:
    def test_clear_with_documents(self, engine):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.get.return_value = {"ids": ["1", "2", "3", "4", "5"]}
        engine.vectorstore._collection = mock_collection

        result = engine.clear_documents()
        assert result["cleared"] is True
        assert result["chunks_removed"] == 5
        mock_collection.delete.assert_called_once()

    def test_clear_empty_db(self, engine):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        engine.vectorstore._collection = mock_collection

        result = engine.clear_documents()
        assert result["cleared"] is True
        assert result["chunks_removed"] == 0


class TestAskQuestion:
    def test_ask_no_documents_found(self, engine):
        """When no documents are found, return a helpful message."""
        engine.vectorstore.as_retriever.return_value.invoke.return_value = []

        result = engine.ask_question("What is Python?")
        assert "No documents found" in result["answer"]
        assert result["sources"] == []
        assert result["chunks_used"] == 0
