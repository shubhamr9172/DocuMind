"""
Tests for app/main.py (Flask API)
==================================
These tests verify all API endpoints work correctly.

KEY CI/CD CONCEPT:
    We MOCK the RAG engine so tests don't need:
    - A real API key
    - The embedding model downloaded
    - A ChromaDB instance

    This is essential for CI/CD — tests must run anywhere,
    fast, and without external dependencies!
"""

import pytest
from unittest.mock import MagicMock
from app.main import create_app


# ─── FIXTURES ───────────────────────────────────────────
# Fixtures are reusable test setup. Every test that needs
# a 'client' parameter will automatically get one.


@pytest.fixture
def client():
    """Create a test client WITH a mocked RAG engine."""
    app = create_app(testing=True)
    app.config["TESTING"] = True

    # Create a mock RAG engine (no real AI, no API key needed)
    app.rag_engine = MagicMock()
    
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def client_no_rag():
    """Create a test client WITHOUT a RAG engine (simulates missing API key)."""
    app = create_app(testing=True)
    app.config["TESTING"] = True
    # app.rag_engine is already None from create_app(testing=True)

    with app.test_client() as test_client:
        yield test_client


# ─── HEALTH CHECK TESTS ─────────────────────────────────


class TestHealthCheck:
    def test_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_returns_app_name(self, client):
        data = response = client.get("/").get_json()
        assert data["app"] == "DocuMind"

    def test_returns_status_healthy(self, client):
        data = client.get("/").get_json()
        assert data["status"] == "healthy"

    def test_returns_version(self, client):
        data = client.get("/").get_json()
        assert "version" in data


# ─── INGEST DOCUMENT TESTS ──────────────────────────────


class TestIngestDocument:
    def test_ingest_success(self, client):
        """Successfully ingest a document → 201 Created."""
        client.application.rag_engine.ingest_document.return_value = {
            "filename": "test.txt",
            "chunks_created": 3,
            "total_characters": 1500,
        }

        response = client.post(
            "/documents/ingest",
            json={"text": "Some document content here.", "filename": "test.txt"},
        )
        assert response.status_code == 201
        data = response.get_json()
        assert data["filename"] == "test.txt"
        assert data["chunks_created"] == 3

    def test_ingest_missing_text_field(self, client):
        """Missing 'text' field → 400 Bad Request."""
        response = client.post("/documents/ingest", json={"filename": "test.txt"})
        assert response.status_code == 400

    def test_ingest_empty_text(self, client):
        """Empty text → 400 Bad Request."""
        response = client.post("/documents/ingest", json={"text": "   "})
        assert response.status_code == 400

    def test_ingest_no_body(self, client):
        """No JSON body → 400 Bad Request."""
        response = client.post(
            "/documents/ingest", content_type="application/json"
        )
        assert response.status_code == 400

    def test_ingest_without_rag(self, client_no_rag):
        """RAG not initialized → 503 Service Unavailable."""
        response = client_no_rag.post(
            "/documents/ingest", json={"text": "hello world"}
        )
        assert response.status_code == 503


# ─── ASK QUESTION TESTS ─────────────────────────────────


class TestAskQuestion:
    def test_ask_success(self, client):
        """Successfully ask a question → 200 with answer."""
        client.application.rag_engine.ask_question.return_value = {
            "answer": "Python is a programming language.",
            "sources": ["python.txt"],
            "chunks_used": 2,
        }

        response = client.post("/ask", json={"question": "What is Python?"})
        data = response.get_json()

        assert response.status_code == 200
        assert "answer" in data
        assert "sources" in data
        assert data["answer"] == "Python is a programming language."

    def test_ask_missing_question(self, client):
        """Missing 'question' field → 400."""
        response = client.post("/ask", json={})
        assert response.status_code == 400

    def test_ask_empty_question(self, client):
        """Empty question → 400."""
        response = client.post("/ask", json={"question": "   "})
        assert response.status_code == 400

    def test_ask_without_rag(self, client_no_rag):
        """RAG not initialized → 503."""
        response = client_no_rag.post(
            "/ask", json={"question": "What is Python?"}
        )
        assert response.status_code == 503


# ─── LIST DOCUMENTS TESTS ───────────────────────────────


class TestListDocuments:
    def test_list_success(self, client):
        client.application.rag_engine.list_documents.return_value = {
            "documents": ["doc1.txt", "doc2.txt"],
            "total_chunks": 10,
        }

        response = client.get("/documents")
        data = response.get_json()

        assert response.status_code == 200
        assert len(data["documents"]) == 2

    def test_list_empty(self, client):
        client.application.rag_engine.list_documents.return_value = {
            "documents": [],
            "total_chunks": 0,
        }

        response = client.get("/documents")
        data = response.get_json()

        assert response.status_code == 200
        assert data["documents"] == []


# ─── CLEAR DOCUMENTS TESTS ──────────────────────────────


class TestClearDocuments:
    def test_clear_success(self, client):
        client.application.rag_engine.clear_documents.return_value = {
            "cleared": True,
            "chunks_removed": 5,
        }

        response = client.delete("/documents/clear")
        data = response.get_json()

        assert response.status_code == 200
        assert data["cleared"] is True
        assert data["chunks_removed"] == 5
