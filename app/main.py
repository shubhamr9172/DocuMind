"""
DocuMind Flask API
==================
REST API for the RAG-powered Document Q&A bot.

Endpoints:
    GET  /                  → Health check
    GET  /documents         → List ingested documents
    POST /documents/ingest  → Upload & ingest a document
    POST /ask               → Ask a question (RAG)
    DELETE /documents/clear  → Clear all documents
    GET  /chat              → Web chat interface
"""

from flask import Flask, request, jsonify, send_from_directory
import os
from app.config import Config
from app.rag_engine import RAGEngine


def create_app(testing=False):
    """
    Application Factory Pattern
    ===========================
    Creates and configures the Flask app.

    WHY a factory?
    - Testing: We can create fresh app instances with different configs
    - Flexibility: Different configs for dev/staging/prod
    - CI/CD: Tests create their own app without needing an API key

    Args:
        testing: If True, skips RAG engine init (for unit tests)
    """
    # static_folder points to our frontend files
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    app = Flask(__name__, static_folder=static_dir, static_url_path="/static")

    # Initialize RAG engine (skip during testing)
    if not testing:
        try:
            Config.validate()
            app.rag_engine = RAGEngine()
            print("[OK] RAG Engine initialized successfully!")
        except ValueError as e:
            print(f"[WARNING] RAG Engine not initialized: {e}")
            app.rag_engine = None
    else:
        app.rag_engine = None

    # ─────────────────────────────────────────────────────
    # ENDPOINT 1: Health Check
    # Purpose: Verify the API is running
    # Used by: Docker HEALTHCHECK, CI/CD pipeline, monitoring
    # ─────────────────────────────────────────────────────
    @app.route("/")
    def health_check():
        return jsonify(
            {
                "status": "healthy",
                "app": "DocuMind",
                "version": "1.0.0",
                "rag_enabled": app.rag_engine is not None,
            }
        )

    # ─────────────────────────────────────────────────────
    # ENDPOINT 2: List Documents
    # Purpose: See what documents are in the knowledge base
    # ─────────────────────────────────────────────────────
    @app.route("/documents", methods=["GET"])
    def list_documents():
        if not app.rag_engine:
            return jsonify({"error": "RAG engine not initialized"}), 503

        result = app.rag_engine.list_documents()
        return jsonify(result)

    # ─────────────────────────────────────────────────────
    # ENDPOINT 3: Ingest Document
    # Purpose: Add a document to the knowledge base
    # Body: { "text": "...", "filename": "doc.txt" }
    # ─────────────────────────────────────────────────────
    @app.route("/documents/ingest", methods=["POST"])
    def ingest_document():
        if not app.rag_engine:
            return jsonify({"error": "RAG engine not initialized"}), 503

        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field in request body"}), 400

        if not data["text"].strip():
            return jsonify({"error": "'text' field cannot be empty"}), 400

        filename = data.get("filename", "untitled.txt")
        result = app.rag_engine.ingest_document(data["text"], filename)
        return jsonify(result), 201

    # ─────────────────────────────────────────────────────
    # ENDPOINT 4: Ask Question
    # Purpose: Ask a question → RAG retrieves + LLM answers
    # Body: { "question": "What is Python?" }
    # ─────────────────────────────────────────────────────
    @app.route("/ask", methods=["POST"])
    def ask_question():
        if not app.rag_engine:
            return jsonify({"error": "RAG engine not initialized"}), 503

        data = request.get_json()

        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' field in request body"}), 400

        if not data["question"].strip():
            return jsonify({"error": "'question' field cannot be empty"}), 400

        result = app.rag_engine.ask_question(data["question"])
        return jsonify(result)

    # ─────────────────────────────────────────────────────
    # ENDPOINT 5: Clear Documents
    # Purpose: Remove all documents from the knowledge base
    # ─────────────────────────────────────────────────────
    @app.route("/documents/clear", methods=["DELETE"])
    def clear_documents():
        if not app.rag_engine:
            return jsonify({"error": "RAG engine not initialized"}), 503

        result = app.rag_engine.clear_documents()
        return jsonify(result)

    # ─────────────────────────────────────────────────────
    # FRONTEND: Serve the chat interface
    # ─────────────────────────────────────────────────────
    @app.route("/chat")
    def serve_frontend():
        return send_from_directory(app.static_folder, "index.html")

    @app.errorhandler(Exception)
    def handle_exception(e):
        """Global error handler for unhandled exceptions."""
        # If the error has a status code (like Flask's HTTPException), use it
        code = 500
        if hasattr(e, "code"):
            code = e.code

        return jsonify({
            "error": "Internal Server Error" if code == 500 else "Error",
            "message": str(e)
        }), code

    return app


# ─────────────────────────────────────────────────────────
# Create the app instance
# Flask CLI uses this: flask --app app.main run
# ─────────────────────────────────────────────────────────
app = create_app()
