"""
Configuration Module
====================
Loads settings from environment variables (.env file or system env).

WHY environment variables?
- Secrets (API keys) should NEVER be in your code or Git
- Different environments (dev/staging/prod) need different settings
- Docker and CI/CD inject config via env vars
"""

import os
from dotenv import load_dotenv

# Load variables from .env file (if it exists)
# In Docker/CI, env vars are injected directly — no .env file needed
load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""

    # --- API Keys ---
    # Your Google Gemini API key (required for the LLM)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # --- RAG Settings ---
    # How many characters per text chunk (smaller = more precise retrieval)
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))

    # How many characters overlap between chunks (prevents cutting sentences)
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Which embedding model to use (runs locally, no API key needed)
    # 'all-MiniLM-L6-v2' is small (~80MB), fast, and good quality
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Which Google Gemini model to use
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-flash-latest")

    # Where ChromaDB stores its data on disk
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    @classmethod
    def validate(cls):
        """
        Validate that all required config is present.
        Called at app startup — fails fast if something's missing.
        """
        if not cls.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is required. "
                "Get a free key at https://aistudio.google.com/ "
                "and add it to your .env file."
            )
