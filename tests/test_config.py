"""
Tests for app/config.py
========================
These tests verify that our configuration loads correctly.

WHY test config?
- A missing API key should fail FAST (not when a user asks a question)
- Default values should be sensible
- Type conversions (str → int) should work correctly
"""

import pytest
from unittest.mock import patch
from app.config import Config


class TestConfigValidation:
    """Test the Config.validate() method."""

    def test_validate_raises_when_no_api_key(self):
        """Config.validate() should raise ValueError if GOOGLE_API_KEY is empty."""
        with patch.object(Config, "GOOGLE_API_KEY", ""):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                Config.validate()

    def test_validate_passes_with_api_key(self):
        """Config.validate() should NOT raise if GOOGLE_API_KEY is set."""
        with patch.object(Config, "GOOGLE_API_KEY", "test-api-key-12345"):
            Config.validate()  # Should not raise


class TestConfigDefaults:
    """Test that default values are sensible."""

    def test_chunk_size_is_integer(self):
        assert isinstance(Config.CHUNK_SIZE, int)

    def test_chunk_overlap_is_integer(self):
        assert isinstance(Config.CHUNK_OVERLAP, int)

    def test_chunk_overlap_less_than_chunk_size(self):
        """Overlap must be smaller than chunk size, otherwise chunks would never advance."""
        assert Config.CHUNK_OVERLAP < Config.CHUNK_SIZE

    def test_embedding_model_has_value(self):
        assert len(Config.EMBEDDING_MODEL) > 0

    def test_llm_model_has_value(self):
        assert len(Config.LLM_MODEL) > 0
