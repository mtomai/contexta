"""Tests for context_compressor service."""
import pytest
from unittest.mock import patch, MagicMock

from app.services.context_compressor import (
    estimate_tokens,
    format_chunk_with_source,
    compress_context,
    get_compression_stats,
)


class TestEstimateTokens:
    def test_basic_estimate(self):
        text = "a" * 400
        assert estimate_tokens(text) == 100

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_text(self):
        assert estimate_tokens("hi") == 0  # 2 chars / 4 = 0


class TestFormatChunkWithSource:
    def test_formats_correctly(self):
        chunk = {
            "text": "Chunk content here.",
            "metadata": {"document_name": "report.pdf", "page": 3},
        }
        result = format_chunk_with_source(chunk)
        assert "[report.pdf, pagina 3]" in result
        assert "Chunk content here." in result

    def test_missing_metadata(self):
        chunk = {"text": "Content", "metadata": {}}
        result = format_chunk_with_source(chunk)
        assert "documento" in result
        assert "Content" in result

    def test_alternate_key_chunk_text(self):
        chunk = {"chunk_text": "Alt content", "metadata": {"document_name": "f.pdf", "page": 1}}
        result = format_chunk_with_source(chunk)
        assert "Alt content" in result


class TestCompressContext:
    def test_empty_chunks(self):
        assert compress_context([], "query") == ""

    def test_no_compression_needed(self):
        chunks = [
            {"text": "Short chunk.", "metadata": {"document_name": "f.pdf", "page": 1}}
        ]
        result = compress_context(chunks, "test", max_tokens=10000)
        assert "Short chunk." in result

    def test_force_long_context_bypass(self):
        chunks = [
            {"text": "A" * 10000, "metadata": {"document_name": "f.pdf", "page": 1}}
        ]
        result = compress_context(chunks, "test", max_tokens=10, force_long_context=True)
        assert len(result) > 10  # Not compressed

    @patch("app.services.context_compressor.client")
    def test_compression_with_llm(self, mock_client):
        mock_choice = MagicMock()
        mock_choice.message.content = "Compressed result"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        # Create chunks that exceed the token limit
        chunks = [
            {"text": "A" * 5000, "metadata": {"document_name": "f.pdf", "page": i}}
            for i in range(10)
        ]
        result = compress_context(chunks, "test query", max_tokens=100)
        assert result == "Compressed result"

    @patch("app.services.context_compressor.client")
    def test_compression_api_failure_fallback(self, mock_client):
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        chunks = [
            {"text": "A" * 5000, "metadata": {"document_name": "f.pdf", "page": i}}
            for i in range(10)
        ]
        result = compress_context(chunks, "test query", max_tokens=100)
        # Should fallback to mechanical truncation
        assert isinstance(result, str)


class TestGetCompressionStats:
    def test_basic_stats(self):
        original = "a" * 1000
        compressed = "a" * 400
        stats = get_compression_stats(original, compressed)
        assert stats["original_tokens"] == 250
        assert stats["compressed_tokens"] == 100
        assert stats["tokens_saved"] == 150
        assert stats["reduction_percent"] == 60.0

    def test_no_reduction(self):
        text = "hello"
        stats = get_compression_stats(text, text)
        assert stats["tokens_saved"] == 0
        assert stats["reduction_percent"] == 0

    def test_empty_original(self):
        stats = get_compression_stats("", "")
        assert stats["original_tokens"] == 0
        assert stats["reduction_percent"] == 0
