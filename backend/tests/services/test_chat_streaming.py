"""Tests for chat_streaming service."""
import json
import pytest
from unittest.mock import patch, MagicMock

from app.services.chat_streaming import (
    _format_sse,
    extract_citations_streaming,
)
from app.services.chat_utils import extract_citations


class TestFormatSSE:
    def test_basic_format(self):
        result = _format_sse("token", {"content": "hello"})
        assert result.startswith("event: token\n")
        assert "data:" in result
        assert result.endswith("\n\n")

    def test_json_data(self):
        result = _format_sse("done", {"key": "value"})
        # Extract the data line
        lines = result.strip().split("\n")
        data_line = [l for l in lines if l.startswith("data:")][0]
        data_json = data_line.replace("data: ", "")
        parsed = json.loads(data_json)
        assert parsed["key"] == "value"

    def test_unicode_handling(self):
        result = _format_sse("token", {"content": "ciao è tutto"})
        assert "ciao è tutto" in result

    def test_error_event(self):
        result = _format_sse("error", {"message": "Something failed"})
        assert "event: error" in result
        assert "Something failed" in result


class TestExtractCitationsStreamingAlias:
    def test_alias_same_function(self):
        # extract_citations_streaming should be the same as extract_citations
        assert extract_citations_streaming is extract_citations


class TestStreamSimpleResponse:
    @pytest.mark.asyncio
    async def test_stream_words(self):
        from app.services.chat_streaming import stream_simple_response
        chunks = []
        async for chunk in stream_simple_response("Hello world test"):
            chunks.append(chunk)
        # Should have word tokens + done event
        assert any("done" in c for c in chunks)
        assert len(chunks) >= 2  # at least some tokens + done
