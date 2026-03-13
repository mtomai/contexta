"""Tests for embeddings service."""
import pytest
from unittest.mock import patch, MagicMock

from app.services.embeddings import (
    truncate_text_for_embedding,
    count_tokens,
    create_embedding,
    create_embeddings,
    embed_document_chunks,
)


class TestCountTokens:
    def test_counts_tokens(self):
        result = count_tokens("Hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_empty_string(self):
        assert count_tokens("") == 0


class TestTruncateTextForEmbedding:
    def test_short_text_unchanged(self):
        text = "Short text."
        result = truncate_text_for_embedding(text, max_tokens=1000)
        assert result == text

    def test_long_text_truncated(self):
        # Generate a very long text
        text = "word " * 50000
        result = truncate_text_for_embedding(text, max_tokens=100)
        assert len(result) < len(text)

    def test_breaks_at_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence. " * 500
        result = truncate_text_for_embedding(text, max_tokens=50)
        # Should end at a sentence boundary if possible
        assert result.endswith(".") or result.endswith(" ")


class TestCreateEmbeddings:
    @patch("app.services.embeddings.client")
    def test_create_embeddings_success(self, mock_client):
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        result = create_embeddings(["test text"])
        assert result[0] == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

    @patch("app.services.embeddings.client")
    def test_create_embeddings_batch(self, mock_client):
        emb1 = MagicMock()
        emb1.embedding = [0.1]
        emb2 = MagicMock()
        emb2.embedding = [0.2]
        mock_response = MagicMock()
        mock_response.data = [emb1, emb2]
        mock_client.embeddings.create.return_value = mock_response

        result = create_embeddings(["text1", "text2"])
        assert result[0] == [0.1]
        assert result[1] == [0.2]

    @patch("app.services.embeddings.client")
    def test_create_embeddings_filters_empty(self, mock_client):
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1]
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        result = create_embeddings(["valid", "", None, "  "])
        # Only "valid" should be sent to API
        call_args = mock_client.embeddings.create.call_args
        assert len(call_args.kwargs["input"]) == 1

    def test_create_embeddings_all_empty(self):
        with pytest.raises(RuntimeError, match="No valid texts"):
            create_embeddings(["", "  ", None])

    @patch("app.services.embeddings.client")
    def test_create_embeddings_api_error(self, mock_client):
        mock_client.embeddings.create.side_effect = Exception("API Error")
        with pytest.raises(RuntimeError, match="Error creating embeddings"):
            create_embeddings(["test"])


class TestCreateEmbedding:
    @patch("app.services.embeddings.client")
    def test_create_single_embedding(self, mock_client):
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.5, 0.6]
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        result = create_embedding("test")
        assert result == [0.5, 0.6]


class TestEmbedDocumentChunks:
    @patch("app.services.embeddings.client")
    def test_embed_chunks(self, mock_client):
        emb1 = MagicMock()
        emb1.embedding = [0.1]
        emb2 = MagicMock()
        emb2.embedding = [0.2]
        mock_response = MagicMock()
        mock_response.data = [emb1, emb2]
        mock_client.embeddings.create.return_value = mock_response

        chunks = [
            {"text": "chunk 1", "metadata": {"page": 1}},
            {"text": "chunk 2", "metadata": {"page": 2}},
        ]
        result = embed_document_chunks(chunks)
        assert len(result) == 2
        assert result[0]["embedding"] == [0.1]
        assert result[1]["embedding"] == [0.2]

    @patch("app.services.embeddings.client")
    def test_embed_chunks_filters_none_embeddings(self, mock_client):
        emb1 = MagicMock()
        emb1.embedding = [0.1]
        mock_response = MagicMock()
        mock_response.data = [emb1]
        mock_client.embeddings.create.return_value = mock_response

        # Second chunk text is empty, will get None embedding
        chunks = [
            {"text": "valid", "metadata": {"page": 1}},
            {"text": "", "metadata": {"page": 2}},
        ]
        result = embed_document_chunks(chunks)
        # Only the valid chunk should be returned
        assert all(c["embedding"] is not None for c in result)
