"""Tests for title_generator service."""
import pytest
from unittest.mock import patch, MagicMock

from app.services.title_generator import (
    generate_conversation_title,
    _fallback_title,
)


class TestFallbackTitle:
    def test_short_message(self):
        assert _fallback_title("Hello") == "Hello"

    def test_exact_length(self):
        text = "a" * 50
        assert _fallback_title(text) == text

    def test_long_message_truncated(self):
        text = "a" * 100
        result = _fallback_title(text)
        assert len(result) == 53  # 50 + "..."
        assert result.endswith("...")


class TestGenerateConversationTitle:
    @patch("app.services.title_generator.client")
    def test_success(self, mock_client):
        mock_choice = MagicMock()
        mock_choice.message.content = "Generated Title"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        result = generate_conversation_title("Parlami del machine learning")
        assert result == "Generated Title"

    @patch("app.services.title_generator.client")
    def test_strips_quotes(self, mock_client):
        mock_choice = MagicMock()
        mock_choice.message.content = '"Quoted Title"'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        result = generate_conversation_title("test")
        assert result == "Quoted Title"

    @patch("app.services.title_generator.client")
    def test_truncates_long_title(self, mock_client):
        mock_choice = MagicMock()
        mock_choice.message.content = "A" * 200
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        result = generate_conversation_title("test")
        assert len(result) <= 100

    @patch("app.services.title_generator.client")
    def test_empty_response_fallback(self, mock_client):
        mock_choice = MagicMock()
        mock_choice.message.content = ""
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        result = generate_conversation_title("My first message")
        assert result == "My first message"

    @patch("app.services.title_generator.client")
    def test_api_failure_fallback(self, mock_client):
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        result = generate_conversation_title("Fallback message content here")
        assert result == "Fallback message content here"

    @patch("app.services.title_generator.client")
    def test_api_failure_long_message_fallback(self, mock_client):
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        long_msg = "x" * 100
        result = generate_conversation_title(long_msg)
        assert len(result) == 53
        assert result.endswith("...")
