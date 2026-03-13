"""Tests for chat service."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.services.chat import (
    _build_messages,
    generate_response,
)
from app.services.query_router import QueryType


class TestBuildMessages:
    @patch("app.services.chat.get_settings")
    def test_basic_messages(self, mock_get_settings):
        mock_settings = MagicMock()
        mock_settings.chat_system_prompt = "System prompt"
        mock_settings.fallback_system_prompt = "Fallback"
        mock_settings.chat_user_prompt_template = "Context: {context_text}\nQuery: {query}"
        mock_settings.full_document_suffix = " [FULL]"
        mock_get_settings.return_value = mock_settings

        messages = _build_messages(
            query="test question",
            context="some context",
        )
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert "test question" in messages[-1]["content"]

    @patch("app.services.chat.get_settings")
    def test_with_conversation_history(self, mock_get_settings):
        mock_settings = MagicMock()
        mock_settings.chat_system_prompt = "System"
        mock_settings.fallback_system_prompt = "Fallback"
        mock_settings.chat_user_prompt_template = "C: {context_text}\nQ: {query}"
        mock_settings.full_document_suffix = ""
        mock_get_settings.return_value = mock_settings

        history = [
            {"role": "user", "content": "prev question"},
            {"role": "assistant", "content": "prev answer"},
        ]
        messages = _build_messages(
            query="new question",
            context="ctx",
            conversation_history=history,
        )
        # system + history (2) + user = 4
        assert len(messages) == 4

    @patch("app.services.chat.get_settings")
    def test_full_document_suffix(self, mock_get_settings):
        mock_settings = MagicMock()
        mock_settings.chat_system_prompt = "System"
        mock_settings.fallback_system_prompt = "Fallback"
        mock_settings.chat_user_prompt_template = "C: {context_text}\nQ: {query}"
        mock_settings.full_document_suffix = " [FULL DOC]"
        mock_get_settings.return_value = mock_settings

        messages = _build_messages(
            query="q", context="c", query_type=QueryType.FULL_DOCUMENT
        )
        assert "[FULL DOC]" in messages[0]["content"]

    @patch("app.services.chat.get_settings")
    def test_history_limited_to_6(self, mock_get_settings):
        mock_settings = MagicMock()
        mock_settings.chat_system_prompt = "System"
        mock_settings.fallback_system_prompt = "Fallback"
        mock_settings.chat_user_prompt_template = "C: {context_text}\nQ: {query}"
        mock_settings.full_document_suffix = ""
        mock_get_settings.return_value = mock_settings

        history = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        messages = _build_messages(query="q", context="c", conversation_history=history)
        # system + 6 history + user = 8
        assert len(messages) == 8

    @patch("app.services.chat.get_settings")
    def test_fallback_system_prompt(self, mock_get_settings):
        mock_settings = MagicMock()
        mock_settings.chat_system_prompt = ""  # Empty = falsy
        mock_settings.fallback_system_prompt = "Fallback used"
        mock_settings.chat_user_prompt_template = "C: {context_text}\nQ: {query}"
        mock_settings.full_document_suffix = ""
        mock_get_settings.return_value = mock_settings

        messages = _build_messages(query="q", context="c")
        assert messages[0]["content"] == "Fallback used"
