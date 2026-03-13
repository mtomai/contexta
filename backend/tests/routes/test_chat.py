"""
Unit tests for the chat route endpoints.

Tests cover:
- POST "" (chat endpoint): valid queries, empty queries, conversation management,
  refinement mode, error handling, title generation
- POST "/stream" (streaming endpoint): valid queries, empty queries, refinement fallback
- GET "/health": health check
- GET "/cache/stats": cache statistics
- POST "/cache/clear": cache clearing
- get_conversation_history helper function
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

os.environ.setdefault("OPENAI_API_KEY", "fake-key-for-tests")

from app.models.chat import Source
from app.routes.chat import get_conversation_history


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _fake_source():
    return Source(
        document="report.pdf",
        page=3,
        chunk_index=0,
        chunk_text="Revenue grew by 15% in Q3.",
        relevance_score=0.91,
    )


def _fake_generate_response_result():
    return {
        "answer": "Revenue grew by 15% in Q3 [report.pdf, pagina 3].",
        "sources": [_fake_source()],
        "metadata": {
            "query_type": "synthesis",
            "confidence": 0.88,
            "model_used": "gpt-4o-mini",
            "processing_time": 0.35,
            "chunks_retrieved": 4,
            "context_compressed": False,
            "routing_explanation": "Standard query requiring LLM",
        },
    }


def _mock_conv_db(
    conversation_id="conv-001",
    conversation=None,
    message_count=0,
    notebook_id=None,
    document_ids=None,
):
    """Build a MagicMock that behaves like ConversationDB."""
    mock_db = MagicMock()
    mock_db.create_conversation.return_value = conversation_id

    if conversation is not None:
        mock_db.get_conversation.return_value = conversation
    else:
        # Default: conversation not found (for new conversation flow)
        mock_db.get_conversation.return_value = None

    mock_db.add_message.return_value = None
    mock_db.update_conversation_title.return_value = None
    mock_db.get_messages.return_value = []
    return mock_db


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST "" — Standard chat, new conversation
# ═══════════════════════════════════════════════════════════════════════════════

def test_chat_new_conversation_returns_200(app_client):
    """POST /api/chat with a valid query and no conversation_id creates a new
    conversation and returns 200 with answer, sources, conversation_id."""
    fake_result = _fake_generate_response_result()
    mock_db = _mock_conv_db()

    with patch("app.routes.chat.generate_response", new_callable=AsyncMock, return_value=fake_result), \
         patch("app.routes.chat.get_conversation_db", return_value=mock_db), \
         patch("app.routes.chat.get_embedding_cache", return_value=MagicMock()), \
         patch("app.routes.chat.generate_conversation_title", return_value="Test Title"):

        response = app_client.post("/api/chat", json={"query": "What is the revenue?"})

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "conversation_id" in data
    assert data["conversation_id"] == "conv-001"
    assert "Revenue" in data["answer"]


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST "" — Empty query → 400
# ═══════════════════════════════════════════════════════════════════════════════

def test_chat_empty_query_returns_400(app_client):
    """POST /api/chat with empty/whitespace query returns 400."""
    response = app_client.post("/api/chat", json={"query": "   "})
    assert response.status_code == 400


def test_chat_blank_query_returns_400(app_client):
    """POST /api/chat with empty string query returns 400."""
    response = app_client.post("/api/chat", json={"query": ""})
    assert response.status_code == 400


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST "" — Existing conversation
# ═══════════════════════════════════════════════════════════════════════════════

def test_chat_existing_conversation(app_client):
    """POST /api/chat with an existing conversation_id reuses the conversation
    and retrieves history."""
    fake_result = _fake_generate_response_result()
    mock_db = _mock_conv_db(conversation_id="conv-existing")
    mock_db.get_conversation.return_value = {
        "id": "conv-existing",
        "message_count": 2,
        "notebook_id": None,
        "document_ids": None,
    }
    mock_db.get_messages.return_value = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    with patch("app.routes.chat.generate_response", new_callable=AsyncMock, return_value=fake_result), \
         patch("app.routes.chat.get_conversation_db", return_value=mock_db), \
         patch("app.routes.chat.get_embedding_cache", return_value=MagicMock()), \
         patch("app.routes.chat.generate_conversation_title", return_value="Title"):

        response = app_client.post(
            "/api/chat",
            json={"query": "Follow up question", "conversation_id": "conv-existing"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == "conv-existing"
    # Title should NOT be regenerated since message_count > 0
    mock_db.update_conversation_title.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST "" — Non-existent conversation_id → 404
# ═══════════════════════════════════════════════════════════════════════════════

def test_chat_nonexistent_conversation_returns_404(app_client):
    """POST /api/chat with a conversation_id that doesn't exist returns 404."""
    mock_db = _mock_conv_db()
    mock_db.get_conversation.return_value = None

    with patch("app.routes.chat.get_conversation_db", return_value=mock_db):
        response = app_client.post(
            "/api/chat",
            json={"query": "Something", "conversation_id": "nonexistent-id"},
        )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST "" — Refinement mode
# ═══════════════════════════════════════════════════════════════════════════════

def test_chat_refinement_mode(app_client):
    """When conversation has document_ids, refinement mode is used."""
    fake_result = _fake_generate_response_result()
    mock_db = _mock_conv_db()
    mock_db.get_conversation.return_value = {
        "id": "conv-refine",
        "message_count": 0,
        "notebook_id": None,
        "document_ids": ["doc-1", "doc-2"],
    }

    with patch("app.routes.chat.generate_refinement_response", new_callable=AsyncMock, return_value=fake_result) as mock_refine, \
         patch("app.routes.chat.generate_response", new_callable=AsyncMock) as mock_rag, \
         patch("app.routes.chat.get_conversation_db", return_value=mock_db), \
         patch("app.routes.chat.get_embedding_cache", return_value=MagicMock()), \
         patch("app.routes.chat.generate_conversation_title", return_value="Refined Title"):

        response = app_client.post(
            "/api/chat",
            json={"query": "Refine this", "conversation_id": "conv-refine"},
        )

    assert response.status_code == 200
    mock_refine.assert_called_once()
    mock_rag.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST "" — First message triggers title generation
# ═══════════════════════════════════════════════════════════════════════════════

def test_chat_first_message_generates_title(app_client):
    """First message in a new conversation triggers title generation."""
    fake_result = _fake_generate_response_result()
    mock_db = _mock_conv_db()

    with patch("app.routes.chat.generate_response", new_callable=AsyncMock, return_value=fake_result), \
         patch("app.routes.chat.get_conversation_db", return_value=mock_db), \
         patch("app.routes.chat.get_embedding_cache", return_value=MagicMock()), \
         patch("app.routes.chat.generate_conversation_title", return_value="Auto Title") as mock_title:

        response = app_client.post("/api/chat", json={"query": "First question"})

    assert response.status_code == 200
    mock_title.assert_called_once_with("First question")
    mock_db.update_conversation_title.assert_called_once_with("conv-001", "Auto Title")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST "" — Sources have required fields
# ═══════════════════════════════════════════════════════════════════════════════

def test_chat_sources_have_required_fields(app_client):
    """Each source in the response contains document, page, chunk_text, relevance_score."""
    fake_result = _fake_generate_response_result()
    mock_db = _mock_conv_db()

    with patch("app.routes.chat.generate_response", new_callable=AsyncMock, return_value=fake_result), \
         patch("app.routes.chat.get_conversation_db", return_value=mock_db), \
         patch("app.routes.chat.get_embedding_cache", return_value=MagicMock()), \
         patch("app.routes.chat.generate_conversation_title", return_value="Title"):

        response = app_client.post("/api/chat", json={"query": "Details?"})

    assert response.status_code == 200
    sources = response.json()["sources"]
    assert isinstance(sources, list)
    assert len(sources) >= 1
    for source in sources:
        assert "document" in source
        assert "page" in source
        assert "chunk_text" in source
        assert "relevance_score" in source


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST "" — RuntimeError → 500
# ═══════════════════════════════════════════════════════════════════════════════

def test_chat_runtime_error_returns_500(app_client):
    """RuntimeError during generate_response should result in 500."""
    mock_db = _mock_conv_db()

    with patch("app.routes.chat.generate_response", new_callable=AsyncMock, side_effect=RuntimeError("LLM unavailable")), \
         patch("app.routes.chat.get_conversation_db", return_value=mock_db), \
         patch("app.routes.chat.get_embedding_cache", return_value=MagicMock()):

        response = app_client.post("/api/chat", json={"query": "Will fail"})

    assert response.status_code == 500
    assert "LLM unavailable" in response.json()["detail"]


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST "" — Generic exception → 500
# ═══════════════════════════════════════════════════════════════════════════════

def test_chat_generic_exception_returns_500(app_client):
    """Unexpected exceptions are caught and return 500."""
    mock_db = _mock_conv_db()

    with patch("app.routes.chat.generate_response", new_callable=AsyncMock, side_effect=ValueError("Unexpected")), \
         patch("app.routes.chat.get_conversation_db", return_value=mock_db), \
         patch("app.routes.chat.get_embedding_cache", return_value=MagicMock()):

        response = app_client.post("/api/chat", json={"query": "Will fail too"})

    assert response.status_code == 500
    assert "Unexpected" in response.json()["detail"]


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST "" — User message is saved to DB
# ═══════════════════════════════════════════════════════════════════════════════

def test_chat_saves_user_and_assistant_messages(app_client):
    """Both user query and assistant response are saved to the database."""
    fake_result = _fake_generate_response_result()
    mock_db = _mock_conv_db()

    with patch("app.routes.chat.generate_response", new_callable=AsyncMock, return_value=fake_result), \
         patch("app.routes.chat.get_conversation_db", return_value=mock_db), \
         patch("app.routes.chat.get_embedding_cache", return_value=MagicMock()), \
         patch("app.routes.chat.generate_conversation_title", return_value="T"):

        app_client.post("/api/chat", json={"query": "Save me"})

    # User message saved
    calls = mock_db.add_message.call_args_list
    assert len(calls) == 2

    user_call = calls[0]
    assert user_call.kwargs["role"] == "user" or user_call[1]["role"] == "user"

    assistant_call = calls[1]
    assert assistant_call.kwargs["role"] == "assistant" or assistant_call[1]["role"] == "assistant"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: GET /health
# ═══════════════════════════════════════════════════════════════════════════════

def test_chat_health_endpoint(app_client):
    """GET /api/chat/health returns status message."""
    response = app_client.get("/api/chat/health")
    assert response.status_code == 200
    assert response.json()["status"] == "chat service is running"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: GET /cache/stats
# ═══════════════════════════════════════════════════════════════════════════════

def test_cache_stats_endpoint(app_client):
    """GET /api/chat/cache/stats returns cache statistics."""
    mock_cache = MagicMock()
    mock_cache.get_stats.return_value = {"hits": 10, "misses": 5, "hit_rate": 0.67}

    with patch("app.routes.chat.get_embedding_cache", return_value=mock_cache):
        response = app_client.get("/api/chat/cache/stats")

    assert response.status_code == 200
    data = response.json()
    assert "hits" in data
    assert "hit_rate" in data


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST /cache/clear
# ═══════════════════════════════════════════════════════════════════════════════

def test_cache_clear_endpoint(app_client):
    """POST /api/chat/cache/clear clears the cache and returns confirmation."""
    mock_cache = MagicMock()

    with patch("app.routes.chat.get_embedding_cache", return_value=mock_cache):
        response = app_client.post("/api/chat/cache/clear")

    assert response.status_code == 200
    assert "cleared" in response.json()["message"].lower()
    mock_cache.clear.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST /stream — Empty query → 400
# ═══════════════════════════════════════════════════════════════════════════════

def test_stream_empty_query_returns_400(app_client):
    """POST /api/chat/stream with empty query returns 400."""
    response = app_client.post("/api/chat/stream", json={"query": ""})
    assert response.status_code == 400


def test_stream_whitespace_query_returns_400(app_client):
    """POST /api/chat/stream with whitespace-only query returns 400."""
    response = app_client.post("/api/chat/stream", json={"query": "   "})
    assert response.status_code == 400


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST /stream — Non-existent conversation → 404
# ═══════════════════════════════════════════════════════════════════════════════

def test_stream_nonexistent_conversation_returns_404(app_client):
    """POST /api/chat/stream with non-existent conversation_id returns 404."""
    mock_db = _mock_conv_db()
    mock_db.get_conversation.return_value = None

    with patch("app.routes.chat.get_conversation_db", return_value=mock_db):
        response = app_client.post(
            "/api/chat/stream",
            json={"query": "Stream me", "conversation_id": "no-exist"},
        )

    assert response.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST /stream — Refinement mode returns SSE
# ═══════════════════════════════════════════════════════════════════════════════

def test_stream_refinement_mode_returns_sse(app_client):
    """Streaming refinement mode sends sources, tokens, and done events."""
    fake_result = _fake_generate_response_result()
    mock_db = _mock_conv_db()
    mock_db.get_conversation.return_value = {
        "id": "conv-stream-refine",
        "message_count": 0,
        "notebook_id": None,
        "document_ids": ["doc-a"],
    }

    with patch("app.routes.chat.generate_refinement_response", new_callable=AsyncMock, return_value=fake_result), \
         patch("app.routes.chat.get_conversation_db", return_value=mock_db), \
         patch("app.routes.chat.get_embedding_cache", return_value=MagicMock()), \
         patch("app.routes.chat.generate_conversation_title", return_value="Stream Title"):

        response = app_client.post(
            "/api/chat/stream",
            json={"query": "Refine stream", "conversation_id": "conv-stream-refine"},
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    body = response.text
    assert "event: sources" in body
    assert "event: token" in body
    assert "event: done" in body


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST /stream — Standard RAG streaming
# ═══════════════════════════════════════════════════════════════════════════════

def test_stream_standard_rag_returns_sse(app_client):
    """Standard RAG streaming returns SSE with tokens and done event."""
    mock_db = _mock_conv_db()

    async def fake_stream(**kwargs):
        yield 'event: token\ndata: {"token": "Hello"}\n\n'
        yield 'event: token\ndata: {"token": " world"}\n\n'
        yield 'event: done\ndata: {"full_response": "Hello world", "sources": [{"document": "test.pdf", "page": 1, "chunk_text": "test", "relevance_score": 0.9}]}\n\n'

    with patch("app.routes.chat.generate_response_stream", side_effect=fake_stream), \
         patch("app.routes.chat.get_conversation_db", return_value=mock_db), \
         patch("app.routes.chat.get_embedding_cache", return_value=MagicMock()), \
         patch("app.routes.chat.generate_conversation_title", return_value="Stream RAG Title"):

        response = app_client.post("/api/chat/stream", json={"query": "Hello?"})

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    body = response.text
    assert "Hello" in body
    assert "event: done" in body

    # Verify assistant message was saved
    add_msg_calls = mock_db.add_message.call_args_list
    # user message + assistant message
    assert len(add_msg_calls) == 2
    assistant_call = add_msg_calls[1]
    assert assistant_call.kwargs.get("role", assistant_call[1].get("role")) == "assistant"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST /stream — Title generated on first message
# ═══════════════════════════════════════════════════════════════════════════════

def test_stream_first_message_generates_title(app_client):
    """First message in streaming mode triggers title generation after streaming."""
    mock_db = _mock_conv_db()

    async def fake_stream(**kwargs):
        yield 'event: done\ndata: {"full_response": "Answer", "sources": []}\n\n'

    with patch("app.routes.chat.generate_response_stream", side_effect=fake_stream), \
         patch("app.routes.chat.get_conversation_db", return_value=mock_db), \
         patch("app.routes.chat.get_embedding_cache", return_value=MagicMock()), \
         patch("app.routes.chat.generate_conversation_title", return_value="Generated Title") as mock_title:

        app_client.post("/api/chat/stream", json={"query": "First stream"})

    mock_title.assert_called_once_with("First stream")
    mock_db.update_conversation_title.assert_called_once_with("conv-001", "Generated Title")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: get_conversation_history helper
# ═══════════════════════════════════════════════════════════════════════════════

def test_get_conversation_history_returns_formatted_messages():
    """get_conversation_history returns messages in [{role, content}] format."""
    mock_db = MagicMock()
    mock_db.get_messages.return_value = [
        {"role": "user", "content": "Q1", "timestamp": "2024-01-01"},
        {"role": "assistant", "content": "A1", "timestamp": "2024-01-01"},
        {"role": "user", "content": "Q2", "timestamp": "2024-01-01"},
    ]

    with patch("app.routes.chat.get_conversation_db", return_value=mock_db):
        history = get_conversation_history("conv-123")

    assert len(history) == 3
    assert history[0] == {"role": "user", "content": "Q1"}
    assert history[1] == {"role": "assistant", "content": "A1"}
    assert history[2] == {"role": "user", "content": "Q2"}


def test_get_conversation_history_limits_to_max_messages():
    """get_conversation_history respects max_messages limit."""
    mock_db = MagicMock()
    mock_db.get_messages.return_value = [
        {"role": "user", "content": f"msg-{i}"} for i in range(20)
    ]

    with patch("app.routes.chat.get_conversation_db", return_value=mock_db):
        history = get_conversation_history("conv-123", max_messages=5)

    assert len(history) == 5
    # Should be the LAST 5 messages
    assert history[0]["content"] == "msg-15"
    assert history[4]["content"] == "msg-19"


def test_get_conversation_history_empty():
    """get_conversation_history returns empty list when no messages exist."""
    mock_db = MagicMock()
    mock_db.get_messages.return_value = []

    with patch("app.routes.chat.get_conversation_db", return_value=mock_db):
        history = get_conversation_history("conv-empty")

    assert history == []


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST "" — Conversation with existing history doesn't regenerate title
# ═══════════════════════════════════════════════════════════════════════════════

def test_chat_existing_conversation_first_message_generates_title(app_client):
    """When conversation exists but message_count == 0, title is generated."""
    fake_result = _fake_generate_response_result()
    mock_db = _mock_conv_db()
    mock_db.get_conversation.return_value = {
        "id": "conv-empty-existing",
        "message_count": 0,
        "notebook_id": None,
        "document_ids": None,
    }

    with patch("app.routes.chat.generate_response", new_callable=AsyncMock, return_value=fake_result), \
         patch("app.routes.chat.get_conversation_db", return_value=mock_db), \
         patch("app.routes.chat.get_embedding_cache", return_value=MagicMock()), \
         patch("app.routes.chat.generate_conversation_title", return_value="New Title") as mock_title:

        response = app_client.post(
            "/api/chat",
            json={"query": "First in existing", "conversation_id": "conv-empty-existing"},
        )

    assert response.status_code == 200
    mock_title.assert_called_once_with("First in existing")
    mock_db.update_conversation_title.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST "" — Assistant sources saved to DB correctly
# ═══════════════════════════════════════════════════════════════════════════════

def test_chat_saves_sources_to_db(app_client):
    """Sources are converted and saved with the assistant message."""
    fake_result = _fake_generate_response_result()
    mock_db = _mock_conv_db()

    with patch("app.routes.chat.generate_response", new_callable=AsyncMock, return_value=fake_result), \
         patch("app.routes.chat.get_conversation_db", return_value=mock_db), \
         patch("app.routes.chat.get_embedding_cache", return_value=MagicMock()), \
         patch("app.routes.chat.generate_conversation_title", return_value="T"):

        app_client.post("/api/chat", json={"query": "With sources"})

    # Second add_message call is the assistant message
    assistant_call = mock_db.add_message.call_args_list[1]
    saved_sources = assistant_call.kwargs.get("sources", assistant_call[1].get("sources"))
    assert len(saved_sources) == 1
    assert saved_sources[0]["document"] == "report.pdf"
    assert saved_sources[0]["page"] == 3
    assert saved_sources[0]["relevance_score"] == 0.91


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST /stream — Saves user message to DB
# ═══════════════════════════════════════════════════════════════════════════════

def test_stream_saves_user_message(app_client):
    """Streaming endpoint saves the user message before streaming starts."""
    mock_db = _mock_conv_db()

    async def fake_stream(**kwargs):
        yield 'event: done\ndata: {"full_response": "Done", "sources": []}\n\n'

    with patch("app.routes.chat.generate_response_stream", side_effect=fake_stream), \
         patch("app.routes.chat.get_conversation_db", return_value=mock_db), \
         patch("app.routes.chat.get_embedding_cache", return_value=MagicMock()), \
         patch("app.routes.chat.generate_conversation_title", return_value="T"):

        app_client.post("/api/chat/stream", json={"query": "Stream query"})

    user_call = mock_db.add_message.call_args_list[0]
    assert user_call.kwargs.get("role", user_call[1].get("role")) == "user"
    assert user_call.kwargs.get("content", user_call[1].get("content")) == "Stream query"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST /stream — Existing conversation with history
# ═══════════════════════════════════════════════════════════════════════════════

def test_stream_existing_conversation_gets_history(app_client):
    """Streaming with existing conversation retrieves history."""
    mock_db = _mock_conv_db()
    mock_db.get_conversation.return_value = {
        "id": "conv-stream-hist",
        "message_count": 2,
        "notebook_id": None,
        "document_ids": None,
    }
    mock_db.get_messages.return_value = [
        {"role": "user", "content": "Previous Q"},
        {"role": "assistant", "content": "Previous A"},
    ]

    async def fake_stream(**kwargs):
        # Verify conversation_history was passed
        assert kwargs.get("conversation_history") is not None
        yield 'event: done\ndata: {"full_response": "Follow up", "sources": []}\n\n'

    with patch("app.routes.chat.generate_response_stream", side_effect=fake_stream), \
         patch("app.routes.chat.get_conversation_db", return_value=mock_db), \
         patch("app.routes.chat.get_embedding_cache", return_value=MagicMock()), \
         patch("app.routes.chat.generate_conversation_title", return_value="T"):

        response = app_client.post(
            "/api/chat/stream",
            json={"query": "Continue", "conversation_id": "conv-stream-hist"},
        )

    assert response.status_code == 200
    # Title should not be regenerated for non-first message
    mock_db.update_conversation_title.assert_not_called()