"""
Unit tests for the conversations route endpoints.

Tests cover:
- POST "" (create_conversation): valid requests, default title, notebook_id handling
- GET "" (list_conversations): returns list of conversations
- GET "/{conversation_id}": valid ID, non-existent ID (404)
- DELETE "/{conversation_id}": valid deletion, non-existent ID (404)
- PATCH "/{conversation_id}/title": valid update, non-existent ID (404)
"""

import os
from unittest.mock import patch, MagicMock

os.environ.setdefault("OPENAI_API_KEY", "fake-key-for-tests")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _mock_conversation_db():
    """Build a MagicMock that behaves like ConversationDB."""
    mock_db = MagicMock()
    mock_db.create_conversation.return_value = "conv-001"
    mock_db.get_conversation.return_value = {
        "id": "conv-001",
        "title": "Test Conversation",
        "notebook_id": None,
        "document_ids": None,
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "message_count": 0
    }
    mock_db.list_conversations.return_value = []
    mock_db.get_conversation_with_messages.return_value = None
    mock_db.delete_conversation.return_value = None
    mock_db.update_conversation_title.return_value = None
    return mock_db


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST "" — Create conversation with title
# ═══════════════════════════════════════════════════════════════════════════════

def test_create_conversation_with_title(app_client):
    """POST /api/conversations with title creates conversation and returns 201."""
    mock_db = _mock_conversation_db()

    with patch("app.routes.conversations.get_conversation_db", return_value=mock_db):
        response = app_client.post(
            "/api/conversations",
            json={"title": "My Custom Title"}
        )

    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["title"] == "Test Conversation"
    assert data["id"] == "conv-001"
    mock_db.create_conversation.assert_called_once_with(
        title="My Custom Title",
        notebook_id=None
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST "" — Create conversation with default title
# ═══════════════════════════════════════════════════════════════════════════════

def test_create_conversation_default_title(app_client):
    """POST /api/conversations without title uses default 'Nuova Conversazione'."""
    mock_db = _mock_conversation_db()

    with patch("app.routes.conversations.get_conversation_db", return_value=mock_db):
        response = app_client.post(
            "/api/conversations",
            json={}
        )

    assert response.status_code == 201
    mock_db.create_conversation.assert_called_once_with(
        title="Nuova Conversazione",
        notebook_id=None
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST "" — Create conversation with notebook_id
# ═══════════════════════════════════════════════════════════════════════════════

def test_create_conversation_with_notebook_id(app_client):
    """POST /api/conversations with notebook_id associates conversation with notebook."""
    mock_db = _mock_conversation_db()
    mock_db.get_conversation.return_value = {
        "id": "conv-001",
        "title": "Test",
        "notebook_id": "notebook-123",
        "document_ids": None,
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "message_count": 0
    }

    with patch("app.routes.conversations.get_conversation_db", return_value=mock_db):
        response = app_client.post(
            "/api/conversations",
            json={"title": "Notebook Chat", "notebook_id": "notebook-123"}
        )

    assert response.status_code == 201
    data = response.json()
    assert data["notebook_id"] == "notebook-123"
    mock_db.create_conversation.assert_called_once_with(
        title="Notebook Chat",
        notebook_id="notebook-123"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: GET "" — List conversations
# ═══════════════════════════════════════════════════════════════════════════════

def test_list_conversations_returns_empty_list(app_client):
    """GET /api/conversations with no conversations returns empty list."""
    mock_db = _mock_conversation_db()
    mock_db.list_conversations.return_value = []

    with patch("app.routes.conversations.get_conversation_db", return_value=mock_db):
        response = app_client.get("/api/conversations")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


def test_list_conversations_returns_multiple(app_client):
    """GET /api/conversations returns list of all conversations."""
    mock_db = _mock_conversation_db()
    mock_db.list_conversations.return_value = [
        {
            "id": "conv-001",
            "title": "First",
            "notebook_id": None,
            "document_ids": None,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "message_count": 3
        },
        {
            "id": "conv-002",
            "title": "Second",
            "notebook_id": "notebook-1",
            "document_ids": None,
            "created_at": "2024-01-02T00:00:00",
            "updated_at": "2024-01-02T00:00:00",
            "message_count": 1
        }
    ]

    with patch("app.routes.conversations.get_conversation_db", return_value=mock_db):
        response = app_client.get("/api/conversations")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["id"] == "conv-001"
    assert data[1]["id"] == "conv-002"
    assert data[0]["message_count"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: GET "/{conversation_id}" — Valid conversation
# ═══════════════════════════════════════════════════════════════════════════════

def test_get_conversation_valid_id(app_client):
    """GET /api/conversations/{id} with valid ID returns conversation with messages."""
    mock_db = _mock_conversation_db()
    mock_db.get_conversation_with_messages.return_value = {
        "id": "conv-123",
        "title": "Test Conversation",
        "notebook_id": None,
        "document_ids": None,
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "message_count": 2,
        "messages": [
            {
                "id": "msg-1",
                "role": "user",
                "content": "Hello",
                "timestamp": "2024-01-01T00:00:00",
                "is_error": False,
                "sources": None
            },
            {
                "id": "msg-2",
                "role": "assistant",
                "content": "Hi there!",
                "timestamp": "2024-01-01T00:00:01",
                "is_error": False,
                "sources": []
            }
        ]
    }

    with patch("app.routes.conversations.get_conversation_db", return_value=mock_db):
        response = app_client.get("/api/conversations/conv-123")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "conv-123"
    assert "messages" in data
    assert len(data["messages"]) == 2
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][1]["role"] == "assistant"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: GET "/{conversation_id}" — Non-existent ID returns 404
# ═══════════════════════════════════════════════════════════════════════════════

def test_get_conversation_not_found(app_client):
    """GET /api/conversations/{id} with non-existent ID returns 404."""
    mock_db = _mock_conversation_db()
    mock_db.get_conversation_with_messages.return_value = None

    with patch("app.routes.conversations.get_conversation_db", return_value=mock_db):
        response = app_client.get("/api/conversations/nonexistent-id")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: DELETE "/{conversation_id}" — Valid deletion
# ═══════════════════════════════════════════════════════════════════════════════

def test_delete_conversation_valid_id(app_client):
    """DELETE /api/conversations/{id} with valid ID deletes and returns 204."""
    mock_db = _mock_conversation_db()
    mock_db.get_conversation.return_value = {
        "id": "conv-delete",
        "title": "To Delete"
    }

    with patch("app.routes.conversations.get_conversation_db", return_value=mock_db):
        response = app_client.delete("/api/conversations/conv-delete")

    assert response.status_code == 204
    mock_db.delete_conversation.assert_called_once_with("conv-delete")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: DELETE "/{conversation_id}" — Non-existent ID returns 404
# ═══════════════════════════════════════════════════════════════════════════════

def test_delete_conversation_not_found(app_client):
    """DELETE /api/conversations/{id} with non-existent ID returns 404."""
    mock_db = _mock_conversation_db()
    mock_db.get_conversation.return_value = None

    with patch("app.routes.conversations.get_conversation_db", return_value=mock_db):
        response = app_client.delete("/api/conversations/nonexistent-id")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
    mock_db.delete_conversation.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: PATCH "/{conversation_id}/title" — Valid update
# ═══════════════════════════════════════════════════════════════════════════════

def test_update_conversation_title_valid(app_client):
    """PATCH /api/conversations/{id}/title with valid ID updates title."""
    mock_db = _mock_conversation_db()
    mock_db.get_conversation.return_value = {
        "id": "conv-update",
        "title": "Old Title"
    }

    with patch("app.routes.conversations.get_conversation_db", return_value=mock_db):
        response = app_client.patch(
            "/api/conversations/conv-update/title?title=New Title"
        )

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "updated" in data["message"].lower()
    mock_db.update_conversation_title.assert_called_once_with("conv-update", "New Title")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: PATCH "/{conversation_id}/title" — Non-existent ID returns 404
# ═══════════════════════════════════════════════════════════════════════════════

def test_update_conversation_title_not_found(app_client):
    """PATCH /api/conversations/{id}/title with non-existent ID returns 404."""
    mock_db = _mock_conversation_db()
    mock_db.get_conversation.return_value = None

    with patch("app.routes.conversations.get_conversation_db", return_value=mock_db):
        response = app_client.patch(
            "/api/conversations/nonexistent-id/title?title=New Title"
        )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
    mock_db.update_conversation_title.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST "" — Empty request body creates conversation with defaults
# ═══════════════════════════════════════════════════════════════════════════════

def test_create_conversation_empty_body(app_client):
    """POST /api/conversations with empty body uses all defaults."""
    mock_db = _mock_conversation_db()

    with patch("app.routes.conversations.get_conversation_db", return_value=mock_db):
        response = app_client.post("/api/conversations", json={})

    assert response.status_code == 201
    data = response.json()
    assert data["id"] == "conv-001"
    mock_db.create_conversation.assert_called_once_with(
        title="Nuova Conversazione",
        notebook_id=None
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: GET "/{conversation_id}" — Conversation with sources in messages
# ═══════════════════════════════════════════════════════════════════════════════

def test_get_conversation_with_sources(app_client):
    """GET /api/conversations/{id} returns messages with sources when available."""
    mock_db = _mock_conversation_db()
    mock_db.get_conversation_with_messages.return_value = {
        "id": "conv-sources",
        "title": "Conv with Sources",
        "notebook_id": None,
        "document_ids": None,
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "message_count": 2,
        "messages": [
            {
                "id": "msg-1",
                "role": "user",
                "content": "Question",
                "timestamp": "2024-01-01T00:00:00",
                "is_error": False,
                "sources": None
            },
            {
                "id": "msg-2",
                "role": "assistant",
                "content": "Answer with sources",
                "timestamp": "2024-01-01T00:00:01",
                "is_error": False,
                "sources": [
                    {
                        "document": "test.pdf",
                        "page": 1,
                        "chunk_text": "Source text",
                        "relevance_score": 0.95
                    }
                ]
            }
        ]
    }

    with patch("app.routes.conversations.get_conversation_db", return_value=mock_db):
        response = app_client.get("/api/conversations/conv-sources")

    assert response.status_code == 200
    data = response.json()
    assistant_message = data["messages"][1]
    assert len(assistant_message["sources"]) == 1
    assert assistant_message["sources"][0]["document"] == "test.pdf"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: PATCH "/{conversation_id}/title" — Empty title handling
# ═══════════════════════════════════════════════════════════════════════════════

def test_update_conversation_title_empty_string(app_client):
    """PATCH /api/conversations/{id}/title with empty title is accepted (backend validates)."""
    mock_db = _mock_conversation_db()
    mock_db.get_conversation.return_value = {
        "id": "conv-update",
        "title": "Old Title"
    }

    with patch("app.routes.conversations.get_conversation_db", return_value=mock_db):
        response = app_client.patch(
            "/api/conversations/conv-update/title?title="
        )

    # The endpoint doesn't validate empty title, passes it to DB
    assert response.status_code == 200
    mock_db.update_conversation_title.assert_called_once_with("conv-update", "")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POST "" — Conversation has correct response structure
# ═══════════════════════════════════════════════════════════════════════════════

def test_create_conversation_response_structure(app_client):
    """POST /api/conversations returns response matching Conversation model."""
    mock_db = _mock_conversation_db()
    mock_db.get_conversation.return_value = {
        "id": "conv-001",
        "title": "Test",
        "notebook_id": "nb-1",
        "document_ids": ["doc-1", "doc-2"],
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "message_count": 0
    }

    with patch("app.routes.conversations.get_conversation_db", return_value=mock_db):
        response = app_client.post(
            "/api/conversations",
            json={"title": "Test", "notebook_id": "nb-1"}
        )

    assert response.status_code == 201
    data = response.json()
    # Check all expected fields from Conversation model
    assert "id" in data
    assert "title" in data
    assert "notebook_id" in data
    assert "created_at" in data
    assert "updated_at" in data
    assert "message_count" in data
    assert data["notebook_id"] == "nb-1"