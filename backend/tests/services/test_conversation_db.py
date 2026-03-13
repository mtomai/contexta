"""Tests for ConversationDB service."""
import os
import tempfile
import pytest

from app.services.conversation_db import ConversationDB


@pytest.fixture()
def db():
    """Create a temporary ConversationDB for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        instance = ConversationDB(db_path=path)
        # Create notebooks table so foreign keys work
        with instance.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notebooks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """)
        yield instance
    finally:
        os.unlink(path)


class TestConversationDB:
    def test_create_conversation_default_title(self, db):
        conv_id = db.create_conversation()
        assert conv_id is not None
        conv = db.get_conversation(conv_id)
        assert conv["title"] == "Nuova Conversazione"

    def test_create_conversation_custom_title(self, db):
        conv_id = db.create_conversation(title="My Conv")
        conv = db.get_conversation(conv_id)
        assert conv["title"] == "My Conv"

    def test_create_conversation_with_notebook_id(self, db):
        # First create a notebook
        with db.get_connection() as conn:
            from datetime import datetime
            conn.execute(
                "INSERT INTO notebooks (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                ("nb1", "Test NB", datetime.now(), datetime.now()),
            )
        conv_id = db.create_conversation(title="T", notebook_id="nb1")
        conv = db.get_conversation(conv_id)
        assert conv["notebook_id"] == "nb1"

    def test_create_conversation_with_document_ids(self, db):
        conv_id = db.create_conversation(
            title="T", document_ids=["doc1", "doc2"]
        )
        conv = db.get_conversation(conv_id)
        assert conv["document_ids"] == ["doc1", "doc2"]

    def test_get_conversation_not_found(self, db):
        assert db.get_conversation("nonexistent") is None

    def test_list_conversations(self, db):
        db.create_conversation(title="C1")
        db.create_conversation(title="C2")
        convs = db.list_conversations()
        assert len(convs) == 2

    def test_list_conversations_empty(self, db):
        assert db.list_conversations() == []

    def test_list_conversations_by_notebook(self, db):
        with db.get_connection() as conn:
            from datetime import datetime
            conn.execute(
                "INSERT INTO notebooks (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                ("nb1", "NB1", datetime.now(), datetime.now()),
            )
        db.create_conversation(title="C1", notebook_id="nb1")
        db.create_conversation(title="C2")
        result = db.list_conversations(notebook_id="nb1")
        assert len(result) == 1
        assert result[0]["title"] == "C1"

    def test_update_conversation_title(self, db):
        conv_id = db.create_conversation(title="Old")
        db.update_conversation_title(conv_id, "New")
        conv = db.get_conversation(conv_id)
        assert conv["title"] == "New"

    def test_delete_conversation(self, db):
        conv_id = db.create_conversation(title="ToDelete")
        db.delete_conversation(conv_id)
        assert db.get_conversation(conv_id) is None

    def test_add_message(self, db):
        conv_id = db.create_conversation(title="T")
        msg_id = db.add_message(conv_id, role="user", content="Hello")
        assert msg_id is not None

        messages = db.get_messages(conv_id)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_add_message_updates_count(self, db):
        conv_id = db.create_conversation(title="T")
        db.add_message(conv_id, role="user", content="msg1")
        db.add_message(conv_id, role="assistant", content="msg2")
        conv = db.get_conversation(conv_id)
        assert conv["message_count"] == 2

    def test_add_message_with_sources(self, db):
        conv_id = db.create_conversation(title="T")
        sources = [
            {"document": "test.pdf", "page": 1, "chunk_text": "chunk", "relevance_score": 0.9}
        ]
        msg_id = db.add_message(
            conv_id, role="assistant", content="Response", sources=sources
        )
        messages = db.get_messages(conv_id)
        assert len(messages[0]["sources"]) == 1
        assert messages[0]["sources"][0]["document"] == "test.pdf"

    def test_add_message_error_flag(self, db):
        conv_id = db.create_conversation(title="T")
        db.add_message(conv_id, role="assistant", content="err", is_error=True)
        messages = db.get_messages(conv_id)
        assert messages[0]["is_error"] == 1

    def test_get_messages_ordered(self, db):
        conv_id = db.create_conversation(title="T")
        db.add_message(conv_id, role="user", content="first")
        db.add_message(conv_id, role="assistant", content="second")
        messages = db.get_messages(conv_id)
        assert messages[0]["content"] == "first"
        assert messages[1]["content"] == "second"

    def test_get_conversation_with_messages(self, db):
        conv_id = db.create_conversation(title="T")
        db.add_message(conv_id, role="user", content="Hello")
        result = db.get_conversation_with_messages(conv_id)
        assert result is not None
        assert result["id"] == conv_id
        assert len(result["messages"]) == 1

    def test_get_conversation_with_messages_not_found(self, db):
        assert db.get_conversation_with_messages("nonexistent") is None

    def test_list_conversations_by_notebook_alias(self, db):
        with db.get_connection() as conn:
            from datetime import datetime
            conn.execute(
                "INSERT INTO notebooks (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                ("nb1", "NB", datetime.now(), datetime.now()),
            )
        db.create_conversation(title="C1", notebook_id="nb1")
        result = db.list_conversations_by_notebook("nb1")
        assert len(result) == 1

    def test_cascade_delete_messages(self, db):
        conv_id = db.create_conversation(title="T")
        db.add_message(conv_id, role="user", content="msg")
        db.delete_conversation(conv_id)
        messages = db.get_messages(conv_id)
        assert messages == []
