"""Tests for NotebookDB service."""
import os
import tempfile
import pytest

from app.services.notebook_db import NotebookDB


@pytest.fixture()
def db():
    """Create a temporary NotebookDB for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        yield NotebookDB(db_path=path)
    finally:
        os.unlink(path)


class TestNotebookDB:
    def test_create_notebook(self, db):
        nb_id = db.create_notebook(name="Test Notebook")
        assert nb_id is not None
        assert isinstance(nb_id, str)

    def test_create_notebook_with_description(self, db):
        nb_id = db.create_notebook(name="NB", description="A description")
        result = db.get_notebook(nb_id)
        assert result["description"] == "A description"

    def test_get_notebook(self, db):
        nb_id = db.create_notebook(name="My NB")
        result = db.get_notebook(nb_id)
        assert result is not None
        assert result["name"] == "My NB"
        assert result["id"] == nb_id

    def test_get_notebook_not_found(self, db):
        assert db.get_notebook("nonexistent") is None

    def test_list_notebooks(self, db):
        db.create_notebook(name="NB1")
        db.create_notebook(name="NB2")
        notebooks = db.list_notebooks()
        assert len(notebooks) == 2

    def test_list_notebooks_empty(self, db):
        assert db.list_notebooks() == []

    def test_list_notebooks_ordered_by_updated(self, db):
        id1 = db.create_notebook(name="First")
        id2 = db.create_notebook(name="Second")
        notebooks = db.list_notebooks()
        # Most recently updated first
        assert notebooks[0]["name"] == "Second"

    def test_update_notebook_name(self, db):
        nb_id = db.create_notebook(name="Old")
        result = db.update_notebook(nb_id, name="New")
        assert result is True
        nb = db.get_notebook(nb_id)
        assert nb["name"] == "New"

    def test_update_notebook_description(self, db):
        nb_id = db.create_notebook(name="NB")
        db.update_notebook(nb_id, description="Updated desc")
        nb = db.get_notebook(nb_id)
        assert nb["description"] == "Updated desc"

    def test_update_notebook_not_found(self, db):
        assert db.update_notebook("nonexistent", name="X") is False

    def test_update_notebook_no_fields(self, db):
        nb_id = db.create_notebook(name="NB")
        result = db.update_notebook(nb_id)
        assert result is True  # Returns True even with no updates

    def test_delete_notebook(self, db):
        nb_id = db.create_notebook(name="To Delete")
        assert db.delete_notebook(nb_id) is True
        assert db.get_notebook(nb_id) is None

    def test_delete_notebook_not_found(self, db):
        assert db.delete_notebook("nonexistent") is False

    def test_notebook_exists(self, db):
        nb_id = db.create_notebook(name="NB")
        assert db.notebook_exists(nb_id) is True
        assert db.notebook_exists("nonexistent") is False

    def test_get_notebook_with_stats(self, db):
        nb_id = db.create_notebook(name="NB")
        # Need conversations table for stats
        with db.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    notebook_id TEXT
                )
            """)
        result = db.get_notebook_with_stats(nb_id)
        assert result is not None
        assert result["conversation_count"] == 0
        assert result["document_count"] == 0

    def test_get_notebook_with_stats_not_found(self, db):
        assert db.get_notebook_with_stats("nonexistent") is None
