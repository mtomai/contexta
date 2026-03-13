"""Tests for NoteDB service."""
import os
import tempfile
import pytest

from app.services.note_db import NoteDB


@pytest.fixture()
def db():
    """Create a temporary NoteDB for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        instance = NoteDB(db_path=path)
        # Create notebooks table so FK constraints work
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
            from datetime import datetime
            conn.execute(
                "INSERT INTO notebooks (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                ("nb1", "Test NB", datetime.now(), datetime.now()),
            )
        yield instance
    finally:
        os.unlink(path)


class TestNoteDB:
    def test_create_note(self, db):
        note_id = db.create_note(notebook_id="nb1", content="Test note content")
        assert note_id is not None
        assert isinstance(note_id, str)

    def test_get_note(self, db):
        note_id = db.create_note(notebook_id="nb1", content="My note")
        result = db.get_note(note_id)
        assert result is not None
        assert result["content"] == "My note"
        assert result["notebook_id"] == "nb1"
        assert result["id"] == note_id

    def test_get_note_not_found(self, db):
        assert db.get_note("nonexistent") is None

    def test_list_notes(self, db):
        db.create_note(notebook_id="nb1", content="Note 1")
        db.create_note(notebook_id="nb1", content="Note 2")
        notes = db.list_notes("nb1")
        assert len(notes) == 2

    def test_list_notes_empty(self, db):
        assert db.list_notes("nb1") == []

    def test_list_notes_ordered_desc(self, db):
        id1 = db.create_note(notebook_id="nb1", content="First")
        id2 = db.create_note(notebook_id="nb1", content="Second")
        notes = db.list_notes("nb1")
        # Most recent first
        assert notes[0]["content"] == "Second"
        assert notes[1]["content"] == "First"

    def test_delete_note(self, db):
        note_id = db.create_note(notebook_id="nb1", content="To delete")
        assert db.delete_note(note_id) is True
        assert db.get_note(note_id) is None

    def test_delete_note_not_found(self, db):
        assert db.delete_note("nonexistent") is False

    def test_count_notes(self, db):
        assert db.count_notes("nb1") == 0
        db.create_note(notebook_id="nb1", content="N1")
        db.create_note(notebook_id="nb1", content="N2")
        assert db.count_notes("nb1") == 2

    def test_notes_isolated_by_notebook(self, db):
        # Create second notebook
        with db.get_connection() as conn:
            from datetime import datetime
            conn.execute(
                "INSERT INTO notebooks (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                ("nb2", "NB2", datetime.now(), datetime.now()),
            )
        db.create_note(notebook_id="nb1", content="NB1 note")
        db.create_note(notebook_id="nb2", content="NB2 note")
        assert len(db.list_notes("nb1")) == 1
        assert len(db.list_notes("nb2")) == 1
        assert db.count_notes("nb1") == 1
