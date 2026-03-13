"""Tests for CustomPromptsDB service."""
import os
import tempfile
import pytest

from app.services.custom_prompts_db import CustomPromptsDB


@pytest.fixture()
def db():
    """Create a temporary CustomPromptsDB for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        instance = CustomPromptsDB(db_path=path)
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


class TestCustomPromptsDB:
    def test_create_custom_prompt(self, db):
        prompt_id = db.create_custom_prompt(
            notebook_id="nb1",
            name="Riassunto",
            prompt_template="Riassumi il documento {{context}}",
        )
        assert prompt_id is not None
        assert isinstance(prompt_id, str)

    def test_get_custom_prompts(self, db):
        db.create_custom_prompt(
            notebook_id="nb1", name="P1", prompt_template="t1"
        )
        db.create_custom_prompt(
            notebook_id="nb1", name="P2", prompt_template="t2"
        )
        results = db.get_custom_prompts("nb1")
        assert len(results) == 2
        assert results[0]["name"] == "P1"

    def test_get_custom_prompts_empty(self, db):
        results = db.get_custom_prompts("nb1")
        assert results == []

    def test_get_custom_prompt(self, db):
        pid = db.create_custom_prompt(
            notebook_id="nb1",
            name="Test",
            prompt_template="tmpl",
            icon="Wand",
        )
        result = db.get_custom_prompt(pid)
        assert result is not None
        assert result["name"] == "Test"
        assert result["prompt_template"] == "tmpl"
        assert result["icon"] == "Wand"
        assert result["notebook_id"] == "nb1"

    def test_get_custom_prompt_not_found(self, db):
        assert db.get_custom_prompt("nonexistent") is None

    def test_update_custom_prompt(self, db):
        pid = db.create_custom_prompt(
            notebook_id="nb1", name="Old", prompt_template="old"
        )
        updated = db.update_custom_prompt(pid, name="New", prompt_template="new")
        assert updated is True
        result = db.get_custom_prompt(pid)
        assert result["name"] == "New"
        assert result["prompt_template"] == "new"

    def test_update_custom_prompt_partial(self, db):
        pid = db.create_custom_prompt(
            notebook_id="nb1", name="Name", prompt_template="tmpl"
        )
        db.update_custom_prompt(pid, icon="Star")
        result = db.get_custom_prompt(pid)
        assert result["icon"] == "Star"
        assert result["name"] == "Name"  # unchanged

    def test_update_custom_prompt_not_found(self, db):
        assert db.update_custom_prompt("nonexistent", name="X") is False

    def test_update_custom_prompt_no_fields(self, db):
        pid = db.create_custom_prompt(
            notebook_id="nb1", name="T", prompt_template="t"
        )
        assert db.update_custom_prompt(pid) is False

    def test_delete_custom_prompt(self, db):
        pid = db.create_custom_prompt(
            notebook_id="nb1", name="T", prompt_template="t"
        )
        assert db.delete_custom_prompt(pid) is True
        assert db.get_custom_prompt(pid) is None

    def test_delete_custom_prompt_not_found(self, db):
        assert db.delete_custom_prompt("nonexistent") is False

    def test_default_icon(self, db):
        pid = db.create_custom_prompt(
            notebook_id="nb1", name="T", prompt_template="t"
        )
        result = db.get_custom_prompt(pid)
        assert result["icon"] == "Sparkles"
