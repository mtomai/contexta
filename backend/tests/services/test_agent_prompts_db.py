"""Tests for AgentPromptsDB service."""
import os
import tempfile
import pytest

from app.services.agent_prompts_db import AgentPromptsDB


@pytest.fixture()
def db():
    """Create a temporary AgentPromptsDB for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        yield AgentPromptsDB(db_path=path)
    finally:
        os.unlink(path)


class TestAgentPromptsDB:
    def test_create_agent_prompt(self, db):
        prompt_id = db.create_agent_prompt(
            name="Test Agent",
            system_prompt="You are a test agent.",
            user_prompt="Analyze {{context}}",
        )
        assert prompt_id is not None
        assert isinstance(prompt_id, str)

    def test_get_agent_prompt(self, db):
        prompt_id = db.create_agent_prompt(
            name="Test Agent",
            system_prompt="sys",
            user_prompt="usr",
            description="desc",
            icon="Zap",
            template_prompt="tmpl",
            variables=[{"name": "var1", "label": "Var 1"}],
        )
        result = db.get_agent_prompt(prompt_id)
        assert result is not None
        assert result["name"] == "Test Agent"
        assert result["system_prompt"] == "sys"
        assert result["user_prompt"] == "usr"
        assert result["description"] == "desc"
        assert result["icon"] == "Zap"
        assert result["template_prompt"] == "tmpl"
        assert result["variables"] == [{"name": "var1", "label": "Var 1"}]

    def test_get_agent_prompt_not_found(self, db):
        assert db.get_agent_prompt("nonexistent-id") is None

    def test_get_all_agent_prompts(self, db):
        db.create_agent_prompt(name="A1", system_prompt="s1", user_prompt="u1")
        db.create_agent_prompt(name="A2", system_prompt="s2", user_prompt="u2")
        results = db.get_all_agent_prompts()
        assert len(results) == 2
        assert results[0]["name"] == "A1"
        assert results[1]["name"] == "A2"

    def test_get_all_agent_prompts_empty(self, db):
        results = db.get_all_agent_prompts()
        assert results == []

    def test_update_agent_prompt(self, db):
        prompt_id = db.create_agent_prompt(
            name="Original",
            system_prompt="sys",
            user_prompt="usr",
        )
        updated = db.update_agent_prompt(prompt_id, name="Updated", icon="Star")
        assert updated is True

        result = db.get_agent_prompt(prompt_id)
        assert result["name"] == "Updated"
        assert result["icon"] == "Star"

    def test_update_agent_prompt_not_found(self, db):
        result = db.update_agent_prompt("nonexistent-id", name="X")
        assert result is False

    def test_update_agent_prompt_no_fields(self, db):
        prompt_id = db.create_agent_prompt(
            name="Test", system_prompt="s", user_prompt="u",
        )
        result = db.update_agent_prompt(prompt_id)
        assert result is False

    def test_update_agent_prompt_variables(self, db):
        prompt_id = db.create_agent_prompt(
            name="Test", system_prompt="s", user_prompt="u",
        )
        db.update_agent_prompt(
            prompt_id,
            variables=[{"name": "v1", "label": "V1"}],
        )
        result = db.get_agent_prompt(prompt_id)
        assert result["variables"] == [{"name": "v1", "label": "V1"}]

    def test_delete_agent_prompt(self, db):
        prompt_id = db.create_agent_prompt(
            name="ToDelete", system_prompt="s", user_prompt="u",
        )
        assert db.delete_agent_prompt(prompt_id) is True
        assert db.get_agent_prompt(prompt_id) is None

    def test_delete_agent_prompt_not_found(self, db):
        assert db.delete_agent_prompt("nonexistent-id") is False

    def test_default_icon(self, db):
        prompt_id = db.create_agent_prompt(
            name="Test", system_prompt="s", user_prompt="u",
        )
        result = db.get_agent_prompt(prompt_id)
        assert result["icon"] == "Bot"

    def test_default_variables_empty_list(self, db):
        prompt_id = db.create_agent_prompt(
            name="Test", system_prompt="s", user_prompt="u",
        )
        result = db.get_agent_prompt(prompt_id)
        assert result["variables"] == []
