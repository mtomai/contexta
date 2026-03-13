"""
Tests for agent_prompts router (app/routes/agent_prompts.py)

Each endpoint is tested for success, not-found, validation errors,
and internal-error scenarios. All external dependencies are mocked.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

os.environ.setdefault("OPENAI_API_KEY", "fake-key-for-tests")

import httpx
from fastapi import FastAPI

from app.routes.agent_prompts import router


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


APP = _make_app()


@pytest.fixture
def async_client():
    transport = httpx.ASGITransport(app=APP)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


FAKE_PROMPT = {
    "id": "prompt-uuid-1",
    "name": "Summarizer",
    "description": "Summarize documents",
    "icon": "Bot",
    "system_prompt": "You are a summarizer.",
    "user_prompt": "Summarize the following.",
    "template_prompt": "{{content}}",
    "variables": [],
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00",
}

FAKE_PROMPT_2 = {
    **FAKE_PROMPT,
    "id": "prompt-uuid-2",
    "name": "Translator",
}

DB_PATH = "app.routes.agent_prompts.get_agent_prompts_db"
EXEC_PATH = "app.routes.agent_prompts.execute_agent_prompt"
EXEC_STREAM_PATH = "app.routes.agent_prompts.execute_agent_prompt_stream"


def _mock_db(**overrides) -> MagicMock:
    db = MagicMock()
    db.get_all_agent_prompts.return_value = overrides.get("all_prompts", [])
    db.get_agent_prompt.return_value = overrides.get("get_prompt", None)
    db.create_agent_prompt.return_value = overrides.get("create_id", "new-id")
    db.update_agent_prompt.return_value = overrides.get("update_ok", True)
    db.delete_agent_prompt.return_value = overrides.get("delete_ok", True)
    return db


# ═══════════════════════════════════════════════════════════════════════════════
# GET /agent-prompts
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_list_agent_prompts_success(async_client):
    """Returns a list of agent prompts with status 200."""
    db = _mock_db(all_prompts=[FAKE_PROMPT, FAKE_PROMPT_2])
    with patch(DB_PATH, return_value=db):
        resp = await async_client.get("/agent-prompts")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["name"] == "Summarizer"
    assert data[1]["name"] == "Translator"


@pytest.mark.asyncio
async def test_list_agent_prompts_empty(async_client):
    """Returns an empty list when no prompts exist."""
    db = _mock_db(all_prompts=[])
    with patch(DB_PATH, return_value=db):
        resp = await async_client.get("/agent-prompts")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_agent_prompts_internal_error(async_client):
    """Returns 500 when the DB raises an exception."""
    db = _mock_db()
    db.get_all_agent_prompts.side_effect = RuntimeError("DB down")
    with patch(DB_PATH, return_value=db):
        resp = await async_client.get("/agent-prompts")
    assert resp.status_code == 500
    assert "Error listing agent prompts" in resp.json()["detail"]


# ═══════════════════════════════════════════════════════════════════════════════
# POST /agent-prompts
# ═══════════════════════════════════════════════════════════════════════════════

CREATE_BODY = {
    "name": "New Prompt",
    "description": "desc",
    "icon": "Star",
    "system_prompt": "sys",
    "user_prompt": "usr",
    "template_prompt": "tmpl",
    "variables": [],
}


@pytest.mark.asyncio
async def test_create_agent_prompt_success(async_client):
    """Creates a prompt and returns 201 with the created object."""
    db = _mock_db(create_id="new-id", get_prompt=FAKE_PROMPT)
    with patch(DB_PATH, return_value=db):
        resp = await async_client.post("/agent-prompts", json=CREATE_BODY)
    assert resp.status_code == 201
    assert resp.json()["id"] == FAKE_PROMPT["id"]
    db.create_agent_prompt.assert_called_once()


@pytest.mark.asyncio
async def test_create_agent_prompt_default_icon(async_client):
    """Uses 'Bot' as default icon when icon is None."""
    body = {**CREATE_BODY, "icon": None}
    db = _mock_db(create_id="new-id", get_prompt=FAKE_PROMPT)
    with patch(DB_PATH, return_value=db):
        resp = await async_client.post("/agent-prompts", json=body)
    assert resp.status_code == 201
    call_kwargs = db.create_agent_prompt.call_args
    assert call_kwargs.kwargs.get("icon") == "Bot" or call_kwargs[1].get("icon") == "Bot"


@pytest.mark.asyncio
async def test_create_agent_prompt_retrieval_failure(async_client):
    """Returns 500 when the prompt cannot be retrieved after creation."""
    db = _mock_db(create_id="new-id", get_prompt=None)
    with patch(DB_PATH, return_value=db):
        resp = await async_client.post("/agent-prompts", json=CREATE_BODY)
    assert resp.status_code == 500
    assert "Error retrieving created agent prompt" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_create_agent_prompt_internal_error(async_client):
    """Returns 500 when create raises an exception."""
    db = _mock_db()
    db.create_agent_prompt.side_effect = RuntimeError("insert failed")
    with patch(DB_PATH, return_value=db):
        resp = await async_client.post("/agent-prompts", json=CREATE_BODY)
    assert resp.status_code == 500
    assert "Error creating agent prompt" in resp.json()["detail"]


# ═══════════════════════════════════════════════════════════════════════════════
# GET /agent-prompts/{prompt_id}
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_get_agent_prompt_success(async_client):
    """Returns the prompt with status 200."""
    db = _mock_db(get_prompt=FAKE_PROMPT)
    with patch(DB_PATH, return_value=db):
        resp = await async_client.get("/agent-prompts/prompt-uuid-1")
    assert resp.status_code == 200
    assert resp.json()["id"] == "prompt-uuid-1"


@pytest.mark.asyncio
async def test_get_agent_prompt_not_found(async_client):
    """Returns 404 when the prompt does not exist."""
    db = _mock_db(get_prompt=None)
    with patch(DB_PATH, return_value=db):
        resp = await async_client.get("/agent-prompts/nonexistent")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_get_agent_prompt_internal_error(async_client):
    """Returns 500 when the DB raises an exception."""
    db = _mock_db()
    db.get_agent_prompt.side_effect = RuntimeError("read error")
    with patch(DB_PATH, return_value=db):
        resp = await async_client.get("/agent-prompts/prompt-uuid-1")
    assert resp.status_code == 500
    assert "Error retrieving agent prompt" in resp.json()["detail"]


# ═══════════════════════════════════════════════════════════════════════════════
# PUT /agent-prompts/{prompt_id}
# ═══════════════════════════════════════════════════════════════════════════════

UPDATE_BODY = {"name": "Updated Name", "description": "Updated desc"}


@pytest.mark.asyncio
async def test_update_agent_prompt_success(async_client):
    """Updates the prompt and returns 200 with updated data."""
    updated = {**FAKE_PROMPT, "name": "Updated Name"}
    db = _mock_db(get_prompt=FAKE_PROMPT, update_ok=True)
    # First call: existence check, second call: return updated
    db.get_agent_prompt.side_effect = [FAKE_PROMPT, updated]
    with patch(DB_PATH, return_value=db):
        resp = await async_client.put("/agent-prompts/prompt-uuid-1", json=UPDATE_BODY)
    assert resp.status_code == 200
    assert resp.json()["name"] == "Updated Name"


@pytest.mark.asyncio
async def test_update_agent_prompt_not_found(async_client):
    """Returns 404 when the prompt doesn't exist."""
    db = _mock_db(get_prompt=None)
    with patch(DB_PATH, return_value=db):
        resp = await async_client.put("/agent-prompts/nonexistent", json=UPDATE_BODY)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_agent_prompt_update_failure(async_client):
    """Returns 500 when db.update_agent_prompt returns False."""
    db = _mock_db(get_prompt=FAKE_PROMPT, update_ok=False)
    with patch(DB_PATH, return_value=db):
        resp = await async_client.put("/agent-prompts/prompt-uuid-1", json=UPDATE_BODY)
    assert resp.status_code == 500
    assert "Error updating agent prompt" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_update_agent_prompt_partial_fields(async_client):
    """Only specified fields are passed to db.update_agent_prompt."""
    db = _mock_db(get_prompt=FAKE_PROMPT, update_ok=True)
    db.get_agent_prompt.side_effect = [FAKE_PROMPT, FAKE_PROMPT]
    partial = {"name": "Only Name"}
    with patch(DB_PATH, return_value=db):
        resp = await async_client.put("/agent-prompts/prompt-uuid-1", json=partial)
    assert resp.status_code == 200
    call_kwargs = db.update_agent_prompt.call_args[1]
    assert "name" in call_kwargs
    assert "description" not in call_kwargs


@pytest.mark.asyncio
async def test_update_agent_prompt_internal_error(async_client):
    """Returns 500 when the DB raises an exception."""
    db = _mock_db()
    db.get_agent_prompt.side_effect = RuntimeError("boom")
    with patch(DB_PATH, return_value=db):
        resp = await async_client.put("/agent-prompts/prompt-uuid-1", json=UPDATE_BODY)
    assert resp.status_code == 500
    assert "Error updating agent prompt" in resp.json()["detail"]


# ═══════════════════════════════════════════════════════════════════════════════
# DELETE /agent-prompts/{prompt_id}
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_delete_agent_prompt_success(async_client):
    """Deletes the prompt and returns 204."""
    db = _mock_db(get_prompt=FAKE_PROMPT, delete_ok=True)
    with patch(DB_PATH, return_value=db):
        resp = await async_client.delete("/agent-prompts/prompt-uuid-1")
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_delete_agent_prompt_not_found(async_client):
    """Returns 404 when the prompt doesn't exist."""
    db = _mock_db(get_prompt=None)
    with patch(DB_PATH, return_value=db):
        resp = await async_client.delete("/agent-prompts/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_agent_prompt_delete_failure(async_client):
    """Returns 500 when db.delete_agent_prompt returns False."""
    db = _mock_db(get_prompt=FAKE_PROMPT, delete_ok=False)
    with patch(DB_PATH, return_value=db):
        resp = await async_client.delete("/agent-prompts/prompt-uuid-1")
    assert resp.status_code == 500
    assert "Error deleting agent prompt" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_delete_agent_prompt_internal_error(async_client):
    """Returns 500 when the DB raises an exception."""
    db = _mock_db()
    db.get_agent_prompt.side_effect = RuntimeError("boom")
    with patch(DB_PATH, return_value=db):
        resp = await async_client.delete("/agent-prompts/prompt-uuid-1")
    assert resp.status_code == 500
    assert "Error deleting agent prompt" in resp.json()["detail"]


# ═══════════════════════════════════════════════════════════════════════════════
# POST /agent-prompts/{prompt_id}/execute
# ═══════════════════════════════════════════════════════════════════════════════

EXECUTE_BODY = {
    "document_ids": ["doc-1", "doc-2"],
    "notebook_id": "nb-1",
    "variable_values": {"key": "value"},
}


@pytest.mark.asyncio
async def test_execute_agent_prompt_success(async_client):
    """Executes and returns conversation_id and title."""
    db = _mock_db(get_prompt=FAKE_PROMPT)
    exec_result = {"conversation_id": "conv-1", "title": "Summary"}
    with patch(DB_PATH, return_value=db), \
         patch(EXEC_PATH, new_callable=AsyncMock, return_value=exec_result):
        resp = await async_client.post("/agent-prompts/prompt-uuid-1/execute", json=EXECUTE_BODY)
    assert resp.status_code == 200
    data = resp.json()
    assert data["conversation_id"] == "conv-1"
    assert data["title"] == "Summary"


@pytest.mark.asyncio
async def test_execute_agent_prompt_not_found(async_client):
    """Returns 404 when the prompt doesn't exist."""
    db = _mock_db(get_prompt=None)
    with patch(DB_PATH, return_value=db):
        resp = await async_client.post("/agent-prompts/nonexistent/execute", json=EXECUTE_BODY)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_execute_agent_prompt_no_documents(async_client):
    """Returns 400 when document_ids is empty."""
    db = _mock_db(get_prompt=FAKE_PROMPT)
    body = {**EXECUTE_BODY, "document_ids": []}
    with patch(DB_PATH, return_value=db):
        resp = await async_client.post("/agent-prompts/prompt-uuid-1/execute", json=body)
    assert resp.status_code == 400
    assert "No documents selected" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_execute_agent_prompt_internal_error(async_client):
    """Returns 500 when execute_agent_prompt raises an exception."""
    db = _mock_db(get_prompt=FAKE_PROMPT)
    with patch(DB_PATH, return_value=db), \
         patch(EXEC_PATH, new_callable=AsyncMock, side_effect=RuntimeError("LLM error")):
        resp = await async_client.post("/agent-prompts/prompt-uuid-1/execute", json=EXECUTE_BODY)
    assert resp.status_code == 500
    assert "Error executing agent prompt" in resp.json()["detail"]


# ═══════════════════════════════════════════════════════════════════════════════
# POST /agent-prompts/{prompt_id}/execute_stream
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_execute_stream_success(async_client):
    """Returns a streaming response with status 200."""
    db = _mock_db(get_prompt=FAKE_PROMPT)

    async def fake_stream(**kwargs):
        yield "data: token1\n\n"
        yield "data: token2\n\n"

    with patch(DB_PATH, return_value=db), \
         patch(EXEC_STREAM_PATH, return_value=fake_stream()):
        resp = await async_client.post(
            "/agent-prompts/prompt-uuid-1/execute_stream", json=EXECUTE_BODY
        )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "token1" in resp.text
    assert "token2" in resp.text


@pytest.mark.asyncio
async def test_execute_stream_not_found(async_client):
    """Returns 404 when the prompt doesn't exist."""
    db = _mock_db(get_prompt=None)
    with patch(DB_PATH, return_value=db):
        resp = await async_client.post(
            "/agent-prompts/nonexistent/execute_stream", json=EXECUTE_BODY
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_execute_stream_no_documents(async_client):
    """Returns 400 when document_ids is empty."""
    db = _mock_db(get_prompt=FAKE_PROMPT)
    body = {**EXECUTE_BODY, "document_ids": []}
    with patch(DB_PATH, return_value=db):
        resp = await async_client.post(
            "/agent-prompts/prompt-uuid-1/execute_stream", json=body
        )
    assert resp.status_code == 400
    assert "No documents selected" in resp.json()["detail"]