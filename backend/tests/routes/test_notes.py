import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, status

from app.routes.notes import list_notes, create_note, delete_note
from app.routes import notes as notes_module


class DummyNote:
    """Minimal replacement for Note pydantic model in pure unit tests."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _request(content):
    """Create a lightweight request-like object with `.content`."""
    return SimpleNamespace(content=content)


def test_list_notes_success_returns_note_objects():
    mock_db = MagicMock()
    mock_db.list_notes.return_value = [
        {"id": "n1", "notebook_id": "nb1", "content": "First"},
        {"id": "n2", "notebook_id": "nb1", "content": "Second"},
    ]

    with patch.object(notes_module, "get_note_db", return_value=mock_db), patch.object(
        notes_module, "Note", DummyNote
    ):
        result = asyncio.run(list_notes("nb1"))

    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], DummyNote)
    assert result[0].id == "n1"
    assert result[1].content == "Second"
    mock_db.list_notes.assert_called_once_with("nb1")


def test_list_notes_db_error_raises_http_500():
    mock_db = MagicMock()
    mock_db.list_notes.side_effect = Exception("db down")

    with patch.object(notes_module, "get_note_db", return_value=mock_db):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(list_notes("nb1"))

    assert exc.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc.value.detail == "Error listing notes: db down"


@pytest.mark.parametrize("bad_content", [None, "", "   "])
def test_create_note_rejects_empty_content(bad_content):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(create_note("nb1", _request(bad_content)))

    assert exc.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc.value.detail == "Note content cannot be empty"


def test_create_note_success_returns_note_object():
    mock_db = MagicMock()
    mock_db.create_note.return_value = "note-123"
    mock_db.get_note.return_value = {
        "id": "note-123",
        "notebook_id": "nb1",
        "content": "Saved note",
    }

    with patch.object(notes_module, "get_note_db", return_value=mock_db), patch.object(
        notes_module, "Note", DummyNote
    ):
        result = asyncio.run(create_note("nb1", _request("Saved note")))

    assert isinstance(result, DummyNote)
    assert result.id == "note-123"
    assert result.notebook_id == "nb1"
    assert result.content == "Saved note"

    mock_db.create_note.assert_called_once_with(notebook_id="nb1", content="Saved note")
    mock_db.get_note.assert_called_once_with("note-123")


def test_create_note_when_created_note_not_found_raises_http_500():
    mock_db = MagicMock()
    mock_db.create_note.return_value = "note-123"
    mock_db.get_note.return_value = None

    with patch.object(notes_module, "get_note_db", return_value=mock_db):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(create_note("nb1", _request("hello")))

    assert exc.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc.value.detail == "Error retrieving created note"


def test_create_note_wraps_generic_exception_as_http_500():
    mock_db = MagicMock()
    mock_db.create_note.side_effect = Exception("write failed")

    with patch.object(notes_module, "get_note_db", return_value=mock_db):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(create_note("nb1", _request("hello")))

    assert exc.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc.value.detail == "Error creating note: write failed"


def test_create_note_propagates_http_exception():
    mock_db = MagicMock()
    mock_db.create_note.side_effect = HTTPException(
        status_code=status.HTTP_409_CONFLICT, detail="conflict"
    )

    with patch.object(notes_module, "get_note_db", return_value=mock_db):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(create_note("nb1", _request("hello")))

    assert exc.value.status_code == status.HTTP_409_CONFLICT
    assert exc.value.detail == "conflict"


def test_delete_note_success_returns_confirmation_message():
    mock_db = MagicMock()
    mock_db.delete_note.return_value = True

    with patch.object(notes_module, "get_note_db", return_value=mock_db):
        result = asyncio.run(delete_note("note-1"))

    assert result == {"message": "Note deleted successfully", "note_id": "note-1"}
    mock_db.delete_note.assert_called_once_with("note-1")


def test_delete_note_not_found_raises_http_404():
    mock_db = MagicMock()
    mock_db.delete_note.return_value = False

    with patch.object(notes_module, "get_note_db", return_value=mock_db):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(delete_note("note-missing"))

    assert exc.value.status_code == status.HTTP_404_NOT_FOUND
    assert exc.value.detail == "Note note-missing not found"


def test_delete_note_wraps_generic_exception_as_http_500():
    mock_db = MagicMock()
    mock_db.delete_note.side_effect = Exception("delete failed")

    with patch.object(notes_module, "get_note_db", return_value=mock_db):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(delete_note("note-1"))

    assert exc.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc.value.detail == "Error deleting note: delete failed"


def test_delete_note_propagates_http_exception():
    mock_db = MagicMock()
    mock_db.delete_note.side_effect = HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="forbidden"
    )

    with patch.object(notes_module, "get_note_db", return_value=mock_db):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(delete_note("note-1"))

    assert exc.value.status_code == status.HTTP_403_FORBIDDEN
    assert exc.value.detail == "forbidden"