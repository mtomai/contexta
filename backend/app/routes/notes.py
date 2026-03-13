from typing import List
from fastapi import APIRouter, HTTPException, status

from app.models.note import Note, NoteCreate
from app.services.note_db import get_note_db

router = APIRouter()


@router.get(
    "/notebooks/{notebook_id}/notes",
    response_model=List[Note],
    tags=["notes"],
)
async def list_notes(notebook_id: str):
    """
    List all saved notes for a notebook, ordered by most recent first.

    Args:
        notebook_id: Notebook UUID

    Returns:
        List of notes
    """
    try:
        db = get_note_db()
        notes = db.list_notes(notebook_id)
        return [Note(**n) for n in notes]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing notes: {str(e)}",
        )


@router.post(
    "/notebooks/{notebook_id}/notes",
    response_model=Note,
    status_code=status.HTTP_201_CREATED,
    tags=["notes"],
)
async def create_note(notebook_id: str, request: NoteCreate):
    """
    Save a new note (pinned AI response) to the notebook.

    Args:
        notebook_id: Notebook UUID
        request: NoteCreate with content

    Returns:
        Created note
    """
    if not request.content or not request.content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Note content cannot be empty",
        )

    try:
        db = get_note_db()
        note_id = db.create_note(notebook_id=notebook_id, content=request.content)
        note = db.get_note(note_id)
        if not note:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error retrieving created note",
            )
        return Note(**note)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating note: {str(e)}",
        )


@router.delete(
    "/notes/{note_id}",
    tags=["notes"],
)
async def delete_note(note_id: str):
    """
    Delete a saved note by ID.

    Args:
        note_id: Note UUID

    Returns:
        Confirmation message
    """
    try:
        db = get_note_db()
        deleted = db.delete_note(note_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Note {note_id} not found",
            )
        return {"message": "Note deleted successfully", "note_id": note_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting note: {str(e)}",
        )
