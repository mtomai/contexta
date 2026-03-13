from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class NoteCreate(BaseModel):
    """Model for creating a new note (pinned AI response)."""
    content: str


class Note(BaseModel):
    """Model for a saved note."""
    id: str
    notebook_id: str
    content: str
    created_at: datetime


class NoteUpdate(BaseModel):
    """Model for updating a note."""
    content: Optional[str] = None
