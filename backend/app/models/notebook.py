from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class NotebookCreate(BaseModel):
    """Model for creating a new notebook."""
    name: str
    description: Optional[str] = None


class NotebookUpdate(BaseModel):
    """Model for updating a notebook."""
    name: Optional[str] = None
    description: Optional[str] = None


class Notebook(BaseModel):
    """Model for notebook metadata."""
    id: str
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime


class NotebookWithStats(Notebook):
    """Model for notebook with statistics."""
    document_count: int
    conversation_count: int
