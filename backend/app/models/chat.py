from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str
    conversation_id: Optional[str] = None  # Optional conversation context


class Source(BaseModel):
    """Model for a source citation."""
    document: str
    page: int
    chunk_index: int  # Index of the chunk within the document
    chunk_text: str
    relevance_score: float


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    sources: list[Source]
    conversation_id: str  # ID of conversation where message was saved
