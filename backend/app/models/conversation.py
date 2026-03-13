from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class ConversationCreate(BaseModel):
    """Model for creating a new conversation."""
    title: Optional[str] = None  # Auto-generated if not provided
    notebook_id: Optional[str] = None  # Optional notebook to associate with


class Conversation(BaseModel):
    """Model for conversation metadata."""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    notebook_id: Optional[str] = None  # Optional notebook association


class MessageSource(BaseModel):
    """Model for message source citation."""
    document: str
    page: int
    chunk_text: str
    relevance_score: float


class Message(BaseModel):
    """Model for a complete message."""
    id: str
    conversation_id: Optional[str] = None
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    is_error: bool
    sources: Optional[List[MessageSource]] = []


class ConversationWithMessages(BaseModel):
    """Model for conversation with full message history."""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    notebook_id: Optional[str] = None
    messages: List[Message] = []
