from fastapi import APIRouter, HTTPException, status
from typing import List

from app.models.conversation import (
    ConversationCreate,
    Conversation,
    ConversationWithMessages
)
from app.services.conversation_db import get_conversation_db

router = APIRouter()


@router.post("", response_model=Conversation, status_code=status.HTTP_201_CREATED)
async def create_conversation(request: ConversationCreate):
    """
    Create a new conversation.

    Args:
        request: ConversationCreate with optional title and notebook_id

    Returns:
        Created conversation metadata
    """
    db = get_conversation_db()

    # Create conversation with provided or default title and notebook_id
    conversation_id = db.create_conversation(
        title=request.title or "Nuova Conversazione",
        notebook_id=request.notebook_id
    )

    # Retrieve and return created conversation
    conversation = db.get_conversation(conversation_id)
    return conversation


@router.get("", response_model=List[Conversation])
async def list_conversations():
    """
    List all conversations ordered by recency (most recent first).

    Returns:
        List of conversation metadata
    """
    db = get_conversation_db()
    conversations = db.list_conversations()
    return conversations


@router.get("/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation(conversation_id: str):
    """
    Get conversation with full message history.

    Args:
        conversation_id: Conversation UUID

    Returns:
        Conversation metadata with all messages and sources

    Raises:
        HTTPException: 404 if conversation not found
    """
    db = get_conversation_db()
    conversation_data = db.get_conversation_with_messages(conversation_id)

    if not conversation_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found"
        )

    return conversation_data


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation and all its messages.

    Args:
        conversation_id: Conversation UUID

    Raises:
        HTTPException: 404 if conversation not found
    """
    db = get_conversation_db()

    # Check if conversation exists
    if not db.get_conversation(conversation_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found"
        )

    # Delete conversation (messages deleted via CASCADE)
    db.delete_conversation(conversation_id)
    return None


@router.patch("/{conversation_id}/title")
async def update_conversation_title(conversation_id: str, title: str):
    """
    Update conversation title.

    Args:
        conversation_id: Conversation UUID
        title: New title

    Returns:
        Success message

    Raises:
        HTTPException: 404 if conversation not found
    """
    db = get_conversation_db()

    # Check if conversation exists
    if not db.get_conversation(conversation_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found"
        )

    # Update title
    db.update_conversation_title(conversation_id, title)
    return {"message": "Title updated successfully"}
