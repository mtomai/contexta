import json
import logging
from typing import List, Dict

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.models.chat import ChatRequest, ChatResponse
from app.services.chat import generate_response, generate_refinement_response
from app.services.chat_streaming import generate_response_stream
from app.services.conversation_db import get_conversation_db
from app.services.title_generator import generate_conversation_title
from app.services.embedding_cache import get_embedding_cache

logger = logging.getLogger(__name__)

router = APIRouter()

# Maximum number of previous messages to include for context
MAX_HISTORY_MESSAGES = 10


def get_conversation_history(conversation_id: str, max_messages: int = MAX_HISTORY_MESSAGES) -> List[Dict[str, str]]:
    """
    Retrieve conversation history for context-aware responses.

    Args:
        conversation_id: Conversation UUID
        max_messages: Maximum number of messages to include (default 10)

    Returns:
        List of messages in format [{"role": "user/assistant", "content": "..."}]
    """
    db = get_conversation_db()
    messages = db.get_messages(conversation_id)

    # Convert to simple format for LLM and limit to recent messages
    history = []
    for msg in messages[-max_messages:]:
        history.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    return history


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for asking questions about uploaded documents.

    Uses RAG (Retrieval Augmented Generation) to:
    1. Create or retrieve conversation
    2. Save user message
    3. Embed the query and find relevant document chunks
    4. Generate response using GPT-4 with context
    5. Save assistant response with sources
    6. Auto-generate title for first message
    7. Return answer with sources and conversation_id

    Supports refinement mode for conversations with associated documents.

    Args:
        request: ChatRequest with user query and optional conversation_id

    Returns:
        ChatResponse with answer, sources, and conversation_id
    """
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )

    try:
        db = get_conversation_db()

        logger.info("── CHAT ── query=%.80s...", request.query)

        # Step 1: Get or create conversation
        if request.conversation_id:
            # Verify conversation exists
            conversation = db.get_conversation(request.conversation_id)
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Conversation {request.conversation_id} not found"
                )
            conversation_id = request.conversation_id
            is_first_message = conversation["message_count"] == 0
            notebook_id = conversation.get("notebook_id")
            document_ids = conversation.get("document_ids")  # For refinement mode
        else:
            # Create new conversation with temporary title
            conversation_id = db.create_conversation(title="Nuova Conversazione")
            is_first_message = True
            notebook_id = None
            document_ids = None

        # Step 2: Get conversation history BEFORE saving current message
        # This ensures we don't include the current query in the history
        conversation_history = []
        if not is_first_message:
            conversation_history = get_conversation_history(conversation_id)

        # Step 3: Save user message
        db.add_message(
            conversation_id=conversation_id,
            role="user",
            content=request.query,
            is_error=False
        )

        # Step 4: Generate response
        # Use refinement mode if conversation has associated documents
        if document_ids:
            logger.info("── CHAT ── mode=refinement | docs=%d | conv=%s", len(document_ids), conversation_id)
            # Refinement mode: use original documents as context
            result = await generate_refinement_response(
                query=request.query,
                document_ids=document_ids,
                conversation_history=conversation_history if conversation_history else []
            )
        else:
            logger.info("── CHAT ── mode=RAG | notebook=%s | conv=%s", notebook_id, conversation_id)
            # Standard RAG mode
            result = await generate_response(
                query=request.query,
                notebook_id=notebook_id,
                conversation_history=conversation_history if conversation_history else None
            )

        # Step 4: Save assistant message with sources
        sources_for_db = [
            {
                "document": source.document,
                "page": source.page,
                "chunk_text": source.chunk_text,
                "relevance_score": source.relevance_score
            }
            for source in result["sources"]
        ]

        db.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=result["answer"],
            is_error=False,
            sources=sources_for_db
        )

        # Step 5: Auto-generate title for first message
        if is_first_message:
            generated_title = generate_conversation_title(request.query)
            db.update_conversation_title(conversation_id, generated_title)

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            conversation_id=conversation_id
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except RuntimeError as e:
        logger.error("── CHAT ERROR ── %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error("── CHAT ERROR ── %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )


@router.get("/health")
async def chat_health():
    """
    Health check for chat service.

    Returns:
        Status message
    """
    return {"status": "chat service is running"}


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint for real-time responses.

    Uses Server-Sent Events (SSE) to stream response tokens.
    Better perceived latency for users.

    Now also saves messages to database and generates titles like non-streaming endpoint.
    For refinement mode (conversations with document_ids), falls back to non-streaming.

    Event types:
    - sources: Initial sources data
    - token: Individual response tokens
    - done: Completion signal with full response
    - error: Error message

    Args:
        request: ChatRequest with user query

    Returns:
        StreamingResponse with SSE events
    """
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )

    db = get_conversation_db()

    logger.info("── STREAM ── query=%.80s...", request.query)

    # Step 1: Get or create conversation (same as non-streaming)
    if request.conversation_id:
        conversation = db.get_conversation(request.conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {request.conversation_id} not found"
            )
        conversation_id = request.conversation_id
        is_first_message = conversation["message_count"] == 0
        notebook_id = conversation.get("notebook_id")
        document_ids = conversation.get("document_ids")  # For refinement mode
    else:
        conversation_id = db.create_conversation(title="Nuova Conversazione")
        is_first_message = True
        notebook_id = None
        document_ids = None

    # Step 2: Get conversation history BEFORE saving current message
    conversation_history = []
    if not is_first_message:
        conversation_history = get_conversation_history(conversation_id)

    # Step 3: Save user message
    db.add_message(
        conversation_id=conversation_id,
        role="user",
        content=request.query,
        is_error=False
    )

    # For refinement mode, use non-streaming response
    if document_ids:
        logger.info("── STREAM ── mode=refinement | docs=%d", len(document_ids))
        result = await generate_refinement_response(
            query=request.query,
            document_ids=document_ids,
            conversation_history=conversation_history if conversation_history else []
        )

        # Save assistant message
        sources_for_db = [
            {
                "document": s.document,
                "page": s.page,
                "chunk_text": s.chunk_text,
                "relevance_score": s.relevance_score
            }
            for s in result["sources"]
        ]

        db.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=result["answer"],
            is_error=False,
            sources=sources_for_db
        )

        # Create a fake streaming response that sends the complete response at once
        async def refinement_stream():
            # Send sources event
            sources_data = [
                {
                    "document": s.document,
                    "page": s.page,
                    "chunk_text": s.chunk_text,
                    "relevance_score": s.relevance_score
                }
                for s in result["sources"]
            ]
            yield f"event: sources\ndata: {json.dumps({'sources': sources_data})}\n\n"

            # Send the full response as tokens (in chunks for smoother display)
            answer = result["answer"]
            chunk_size = 20
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i+chunk_size]
                yield f"event: token\ndata: {json.dumps({'token': chunk})}\n\n"

            # Send done event
            yield f"event: done\ndata: {json.dumps({'full_response': answer, 'sources': sources_data})}\n\n"

        return StreamingResponse(
            refinement_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # Step 4: Create wrapper generator to save response after streaming (standard RAG)
    logger.info("── STREAM ── mode=RAG | notebook=%s | conv=%s", notebook_id, conversation_id)

    async def stream_and_save():
        full_response = ""
        sources_for_db = []

        async for event in generate_response_stream(
            query=request.query,
            notebook_id=notebook_id,
            conversation_history=conversation_history if conversation_history else None,
            top_k=8,
            compress=True
        ):
            # Parse event to capture sources and full response
            if event.startswith("event: done"):
                # Extract full response and filtered sources from done event
                lines = event.strip().split("\n")
                for line in lines:
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        full_response = data.get("full_response", full_response)
                        # Sources are now sent with done event (already filtered)
                        sources_for_db = [
                            {
                                "document": s.get("document", ""),
                                "page": s.get("page", 0),
                                "chunk_text": s.get("chunk_text", ""),
                                "relevance_score": s.get("relevance_score", 0.0)
                            }
                            for s in data.get("sources", [])
                        ]
            elif event.startswith("event: token"):
                # Accumulate tokens
                lines = event.strip().split("\n")
                for line in lines:
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        full_response += data.get("token", "")

            yield event

        # Step 4: Save assistant message with sources after streaming completes
        if full_response:
            db.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_response,
                is_error=False,
                sources=sources_for_db
            )

            # Step 5: Auto-generate title for first message
            if is_first_message:
                generated_title = generate_conversation_title(request.query)
                db.update_conversation_title(conversation_id, generated_title)

    return StreamingResponse(
        stream_and_save(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/cache/stats")
async def get_cache_stats():
    """
    Get embedding cache statistics.
    
    Useful for monitoring cache efficiency.
    
    Returns:
        Cache statistics including hit rate
    """
    cache = get_embedding_cache()
    return cache.get_stats()


@router.post("/cache/clear")
async def clear_cache():
    """
    Clear the embedding cache.
    
    Use when you want to force fresh embeddings.
    
    Returns:
        Confirmation message
    """
    cache = get_embedding_cache()
    cache.clear()
    return {"message": "Cache cleared successfully"}
