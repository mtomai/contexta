from typing import List
from pathlib import Path
from fastapi import APIRouter, HTTPException, status

from app.models.notebook import (
    Notebook,
    NotebookCreate,
    NotebookUpdate,
    NotebookWithStats
)
from app.models.document import DocumentListResponse, DocumentInfo
from app.models.conversation import Conversation
from app.services.notebook_db import get_notebook_db
from app.services.conversation_db import get_conversation_db
from app.services.vector_store import get_vector_store
from app.services.parent_chunk_store import get_parent_chunk_store
from app.services.bm25_search import get_bm25_engine
from app.config import get_settings
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


@router.post("", response_model=Notebook, status_code=status.HTTP_201_CREATED)
async def create_notebook(request: NotebookCreate):
    """
    Create a new notebook.

    Args:
        request: Notebook creation data

    Returns:
        Created notebook
    """
    try:
        db = get_notebook_db()
        notebook_id = db.create_notebook(
            name=request.name,
            description=request.description
        )

        notebook = db.get_notebook(notebook_id)
        if not notebook:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error retrieving created notebook"
            )

        return Notebook(**notebook)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating notebook: {str(e)}"
        )


@router.get("", response_model=List[NotebookWithStats])
async def list_notebooks():
    """
    List all notebooks ordered by most recently updated.
    Includes document and conversation counts for each notebook.

    Returns:
        List of notebooks with statistics
    """
    try:
        db = get_notebook_db()
        vector_store = get_vector_store()
        notebooks = db.list_notebooks()

        # Add stats for each notebook
        notebooks_with_stats = []
        for nb in notebooks:
            # Get conversation count
            conv_db = get_conversation_db()
            conversations = conv_db.list_conversations_by_notebook(nb["id"])
            nb["conversation_count"] = len(conversations)

            # Get document count from vector store
            documents = vector_store.list_documents(notebook_id=nb["id"])
            nb["document_count"] = len(documents)

            notebooks_with_stats.append(NotebookWithStats(**nb))

        return notebooks_with_stats

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing notebooks: {str(e)}"
        )


@router.get("/{notebook_id}", response_model=NotebookWithStats)
async def get_notebook(notebook_id: str):
    """
    Get notebook details with statistics.

    Args:
        notebook_id: Notebook UUID

    Returns:
        Notebook with document and conversation counts
    """
    try:
        db = get_notebook_db()
        notebook = db.get_notebook_with_stats(notebook_id)

        if not notebook:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Notebook not found"
            )

        # Get document count from vector store
        vector_store = get_vector_store()
        documents = vector_store.list_documents(notebook_id=notebook_id)
        notebook["document_count"] = len(documents)

        # Get conversation count
        conv_db = get_conversation_db()
        conversations = conv_db.list_conversations_by_notebook(notebook_id)
        notebook["conversation_count"] = len(conversations)

        return NotebookWithStats(**notebook)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving notebook: {str(e)}"
        )


@router.put("/{notebook_id}", response_model=Notebook)
async def update_notebook(notebook_id: str, request: NotebookUpdate):
    """
    Update notebook name and/or description.

    Args:
        notebook_id: Notebook UUID
        request: Update data

    Returns:
        Updated notebook
    """
    try:
        db = get_notebook_db()

        # Check if notebook exists
        if not db.notebook_exists(notebook_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Notebook not found"
            )

        # Update notebook
        success = db.update_notebook(
            notebook_id=notebook_id,
            name=request.name,
            description=request.description
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error updating notebook"
            )

        # Get updated notebook
        notebook = db.get_notebook(notebook_id)
        return Notebook(**notebook)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating notebook: {str(e)}"
        )


@router.delete("/{notebook_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_notebook(notebook_id: str):
    """
    Delete notebook. Conversations will be deleted (CASCADE).
    Documents will remain but can be reassigned.

    Args:
        notebook_id: Notebook UUID
    """
    try:
        db = get_notebook_db()

        # Check if notebook exists
        if not db.notebook_exists(notebook_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Notebook not found"
            )

        # Delete notebook (conversations CASCADE deleted)
        success = db.delete_notebook(notebook_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error deleting notebook"
            )

        # Delete all documents belonging to this notebook
        vector_store = get_vector_store()
        documents = vector_store.list_documents(notebook_id=notebook_id)
        uploads_dir = Path(settings.uploads_path)

        for doc in documents:
            doc_id = doc["document_id"]

            # Delete chunks from vector store
            vector_store.delete_document(doc_id)

            # Delete parent chunks from SQLite
            try:
                parent_store = get_parent_chunk_store()
                parent_store.delete_document(doc_id)
            except Exception:
                pass  # Non-fatal

            # Delete physical files from uploads/
            for file_path in uploads_dir.glob(f"{doc_id}_*"):
                try:
                    file_path.unlink()
                    logger.info("Deleted upload file: %s", file_path.name)
                except Exception as e:
                    logger.warning("Could not delete file %s: %s", file_path, e)

        # Invalidate BM25 index
        if documents:
            try:
                bm25_engine = get_bm25_engine()
                bm25_engine.invalidate()
            except Exception:
                pass  # Non-fatal

        logger.info(
            "Notebook %s deleted: %d documents removed",
            notebook_id, len(documents)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting notebook: {str(e)}"
        )


@router.get("/{notebook_id}/documents", response_model=DocumentListResponse)
async def list_notebook_documents(notebook_id: str):
    """
    List all documents in a notebook.

    Args:
        notebook_id: Notebook UUID

    Returns:
        List of documents with metadata
    """
    try:
        # Verify notebook exists
        db = get_notebook_db()
        if not db.notebook_exists(notebook_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Notebook not found"
            )

        # Get documents from vector store
        vector_store = get_vector_store()
        documents = vector_store.list_documents(notebook_id=notebook_id)
        logger.info(
            "Listing documents for notebook %s: found %d documents",
            notebook_id, len(documents)
        )

        # Convert to DocumentInfo objects
        document_infos = [
            DocumentInfo(
                document_id=doc["document_id"],
                document_name=doc["document_name"],
                upload_timestamp=datetime.fromisoformat(doc["upload_timestamp"]),
                page_count=doc["page_count"],
                chunk_count=doc["chunk_count"]
            )
            for doc in documents
        ]

        return DocumentListResponse(
            documents=document_infos,
            total=len(document_infos)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing notebook documents: {str(e)}"
        )


@router.get("/{notebook_id}/conversations", response_model=List[Conversation])
async def list_notebook_conversations(notebook_id: str):
    """
    List all conversations in a notebook.

    Args:
        notebook_id: Notebook UUID

    Returns:
        List of conversations
    """
    try:
        # Verify notebook exists
        notebook_db = get_notebook_db()
        if not notebook_db.notebook_exists(notebook_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Notebook not found"
            )

        # Get conversations
        conv_db = get_conversation_db()
        conversations = conv_db.list_conversations_by_notebook(notebook_id)

        return [Conversation(**conv) for conv in conversations]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing notebook conversations: {str(e)}"
        )



