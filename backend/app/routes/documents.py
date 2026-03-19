import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from app.config import get_settings
from app.models.document import DocumentUploadResponse, DocumentListResponse, DocumentInfo
from app.services.document_parser import parse_document, get_document_page_count
from app.services.embeddings import embed_document_chunks_parallel
from app.services.vector_store import get_vector_store
from app.services.parent_chunk_store import get_parent_chunk_store
from app.services.bm25_search import get_bm25_engine

settings = get_settings()
router = APIRouter()
logger = logging.getLogger(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".md"}


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    notebook_id: str = Form(None)
):
    """
    Upload and process a document (PDF or Word).

    Steps:
    1. Validate file type and size
    2. Save file temporarily
    3. Parse document with layout-aware parser and extract structured text
    4. Create parent-child chunks (if enabled)
    5. Create embeddings for child chunks
    6. Store child chunks in vector database
    7. Store parent chunks in SQLite
    8. Invalidate BM25 index for rebuild
    9. Return document info

    Args:
        file: Uploaded file
        notebook_id: Optional notebook UUID to associate document with
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not supported. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Validate file size
    file.file.seek(0, 2)  # Move to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning

    max_size = settings.max_file_size_mb * 1024 * 1024  # Convert to bytes
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
        )

    # Generate unique document ID
    document_id = str(uuid.uuid4())
    upload_timestamp = datetime.now(timezone.utc)

    # Save file temporarily
    file_path = Path(settings.uploads_path) / f"{document_id}_{file.filename}"
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving file: {str(e)}"
        )

    # Parse document (returns parent + child chunks)
    try:
        file_type = file_ext.lstrip('.')
        parse_result = parse_document(str(file_path), file_type)
        page_count = get_document_page_count(str(file_path), file_type)
    except Exception as e:
        # Clean up file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Error parsing document: {str(e)}"
        )

    # The child chunks are what we embed and store in ChromaDB
    child_chunks = parse_result["child_chunks"]
    parent_chunks = parse_result["parent_chunks"]
    mode = parse_result["mode"]

    # Create embeddings for child chunks (parallel for better performance)
    try:
        chunks_with_embeddings = await embed_document_chunks_parallel(child_chunks)
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating embeddings: {str(e)}"
        )

    # Store child chunks in vector database
    try:
        vector_store = get_vector_store()
        chunk_count = vector_store.add_document(
            document_id=document_id,
            document_name=file.filename,
            chunks=chunks_with_embeddings,
            notebook_id=notebook_id
        )
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error storing document: {str(e)}"
        )

    # Store parent chunks in SQLite (if parent-child mode)
    if mode == "parent_child" and parent_chunks:
        try:
            parent_store = get_parent_chunk_store()
            parent_store.add_parent_chunks(
                document_id=document_id,
                document_name=file.filename,
                parent_chunks=parent_chunks,
                notebook_id=notebook_id
            )
        except Exception as e:
            # Non-fatal: parent chunks are an optimization, not critical
            print(f"Warning: Failed to store parent chunks: {e}")

    # Invalidate BM25 index so it rebuilds with the new document
    try:
        bm25_engine = get_bm25_engine()
        bm25_engine.invalidate()
    except Exception:
        pass  # Non-fatal

    logger.info(
        "Document uploaded: id=%s name=%s notebook_id=%s chunks=%d pages=%d",
        document_id, file.filename, notebook_id, chunk_count, page_count
    )

    return DocumentUploadResponse(
        document_id=document_id,
        document_name=file.filename,
        page_count=page_count,
        chunk_count=chunk_count,
        upload_timestamp=upload_timestamp,
        notebook_id=notebook_id
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """
    List all uploaded documents.

    Returns:
        List of documents with metadata
    """
    try:
        vector_store = get_vector_store()
        documents = vector_store.list_documents()

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

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document from the system.

    Args:
        document_id: UUID of the document to delete

    Returns:
        Success message
    """
    try:
        vector_store = get_vector_store()

        # Check if document exists
        if not vector_store.document_exists(document_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        # Delete from vector store
        chunks_deleted = vector_store.delete_document(document_id)

        # Delete parent chunks from SQLite
        try:
            parent_store = get_parent_chunk_store()
            parent_store.delete_document(document_id)
        except Exception:
            pass  # Non-fatal

        # Invalidate BM25 index
        try:
            bm25_engine = get_bm25_engine()
            bm25_engine.invalidate()
        except Exception:
            pass  # Non-fatal

        # Delete physical files
        uploads_dir = Path(settings.uploads_path)
        for file_path in uploads_dir.glob(f"{document_id}_*"):
            try:
                file_path.unlink()
            except Exception as e:
                print(f"Warning: Could not delete file {file_path}: {e}")

        return {
            "message": "Document deleted successfully",
            "document_id": document_id,
            "chunks_deleted": chunks_deleted
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )


@router.get("/stats")
async def get_stats():
    """
    Get statistics about the document store.

    Returns:
        Statistics about documents and storage
    """
    try:
        vector_store = get_vector_store()
        stats = vector_store.get_collection_stats()
        documents = vector_store.list_documents()

        # BM25 stats
        bm25_stats = {}
        try:
            bm25_engine = get_bm25_engine()
            bm25_stats = bm25_engine.get_stats()
        except Exception:
            pass

        return {
            "total_documents": len(documents),
            "total_chunks": stats["total_chunks"],
            "collection_name": stats["collection_name"],
            "bm25_index": bm25_stats
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting stats: {str(e)}"
        )


@router.put("/{document_id}/notebook")
async def assign_document_to_notebook(
    document_id: str,
    notebook_id: str = None
):
    """
    Assign or unassign a document to/from a notebook.

    Args:
        document_id: Document UUID
        notebook_id: Notebook UUID (None to unassign)

    Returns:
        Success message with number of chunks updated
    """
    try:
        vector_store = get_vector_store()

        # Check if document exists
        if not vector_store.document_exists(document_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        # Update notebook assignment in vector store
        chunks_updated = vector_store.update_document_notebook(
            document_id=document_id,
            notebook_id=notebook_id
        )

        # Update parent chunks notebook assignment
        try:
            parent_store = get_parent_chunk_store()
            parent_store.update_notebook(document_id, notebook_id)
        except Exception:
            pass  # Non-fatal

        # Invalidate BM25 index (notebook metadata changed)
        try:
            bm25_engine = get_bm25_engine()
            bm25_engine.invalidate()
        except Exception:
            pass

        return {
            "message": "Document assignment updated successfully",
            "document_id": document_id,
            "notebook_id": notebook_id,
            "chunks_updated": chunks_updated
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error assigning document: {str(e)}"
        )
