"""
Test suite for app/routes/notebooks.py

Tests all CRUD operations and document/conversation listing endpoints.
Uses pytest with mock to isolate route handlers from external dependencies
(database, vector store, file system, BM25 index).
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
from datetime import datetime
from fastapi import HTTPException, status

os.environ.setdefault("OPENAI_API_KEY", "fake-key-for-tests")

from app.routes.notebooks import (
    create_notebook,
    list_notebooks,
    get_notebook,
    update_notebook,
    delete_notebook,
    list_notebook_documents,
    list_notebook_conversations,
)
from app.models.notebook import Notebook, NotebookCreate, NotebookUpdate, NotebookWithStats
from app.models.document import DocumentListResponse, DocumentInfo
from app.models.conversation import Conversation


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures: Reusable test data
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def notebook_id():
    """Standard test notebook ID."""
    return "550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture
def notebook_data(notebook_id):
    """Sample notebook data."""
    return {
        "id": notebook_id,
        "name": "Test Notebook",
        "description": "A test notebook",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }


@pytest.fixture
def notebook_with_stats(notebook_data):
    """Notebook data with stats."""
    return {
        **notebook_data,
        "conversation_count": 2,
        "document_count": 3,
    }


@pytest.fixture
def sample_documents():
    """Sample document records from vector store."""
    return [
        {
            "document_id": "doc1",
            "document_name": "sample1.pdf",
            "upload_timestamp": datetime.now().isoformat(),
            "page_count": 10,
            "chunk_count": 25,
        },
        {
            "document_id": "doc2",
            "document_name": "sample2.pdf",
            "upload_timestamp": datetime.now().isoformat(),
            "page_count": 5,
            "chunk_count": 12,
        },
    ]


@pytest.fixture
def sample_conversations():
    """Sample conversation records."""
    return [
        {
            "id": "conv1",
            "notebook_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Conversation 1",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_count": 0,
        },
        {
            "id": "conv2",
            "notebook_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Conversation 2",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_count": 0,
        },
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: create_notebook - Success
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_create_notebook_success(notebook_id, notebook_data):
    """
    Creating a notebook with valid data should return the created Notebook object.
    Verifies that db.create_notebook() and db.get_notebook() are called correctly.
    """
    request = NotebookCreate(name="Test Notebook", description="A test notebook")

    with patch("app.routes.notebooks.get_notebook_db") as mock_get_db:
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.create_notebook.return_value = notebook_id
        mock_db.get_notebook.return_value = notebook_data

        result = await create_notebook(request)

        assert isinstance(result, Notebook)
        assert result.name == "Test Notebook"
        assert result.description == "A test notebook"
        mock_db.create_notebook.assert_called_once_with(
            name="Test Notebook", description="A test notebook"
        )
        mock_db.get_notebook.assert_called_once_with(notebook_id)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: create_notebook - Error retrieving created notebook
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_create_notebook_retrieval_error(notebook_id):
    """
    If db.get_notebook() returns None after creation, should raise HTTPException
    with 500 status code.
    """
    request = NotebookCreate(name="Test Notebook", description="A test notebook")

    with patch("app.routes.notebooks.get_notebook_db") as mock_get_db:
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.create_notebook.return_value = notebook_id
        mock_db.get_notebook.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await create_notebook(request)

        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: create_notebook - Database exception
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_create_notebook_db_exception():
    """
    If database raises an exception, should return 500 error with error message.
    """
    request = NotebookCreate(name="Test Notebook", description="A test notebook")

    with patch("app.routes.notebooks.get_notebook_db") as mock_get_db:
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.create_notebook.side_effect = Exception("Database connection failed")

        with pytest.raises(HTTPException) as exc_info:
            await create_notebook(request)

        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Database connection failed" in str(exc_info.value.detail)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: list_notebooks - Success with multiple notebooks
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_list_notebooks_success(notebook_data, sample_documents, sample_conversations):
    """
    Listing notebooks should return list of NotebookWithStats with correct counts.
    Verifies document and conversation counts are aggregated correctly.
    """
    notebooks_db_result = [notebook_data]

    with patch("app.routes.notebooks.get_notebook_db") as mock_get_nb_db, \
         patch("app.routes.notebooks.get_vector_store") as mock_get_vs, \
         patch("app.routes.notebooks.get_conversation_db") as mock_get_conv_db:

        mock_nb_db = Mock()
        mock_get_nb_db.return_value = mock_nb_db
        mock_nb_db.list_notebooks.return_value = notebooks_db_result

        mock_vs = Mock()
        mock_get_vs.return_value = mock_vs
        mock_vs.list_documents.return_value = sample_documents

        mock_conv_db = Mock()
        mock_get_conv_db.return_value = mock_conv_db
        mock_conv_db.list_conversations_by_notebook.return_value = sample_conversations

        result = await list_notebooks()

        assert len(result) == 1
        assert isinstance(result[0], NotebookWithStats)
        assert result[0].document_count == len(sample_documents)
        assert result[0].conversation_count == len(sample_conversations)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: list_notebooks - Empty result
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_list_notebooks_empty():
    """
    Listing notebooks when none exist should return an empty list.
    """
    with patch("app.routes.notebooks.get_notebook_db") as mock_get_nb_db, \
         patch("app.routes.notebooks.get_vector_store") as mock_get_vs:

        mock_nb_db = Mock()
        mock_get_nb_db.return_value = mock_nb_db
        mock_nb_db.list_notebooks.return_value = []

        result = await list_notebooks()

        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: list_notebooks - Database exception
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_list_notebooks_db_exception():
    """
    If database raises exception, should return 500 error.
    """
    with patch("app.routes.notebooks.get_notebook_db") as mock_get_db:
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.list_notebooks.side_effect = Exception("DB error")

        with pytest.raises(HTTPException) as exc_info:
            await list_notebooks()

        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: get_notebook - Success
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_get_notebook_success(notebook_id, notebook_with_stats, sample_documents):
    """
    Getting an existing notebook should return NotebookWithStats with correct counts.
    """
    with patch("app.routes.notebooks.get_notebook_db") as mock_get_nb_db, \
         patch("app.routes.notebooks.get_vector_store") as mock_get_vs, \
         patch("app.routes.notebooks.get_conversation_db") as mock_get_conv_db:

        mock_nb_db = Mock()
        mock_get_nb_db.return_value = mock_nb_db
        mock_nb_db.get_notebook_with_stats.return_value = {
            k: v for k, v in notebook_with_stats.items() if k != "conversation_count"
        }

        mock_vs = Mock()
        mock_get_vs.return_value = mock_vs
        mock_vs.list_documents.return_value = sample_documents

        mock_conv_db = Mock()
        mock_get_conv_db.return_value = mock_conv_db
        mock_conv_db.list_conversations_by_notebook.return_value = [{"id": "c1"}, {"id": "c2"}]

        result = await get_notebook(notebook_id)

        assert isinstance(result, NotebookWithStats)
        assert result.id == notebook_id
        assert result.document_count == len(sample_documents)
        assert result.conversation_count == 2
        mock_nb_db.get_notebook_with_stats.assert_called_once_with(notebook_id)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8: get_notebook - Not found
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_get_notebook_not_found(notebook_id):
    """
    Getting a non-existent notebook should raise HTTPException with 404 status.
    """
    with patch("app.routes.notebooks.get_notebook_db") as mock_get_db:
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.get_notebook_with_stats.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_notebook(notebook_id)

        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert "Notebook not found" in str(exc_info.value.detail)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 9: get_notebook - Database exception
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_get_notebook_db_exception(notebook_id):
    """
    If database raises exception, should return 500 error.
    """
    with patch("app.routes.notebooks.get_notebook_db") as mock_get_db:
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.get_notebook_with_stats.side_effect = Exception("Connection timeout")

        with pytest.raises(HTTPException) as exc_info:
            await get_notebook(notebook_id)

        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 10: update_notebook - Success
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_update_notebook_success(notebook_id, notebook_data):
    """
    Updating an existing notebook should return the updated Notebook object.
    """
    request = NotebookUpdate(name="Updated Name", description="Updated description")
    updated_data = {**notebook_data, "name": "Updated Name", "description": "Updated description"}

    with patch("app.routes.notebooks.get_notebook_db") as mock_get_db:
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.notebook_exists.return_value = True
        mock_db.update_notebook.return_value = True
        mock_db.get_notebook.return_value = updated_data

        result = await update_notebook(notebook_id, request)

        assert isinstance(result, Notebook)
        assert result.name == "Updated Name"
        assert result.description == "Updated description"
        mock_db.update_notebook.assert_called_once_with(
            notebook_id=notebook_id, name="Updated Name", description="Updated description"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 11: update_notebook - Not found
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_update_notebook_not_found(notebook_id):
    """
    Updating a non-existent notebook should raise HTTPException with 404 status.
    """
    request = NotebookUpdate(name="Updated Name", description="Updated description")

    with patch("app.routes.notebooks.get_notebook_db") as mock_get_db:
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.notebook_exists.return_value = False

        with pytest.raises(HTTPException) as exc_info:
            await update_notebook(notebook_id, request)

        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 12: update_notebook - Update fails
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_update_notebook_update_fails(notebook_id):
    """
    If db.update_notebook() returns False, should raise HTTPException with 500 status.
    """
    request = NotebookUpdate(name="Updated Name", description="Updated description")

    with patch("app.routes.notebooks.get_notebook_db") as mock_get_db:
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.notebook_exists.return_value = True
        mock_db.update_notebook.return_value = False

        with pytest.raises(HTTPException) as exc_info:
            await update_notebook(notebook_id, request)

        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 13: delete_notebook - Success with cleanup
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_delete_notebook_success(notebook_id, sample_documents):
    """
    Deleting a notebook should:
    1. Delete from database
    2. Delete all documents from vector store
    3. Delete parent chunks
    4. Delete physical files
    5. Invalidate BM25 index
    """
    with patch("app.routes.notebooks.get_notebook_db") as mock_get_nb_db, \
         patch("app.routes.notebooks.get_vector_store") as mock_get_vs, \
         patch("app.routes.notebooks.get_parent_chunk_store") as mock_get_pcs, \
         patch("app.routes.notebooks.get_bm25_engine") as mock_get_bm25, \
         patch("app.routes.notebooks.Path") as mock_path_cls:

        mock_nb_db = Mock()
        mock_get_nb_db.return_value = mock_nb_db
        mock_nb_db.notebook_exists.return_value = True
        mock_nb_db.delete_notebook.return_value = True

        mock_vs = Mock()
        mock_get_vs.return_value = mock_vs
        mock_vs.list_documents.return_value = sample_documents

        mock_pcs = Mock()
        mock_get_pcs.return_value = mock_pcs

        mock_bm25 = Mock()
        mock_get_bm25.return_value = mock_bm25

        mock_uploads_dir = MagicMock()
        mock_path_cls.return_value = mock_uploads_dir
        mock_uploads_dir.glob.return_value = []

        # This endpoint returns 204, so no content to check
        result = await delete_notebook(notebook_id)

        # Verify calls
        mock_nb_db.delete_notebook.assert_called_once_with(notebook_id)
        assert mock_vs.delete_document.call_count == len(sample_documents)
        mock_bm25.invalidate.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 14: delete_notebook - Not found
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_delete_notebook_not_found(notebook_id):
    """
    Deleting a non-existent notebook should raise HTTPException with 404 status.
    """
    with patch("app.routes.notebooks.get_notebook_db") as mock_get_db:
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.notebook_exists.return_value = False

        with pytest.raises(HTTPException) as exc_info:
            await delete_notebook(notebook_id)

        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 15: delete_notebook - Delete fails
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_delete_notebook_delete_fails(notebook_id):
    """
    If db.delete_notebook() returns False, should raise HTTPException with 500 status.
    """
    with patch("app.routes.notebooks.get_notebook_db") as mock_get_db:
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.notebook_exists.return_value = True
        mock_db.delete_notebook.return_value = False

        with pytest.raises(HTTPException) as exc_info:
            await delete_notebook(notebook_id)

        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 16: delete_notebook - File deletion errors (non-fatal)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_delete_notebook_file_deletion_errors(notebook_id, sample_documents):
    """
    File deletion errors should not prevent notebook deletion (non-fatal).
    Verify that deletion continues despite file errors.
    """
    with patch("app.routes.notebooks.get_notebook_db") as mock_get_nb_db, \
         patch("app.routes.notebooks.get_vector_store") as mock_get_vs, \
         patch("app.routes.notebooks.get_parent_chunk_store") as mock_get_pcs, \
         patch("app.routes.notebooks.get_bm25_engine") as mock_get_bm25, \
         patch("app.routes.notebooks.Path") as mock_path_cls:

        mock_nb_db = Mock()
        mock_get_nb_db.return_value = mock_nb_db
        mock_nb_db.notebook_exists.return_value = True
        mock_nb_db.delete_notebook.return_value = True

        mock_vs = Mock()
        mock_get_vs.return_value = mock_vs
        mock_vs.list_documents.return_value = sample_documents

        mock_uploads_dir = MagicMock()
        mock_path_cls.return_value = mock_uploads_dir
        mock_file = MagicMock()
        mock_file.unlink.side_effect = OSError("Permission denied")
        mock_uploads_dir.glob.return_value = [mock_file]

        result = await delete_notebook(notebook_id)

        # Should not raise despite file error
        mock_nb_db.delete_notebook.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 17: list_notebook_documents - Success
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_list_notebook_documents_success(notebook_id, sample_documents):
    """
    Listing documents in a notebook should return DocumentListResponse with
    correct document count and info objects.
    """
    with patch("app.routes.notebooks.get_notebook_db") as mock_get_nb_db, \
         patch("app.routes.notebooks.get_vector_store") as mock_get_vs:

        mock_nb_db = Mock()
        mock_get_nb_db.return_value = mock_nb_db
        mock_nb_db.notebook_exists.return_value = True

        mock_vs = Mock()
        mock_get_vs.return_value = mock_vs
        mock_vs.list_documents.return_value = sample_documents

        result = await list_notebook_documents(notebook_id)

        assert isinstance(result, DocumentListResponse)
        assert result.total == len(sample_documents)
        assert len(result.documents) == len(sample_documents)
        assert all(isinstance(d, DocumentInfo) for d in result.documents)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 18: list_notebook_documents - Notebook not found
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_list_notebook_documents_not_found(notebook_id):
    """
    Listing documents for a non-existent notebook should raise HTTPException with 404.
    """
    with patch("app.routes.notebooks.get_notebook_db") as mock_get_db:
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.notebook_exists.return_value = False

        with pytest.raises(HTTPException) as exc_info:
            await list_notebook_documents(notebook_id)

        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 19: list_notebook_documents - Empty list
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_list_notebook_documents_empty(notebook_id):
    """
    Listing documents when none exist should return empty DocumentListResponse.
    """
    with patch("app.routes.notebooks.get_notebook_db") as mock_get_nb_db, \
         patch("app.routes.notebooks.get_vector_store") as mock_get_vs:

        mock_nb_db = Mock()
        mock_get_nb_db.return_value = mock_nb_db
        mock_nb_db.notebook_exists.return_value = True

        mock_vs = Mock()
        mock_get_vs.return_value = mock_vs
        mock_vs.list_documents.return_value = []

        result = await list_notebook_documents(notebook_id)

        assert result.total == 0
        assert len(result.documents) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 20: list_notebook_conversations - Success
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_list_notebook_conversations_success(notebook_id, sample_conversations):
    """
    Listing conversations in a notebook should return list of Conversation objects.
    """
    with patch("app.routes.notebooks.get_notebook_db") as mock_get_nb_db, \
         patch("app.routes.notebooks.get_conversation_db") as mock_get_conv_db:

        mock_nb_db = Mock()
        mock_get_nb_db.return_value = mock_nb_db
        mock_nb_db.notebook_exists.return_value = True

        mock_conv_db = Mock()
        mock_get_conv_db.return_value = mock_conv_db
        mock_conv_db.list_conversations_by_notebook.return_value = sample_conversations

        result = await list_notebook_conversations(notebook_id)

        assert len(result) == len(sample_conversations)
        assert all(isinstance(c, Conversation) for c in result)
        mock_conv_db.list_conversations_by_notebook.assert_called_once_with(notebook_id)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 21: list_notebook_conversations - Notebook not found
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_list_notebook_conversations_not_found(notebook_id):
    """
    Listing conversations for a non-existent notebook should raise HTTPException with 404.
    """
    with patch("app.routes.notebooks.get_notebook_db") as mock_get_db:
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.notebook_exists.return_value = False

        with pytest.raises(HTTPException) as exc_info:
            await list_notebook_conversations(notebook_id)

        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 22: list_notebook_conversations - Empty list
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_list_notebook_conversations_empty(notebook_id):
    """
    Listing conversations when none exist should return empty list.
    """
    with patch("app.routes.notebooks.get_notebook_db") as mock_get_nb_db, \
         patch("app.routes.notebooks.get_conversation_db") as mock_get_conv_db:

        mock_nb_db = Mock()
        mock_get_nb_db.return_value = mock_nb_db
        mock_nb_db.notebook_exists.return_value = True

        mock_conv_db = Mock()
        mock_get_conv_db.return_value = mock_conv_db
        mock_conv_db.list_conversations_by_notebook.return_value = []

        result = await list_notebook_conversations(notebook_id)

        assert result == []