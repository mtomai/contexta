"""
Unit tests for app.routes.documents.upload_document.

Covers:
- Extension validation
- File-size validation
- Save/parse/embed/store error handling and cleanup
- Successful upload flow (including parent chunks + BM25 invalidation)
- Non-fatal failures (parent chunk store and BM25)
"""

import asyncio
import io
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

os.environ.setdefault("OPENAI_API_KEY", "fake-key-for-tests")

from app.routes import documents as documents_module
from app.routes.documents import upload_document


class DummyUploadFile:
    """Minimal upload-file test double compatible with upload_document()."""

    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self.file = io.BytesIO(content)

    async def read(self) -> bytes:
        return self.file.read()


def _make_upload_file(filename: str = "test.pdf", content: bytes = b"dummy-content") -> DummyUploadFile:
    return DummyUploadFile(filename=filename, content=content)


def _run_upload(file_obj: DummyUploadFile, notebook_id=None):
    return asyncio.run(upload_document(file=file_obj, notebook_id=notebook_id))


def _parse_result(mode: str = "parent_child", with_parent_chunks: bool = True):
    return {
        "child_chunks": [{"chunk_id": "c1", "text": "child text", "page": 1}],
        "parent_chunks": [{"parent_id": "p1", "text": "parent text"}] if with_parent_chunks else [],
        "mode": mode,
    }


def test_upload_document_accepts_markdown_extension(tmp_path):
    """Markdown (.md) file passes extension validation (parse is mocked)."""
    file_obj = _make_upload_file(filename="readme.md", content=b"# Hello")

    with patch.object(
        documents_module,
        "settings",
        SimpleNamespace(uploads_path=str(tmp_path), max_file_size_mb=10),
    ), patch.object(
        documents_module, "parse_document", return_value=_parse_result()
    ), patch.object(
        documents_module, "get_document_page_count", return_value=1
    ), patch.object(
        documents_module, "embed_document_chunks_parallel", new_callable=AsyncMock,
        return_value=[{"text": "child text", "embedding": [0.1], "metadata": {"chunk_index": 0, "page": 1}}],
    ), patch.object(
        documents_module, "get_vector_store",
        return_value=MagicMock(add_document=MagicMock(return_value=1)),
    ), patch.object(
        documents_module, "get_parent_chunk_store",
        return_value=MagicMock(),
    ), patch.object(
        documents_module, "get_bm25_engine",
        return_value=MagicMock(),
    ):
        result = _run_upload(file_obj)

    assert result.document_name == "readme.md"


def test_upload_document_rejects_unsupported_file_extension(tmp_path):
    """Unsupported extension returns 400 and exits early."""
    file_obj = _make_upload_file(filename="notes.txt", content=b"abc")

    with patch.object(
        documents_module,
        "settings",
        SimpleNamespace(uploads_path=str(tmp_path), max_file_size_mb=10),
    ), patch.object(documents_module, "parse_document") as parse_mock:
        with pytest.raises(HTTPException) as exc:
            _run_upload(file_obj)

    assert exc.value.status_code == 400
    assert "File type not supported" in exc.value.detail
    parse_mock.assert_not_called()


def test_upload_document_rejects_when_file_exceeds_max_size(tmp_path):
    """File larger than configured max size returns 400."""
    file_obj = _make_upload_file(filename="big.pdf", content=b"x")

    with patch.object(
        documents_module,
        "settings",
        SimpleNamespace(uploads_path=str(tmp_path), max_file_size_mb=0),
    ), patch.object(documents_module, "parse_document") as parse_mock:
        with pytest.raises(HTTPException) as exc:
            _run_upload(file_obj)

    assert exc.value.status_code == 400
    assert "File too large" in exc.value.detail
    parse_mock.assert_not_called()


def test_upload_document_returns_500_when_file_save_fails(tmp_path):
    """I/O error while saving file returns 500."""
    file_obj = _make_upload_file(filename="test.pdf", content=b"abc")

    with patch.object(
        documents_module,
        "settings",
        SimpleNamespace(uploads_path=str(tmp_path), max_file_size_mb=10),
    ), patch("builtins.open", side_effect=OSError("disk full")):
        with pytest.raises(HTTPException) as exc:
            _run_upload(file_obj)

    assert exc.value.status_code == 500
    assert "Error saving file" in exc.value.detail


def test_upload_document_returns_422_and_cleans_file_on_parse_error(tmp_path):
    """Parse failure returns 422 and deletes temporary file."""
    file_obj = _make_upload_file(filename="parse_fail.pdf", content=b"abc")
    document_id = "doc-parse-fail"
    expected_file_path = tmp_path / f"{document_id}_parse_fail.pdf"

    with patch.object(
        documents_module,
        "settings",
        SimpleNamespace(uploads_path=str(tmp_path), max_file_size_mb=10),
    ), patch.object(documents_module.uuid, "uuid4", return_value=document_id), patch.object(
        documents_module, "parse_document", side_effect=Exception("bad parse")
    ):
        with pytest.raises(HTTPException) as exc:
            _run_upload(file_obj)

    assert exc.value.status_code == 422
    assert "Error parsing document" in exc.value.detail
    assert not expected_file_path.exists()


def test_upload_document_returns_500_and_cleans_file_on_embedding_error(tmp_path):
    """Embedding failure returns 500 and deletes temporary file."""
    file_obj = _make_upload_file(filename="embed_fail.pdf", content=b"abc")
    document_id = "doc-embed-fail"
    expected_file_path = tmp_path / f"{document_id}_embed_fail.pdf"

    with patch.object(
        documents_module,
        "settings",
        SimpleNamespace(uploads_path=str(tmp_path), max_file_size_mb=10),
    ), patch.object(documents_module.uuid, "uuid4", return_value=document_id), patch.object(
        documents_module, "parse_document", return_value=_parse_result()
    ), patch.object(
        documents_module, "get_document_page_count", return_value=2
    ), patch.object(
        documents_module,
        "embed_document_chunks_parallel",
        new=AsyncMock(side_effect=Exception("embedding failed")),
    ):
        with pytest.raises(HTTPException) as exc:
            _run_upload(file_obj)

    assert exc.value.status_code == 500
    assert "Error creating embeddings" in exc.value.detail
    assert not expected_file_path.exists()


def test_upload_document_returns_500_and_cleans_file_on_vector_store_error(tmp_path):
    """Vector-store add failure returns 500 and deletes temporary file."""
    file_obj = _make_upload_file(filename="store_fail.pdf", content=b"abc")
    document_id = "doc-store-fail"
    expected_file_path = tmp_path / f"{document_id}_store_fail.pdf"

    mock_vector_store = MagicMock()
    mock_vector_store.add_document.side_effect = Exception("db write error")

    with patch.object(
        documents_module,
        "settings",
        SimpleNamespace(uploads_path=str(tmp_path), max_file_size_mb=10),
    ), patch.object(documents_module.uuid, "uuid4", return_value=document_id), patch.object(
        documents_module, "parse_document", return_value=_parse_result()
    ), patch.object(
        documents_module, "get_document_page_count", return_value=3
    ), patch.object(
        documents_module,
        "embed_document_chunks_parallel",
        new=AsyncMock(return_value=[{"chunk_id": "c1", "embedding": [0.1]}]),
    ), patch.object(
        documents_module, "get_vector_store", return_value=mock_vector_store
    ):
        with pytest.raises(HTTPException) as exc:
            _run_upload(file_obj)

    assert exc.value.status_code == 500
    assert "Error storing document" in exc.value.detail
    assert not expected_file_path.exists()


def test_upload_document_success_parent_child_stores_all_and_invalidates_bm25(tmp_path):
    """Successful upload executes full parent-child path and returns expected response."""
    file_obj = _make_upload_file(filename="REPORT.PDF", content=b"%PDF-test")
    document_id = "doc-success"
    expected_file_path = tmp_path / f"{document_id}_REPORT.PDF"

    parse_result = _parse_result(mode="parent_child", with_parent_chunks=True)
    embedded_chunks = [{"chunk_id": "c1", "text": "child text", "embedding": [0.11, 0.22]}]

    mock_vector_store = MagicMock()
    mock_vector_store.add_document.return_value = 1

    mock_parent_store = MagicMock()
    mock_parent_store.add_parent_chunks.return_value = 1

    mock_bm25 = MagicMock()

    with patch.object(
        documents_module,
        "settings",
        SimpleNamespace(uploads_path=str(tmp_path), max_file_size_mb=10),
    ), patch.object(documents_module.uuid, "uuid4", return_value=document_id), patch.object(
        documents_module, "parse_document", return_value=parse_result
    ), patch.object(
        documents_module, "get_document_page_count", return_value=7
    ), patch.object(
        documents_module,
        "embed_document_chunks_parallel",
        new=AsyncMock(return_value=embedded_chunks),
    ), patch.object(
        documents_module, "get_vector_store", return_value=mock_vector_store
    ), patch.object(
        documents_module, "get_parent_chunk_store", return_value=mock_parent_store
    ), patch.object(
        documents_module, "get_bm25_engine", return_value=mock_bm25
    ):
        response = _run_upload(file_obj, notebook_id="nb-123")

    assert response.document_id == document_id
    assert response.document_name == "REPORT.PDF"
    assert response.page_count == 7
    assert response.chunk_count == 1
    assert response.notebook_id == "nb-123"

    mock_vector_store.add_document.assert_called_once_with(
        document_id=document_id,
        document_name="REPORT.PDF",
        chunks=embedded_chunks,
        notebook_id="nb-123",
    )
    mock_parent_store.add_parent_chunks.assert_called_once_with(
        document_id=document_id,
        document_name="REPORT.PDF",
        parent_chunks=parse_result["parent_chunks"],
        notebook_id="nb-123",
    )
    mock_bm25.invalidate.assert_called_once()
    assert expected_file_path.exists()


def test_upload_document_parent_store_failure_is_non_fatal(tmp_path):
    """Parent chunk storage failure does not fail upload."""
    file_obj = _make_upload_file(filename="parent_nonfatal.pdf", content=b"abc")
    document_id = "doc-parent-nonfatal"

    mock_vector_store = MagicMock()
    mock_vector_store.add_document.return_value = 1

    mock_parent_store = MagicMock()
    mock_parent_store.add_parent_chunks.side_effect = Exception("sqlite error")

    mock_bm25 = MagicMock()

    with patch.object(
        documents_module,
        "settings",
        SimpleNamespace(uploads_path=str(tmp_path), max_file_size_mb=10),
    ), patch.object(documents_module.uuid, "uuid4", return_value=document_id), patch.object(
        documents_module, "parse_document", return_value=_parse_result(mode="parent_child", with_parent_chunks=True)
    ), patch.object(
        documents_module, "get_document_page_count", return_value=1
    ), patch.object(
        documents_module,
        "embed_document_chunks_parallel",
        new=AsyncMock(return_value=[{"chunk_id": "c1", "embedding": [0.1]}]),
    ), patch.object(
        documents_module, "get_vector_store", return_value=mock_vector_store
    ), patch.object(
        documents_module, "get_parent_chunk_store", return_value=mock_parent_store
    ), patch.object(
        documents_module, "get_bm25_engine", return_value=mock_bm25
    ):
        response = _run_upload(file_obj)

    assert response.document_id == document_id
    assert response.chunk_count == 1
    mock_vector_store.add_document.assert_called_once()
    mock_parent_store.add_parent_chunks.assert_called_once()
    mock_bm25.invalidate.assert_called_once()


def test_upload_document_bm25_invalidate_failure_is_non_fatal(tmp_path):
    """BM25 invalidate errors are swallowed and upload still succeeds."""
    file_obj = _make_upload_file(filename="bm25_nonfatal.pdf", content=b"abc")
    document_id = "doc-bm25-nonfatal"

    mock_vector_store = MagicMock()
    mock_vector_store.add_document.return_value = 2

    with patch.object(
        documents_module,
        "settings",
        SimpleNamespace(uploads_path=str(tmp_path), max_file_size_mb=10),
    ), patch.object(documents_module.uuid, "uuid4", return_value=document_id), patch.object(
        documents_module, "parse_document", return_value=_parse_result(mode="flat", with_parent_chunks=False)
    ), patch.object(
        documents_module, "get_document_page_count", return_value=4
    ), patch.object(
        documents_module,
        "embed_document_chunks_parallel",
        new=AsyncMock(return_value=[{"chunk_id": "c1", "embedding": [0.1]}, {"chunk_id": "c2", "embedding": [0.2]}]),
    ), patch.object(
        documents_module, "get_vector_store", return_value=mock_vector_store
    ), patch.object(
        documents_module, "get_bm25_engine", side_effect=Exception("bm25 unavailable")
    ):
        response = _run_upload(file_obj)

    assert response.document_id == document_id
    assert response.chunk_count == 2
    mock_vector_store.add_document.assert_called_once()


def test_upload_document_skips_parent_store_when_mode_is_not_parent_child(tmp_path):
    """If parse mode is not parent_child, parent store is not touched."""
    file_obj = _make_upload_file(filename="flat_mode.pdf", content=b"abc")
    document_id = "doc-flat-mode"

    mock_vector_store = MagicMock()
    mock_vector_store.add_document.return_value = 1
    mock_bm25 = MagicMock()

    with patch.object(
        documents_module,
        "settings",
        SimpleNamespace(uploads_path=str(tmp_path), max_file_size_mb=10),
    ), patch.object(documents_module.uuid, "uuid4", return_value=document_id), patch.object(
        documents_module, "parse_document", return_value=_parse_result(mode="flat", with_parent_chunks=True)
    ), patch.object(
        documents_module, "get_document_page_count", return_value=2
    ), patch.object(
        documents_module,
        "embed_document_chunks_parallel",
        new=AsyncMock(return_value=[{"chunk_id": "c1", "embedding": [0.9]}]),
    ), patch.object(
        documents_module, "get_vector_store", return_value=mock_vector_store
    ), patch.object(
        documents_module, "get_parent_chunk_store"
    ) as parent_store_getter, patch.object(
        documents_module, "get_bm25_engine", return_value=mock_bm25
    ):
        response = _run_upload(file_obj)

    assert response.document_id == document_id
    parent_store_getter.assert_not_called()
    mock_bm25.invalidate.assert_called_once()