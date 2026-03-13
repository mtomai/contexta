"""Tests for ParentChunkStore service."""
import os
import tempfile
import pytest

from app.services.parent_chunk_store import ParentChunkStore


@pytest.fixture()
def store():
    """Create a temporary ParentChunkStore for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        yield ParentChunkStore(db_path=path)
    finally:
        os.unlink(path)


class TestParentChunkStore:
    def test_add_parent_chunks(self, store):
        chunks = [
            {
                "text": "Parent chunk 1",
                "metadata": {"parent_chunk_index": 0, "page": 1},
            },
            {
                "text": "Parent chunk 2",
                "metadata": {"parent_chunk_index": 1, "page": 2},
            },
        ]
        count = store.add_parent_chunks(
            document_id="doc1",
            document_name="test.pdf",
            parent_chunks=chunks,
            notebook_id="nb1",
        )
        assert count == 2

    def test_add_parent_chunks_empty(self, store):
        assert store.add_parent_chunks("doc1", "test.pdf", []) == 0

    def test_get_parent_chunk(self, store):
        chunks = [
            {
                "text": "My parent chunk text",
                "metadata": {"parent_chunk_index": 0, "page": 3},
            }
        ]
        store.add_parent_chunks("doc1", "test.pdf", chunks, notebook_id="nb1")
        result = store.get_parent_chunk("doc1", parent_index=0)
        assert result is not None
        assert result["text"] == "My parent chunk text"
        assert result["page"] == 3
        assert result["document_name"] == "test.pdf"

    def test_get_parent_chunk_not_found(self, store):
        assert store.get_parent_chunk("doc1", parent_index=99) is None

    def test_get_parent_chunks_batch(self, store):
        chunks = [
            {"text": "chunk 0", "metadata": {"parent_chunk_index": 0, "page": 1}},
            {"text": "chunk 1", "metadata": {"parent_chunk_index": 1, "page": 2}},
        ]
        store.add_parent_chunks("doc1", "test.pdf", chunks)

        lookups = [
            {"document_id": "doc1", "parent_index": 0},
            {"document_id": "doc1", "parent_index": 1},
        ]
        results = store.get_parent_chunks_batch(lookups)
        assert "doc1:0" in results
        assert "doc1:1" in results
        assert results["doc1:0"]["text"] == "chunk 0"

    def test_get_parent_chunks_batch_empty(self, store):
        assert store.get_parent_chunks_batch([]) == {}

    def test_delete_document(self, store):
        chunks = [
            {"text": "chunk", "metadata": {"parent_chunk_index": 0, "page": 1}},
        ]
        store.add_parent_chunks("doc1", "test.pdf", chunks)
        deleted = store.delete_document("doc1")
        assert deleted == 1
        assert store.get_parent_chunk("doc1", 0) is None

    def test_delete_document_not_found(self, store):
        assert store.delete_document("nonexistent") == 0

    def test_update_notebook(self, store):
        chunks = [
            {"text": "chunk", "metadata": {"parent_chunk_index": 0, "page": 1}},
        ]
        store.add_parent_chunks("doc1", "test.pdf", chunks, notebook_id="nb1")
        updated = store.update_notebook("doc1", "nb2")
        assert updated == 1
        result = store.get_parent_chunk("doc1", 0)
        assert result["notebook_id"] == "nb2"

    def test_get_document_parent_chunks(self, store):
        chunks = [
            {"text": "chunk 0", "metadata": {"parent_chunk_index": 0, "page": 1}},
            {"text": "chunk 1", "metadata": {"parent_chunk_index": 1, "page": 2}},
        ]
        store.add_parent_chunks("doc1", "test.pdf", chunks)
        result = store.get_document_parent_chunks("doc1")
        assert len(result) == 2
        assert result[0]["parent_index"] == 0
        assert result[1]["parent_index"] == 1

    def test_get_document_parent_chunks_empty(self, store):
        assert store.get_document_parent_chunks("nonexistent") == []

    def test_get_documents_parent_chunks_batch(self, store):
        store.add_parent_chunks(
            "doc1", "a.pdf",
            [{"text": "a0", "metadata": {"parent_chunk_index": 0, "page": 1}}],
        )
        store.add_parent_chunks(
            "doc2", "b.pdf",
            [{"text": "b0", "metadata": {"parent_chunk_index": 0, "page": 1}}],
        )
        results = store.get_documents_parent_chunks_batch(["doc1", "doc2"])
        assert len(results) == 2
        doc_ids = {r["document_id"] for r in results}
        assert doc_ids == {"doc1", "doc2"}

    def test_get_documents_parent_chunks_batch_empty(self, store):
        assert store.get_documents_parent_chunks_batch([]) == []

    def test_add_parent_chunks_replace(self, store):
        chunks = [
            {"text": "original", "metadata": {"parent_chunk_index": 0, "page": 1}},
        ]
        store.add_parent_chunks("doc1", "test.pdf", chunks)
        # Replace with new text
        chunks2 = [
            {"text": "replaced", "metadata": {"parent_chunk_index": 0, "page": 1}},
        ]
        store.add_parent_chunks("doc1", "test.pdf", chunks2)
        result = store.get_parent_chunk("doc1", 0)
        assert result["text"] == "replaced"
