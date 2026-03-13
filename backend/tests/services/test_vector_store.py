"""Tests for VectorStore service."""
import os
import shutil
import tempfile
import pytest
from unittest.mock import patch

from app.services.vector_store import VectorStore


@pytest.fixture()
def store(tmp_path):
    """Create a VectorStore with a temporary ChromaDB path."""
    chroma_path = str(tmp_path / "test_chroma")
    with patch("app.services.vector_store.settings") as mock_settings:
        mock_settings.chroma_db_path = chroma_path
        vs = VectorStore()
    yield vs


def _make_chunks(count=3, doc_id="doc1", page_start=1):
    """Helper to create test chunks with embeddings."""
    chunks = []
    for i in range(count):
        chunks.append({
            "text": f"Chunk {i} content for testing.",
            "embedding": [0.1 * (i + 1)] * 10,  # Simple fake embedding
            "metadata": {
                "chunk_index": i,
                "page": page_start + i,
            },
        })
    return chunks


class TestVectorStore:
    def test_add_document(self, store):
        chunks = _make_chunks(3)
        count = store.add_document("doc1", "test.pdf", chunks, notebook_id="nb1")
        assert count == 3

    def test_add_document_empty(self, store):
        count = store.add_document("doc1", "test.pdf", [])
        assert count == 0

    def test_search_similar(self, store):
        chunks = _make_chunks(3)
        store.add_document("doc1", "test.pdf", chunks, notebook_id="nb1")

        results = store.search_similar(
            query_embedding=[0.1] * 10,
            n_results=2,
            notebook_id="nb1",
        )
        assert "ids" in results
        assert len(results["ids"][0]) <= 2

    def test_search_similar_with_document_filter(self, store):
        store.add_document("doc1", "a.pdf", _make_chunks(2, "doc1"), notebook_id="nb1")
        store.add_document("doc2", "b.pdf", _make_chunks(2, "doc2"), notebook_id="nb1")

        results = store.search_similar(
            query_embedding=[0.1] * 10,
            n_results=10,
            document_ids=["doc1"],
        )
        for meta in results["metadatas"][0]:
            assert meta["document_id"] == "doc1"

    def test_delete_document(self, store):
        store.add_document("doc1", "test.pdf", _make_chunks(3))
        deleted = store.delete_document("doc1")
        assert deleted == 3
        assert store.document_exists("doc1") is False

    def test_delete_document_not_found(self, store):
        assert store.delete_document("nonexistent") == 0

    def test_document_exists(self, store):
        assert store.document_exists("doc1") is False
        store.add_document("doc1", "test.pdf", _make_chunks(1))
        assert store.document_exists("doc1") is True

    def test_list_documents(self, store):
        store.add_document("doc1", "a.pdf", _make_chunks(2, "doc1"), notebook_id="nb1")
        store.add_document("doc2", "b.pdf", _make_chunks(1, "doc2"), notebook_id="nb1")

        docs = store.list_documents(notebook_id="nb1")
        assert len(docs) == 2
        doc_ids = {d["document_id"] for d in docs}
        assert doc_ids == {"doc1", "doc2"}

    def test_list_documents_empty(self, store):
        assert store.list_documents() == []

    def test_list_documents_filter_notebook(self, store):
        store.add_document("doc1", "a.pdf", _make_chunks(1, "doc1"), notebook_id="nb1")
        store.add_document("doc2", "b.pdf", _make_chunks(1, "doc2"), notebook_id="nb2")

        docs = store.list_documents(notebook_id="nb1")
        assert len(docs) == 1
        assert docs[0]["document_id"] == "doc1"

    def test_update_document_notebook(self, store):
        store.add_document("doc1", "test.pdf", _make_chunks(2), notebook_id="nb1")
        updated = store.update_document_notebook("doc1", "nb2")
        assert updated == 2

        docs = store.list_documents(notebook_id="nb2")
        assert len(docs) == 1

    def test_update_document_notebook_not_found(self, store):
        assert store.update_document_notebook("nonexistent", "nb1") == 0

    def test_get_collection_stats(self, store):
        store.add_document("doc1", "test.pdf", _make_chunks(3))
        stats = store.get_collection_stats()
        assert stats["total_chunks"] == 3
        assert "collection_name" in stats

    def test_get_document_chunks(self, store):
        store.add_document("doc1", "test.pdf", _make_chunks(3))
        chunks = store.get_document_chunks(["doc1"])
        assert len(chunks) == 3

    def test_get_document_chunks_empty(self, store):
        assert store.get_document_chunks([]) == []

    def test_get_document_chunks_sorted(self, store):
        store.add_document("doc1", "test.pdf", _make_chunks(3))
        chunks = store.get_document_chunks(["doc1"])
        pages = [c["metadata"]["page"] for c in chunks]
        assert pages == sorted(pages)

    def test_add_document_with_parent_chunk_index(self, store):
        chunks = [{
            "text": "Child chunk",
            "embedding": [0.1] * 10,
            "metadata": {
                "chunk_index": 0,
                "page": 1,
                "parent_chunk_index": 5,
            },
        }]
        store.add_document("doc1", "test.pdf", chunks, notebook_id="nb1")
        result_chunks = store.get_document_chunks(["doc1"])
        assert result_chunks[0]["metadata"]["parent_chunk_index"] == 5
