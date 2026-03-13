"""Tests for chat_utils service."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.services.chat_utils import (
    _fuse_results,
    extract_citations,
    format_context_xml,
    get_formatted_notes_context,
    resolve_parent_context,
)


class TestFuseResults:
    def test_basic_fusion(self):
        vector = [
            {"id": "a", "text": "doc a", "relevance_score": 0.9},
            {"id": "b", "text": "doc b", "relevance_score": 0.8},
        ]
        bm25 = [
            {"id": "b", "text": "doc b", "bm25_score": 5.0},
            {"id": "c", "text": "doc c", "bm25_score": 3.0},
        ]
        fused = _fuse_results(vector, bm25, top_k=10)
        ids = [r["id"] for r in fused]
        assert "a" in ids
        assert "b" in ids
        assert "c" in ids

    def test_empty_inputs(self):
        assert _fuse_results([], [], top_k=5) == []

    def test_top_k_limit(self):
        vector = [{"id": f"v{i}", "text": f"t{i}", "relevance_score": 0.5} for i in range(10)]
        fused = _fuse_results(vector, [], top_k=3)
        assert len(fused) == 3

    def test_weights(self):
        vector = [{"id": "a", "text": "t", "relevance_score": 0.9}]
        bm25 = [{"id": "b", "text": "t", "bm25_score": 5.0}]
        fused = _fuse_results(vector, bm25, top_k=2, vector_weight=0.9, bm25_weight=0.1)
        # "a" should rank higher with higher vector weight
        assert fused[0]["id"] == "a"

    def test_overlapping_ids_get_combined_score(self):
        vector = [{"id": "x", "text": "t", "relevance_score": 0.9}]
        bm25 = [{"id": "x", "text": "t", "bm25_score": 5.0}]
        fused = _fuse_results(vector, bm25, top_k=5)
        assert len(fused) == 1
        fused_score = fused[0]["relevance_score"]
        # Score should be higher than either alone (use fresh dicts to avoid mutation)
        v_only = _fuse_results([{"id": "x", "text": "t", "relevance_score": 0.9}], [], top_k=5)
        assert fused_score > v_only[0]["relevance_score"]


class TestExtractCitations:
    def test_extracts_matching_citation(self):
        answer = "La risposta è qui [test.pdf, pagina 1]."
        sources = [
            {"document": "test.pdf", "page": 1, "chunk_index": 0, "chunk_text": "t", "relevance_score": 0.9},
            {"document": "other.pdf", "page": 2, "chunk_index": 0, "chunk_text": "t", "relevance_score": 0.8},
        ]
        cited = extract_citations(answer, sources)
        assert len(cited) >= 1
        assert any(s["document"] == "test.pdf" for s in cited)

    def test_fallback_top3(self):
        answer = "Risposta senza citazioni."
        sources = [
            {"document": f"doc{i}.pdf", "page": i, "chunk_index": i, "relevance_score": 1.0 - i * 0.1}
            for i in range(5)
        ]
        cited = extract_citations(answer, sources)
        assert len(cited) <= 3

    def test_no_sources(self):
        cited = extract_citations("Test answer", [])
        assert cited == []

    def test_pag_abbreviation(self):
        answer = "Vedi [report.pdf, pag. 3]."
        sources = [
            {"document": "report.pdf", "page": 3, "chunk_index": 0, "chunk_text": "t", "relevance_score": 0.9},
        ]
        cited = extract_citations(answer, sources)
        assert len(cited) == 1

    def test_sorted_by_relevance(self):
        answer = "Citazione [a.pdf, pagina 1] e [b.pdf, pagina 2]."
        sources = [
            {"document": "a.pdf", "page": 1, "chunk_index": 0, "chunk_text": "t", "relevance_score": 0.5},
            {"document": "b.pdf", "page": 2, "chunk_index": 0, "chunk_text": "t", "relevance_score": 0.9},
        ]
        cited = extract_citations(answer, sources)
        scores = [s["relevance_score"] for s in cited]
        assert scores == sorted(scores, reverse=True)


class TestFormatContextXml:
    def test_basic_format(self):
        items = [{
            "context_text": "Chunk content",
            "metadata": {"document_name": "test.pdf", "page": 1},
        }]
        result = format_context_xml(items)
        assert '<document name="test.pdf" page="1">' in result
        assert "Chunk content" in result
        assert "</document>" in result

    def test_skips_none_text(self):
        items = [
            {"context_text": None, "metadata": {"document_name": "a.pdf", "page": 1}},
            {"context_text": "Valid", "metadata": {"document_name": "b.pdf", "page": 2}},
        ]
        result = format_context_xml(items)
        assert "a.pdf" not in result
        assert "Valid" in result

    def test_empty_list(self):
        assert format_context_xml([]) == ""

    def test_missing_metadata(self):
        items = [{"context_text": "Text", "metadata": {}}]
        result = format_context_xml(items)
        assert "Documento Sconosciuto" in result


class TestGetFormattedNotesContext:
    @patch("app.services.chat_utils.get_note_db")
    def test_returns_notes(self, mock_get_note_db):
        mock_db = MagicMock()
        mock_db.list_notes.return_value = [
            {"title": "My Note", "content": "Note content here"},
        ]
        mock_get_note_db.return_value = mock_db

        result = get_formatted_notes_context("nb1")
        assert "Nota: My Note" in result
        assert "Note content here" in result

    @patch("app.services.chat_utils.get_note_db")
    def test_empty_notes(self, mock_get_note_db):
        mock_db = MagicMock()
        mock_db.list_notes.return_value = []
        mock_get_note_db.return_value = mock_db

        result = get_formatted_notes_context("nb1")
        assert result == ""

    @patch("app.services.chat_utils.get_note_db")
    def test_db_error_returns_empty(self, mock_get_note_db):
        mock_get_note_db.side_effect = Exception("DB error")
        result = get_formatted_notes_context("nb1")
        assert result == ""


class TestResolveParentContext:
    @patch("app.services.chat_utils.get_settings")
    def test_parent_child_disabled(self, mock_settings):
        mock_settings.return_value.enable_parent_child = False
        results = [
            {
                "text": "child text",
                "metadata": {"document_id": "d1", "page": 1},
                "relevance_score": 0.9,
            }
        ]
        contexts = resolve_parent_context(results)
        assert len(contexts) == 1
        assert contexts[0]["context_text"] == "child text"
        assert contexts[0]["citation_text"] == "child text"

    @patch("app.services.chat_utils.get_parent_chunk_store")
    @patch("app.services.chat_utils.get_settings")
    def test_parent_child_enabled(self, mock_settings, mock_parent_store):
        mock_settings.return_value.enable_parent_child = True
        mock_store = MagicMock()
        mock_store.get_parent_chunks_batch.return_value = {
            "d1:0": {"text": "parent text", "page": 1, "document_name": "test.pdf"},
        }
        mock_parent_store.return_value = mock_store

        results = [
            {
                "text": "child text",
                "metadata": {"document_id": "d1", "parent_chunk_index": 0, "page": 1},
                "relevance_score": 0.9,
            }
        ]
        contexts = resolve_parent_context(results)
        assert len(contexts) == 1
        assert contexts[0]["context_text"] == "parent text"
        assert contexts[0]["citation_text"] == "child text"
