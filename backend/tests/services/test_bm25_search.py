"""Tests for BM25SearchEngine and related functions."""
import pytest
from unittest.mock import patch, MagicMock

from app.services.bm25_search import (
    _tokenize,
    BM25SearchEngine,
    reciprocal_rank_fusion,
)


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = _tokenize("Questo è un test semplice")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_removes_stopwords(self):
        tokens = _tokenize("il la lo un una di del della")
        # All Italian stopwords should be removed
        assert tokens == []

    def test_removes_short_tokens(self):
        tokens = _tokenize("a b c test")
        # Single-char tokens should be removed; "test" stays (stemmed)
        raw_words = ["a", "b", "c", "test"]
        for t in tokens:
            assert len(t) > 1

    def test_stemming_applied(self):
        tokens1 = _tokenize("fatture")
        tokens2 = _tokenize("fatturato")
        # Both should produce the same stem
        assert tokens1[0] == tokens2[0]

    def test_empty_input(self):
        assert _tokenize("") == []

    def test_lowercasing(self):
        tokens = _tokenize("PROVA GRANDE")
        for t in tokens:
            assert t == t.lower()


class TestBM25SearchEngine:
    def test_init_empty(self):
        engine = BM25SearchEngine()
        assert engine._is_built is False
        assert engine._index is None

    def test_search_without_build_triggers_build(self):
        engine = BM25SearchEngine()
        with patch.object(engine, "build_index") as mock_build:
            engine._is_built = False
            engine.search("test query")
            mock_build.assert_called_once()

    def test_search_empty_index(self):
        engine = BM25SearchEngine()
        engine._is_built = True
        engine._index = None
        engine._corpus = []
        results = engine.search("prova")
        assert results == []

    def test_search_returns_results(self):
        engine = BM25SearchEngine()
        engine._is_built = True

        from rank_bm25 import BM25Okapi
        corpus = [
            _tokenize("La fattura numero 42 è stata pagata"),
            _tokenize("Il contratto scade domani"),
            _tokenize("Il report annuale contiene i dati finanziari"),
        ]
        engine._corpus = corpus
        engine._index = BM25Okapi(corpus)
        engine._chunk_ids = ["c1", "c2", "c3"]
        engine._chunk_texts = [
            "La fattura numero 42 è stata pagata",
            "Il contratto scade domani",
            "Il report annuale contiene i dati finanziari",
        ]
        engine._chunk_metadatas = [
            {"notebook_id": "nb1", "document_id": "d1"},
            {"notebook_id": "nb1", "document_id": "d2"},
            {"notebook_id": "nb2", "document_id": "d3"},
        ]

        results = engine.search("fattura pagata", n_results=2)
        assert len(results) > 0
        assert results[0]["bm25_score"] > 0

    def test_search_notebook_filter(self):
        engine = BM25SearchEngine()
        engine._is_built = True

        from rank_bm25 import BM25Okapi
        corpus = [_tokenize("test chunk one"), _tokenize("test chunk two")]
        engine._corpus = corpus
        engine._index = BM25Okapi(corpus)
        engine._chunk_ids = ["c1", "c2"]
        engine._chunk_texts = ["test chunk one", "test chunk two"]
        engine._chunk_metadatas = [
            {"notebook_id": "nb1", "document_id": "d1"},
            {"notebook_id": "nb2", "document_id": "d2"},
        ]

        results = engine.search("test chunk", notebook_id="nb1")
        assert all(r["metadata"]["notebook_id"] == "nb1" for r in results)

    def test_search_document_ids_filter(self):
        engine = BM25SearchEngine()
        engine._is_built = True

        from rank_bm25 import BM25Okapi
        corpus = [_tokenize("test chunk one"), _tokenize("test chunk two")]
        engine._corpus = corpus
        engine._index = BM25Okapi(corpus)
        engine._chunk_ids = ["c1", "c2"]
        engine._chunk_texts = ["test chunk one", "test chunk two"]
        engine._chunk_metadatas = [
            {"notebook_id": "nb1", "document_id": "d1"},
            {"notebook_id": "nb1", "document_id": "d2"},
        ]

        results = engine.search("test chunk", document_ids=["d1"])
        assert all(r["metadata"]["document_id"] == "d1" for r in results)

    def test_invalidate(self):
        engine = BM25SearchEngine()
        engine._is_built = True
        engine.invalidate()
        assert engine._is_built is False

    def test_get_stats(self):
        engine = BM25SearchEngine()
        stats = engine.get_stats()
        assert "is_built" in stats
        assert "total_chunks" in stats
        assert "corpus_size" in stats


class TestReciprocalRankFusion:
    def test_basic_fusion(self):
        vector = [
            {"id": "a", "text": "doc a", "relevance_score": 0.9},
            {"id": "b", "text": "doc b", "relevance_score": 0.8},
        ]
        bm25 = [
            {"id": "b", "text": "doc b", "bm25_score": 5.0},
            {"id": "c", "text": "doc c", "bm25_score": 3.0},
        ]
        fused = reciprocal_rank_fusion(vector, bm25)
        ids = [r["id"] for r in fused]
        # "b" appears in both lists so it should rank high
        assert "b" in ids
        assert "a" in ids
        assert "c" in ids

    def test_empty_inputs(self):
        result = reciprocal_rank_fusion([], [])
        assert result == []

    def test_vector_only(self):
        vector = [{"id": "a", "text": "doc a", "relevance_score": 0.9}]
        result = reciprocal_rank_fusion(vector, [])
        assert len(result) == 1
        assert result[0]["id"] == "a"

    def test_bm25_only(self):
        bm25 = [{"id": "a", "text": "doc a", "bm25_score": 5.0}]
        result = reciprocal_rank_fusion([], bm25)
        assert len(result) == 1
        assert result[0]["id"] == "a"

    def test_rrf_scores_are_present(self):
        vector = [{"id": "a", "text": "t", "relevance_score": 0.9}]
        bm25 = [{"id": "a", "text": "t", "bm25_score": 5.0}]
        result = reciprocal_rank_fusion(vector, bm25)
        assert result[0]["rrf_score"] > 0
