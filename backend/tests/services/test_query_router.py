"""Tests for QueryRouter service."""
import pytest

from app.services.query_router import (
    QueryType,
    RoutingDecision,
    classify_query,
    _requires_synthesis,
    _try_extract_direct_answer,
    FULL_DOCUMENT_KEYWORDS,
    SYNTHESIS_KEYWORDS,
)


class TestClassifyQuery:
    def _make_results(self, score=0.9):
        return [{
            "id": "c1",
            "text": "Chunk di test con contenuto.",
            "metadata": {
                "document_id": "doc1",
                "document_name": "test.pdf",
                "page": 1,
                "chunk_index": 0,
            },
            "relevance_score": score,
        }]

    def test_full_document_keyword(self):
        results = self._make_results()
        decision = classify_query("riassumi tutto il documento", results, 0.9)
        assert decision.query_type == QueryType.FULL_DOCUMENT

    def test_out_of_scope_low_score(self):
        results = self._make_results(score=0.1)
        decision = classify_query("qualcosa di irrilevante?", results, 0.1)
        assert decision.query_type == QueryType.OUT_OF_SCOPE

    def test_synthesis_keyword(self):
        results = self._make_results()
        decision = classify_query("confronta i due approcci", results, 0.7)
        assert decision.query_type == QueryType.SYNTHESIS

    def test_default_synthesis(self):
        results = self._make_results()
        decision = classify_query("prova generica", results, 0.7)
        assert decision.query_type == QueryType.SYNTHESIS

    def test_direct_answer_definition(self):
        results = [{
            "id": "c1",
            "text": "Il machine learning è una branca dell'intelligenza artificiale.",
            "metadata": {
                "document_id": "doc1",
                "document_name": "test.pdf",
                "page": 1,
                "chunk_index": 0,
            },
            "relevance_score": 0.95,
        }]
        decision = classify_query("cos'è il machine learning?", results, 0.95)
        # Could be DIRECT_ANSWER or SYNTHESIS depending on extraction
        assert decision.query_type in (QueryType.DIRECT_ANSWER, QueryType.SYNTHESIS)

    def test_routing_decision_has_confidence(self):
        results = self._make_results()
        decision = classify_query("test query", results, 0.7)
        assert isinstance(decision.confidence, float)
        assert decision.confidence > 0

    def test_routing_decision_has_reason(self):
        results = self._make_results()
        decision = classify_query("test query", results, 0.7)
        assert isinstance(decision.reason, str)
        assert len(decision.reason) > 0


class TestRequiresSynthesis:
    def test_synthesis_keyword_detected(self):
        assert _requires_synthesis("confronta i risultati") is True

    def test_no_synthesis_needed(self):
        assert _requires_synthesis("prova") is False

    def test_multiple_question_marks(self):
        assert _requires_synthesis("come funziona? e perché?") is True

    def test_long_query(self):
        # More than 15 words
        query = " ".join(["parola"] * 20)
        assert _requires_synthesis(query) is True


class TestTryExtractDirectAnswer:
    def test_extracts_definition(self):
        chunk = {
            "text": "Il DNA è una molecola che contiene le istruzioni genetiche.",
            "metadata": {"document_name": "bio.pdf", "page": 5},
        }
        result = _try_extract_direct_answer(
            "cos'è il dna?", "il dna", chunk, "definition"
        )
        assert result is not None
        assert "DNA" in result or "dna" in result.lower()

    def test_returns_none_no_match(self):
        chunk = {
            "text": "Testo generico senza definizioni.",
            "metadata": {"document_name": "doc.pdf", "page": 1},
        }
        result = _try_extract_direct_answer(
            "cos'è il test?", "il test", chunk, "definition"
        )
        assert result is None

    def test_returns_none_no_chunk(self):
        result = _try_extract_direct_answer(
            "cos'è il test?", "il test", None, "definition"
        )
        assert result is None

    def test_non_definition_pattern_returns_none(self):
        chunk = {
            "text": "Qualcosa.",
            "metadata": {"document_name": "doc.pdf", "page": 1},
        }
        result = _try_extract_direct_answer(
            "chi è Marco?", "Marco", chunk, "entity"
        )
        assert result is None


class TestFullDocumentKeywords:
    def test_keywords_list_not_empty(self):
        assert len(FULL_DOCUMENT_KEYWORDS) > 0

    @pytest.mark.parametrize("keyword", FULL_DOCUMENT_KEYWORDS)
    def test_each_keyword_triggers_full_document(self, keyword):
        results = [{
            "id": "c1", "text": "t",
            "metadata": {"document_id": "d1", "document_name": "f.pdf", "page": 1, "chunk_index": 0},
            "relevance_score": 0.9,
        }]
        decision = classify_query(keyword, results, 0.9)
        assert decision.query_type == QueryType.FULL_DOCUMENT
