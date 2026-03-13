"""Tests for ReRanker service."""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from app.services.reranker import ReRanker


@pytest.fixture()
def reranker():
    """Create a ReRanker with a mocked CrossEncoder model."""
    with patch("app.services.reranker.CrossEncoder") as MockCE:
        mock_model = MagicMock()
        MockCE.return_value = mock_model
        rr = ReRanker(model_name="test-model")
        yield rr, mock_model


class TestReRanker:
    def test_init(self, reranker):
        rr, _ = reranker
        assert rr.model_name == "test-model"
        assert rr.model is not None

    def test_rerank_empty(self, reranker):
        rr, _ = reranker
        result = rr.rerank("query", [], top_k=5)
        assert result == []

    def test_rerank_returns_top_k(self, reranker):
        rr, mock_model = reranker
        mock_model.predict.return_value = np.array([0.9, 0.3, 0.7, 0.1])

        results = [
            {"id": "a", "text": "text a"},
            {"id": "b", "text": "text b"},
            {"id": "c", "text": "text c"},
            {"id": "d", "text": "text d"},
        ]
        reranked = rr.rerank("query", results, top_k=2)
        assert len(reranked) == 2
        assert reranked[0]["id"] == "a"  # highest score
        assert reranked[1]["id"] == "c"  # second highest

    def test_rerank_updates_relevance_score(self, reranker):
        rr, mock_model = reranker
        mock_model.predict.return_value = np.array([0.5])

        results = [{"id": "a", "text": "text a"}]
        reranked = rr.rerank("query", results, top_k=1)
        assert reranked[0]["relevance_score"] == 0.5

    def test_rerank_sorted_descending(self, reranker):
        rr, mock_model = reranker
        mock_model.predict.return_value = np.array([0.2, 0.8, 0.5])

        results = [
            {"id": "a", "text": "a"},
            {"id": "b", "text": "b"},
            {"id": "c", "text": "c"},
        ]
        reranked = rr.rerank("query", results, top_k=3)
        scores = [r["relevance_score"] for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_passes_correct_pairs(self, reranker):
        rr, mock_model = reranker
        mock_model.predict.return_value = np.array([0.5, 0.3])

        results = [
            {"id": "a", "text": "text a"},
            {"id": "b", "text": "text b"},
        ]
        rr.rerank("my query", results, top_k=2)

        call_args = mock_model.predict.call_args[0][0]
        assert call_args == [["my query", "text a"], ["my query", "text b"]]

    def test_rerank_top_k_larger_than_results(self, reranker):
        rr, mock_model = reranker
        mock_model.predict.return_value = np.array([0.5])

        results = [{"id": "a", "text": "a"}]
        reranked = rr.rerank("query", results, top_k=10)
        assert len(reranked) == 1
