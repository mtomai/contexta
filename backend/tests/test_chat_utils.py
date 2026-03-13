"""
Test per le funzioni pure di chat_utils.py

Queste funzioni NON dipendono da stato esterno (DB, API, file system),
quindi possiamo testarle direttamente senza mock pesanti.

Test inclusi:
1. _fuse_results  – Reciprocal Rank Fusion di risultati Vector + BM25
2. format_context_xml – Formattazione XML del contesto per l'LLM
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Impostiamo la variabile d'ambiente PRIMA di importare qualsiasi modulo
# del progetto, perché config.py la richiede al caricamento.
os.environ.setdefault("OPENAI_API_KEY", "fake-key-for-tests")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: _fuse_results (Reciprocal Rank Fusion)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFuseResults:
    """Test suite per la funzione di Reciprocal Rank Fusion."""

    def test_fuse_results_ordering(self):
        """
        Verifica che _fuse_results unisca due liste classificate (vector e BM25)
        e restituisca i risultati ordinati per score RRF decrescente.

        Scenario:
        - Vector: [A (rank 0), B (rank 1)]
        - BM25:   [B (rank 0), C (rank 1)]
        - B appare in ENTRAMBE le liste → ha lo score RRF più alto
        - A appare solo nel vector, C solo in BM25
        """
        from app.services.chat_utils import _fuse_results

        # Risultati dal vector search (rank 0 = più rilevante)
        vector_results = [
            {"id": "chunk_A", "text": "Testo di A", "relevance_score": 0.95},
            {"id": "chunk_B", "text": "Testo di B", "relevance_score": 0.80},
        ]

        # Risultati dal BM25 search (rank 0 = più rilevante)
        bm25_results = [
            {"id": "chunk_B", "text": "Testo di B", "relevance_score": 5.2},
            {"id": "chunk_C", "text": "Testo di C", "relevance_score": 3.1},
        ]

        fused = _fuse_results(
            vector_results=vector_results,
            bm25_results=bm25_results,
            top_k=10,
            vector_weight=0.7,
            bm25_weight=0.3,
        )

        # chunk_B deve essere primo perché appare in ENTRAMBE le liste
        assert fused[0]["id"] == "chunk_B", (
            "Il chunk presente in entrambe le liste deve avere lo score RRF più alto"
        )
        # Tutti e 3 i chunk devono essere presenti
        fused_ids = [r["id"] for r in fused]
        assert set(fused_ids) == {"chunk_A", "chunk_B", "chunk_C"}

    def test_fuse_results_respects_top_k(self):
        """
        Verifica che _fuse_results limiti i risultati a top_k.
        """
        from app.services.chat_utils import _fuse_results

        # Creiamo molti risultati fittizi
        vector_results = [
            {"id": f"v_{i}", "text": f"Vector {i}", "relevance_score": 0.9 - i * 0.1}
            for i in range(5)
        ]
        bm25_results = [
            {"id": f"b_{i}", "text": f"BM25 {i}", "relevance_score": 4.0 - i * 0.5}
            for i in range(5)
        ]

        top_k = 3
        fused = _fuse_results(vector_results, bm25_results, top_k=top_k)

        assert len(fused) <= top_k, (
            f"Il numero di risultati ({len(fused)}) supera top_k={top_k}"
        )

    def test_fuse_results_empty_lists(self):
        """
        Verifica che _fuse_results gestisca liste vuote senza errori.
        """
        from app.services.chat_utils import _fuse_results

        fused = _fuse_results([], [], top_k=5)
        assert fused == [], "La fusione di due liste vuote deve restituire una lista vuota"

    def test_fuse_results_single_source(self):
        """
        Verifica che funzioni correttamente anche con una sola sorgente non vuota.
        """
        from app.services.chat_utils import _fuse_results

        vector_results = [
            {"id": "only_v", "text": "Solo vector", "relevance_score": 0.9},
        ]

        fused = _fuse_results(vector_results, [], top_k=5)
        assert len(fused) == 1
        assert fused[0]["id"] == "only_v"

    def test_fuse_results_has_relevance_score(self):
        """
        Verifica che ogni risultato fuso abbia un campo 'relevance_score'
        aggiornato con lo score RRF calcolato.
        """
        from app.services.chat_utils import _fuse_results

        vector_results = [{"id": "x", "text": "X", "relevance_score": 0.9}]
        bm25_results = [{"id": "x", "text": "X", "relevance_score": 3.0}]

        fused = _fuse_results(vector_results, bm25_results, top_k=5)
        assert "relevance_score" in fused[0]
        # Lo score RRF deve essere > 0
        assert fused[0]["relevance_score"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: format_context_xml
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatContextXml:
    """Test suite per la formattazione XML del contesto LLM."""

    def test_format_context_xml_basic(self):
        """
        Verifica che format_context_xml produca tag <document> corretti
        con gli attributi name e page estratti dai metadati.
        """
        from app.services.chat_utils import format_context_xml

        context_items = [
            {
                "context_text": "Il PIL italiano è cresciuto del 2%.",
                "metadata": {
                    "document_name": "economia.pdf",
                    "page": 5,
                },
            },
            {
                "context_text": "La riforma fiscale è stata approvata.",
                "metadata": {
                    "document_name": "politica.pdf",
                    "page": 12,
                },
            },
        ]

        result = format_context_xml(context_items)

        # Verifica che ogni documento sia racchiuso nel tag XML corretto
        assert '<document name="economia.pdf" page="5">' in result
        assert '<document name="politica.pdf" page="12">' in result

        # Verifica che il testo dei chunk sia presente
        assert "Il PIL italiano è cresciuto del 2%." in result
        assert "La riforma fiscale è stata approvata." in result

        # Verifica che ci siano i tag di chiusura
        assert result.count("</document>") == 2

    def test_format_context_xml_skips_none_text(self):
        """
        Verifica che i chunk con context_text=None vengano ignorati.
        """
        from app.services.chat_utils import format_context_xml

        context_items = [
            {"context_text": None, "metadata": {"document_name": "vuoto.pdf", "page": 1}},
            {"context_text": "Testo valido.", "metadata": {"document_name": "ok.pdf", "page": 3}},
        ]

        result = format_context_xml(context_items)

        # Solo il secondo documento deve apparire
        assert "vuoto.pdf" not in result
        assert '<document name="ok.pdf" page="3">' in result
        assert result.count("</document>") == 1

    def test_format_context_xml_empty_list(self):
        """
        Una lista vuota deve restituire una stringa vuota.
        """
        from app.services.chat_utils import format_context_xml

        assert format_context_xml([]) == ""

    def test_format_context_xml_default_metadata(self):
        """
        Verifica i valori di default quando i metadati sono incompleti:
        - document_name → "Documento Sconosciuto"
        - page → 0
        """
        from app.services.chat_utils import format_context_xml

        context_items = [
            {"context_text": "Testo senza metadati completi.", "metadata": {}},
        ]

        result = format_context_xml(context_items)

        assert '<document name="Documento Sconosciuto" page="0">' in result
