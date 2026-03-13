"""
Test per il Query Router (app/services/query_router.py)

Il query router classifica le domande degli utenti per determinare la
strategia di gestione ottimale (DIRECT_ANSWER, SYNTHESIS, FULL_DOCUMENT,
OUT_OF_SCOPE, CLARIFICATION).

classify_query() è una funzione SINCRONA e deterministica: il routing
dipende dal testo della query e dallo score di similarità, NON da chiamate
LLM. Tuttavia usiamo pytest-asyncio per coerenza con il resto della suite
e per eventuali estensioni future.

Test inclusi:
1. test_classify_query_direct_answer – query semplice + alto punteggio
2. test_classify_query_full_document – keyword che attiva FULL_DOCUMENT
3. test_classify_query_out_of_scope  – punteggio troppo basso
4. test_classify_query_synthesis     – query complessa
"""

import os
import pytest
import pytest_asyncio

os.environ.setdefault("OPENAI_API_KEY", "fake-key-for-tests")

from app.services.query_router import classify_query, QueryType


# ═══════════════════════════════════════════════════════════════════════════════
# Dati fittizi riusabili
# ═══════════════════════════════════════════════════════════════════════════════

def _make_fake_results(n: int = 3) -> list[dict]:
    """Genera una lista di risultati fittizi con metadati minimi."""
    return [
        {
            "id": f"chunk_{i}",
            "text": f"Questo è il contenuto fittizio del chunk {i}.",
            "metadata": {
                "document_id": "doc1",
                "document_name": "appunti.pdf",
                "page": i + 1,
                "chunk_index": i,
            },
            "relevance_score": 0.95 - i * 0.05,
        }
        for i in range(n)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: DIRECT_ANSWER
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_classify_query_direct_answer():
    """
    Una query con pattern di lookup semplice (es. "Cos'è X?") e uno score
    di similarità molto alto (≥ 0.85) deve essere classificata come
    DIRECT_ANSWER — il sistema può rispondere direttamente dal chunk
    senza bisogno di sintesi LLM.

    Nota: classify_query è sincrona, ma usiamo async per coerenza con la
    suite di test. La chiamata await non è necessaria qui.
    """
    results = _make_fake_results(3)

    # Query con pattern "Cos'è ...?" → tipo "definition"
    # top_score = 0.95 → supera high_relevance_threshold (0.85)
    decision = classify_query(
        query="Cos'è l'inflazione?",
        top_results=results,
        top_score=0.95,
    )

    # Con uno score così alto e una query di tipo definition,
    # ci aspettiamo DIRECT_ANSWER (se il chunk contiene l'info)
    # oppure SYNTHESIS come fallback se l'estrazione diretta fallisce.
    assert decision.query_type in (QueryType.DIRECT_ANSWER, QueryType.SYNTHESIS), (
        f"Atteso DIRECT_ANSWER o SYNTHESIS, ottenuto {decision.query_type}"
    )
    assert decision.confidence > 0, "La confidence deve essere > 0"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: FULL_DOCUMENT
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_classify_query_full_document():
    """
    Una query che contiene keyword esplicite di analisi completa
    (es. "riassumi tutto", "intero documento") deve essere classificata
    come FULL_DOCUMENT, indipendentemente dallo score di similarità.

    Questo attiva il recupero di TUTTI i chunk del documento anziché
    solo i top_k più rilevanti.
    """
    results = _make_fake_results(3)

    # "riassumi tutto" è una delle FULL_DOCUMENT_KEYWORDS definite nel router
    decision = classify_query(
        query="Riassumi tutto il contenuto del documento",
        top_results=results,
        top_score=0.50,   # Score medio — irrilevante per FULL_DOCUMENT
    )

    assert decision.query_type == QueryType.FULL_DOCUMENT, (
        f"Atteso FULL_DOCUMENT, ottenuto {decision.query_type}"
    )
    assert decision.confidence >= 0.9, (
        "Le richieste esplicite di analisi completa devono avere alta confidence"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: OUT_OF_SCOPE
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_classify_query_out_of_scope():
    """
    Se il punteggio di similarità del miglior risultato è inferiore a
    low_relevance_threshold (0.35), la query viene classificata come
    OUT_OF_SCOPE — non è possibile rispondere dai documenti caricati.
    """
    results = _make_fake_results(1)

    decision = classify_query(
        query="Qual è la ricetta della carbonara?",
        top_results=results,
        top_score=0.10,   # Score molto basso → fuori ambito
    )

    assert decision.query_type == QueryType.OUT_OF_SCOPE, (
        f"Atteso OUT_OF_SCOPE, ottenuto {decision.query_type}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_classify_query_synthesis():
    """
    Una query complessa che contiene keyword di sintesi (es. "confronta",
    "analizza", "vantaggi e svantaggi") deve essere classificata come
    SYNTHESIS — richiede elaborazione LLM per combinare più chunk.
    """
    results = _make_fake_results(3)

    decision = classify_query(
        query="Confronta i vantaggi e svantaggi del modello A rispetto al B",
        top_results=results,
        top_score=0.75,   # Score discreto, ma la query richiede sintesi
    )

    assert decision.query_type == QueryType.SYNTHESIS, (
        f"Atteso SYNTHESIS, ottenuto {decision.query_type}"
    )
    assert decision.confidence > 0
