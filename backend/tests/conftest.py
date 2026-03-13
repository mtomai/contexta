"""
Fixtures condivise per tutti i test del progetto.

Questo file viene caricato automaticamente da pytest e contiene:
- mock_settings:      sovrascrive get_settings() con configurazioni di test
                      (nessuna API key reale, path temporanei, ecc.)
- mock_openai:        mocka il client AsyncOpenAI restituendo risposte fittizie
- mock_vector_store:  mocka il singleton VectorStore con un metodo search_similar fittizio
- app_client:         TestClient FastAPI pronto all'uso (tutti i mock già attivi)
"""

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi.testclient import TestClient


# ── Fake Settings ─────────────────────────────────────────────────────────────
# Creiamo un oggetto Settings con valori di test PRIMA di importare l'app,
# così nessun modulo tenta di leggere variabili d'ambiente reali.

def _make_test_settings():
    """Costruisce un'istanza Settings con valori sicuri per i test."""
    # Impostiamo la variabile d'ambiente richiesta da pydantic-settings
    os.environ.setdefault("OPENAI_API_KEY", "fake-key-for-tests")

    from app.config import Settings
    return Settings(
        openai_api_key="fake-key-for-tests",
        embedding_model="text-embedding-3-large",
        chat_model="gpt-4o-mini",
        light_model="gpt-4o-mini",
        long_context_model="gpt-4o-mini",
        vision_model="gpt-4o-mini",
        chroma_db_path="./test_chroma_db",
        uploads_path="./test_uploads",
        max_file_size_mb=10,
        chunk_size=500,
        chunk_overlap=50,
        default_top_k=5,
        enable_hybrid_search=True,
        bm25_weight=0.3,
        vector_weight=0.7,
        rrf_k=60,
        high_relevance_threshold=0.85,
        low_relevance_threshold=0.35,
    )


# ── Fixture: mock_settings ────────────────────────────────────────────────────
# Sostituisce globalmente get_settings() in modo che TUTTI i moduli
# ricevano la configurazione di test, senza toccare .env o variabili reali.

@pytest.fixture()
def mock_settings():
    """Override di get_settings() → restituisce Settings di test."""
    fake = _make_test_settings()
    with patch("app.config.get_settings", return_value=fake) as mocked:
        yield fake


# ── Fixture: mock_openai ──────────────────────────────────────────────────────
# Mocka il client AsyncOpenAI ovunque venga istanziato nel progetto.
# La risposta fittizia simula la struttura reale di OpenAI:
#   response.choices[0].message.content  →  stringa di testo
#   response.choices[0].finish_reason    →  "stop"

def _build_fake_completion(content: str = "Risposta fittizia dal mock OpenAI."):
    """Crea un oggetto mock che imita ChatCompletion di OpenAI."""
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = "stop"

    completion = MagicMock()
    completion.choices = [choice]
    return completion


@pytest.fixture()
def mock_openai():
    """
    Mocka AsyncOpenAI in chat.py e chat_utils.py.

    Il mock restituisce una risposta fittizia con finish_reason="stop"
    così il loop di auto-continuation non si attiva.
    """
    fake_completion = _build_fake_completion()

    # Creiamo un client fittizio con il metodo asincrono .chat.completions.create()
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=fake_completion)

    # Patchiamo i factory function che restituiscono il client OpenAI
    # sia nel modulo chat.py che in chat_utils.py
    with patch("app.services.chat.get_openai_client", return_value=fake_client) as m1, \
         patch("app.services.chat_utils._get_openai_client", return_value=fake_client) as m2:
        yield fake_client


# ── Fixture: mock_vector_store ────────────────────────────────────────────────
# Mocka il singleton VectorStore restituendo risultati fittizi da search_similar.

@pytest.fixture()
def mock_vector_store():
    """
    Mocka get_vector_store() → restituisce un VectorStore fittizio.

    search_similar() ritorna una struttura compatibile con ChromaDB:
    {
        "ids": [...], "documents": [...], "metadatas": [...], "distances": [...]
    }
    """
    fake_store = MagicMock()

    # Risposta fittizia di search_similar: un singolo chunk fittizio
    fake_store.search_similar.return_value = {
        "ids": [["doc1_0"]],
        "documents": [["Questo è un chunk di test."]],
        "metadatas": [[{
            "document_id": "doc1",
            "document_name": "test.pdf",
            "page": 1,
            "chunk_index": 0,
            "notebook_id": "nb-test-1",
        }]],
        "distances": [[0.15]],
    }

    fake_store.list_documents.return_value = [
        {"document_id": "doc1", "document_name": "test.pdf"}
    ]

    with patch("app.services.vector_store.get_vector_store", return_value=fake_store), \
         patch("app.services.chat_utils.get_vector_store", return_value=fake_store):
        yield fake_store


# ── Fixture: app_client ───────────────────────────────────────────────────────
# TestClient con mock del reranker (evita il download del modello ML)
# e mock di get_settings (evita la lettura di .env).

@pytest.fixture()
def app_client(mock_settings):
    """
    TestClient FastAPI con i componenti pesanti già mockati.

    - get_reranker() → MagicMock (nessun modello ML caricato)
    - get_settings() → Settings di test (nessuna API key reale)
    - Path.mkdir()  → no-op (nessuna directory creata su disco)
    """
    with patch("app.main.get_reranker", return_value=MagicMock()), \
         patch("app.main.get_settings", return_value=mock_settings), \
         patch("pathlib.Path.mkdir"):

        # Importiamo l'app QUI, dentro il contesto dei mock, così il
        # lifespan non tenta di caricare il modello reranker reale.
        from app.main import app
        client = TestClient(app)
        yield client
