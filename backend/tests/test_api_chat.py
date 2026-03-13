"""
Test di integrazione per l'endpoint POST /api/chat

Usa il TestClient di FastAPI (fixture `app_client` da conftest.py) per
simulare richieste HTTP reali. Tutti i componenti pesanti (OpenAI,
VectorStore, ConversationDB, embedding, reranker, BM25) sono mockati
così i test girano in millisecondi senza rete o disco.

Test inclusi:
1. test_chat_standard_endpoint     – risposta 200 con chiavi "answer" e "sources"
2. test_chat_empty_query           – validazione: query vuota → 400
3. test_chat_response_has_sources  – la lista sources è una lista valida
"""

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

os.environ.setdefault("OPENAI_API_KEY", "fake-key-for-tests")

from app.models.chat import Source


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: risposta fittizia di generate_response
# ═══════════════════════════════════════════════════════════════════════════════

def _fake_generate_response_result():
    """
    Costruisce il dict che generate_response() restituirebbe normalmente.
    Include answer, sources (lista di Source pydantic) e metadata.
    """
    return {
        "answer": "Il PIL è cresciuto del 2% nel 2024 [economia.pdf, pagina 5].",
        "sources": [
            Source(
                document="economia.pdf",
                page=5,
                chunk_index=0,
                chunk_text="Il PIL è cresciuto del 2% nel 2024.",
                relevance_score=0.92,
            )
        ],
        "metadata": {
            "query_type": "synthesis",
            "confidence": 0.85,
            "model_used": "gpt-4o-mini",
            "processing_time": 0.42,
            "chunks_retrieved": 3,
            "context_compressed": False,
            "routing_explanation": "Standard query requiring LLM",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Endpoint standard – risposta 200
# ═══════════════════════════════════════════════════════════════════════════════

def test_chat_standard_endpoint(app_client):
    """
    Verifica che POST /api/chat con una query valida restituisca:
    - status code 200
    - JSON con le chiavi "answer", "sources", "conversation_id"

    Mock attivi:
    - generate_response → risultato fittizio (nessuna chiamata OpenAI)
    - ConversationDB  → database in-memory fittizio
    - EmbeddingCache  → nessuna cache reale
    """
    fake_result = _fake_generate_response_result()

    # Mock del database conversazioni
    mock_conv_db = MagicMock()
    mock_conv_db.create_conversation.return_value = "conv-test-001"
    mock_conv_db.get_conversation.return_value = None
    mock_conv_db.add_message.return_value = None

    # Mock della cache embeddings
    mock_emb_cache = MagicMock()

    with patch("app.routes.chat.generate_response", new_callable=AsyncMock, return_value=fake_result), \
         patch("app.routes.chat.get_conversation_db", return_value=mock_conv_db), \
         patch("app.routes.chat.get_embedding_cache", return_value=mock_emb_cache), \
         patch("app.routes.chat.generate_conversation_title", return_value="Titolo test"):

        response = app_client.post(
            "/api/chat",
            json={
                "query": "Qual è il PIL italiano?",
                # conversation_id omesso → ne verrà creato uno nuovo
            },
        )

    # Asserzioni sullo status code
    assert response.status_code == 200, (
        f"Atteso 200, ottenuto {response.status_code}: {response.text}"
    )

    # Asserzioni sul body JSON
    data = response.json()
    assert "answer" in data, "La risposta deve contenere la chiave 'answer'"
    assert "sources" in data, "La risposta deve contenere la chiave 'sources'"
    assert "conversation_id" in data, "La risposta deve contenere 'conversation_id'"

    # Verifica che la risposta contenga il testo fittizio
    assert "PIL" in data["answer"]


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Query vuota → 400 Bad Request
# ═══════════════════════════════════════════════════════════════════════════════

def test_chat_empty_query(app_client):
    """
    Verifica che una query vuota restituisca HTTP 400.

    L'endpoint ha una validazione esplicita:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, ...)
    """
    response = app_client.post(
        "/api/chat",
        json={"query": "   "},  # Solo spazi → considerata vuota
    )

    assert response.status_code == 400, (
        f"Una query vuota dovrebbe restituire 400, ottenuto {response.status_code}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: La lista sources è valida
# ═══════════════════════════════════════════════════════════════════════════════

def test_chat_response_has_valid_sources(app_client):
    """
    Verifica che ogni elemento della lista 'sources' contenga i campi
    obbligatori definiti dal modello Source: document, page, chunk_text,
    relevance_score.
    """
    fake_result = _fake_generate_response_result()

    mock_conv_db = MagicMock()
    mock_conv_db.create_conversation.return_value = "conv-test-002"
    mock_conv_db.get_conversation.return_value = None
    mock_conv_db.add_message.return_value = None

    mock_emb_cache = MagicMock()

    with patch("app.routes.chat.generate_response", new_callable=AsyncMock, return_value=fake_result), \
         patch("app.routes.chat.get_conversation_db", return_value=mock_conv_db), \
         patch("app.routes.chat.get_embedding_cache", return_value=mock_emb_cache), \
         patch("app.routes.chat.generate_conversation_title", return_value="Titolo"):

        response = app_client.post(
            "/api/chat",
            json={"query": "Spiegami l'inflazione"},
        )

    assert response.status_code == 200
    data = response.json()

    sources = data["sources"]
    assert isinstance(sources, list), "sources deve essere una lista"
    assert len(sources) >= 1, "Deve esserci almeno una source"

    # Verifica i campi obbligatori di ogni Source
    for source in sources:
        assert "document" in source, "Ogni source deve avere 'document'"
        assert "page" in source, "Ogni source deve avere 'page'"
        assert "chunk_text" in source, "Ogni source deve avere 'chunk_text'"
        assert "relevance_score" in source, "Ogni source deve avere 'relevance_score'"
