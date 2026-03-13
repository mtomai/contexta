"""
Re-Ranker Service — Post-retrieval reranking con Cross-Encoder.

Dopo la fase di Hybrid Search (Vector + BM25), questo modulo riordina
i risultati usando un modello Cross-Encoder che valuta la pertinenza
reale di ogni coppia (query, documento).  Il Cross-Encoder è più accurato
del semplice cosine-similarity perché analizza query e documento INSIEME
anziché confrontare embedding indipendenti.
"""

import logging
from typing import List, Dict

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Lightweight reranker model, possesses strong multilingual capabilities, easy to deploy, with fast inference.
_DEFAULT_MODEL_NAME = "nreimers/mmarco-mMiniLMv2-L6-H384-v1"


class ReRanker:
    """
    Wrapper attorno a un Cross-Encoder di sentence-transformers.

    Fornisce un metodo `rerank()` che riceve i risultati della Hybrid Search,
    calcola uno score di pertinenza per ogni coppia (query, chunk) e
    restituisce solo i migliori `top_k` risultati riordinati.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME):
        """
        Inizializza il Cross-Encoder.

        Args:
            model_name: Nome o percorso del modello Cross-Encoder da caricare.
        """
        self.model_name = model_name
        logger.info(f"Caricamento modello Cross-Encoder: {self.model_name}")
        
        # Questa operazione pesante viene fatta SOLO ORA
        self.model = CrossEncoder(self.model_name)
        logger.info("Cross-Encoder caricato con successo ed è pronto all'uso in memoria.")

    def rerank(
        self,
        query: str,
        search_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Riordinamento dei risultati di ricerca tramite Cross-Encoder.

        1. Prepara le coppie [query, testo_chunk] per ogni risultato.
        2. Calcola uno score di pertinenza con il Cross-Encoder.
        3. Aggiorna il campo `relevance_score` di ogni risultato.
        4. Ordina in modo decrescente e restituisce i primi `top_k`.

        Args:
            query: La domanda originale dell'utente.
            search_results: Lista di dizionari con almeno le chiavi
                            "text" e "id" (output della Hybrid Search).
            top_k: Numero massimo di risultati da restituire.

        Returns:
            Lista dei migliori `top_k` risultati, riordinati per pertinenza.
        """
        if not search_results:
            return []

        # Prepara le coppie (query, testo) per il Cross-Encoder
        pairs = [[query, result.get("text", "")] for result in search_results]

        # Calcola gli score di pertinenza con il Cross-Encoder
        scores = self.model.predict(pairs)

        # Aggiorna lo score di ogni risultato con il punteggio del Cross-Encoder
        for i, doc in enumerate(search_results):
            doc["relevance_score"] = float(scores[i])

        # Ordina per score decrescente e restituisci solo i migliori top_k
        reranked = sorted(search_results, key=lambda x: x["relevance_score"], reverse=True)[:top_k]

        logger.info(
            f"Re-Ranking completato: {len(search_results)} risultati -> "
            f"top {len(reranked)} selezionati"
        )

        return reranked


# ---------------------------------------------------------------------------
# Singleton / Lazy Loading  (stesso pattern usato negli altri servizi)
# ---------------------------------------------------------------------------
# Variabile globale per mantenere l'istanza viva
_reranker_instance = None


def get_reranker() -> ReRanker:
    """
    Restituisce l'istanza singleton del ReRanker.
    Il modello viene caricato solo alla prima chiamata (lazy loading).
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = ReRanker()
    return _reranker_instance
