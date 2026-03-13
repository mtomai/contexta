"""
BM25 Search Engine Module

Provides keyword-based search using BM25 (Okapi BM25) algorithm.
Used alongside ChromaDB vector search in a Hybrid Search pipeline.

The BM25 index is built in-memory from all document chunks and rebuilt
when documents are added or removed. This complements the semantic
vector search with exact keyword matching, which is essential for:
- Acronyms, codes, serial numbers
- Proper nouns and domain-specific terms
- Exact phrase matching
"""

import re
import threading
from typing import List, Dict, Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from nltk.stem.snowball import SnowballStemmer

from app.config import get_settings

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

settings = get_settings()
_stemmer_it = SnowballStemmer("italian")
_stemmer_en = SnowballStemmer("english")

# Italian stop words
_ITALIAN_STOPWORDS = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "di", "del", "dello", "della", "dei", "degli", "delle",
    "a", "al", "allo", "alla", "ai", "agli", "alle",
    "da", "dal", "dallo", "dalla", "dai", "dagli", "dalle",
    "in", "nel", "nello", "nella", "nei", "negli", "nelle",
    "su", "sul", "sullo", "sulla", "sui", "sugli", "sulle",
    "per", "con", "tra", "fra",
    "e", "o", "ma", "che", "non", "si", "è", "sono",
    "ha", "ho", "hanno", "essere", "avere", "questo", "quello",
    "come", "più", "anche", "già", "ancora", "molto",
}

# English stop words
_ENGLISH_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "in", "on", "at", "to", "for", "of", "with", "by", "this", "that",
    "it", "from", "as", "be", "have", "has", "had", "not", "are",
}

_ALL_STOPWORDS = _ITALIAN_STOPWORDS | _ENGLISH_STOPWORDS


def _tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25 indexing and search.
    
    Detects language (Italian vs English) based on stopword density
    and applies the appropriate stemmer.
    """
    tokens = re.findall(r'\w+', text.lower())
    
    if not tokens:
        return []

    # 1. Detect language based on stopwords
    it_count = sum(1 for t in tokens if t in _ITALIAN_STOPWORDS)
    en_count = sum(1 for t in tokens if t in _ENGLISH_STOPWORDS)
    
    # Select stemmer (default to Italian if unknown or tie)
    stemmer = _stemmer_en if en_count > it_count else _stemmer_it
    
    # 2. Filter stopwords and short words
    filtered_tokens = [t for t in tokens if len(t) > 1 and t not in _ALL_STOPWORDS]
    
    # 3. Apply stemming
    stemmed_tokens = [stemmer.stem(t) for t in filtered_tokens]
    
    return stemmed_tokens


class BM25SearchEngine:
    """
    In-memory BM25 search engine with lazy index building.

    The index is built from all chunks stored in ChromaDB on first use,
    and can be rebuilt when documents change.
    """

    def __init__(self):
        self._index: Optional[BM25Okapi] = None
        self._corpus: List[List[str]] = []       # tokenized documents
        self._chunk_ids: List[str] = []           # ChromaDB chunk IDs
        self._chunk_texts: List[str] = []         # original texts
        self._chunk_metadatas: List[Dict] = []    # original metadatas
        self._is_built = False
        self._lock = threading.Lock()

    def build_index(self):
        """
        Build BM25 index from all chunks currently in ChromaDB.

        This is called lazily on first search, or explicitly when
        documents are added/removed.
        """
        from app.services.vector_store import get_vector_store

        with self._lock:
            vector_store = get_vector_store()

            try:
                all_items = vector_store.collection.get()
            except Exception:
                self._is_built = True
                return

            if not all_items or not all_items["ids"]:
                self._chunk_ids = []
                self._chunk_texts = []
                self._chunk_metadatas = []
                self._corpus = []
                self._index = None
                self._is_built = True
                return

            self._chunk_ids = all_items["ids"]
            self._chunk_texts = all_items["documents"]
            self._chunk_metadatas = all_items["metadatas"]
            self._corpus = [_tokenize(text) for text in self._chunk_texts]

            # Build the BM25 index
            if self._corpus:
                self._index = BM25Okapi(self._corpus)
            else:
                self._index = None

            self._is_built = True

    def search(
        self,
        query: str,
        n_results: int = 10,
        notebook_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the BM25 index for matching chunks.

        Args:
            query: Search query text
            n_results: Maximum number of results to return
            notebook_id: Optional notebook filter
            document_ids: Optional document ID filter

        Returns:
            List of result dicts with 'id', 'text', 'metadata', 'bm25_score'
        """
        if not self._is_built:
            self.build_index()

        if self._index is None or not self._corpus:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        with self._lock:
            scores = self._index.get_scores(query_tokens)

        # Get indices sorted by score (descending)
        sorted_indices = np.argsort(scores)[::-1]

        results: List[Dict[str, Any]] = []

        for idx in sorted_indices:
            if scores[idx] <= 0:
                break

            metadata = self._chunk_metadatas[idx]

            # Apply notebook filter
            if notebook_id and metadata.get("notebook_id") != notebook_id:
                continue

            # Apply document_ids filter
            if document_ids and metadata.get("document_id") not in document_ids:
                continue

            results.append({
                "id": self._chunk_ids[idx],
                "text": self._chunk_texts[idx],
                "metadata": metadata,
                "bm25_score": float(scores[idx])
            })

            if len(results) >= n_results:
                break

        return results

    def rebuild(self):
        """Force a full rebuild of the BM25 index."""
        self._is_built = False
        self.build_index()

    def invalidate(self):
        """Mark the index as stale so it rebuilds on next search."""
        self._is_built = False

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "is_built": self._is_built,
            "total_chunks": len(self._chunk_ids),
            "corpus_size": len(self._corpus)
        }


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Combine vector search and BM25 results using Reciprocal Rank Fusion (RRF).

    RRF formula: score(d) = sum( weight / (k + rank_i) ) for each ranker i

    This is the same fusion technique used by modern search systems to
    combine dense (vector) and sparse (keyword) retrieval.

    Args:
        vector_results: Results from ChromaDB vector search
        bm25_results: Results from BM25 keyword search
        vector_weight: Weight multiplier for vector results
        bm25_weight: Weight multiplier for BM25 results
        k: RRF constant (higher = more uniform blending)

    Returns:
        Fused results sorted by combined RRF score
    """
    scores: Dict[str, float] = {}
    result_data: Dict[str, Dict[str, Any]] = {}

    # Score vector results
    for rank, result in enumerate(vector_results):
        doc_id = result["id"]
        scores[doc_id] = scores.get(doc_id, 0) + vector_weight / (k + rank + 1)
        if doc_id not in result_data:
            result_data[doc_id] = result

    # Score BM25 results
    for rank, result in enumerate(bm25_results):
        doc_id = result["id"]
        scores[doc_id] = scores.get(doc_id, 0) + bm25_weight / (k + rank + 1)
        if doc_id not in result_data:
            result_data[doc_id] = result

    # Sort by combined score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused_results = []
    for doc_id in sorted_ids:
        result = result_data[doc_id].copy()
        result["rrf_score"] = scores[doc_id]
        fused_results.append(result)

    return fused_results


# Singleton instance
_bm25_engine: Optional[BM25SearchEngine] = None


def get_bm25_engine() -> BM25SearchEngine:
    """Get or create BM25 engine singleton instance."""
    global _bm25_engine
    if _bm25_engine is None:
        _bm25_engine = BM25SearchEngine()
    return _bm25_engine
