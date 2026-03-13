"""
Chat Utilities Module

Shared helper functions used by both chat.py (non-streaming) and
chat_streaming.py (streaming) pipelines to avoid code duplication
and circular imports.

Contents:
- _fuse_results: Reciprocal Rank Fusion of vector + BM25 results
- _fetch_full_document_chunks: retrieve ALL chunks for a document scope
- extract_citations: match inline citations in the answer text to sources
- _generate_multi_queries: LLM-powered query expansion for better recall
- perform_hybrid_search_async: hybrid search (vector + BM25, RRF, reranker)
- resolve_parent_context: map child search results to parent chunks
- format_context_xml: XML-formatted context string for the LLM
"""

import re
import asyncio
import logging
from typing import Any, List, Dict, Optional

from openai import AsyncOpenAI

from app.config import get_settings
from app.services.vector_store import get_vector_store
from app.services.parent_chunk_store import get_parent_chunk_store
from app.services.embedding_cache import create_embedding_cached
from app.services.bm25_search import get_bm25_engine
from app.services.reranker import get_reranker
from app.services.note_db import get_note_db

logger = logging.getLogger(__name__)

# Maximum number of parent chunks to fetch in FULL_DOCUMENT mode
# to prevent memory/token explosions before the API call
MAX_FULL_DOCUMENT_CHUNKS = 500


def _get_openai_client() -> AsyncOpenAI:
    """Get AsyncOpenAI client (used internally by _generate_multi_queries)."""
    return AsyncOpenAI(api_key=get_settings().openai_api_key)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def _fuse_results(
    vector_results: list[dict],
    bm25_results: list[dict],
    top_k: int,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3
) -> list[dict]:
    """
    Reciprocal Rank Fusion of vector and BM25 results.

    Combines two ranked lists using weighted RRF scores, producing a single
    merged ranking that benefits from both retrieval strategies.

    Args:
        vector_results: Ranked list from vector similarity search.
        bm25_results:   Ranked list from BM25 lexical search.
        top_k:          Maximum number of results to return.
        vector_weight:  Weight applied to vector search ranks.
        bm25_weight:    Weight applied to BM25 ranks.

    Returns:
        Fused & sorted list of chunk dicts with updated ``relevance_score``.
    """
    k = 60  # RRF constant
    scores: dict[str, float] = {}
    result_map: dict[str, dict] = {}

    for rank, result in enumerate(vector_results):
        chunk_id = result.get("id", "")
        if chunk_id:
            scores[chunk_id] = scores.get(chunk_id, 0) + vector_weight / (k + rank + 1)
            result_map[chunk_id] = result

    for rank, result in enumerate(bm25_results):
        chunk_id = result.get("id", "")
        if chunk_id:
            scores[chunk_id] = scores.get(chunk_id, 0) + bm25_weight / (k + rank + 1)
            if chunk_id not in result_map:
                result_map[chunk_id] = result

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused = []
    for chunk_id in sorted_ids[:top_k]:
        result = result_map[chunk_id]
        result["relevance_score"] = scores[chunk_id]
        fused.append(result)

    return fused


# ---------------------------------------------------------------------------
# Full-Document Chunk Fetching
# ---------------------------------------------------------------------------

def _fetch_full_document_chunks(
    notebook_id: str,
    document_ids: list[str] | None = None
) -> list[dict]:
    """
    Fetch ALL chunks for the given document scope (full-document strategy).

    Prefers parent chunks (larger, more coherent) and falls back to the
    vector store for documents without parent chunks.

    Returns a list of chunk dicts compatible with the ``search_results``
    format, capped at ``MAX_FULL_DOCUMENT_CHUNKS``.
    """
    vector_store = get_vector_store()
    parent_store = get_parent_chunk_store()

    # Resolve which document IDs to fetch
    if not document_ids:
        all_docs = vector_store.list_documents(notebook_id)
        document_ids = [doc["document_id"] for doc in all_docs]
        logger.info(
            "FULL_DOCUMENT: resolved %d documents from notebook %s",
            len(document_ids), notebook_id,
        )

    if not document_ids:
        logger.warning("FULL_DOCUMENT: no documents found in scope")
        return []

    all_chunks: list[dict] = []

    # --- Batch fetch parent chunks for ALL documents at once ---
    try:
        parent_chunks = parent_store.get_documents_parent_chunks_batch(document_ids)
    except Exception as e:
        logger.debug("Batch parent chunk fetch failed, will fallback per-document: %s", e)
        parent_chunks = []

    # Track which documents already have parent chunks
    docs_with_parents: set[str] = set()
    for pc in parent_chunks:
        doc_id = pc.get("document_id", "")
        docs_with_parents.add(doc_id)
        all_chunks.append({
            "id": pc.get("id", ""),
            "text": pc.get("text", ""),
            "relevance_score": 1.0,
            "metadata": {
                "document_id": doc_id,
                "document_name": pc.get("document_name", ""),
                "page": pc.get("page", 0),
                "parent_index": pc.get("parent_index", 0),
            },
        })

    # Fallback: fetch from vector store for documents without parent chunks
    missing_ids = [did for did in document_ids if did not in docs_with_parents]
    if missing_ids:
        try:
            doc_chunks = vector_store.get_document_chunks(missing_ids)
            for chunk in doc_chunks:
                all_chunks.append({
                    "id": chunk.get("id", ""),
                    "text": chunk.get("text", ""),
                    "relevance_score": 1.0,
                    "metadata": chunk.get("metadata", {}),
                })
        except Exception as e:
            logger.error("Failed to fetch chunks for documents %s: %s", missing_ids, e)

    # Safeguard: cap the number of chunks
    if len(all_chunks) > MAX_FULL_DOCUMENT_CHUNKS:
        logger.warning(
            "FULL_DOCUMENT: %d chunks exceed cap of %d. Truncating.",
            len(all_chunks), MAX_FULL_DOCUMENT_CHUNKS,
        )
        all_chunks = all_chunks[:MAX_FULL_DOCUMENT_CHUNKS]

    return all_chunks


# ---------------------------------------------------------------------------
# Citation Extraction (unified version)
# ---------------------------------------------------------------------------

def extract_citations(
    answer: str,
    sources_info: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Extract inline citations from the generated answer and match them against
    the available sources.

    The function looks for markers like ``[document_name, pagina X]`` or
    ``[document_name, pag. X]`` in *answer*, then collects every source chunk
    whose (document, page) pair was cited.  When no citation is matched it
    falls back to the top-3 sources by relevance score.

    This is the **unified** version used by both the streaming and
    non-streaming pipelines.

    Args:
        answer:       The full LLM-generated answer text.
        sources_info: List of source dicts.  Each dict MUST contain at least
                      ``"document"`` (str) and ``"page"`` (int) keys.  An
                      optional ``"chunk_index"`` is used for deduplication.

    Returns:
        Filtered (and sorted) list of source dicts that were actually cited
        in the answer, or the top-3 sources as fallback.
    """
    # Find all citations in [document_name, pagina X] or [document_name, pag. X]
    citation_pattern = r"\[([^\],]+),\s*(?:pagina|pag\.?|page)\s*(\d+)\]"
    citations = re.findall(citation_pattern, answer, re.IGNORECASE)

    # Collect unique cited pages (document + page combinations)
    cited_pages: set[tuple[str, int]] = set()
    for doc_name_cited, page_cited in citations:
        doc_name_cited = doc_name_cited.strip()
        try:
            cited_pages.add((doc_name_cited, int(page_cited)))
        except ValueError:
            continue

    # Match citations with sources — include ALL chunks from cited pages
    cited_sources: list[dict] = []
    seen_chunks: set[tuple] = set()

    for source in sources_info:
        doc_name = source.get("document", "")
        page = source.get("page", 0)
        chunk_idx = source.get("chunk_index", source.get("chunk_id", ""))
        page_key = (doc_name, page)
        chunk_key = (doc_name, chunk_idx)

        if page_key in cited_pages and chunk_key not in seen_chunks:
            cited_sources.append(source)
            seen_chunks.add(chunk_key)

    # Fallback: if nothing matched, return top 3 by relevance
    if not cited_sources:
        for source in sources_info[:3]:
            doc_name = source.get("document", "")
            chunk_idx = source.get("chunk_index", source.get("chunk_id", ""))
            chunk_key = (doc_name, chunk_idx)
            if chunk_key not in seen_chunks:
                cited_sources.append(source)
                seen_chunks.add(chunk_key)

    # Sort by relevance score descending
    cited_sources.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    return cited_sources


# ---------------------------------------------------------------------------
# Multi-Query Expansion
# ---------------------------------------------------------------------------

async def _generate_multi_queries(query: str, num_queries: int = 3) -> list[str]:
    """
    Generate alternative formulations of the user query to improve recall.

    Uses the lightweight LLM (settings.light_model) to produce ``num_queries - 1``
    rephrasings / synonym-based variants of the original query.  The original
    query is always included as the first element of the returned list.

    Args:
        query: The original user query.
        num_queries: Total number of queries to return (original + alternatives).

    Returns:
        A list of query strings (length == ``num_queries`` when successful,
        length == 1 if the LLM call fails).
    """
    if num_queries <= 1:
        return [query]

    settings = get_settings()
    alternatives_needed = num_queries - 1

    system_prompt = settings.multi_query_system_prompt_template.format(
        alternatives_needed=alternatives_needed
    )

    try:
        client = _get_openai_client()
        response = await client.chat.completions.create(
            model=settings.light_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=1,
            max_completion_tokens=300,
        )

        raw_text = response.choices[0].message.content or ""

        # Clean up: strip numbered/bulleted prefixes and blank lines
        alternatives: list[str] = []
        for line in raw_text.strip().splitlines():
            cleaned = re.sub(r"^\s*[\d\-\*\•\.\)]+\s*", "", line).strip()
            if cleaned:
                alternatives.append(cleaned)

        # Always start with the original query
        queries = [query] + alternatives[:alternatives_needed]
        logger.info("Multi-Query expansion: %d queries generated", len(queries))
        return queries

    except Exception as e:
        logger.warning("Multi-Query generation failed, using original query only: %s", e)
        return [query]


# ---------------------------------------------------------------------------
# Hybrid Search (unified async version)
# ---------------------------------------------------------------------------

async def perform_hybrid_search_async(
    queries: List[str],
    n_results: int,
    notebook_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    min_relevance_score: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search with Multi-Query support.

    Iterates over multiple query variants (original + expansions), accumulates
    the best scores per chunk, then fuses vector and BM25 results and re-ranks.
    After reranking, results below ``min_relevance_score`` are discarded.

    Vector and BM25 searches run concurrently via ``asyncio.to_thread``.

    Args:
        queries: List of query strings (original + alternatives).
        n_results: Number of results to return.
        notebook_id: Optional notebook filter.
        document_ids: Optional list of document IDs to restrict search.
        min_relevance_score: Minimum reranker score to keep a result.

    Returns:
        List of result dicts with search scores, filtered by relevance.
    """
    settings = get_settings()
    vector_store = get_vector_store()
    bm25_engine = get_bm25_engine()

    # Fetch more initial results to feed the Re-Ranking step
    retrieval_k = n_results * 3

    # Accumulators: keep the highest score per chunk ID
    all_vector_results: dict[str, dict] = {}
    all_bm25_results: dict[str, dict] = {}

    for q in queries:
        # --- Embedding (fast if cached) ---
        q_embedding = create_embedding_cached(q)

        # --- Build concurrent tasks ---
        v_task = asyncio.to_thread(
            vector_store.search_similar,
            query_embedding=q_embedding,
            n_results=retrieval_k,
            document_ids=document_ids,
            notebook_id=notebook_id,
        )

        if settings.enable_hybrid_search:
            b_task = asyncio.to_thread(
                bm25_engine.search,
                query=q,
                n_results=retrieval_k,
                notebook_id=notebook_id,
                document_ids=document_ids,
            )
            raw_results, raw_bm25 = await asyncio.gather(
                v_task, b_task, return_exceptions=True
            )
        else:
            raw_results = await v_task
            raw_bm25 = None

        # --- Accumulate vector results ---
        if not isinstance(raw_results, Exception):
            if raw_results["documents"] and raw_results["documents"][0]:
                for i, (doc_text, metadata, distance) in enumerate(zip(
                    raw_results["documents"][0],
                    raw_results["metadatas"][0],
                    raw_results["distances"][0],
                )):
                    chunk_id = raw_results["ids"][0][i]
                    score = round(1 - distance, 3)
                    if chunk_id not in all_vector_results or score > all_vector_results[chunk_id]["relevance_score"]:
                        all_vector_results[chunk_id] = {
                            "id": chunk_id, "text": doc_text,
                            "metadata": metadata, "relevance_score": score,
                        }
        else:
            logger.warning("Vector search failed for query variant '%s': %s", q, raw_results)

        # --- Accumulate BM25 results ---
        if raw_bm25 is not None:
            if isinstance(raw_bm25, Exception):
                logger.warning("BM25 search failed for query variant '%s': %s", q, raw_bm25)
            else:
                for hit in raw_bm25:
                    chunk_id = hit["id"]
                    if chunk_id not in all_bm25_results or hit["bm25_score"] > all_bm25_results[chunk_id]["bm25_score"]:
                        all_bm25_results[chunk_id] = hit

    # --- Fusion (once, outside the loop) ---
    vector_list = list(all_vector_results.values())
    bm25_list = list(all_bm25_results.values())

    vector_list.sort(key=lambda x: x["relevance_score"], reverse=True)
    bm25_list.sort(key=lambda x: x.get("bm25_score", 0), reverse=True)

    if settings.enable_hybrid_search and bm25_list:
        fused = _fuse_results(vector_list, bm25_list, retrieval_k,
                              settings.vector_weight, settings.bm25_weight)
    else:
        fused = vector_list[:retrieval_k]

    # Re-Ranking with Cross-Encoder (eseguito in un thread separato per non bloccare FastAPI!)
    reranker = get_reranker()
    reranked = await asyncio.to_thread(reranker.rerank, queries[0], fused, top_k=n_results)

    # Filter by minimum relevance score
    # DISABILITATO TEMPORANEAMENTE: I modelli Cross-Encoder restituiscono "logits" (da -10 a +10). 
    # Un punteggio di 0.25 tagliava fuori risultati validi che magari avevano punteggio -1.2.
    # reranked = [r for r in reranked if r.get("relevance_score", 0) >= min_relevance_score]

    return reranked


# ---------------------------------------------------------------------------
# Parent-Context Resolution (unified version)
# ---------------------------------------------------------------------------

def resolve_parent_context(
    search_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Map child search results to parent chunks for broader LLM context.

    When parent-child retrieval is disabled (``settings.enable_parent_child``
    is False), child chunks are returned as-is.

    Args:
        search_results: Child chunk search results.

    Returns:
        List of context dicts with keys:
        - ``context_text``: Parent text (or child text if no parent).
        - ``citation_text``: The original child text for citation matching.
        - ``metadata``: Chunk metadata.
        - ``relevance_score``: Score from the search / reranker.
    """
    settings = get_settings()

    if not settings.enable_parent_child:
        return [{
            "context_text": r["text"],
            "citation_text": r["text"],
            "metadata": r["metadata"],
            "relevance_score": r.get("relevance_score", 0.5),
        } for r in search_results]

    parent_store = get_parent_chunk_store()

    # Collect unique parent lookups
    parent_lookups: list[dict] = []
    seen_parents: set[str] = set()
    for result in search_results:
        metadata = result["metadata"]
        doc_id = metadata.get("document_id")
        parent_idx = metadata.get("parent_chunk_index")
        if doc_id and parent_idx is not None:
            key = f"{doc_id}:{parent_idx}"
            if key not in seen_parents:
                seen_parents.add(key)
                parent_lookups.append({
                    "document_id": doc_id,
                    "parent_index": parent_idx,
                })

    # Batch fetch parents
    parent_map = parent_store.get_parent_chunks_batch(parent_lookups) if parent_lookups else {}

    # Build context
    contexts: list[dict] = []
    used_parents: set[str] = set()

    for result in search_results:
        metadata = result["metadata"]
        doc_id = metadata.get("document_id")
        parent_idx = metadata.get("parent_chunk_index")
        parent_key = f"{doc_id}:{parent_idx}" if doc_id and parent_idx is not None else None

        parent = parent_map.get(parent_key) if parent_key else None

        if parent and parent_key not in used_parents:
            contexts.append({
                "context_text": parent["text"],
                "citation_text": result["text"],
                "metadata": metadata,
                "relevance_score": result.get("relevance_score", 0.5),
            })
            used_parents.add(parent_key)
        elif parent_key and parent_key in used_parents:
            contexts.append({
                "context_text": None,
                "citation_text": result["text"],
                "metadata": metadata,
                "relevance_score": result.get("relevance_score", 0.5),
            })
        else:
            contexts.append({
                "context_text": result["text"],
                "citation_text": result["text"],
                "metadata": metadata,
                "relevance_score": result.get("relevance_score", 0.5),
            })

    return contexts


# ---------------------------------------------------------------------------
# Notes Context Helper
# ---------------------------------------------------------------------------

def get_formatted_notes_context(notebook_id: str) -> str:
    """
    Retrieve all notes for a notebook and format them as XML document tags.

    Notes are formatted identically to document chunks so the LLM can cite
    them using the standard citation pattern (e.g. [Nota: Titolo, pagina 1]).

    Args:
        notebook_id: Notebook UUID whose notes should be fetched.

    Returns:
        XML-formatted string of notes, or empty string if none exist.
    """
    try:
        note_db = get_note_db()
        notes = note_db.list_notes(notebook_id)
    except Exception as e:
        logger.warning("Failed to fetch notes for notebook %s: %s", notebook_id, e)
        return ""

    if not notes:
        return ""

    xml_notes: list[str] = []
    for note in notes:
        # note_db.list_notes returns dicts (sqlite3.Row converted via dict())
        title = note.get("title", None) or "Senza Titolo"
        content = note.get("content", "")
        if not content:
            continue
        xml_notes.append(
            f'<document name="Nota: {title}" page="1">\n{content}\n</document>'
        )

    return "\n\n".join(xml_notes)


# ---------------------------------------------------------------------------
# XML Context Formatter
# ---------------------------------------------------------------------------

def format_context_xml(context_items: List[Dict]) -> str:
    """
    Format a list of context items into an XML-style string for the LLM.

    Each chunk is wrapped in a ``<document>`` tag with ``name`` and ``page``
    attributes, making it easier for the model to distinguish sources.

    Args:
        context_items: List of dicts, each containing at least:
            - ``context_text`` (str | None): the text to include.
            - ``metadata`` (dict): must have ``document_name`` and ``page``.

    Returns:
        A single string with all chunks formatted as XML fragments.
    """
    parts: list[str] = []
    for item in context_items:
        text = item.get("context_text")
        if text is None:
            continue
        metadata = item.get("metadata", {})
        doc_name = metadata.get("document_name", "Documento Sconosciuto")
        page = metadata.get("page", 0)
        parts.append(f'<document name="{doc_name}" page="{page}">\n{text}\n</document>')
    return "\n\n".join(parts)
