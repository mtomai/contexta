"""
Chat Streaming Module

Provides streaming responses with the advanced RAG pipeline:
- Hybrid Search (Vector + BM25 with RRF fusion)
- Parent-Child Retrieval (precise child search → broad parent context)
- Context Compression
- Server-Sent Events (SSE) for real-time token delivery
"""

import re
import json
import asyncio
import logging
from typing import AsyncGenerator, List, Dict, Any, Optional
from openai import OpenAI

from app.config import get_settings
from app.services.vector_store import get_vector_store
from app.services.context_compressor import compress_context
from app.services.query_router import classify_query, QueryType
from app.services.chat_utils import (
    _fetch_full_document_chunks,
    extract_citations,
    _generate_multi_queries,
    perform_hybrid_search_async,
    resolve_parent_context,
    format_context_xml,
    get_formatted_notes_context,
)

logger = logging.getLogger(__name__)

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)


# extract_citations_streaming is replaced by the unified extract_citations
# imported from chat_utils (kept as alias for backwards compat if needed)
extract_citations_streaming = extract_citations


async def generate_response_stream(
    query: str,
    notebook_id: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    top_k: int = None,
    compress: bool = True,
    max_context_tokens: int = None
) -> AsyncGenerator[str, None]:
    """
    Generate streaming response using the advanced RAG pipeline.

    Yields Server-Sent Events (SSE) formatted data.

    Event types:
    - sources: Initial sources data
    - token: Individual response tokens
    - done: Completion signal
    - error: Error message

    Pipeline:
    1. Embed query (cached)
    2. Hybrid search (vector + BM25, RRF fusion)
    3. Parent-child retrieval (child → parent context)
    4. Context compression (if needed)
    5. Stream LLM response
    6. Extract citations from complete response

    Args:
        query: User question
        notebook_id: Optional notebook filter
        conversation_history: Optional list of previous messages
        top_k: Number of chunks to retrieve
        compress: Whether to compress context
        max_context_tokens: Max tokens for context

    Yields:
        SSE formatted strings
    """
    try:
        if top_k is None:
            top_k = settings.child_retrieval_top_k if settings.enable_parent_child else settings.default_top_k
        if max_context_tokens is None:
            max_context_tokens = settings.max_context_tokens

        # Step 1: Multi-Query expansion
        logger.info("STREAM START | query='%.50s...' | top_k=%d | notebook=%s", query, top_k, notebook_id)

        yield _format_sse("thought", {"message": "Analisi della domanda e generazione varianti..."})
        queries = await _generate_multi_queries(query)

        # Step 2: Hybrid search (vector + BM25) with multi-query
        yield _format_sse("thought", {"message": "Ricerca vettoriale e testuale nei documenti in corso..."})
        search_results = await perform_hybrid_search_async(
            queries=queries,
            n_results=top_k,
            notebook_id=notebook_id,
        )

        if not search_results:
            logger.warning("STREAM: No relevant results after hybrid search + reranking.")
            yield _format_sse("error", {
                "message": settings.no_results_message
            })
            return

        # Step 2.5 (NEW): Query Routing & FULL_DOCUMENT intercept
        top_score = search_results[0]["relevance_score"] if search_results else 0.0
        routing_decision = classify_query(
            query=query,
            top_results=search_results,
            top_score=top_score,
        )
        logger.info(
            "STREAM routing: type=%s, confidence=%.2f",
            routing_decision.query_type, routing_decision.confidence,
        )

        # Determine which LLM model to use
        model = settings.chat_model

        if routing_decision.query_type == QueryType.FULL_DOCUMENT:
            logger.info("STREAM FULL_DOCUMENT detected — fetching entire document scope")
            search_results = _fetch_full_document_chunks(
                notebook_id=notebook_id,
            )
            compress = False
            model = settings.long_context_model

            # Build sources_info & resolved_contexts_data from full-document chunks
            sources_info = []
            resolved_contexts_data = []
            for result in search_results:
                content = result.get("text", "")
                if not content:
                    continue
                metadata = result.get("metadata", {})
                doc_name = metadata.get("document_name", "Documento Sconosciuto")
                resolved_contexts_data.append({
                    "context_text": content,
                    "citation_text": content,
                    "metadata": metadata,
                    "relevance_score": result.get("relevance_score", 1.0),
                })
                sources_info.append({
                    "document": doc_name,
                    "page": metadata.get("page", 0),
                    "chunk_index": metadata.get("chunk_index", result.get("id", "")),
                    "chunk_text": content[:500],
                    "relevance_score": result.get("relevance_score", 1.0),
                })

            # Skip compression for FULL_DOCUMENT
            context_text = format_context_xml(resolved_contexts_data)
        else:
            # Step 3: Parent-child retrieval (standard RAG path)
            yield _format_sse("thought", {"message": "Riordino semantico dei risultati (Cross-Encoder)..."})
            resolved_contexts = resolve_parent_context(search_results)

            # Build sources_info from child chunks (for citation tracking)
            sources_info = []
            for ctx in resolved_contexts:
                sources_info.append({
                    "document": ctx["metadata"]["document_name"],
                    "page": ctx["metadata"]["page"],
                    "chunk_index": ctx["metadata"]["chunk_index"],
                    "chunk_text": ctx["citation_text"],
                    "relevance_score": ctx["relevance_score"]
                })

            # Step 4: Build context from parent chunks
            compression_chunks = []
            for ctx in resolved_contexts:
                if ctx["context_text"] is not None:
                    compression_chunks.append({
                        "text": ctx["context_text"],
                        "metadata": ctx["metadata"]
                    })

            # Step 4.5: Compress context if needed
            if compress and compression_chunks:
                context_text = compress_context(compression_chunks, query, max_context_tokens)
            else:
                context_text = format_context_xml(resolved_contexts)

        logger.info("STREAM: calling LLM (ctx_len=%d, model=%s)", len(context_text), model)

        # Inject saved notes as priority context
        notes_context = get_formatted_notes_context(notebook_id)
        if notes_context:
            context_text = f"{notes_context}\n\n{context_text}"

        yield _format_sse("thought", {"message": "Lettura del contesto e formulazione della risposta..."})
        # Step 5: Stream response from LLM with auto-continuation
        system_prompt = settings.chat_streaming_system_prompt
        if routing_decision.query_type == QueryType.FULL_DOCUMENT:
            system_prompt += settings.full_document_suffix

        user_prompt = settings.chat_streaming_user_prompt_template.format(
            context_text=context_text,
            query=query
        )

        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_prompt})

        full_response = ""

        for _ in range(settings.max_continuations):
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True
            )

            chunk_response = ""
            finish_reason = None

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    chunk_response += token
                    full_response += token
                    yield _format_sse("token", {"token": token})

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            if finish_reason != "length":
                break

            messages.append({"role": "assistant", "content": chunk_response})
            messages.append({"role": "user", "content": settings.continuation_prompt})

        # Step 6: Extract cited sources from the response (uses unified extract_citations)
        cited_sources = extract_citations(full_response, sources_info)

        logger.info("STREAM DONE | response_len=%d | sources=%d", len(full_response), len(cited_sources))
        yield _format_sse("done", {
            "full_response": full_response,
            "sources": cited_sources,
            "sources_count": len(cited_sources)
        })

    except Exception as e:
        logger.error("STREAM ERROR: %s", str(e), exc_info=True)
        yield _format_sse("error", {"message": str(e)})


def _format_sse(event: str, data: dict) -> str:
    """
    Format data as Server-Sent Event.

    Args:
        event: Event type name
        data: Data dictionary to serialize

    Returns:
        SSE formatted string
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_simple_response(
    message: str,
    event_type: str = "token"
) -> AsyncGenerator[str, None]:
    """
    Stream a simple pre-generated response.
    Useful for cached or direct answers.

    Args:
        message: Message to stream
        event_type: SSE event type

    Yields:
        SSE formatted strings
    """
    words = message.split()

    for i, word in enumerate(words):
        separator = " " if i > 0 else ""
        yield _format_sse(event_type, {"token": separator + word})

    yield _format_sse("done", {"full_response": message})


# ---------------------------------------------------------------------------
# Streaming Refinement (NEW — aligned with generate_refinement_response)
# ---------------------------------------------------------------------------

async def generate_refinement_response_stream(
    query: str,
    document_ids: List[str],
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream a refinement response for conversations tied to specific documents.

    Equivalent to ``generate_refinement_response`` in chat.py but delivers
    tokens via SSE in real-time, with auto-continuation support.

    Pipeline:
    1. Fetch full document chunks for the specified document_ids.
    2. Find the last assistant message for refinement context.
    3. Build messages using ``settings.refinement_user_prompt_template``.
    4. Stream LLM response with auto-continuation.
    5. Extract citations and emit ``done`` event.

    Args:
        query: User's refinement instruction.
        document_ids: List of document IDs to use as context.
        conversation_history: Optional list of previous messages.

    Yields:
        SSE formatted strings (token, done, error).
    """
    try:
        # Step 1: Fetch all chunks for the specified documents
        search_results = _fetch_full_document_chunks(
            notebook_id="",  # Not needed when document_ids are provided
            document_ids=document_ids,
        )

        if not search_results:
            yield _format_sse("error", {"message": settings.no_documents_message})
            return

        # Build context and sources_info
        resolved_contexts_data: List[Dict[str, Any]] = []
        sources_info: List[Dict[str, Any]] = []
        for result in search_results:
            content = result.get("text", "")
            if not content:
                continue
            metadata = result.get("metadata", {})
            doc_name = metadata.get("document_name", "Documento Sconosciuto")
            resolved_contexts_data.append({
                "context_text": content,
                "citation_text": content,
                "metadata": metadata,
                "relevance_score": result.get("relevance_score", 1.0),
            })
            sources_info.append({
                "document": doc_name,
                "page": metadata.get("page", 0),
                "chunk_index": metadata.get("chunk_index", result.get("id", "")),
                "chunk_text": content[:500],
                "relevance_score": result.get("relevance_score", 1.0),
            })

        context_text = format_context_xml(resolved_contexts_data)

        # Step 2: Find the last assistant response for refinement context
        previous_response: Optional[str] = None
        if conversation_history:
            for msg in reversed(conversation_history):
                if msg.get("role") == "assistant":
                    previous_response = msg.get("content", "")
                    break

        # Step 3: Build refinement messages
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": settings.refinement_system_prompt}
        ]

        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

        if previous_response:
            user_prompt = settings.refinement_user_prompt_template.format(
                previous_response=previous_response,
                context_text=context_text,
                query=query,
            )
        else:
            user_prompt = settings.refinement_user_prompt_no_history_template.format(
                context_text=context_text,
                query=query,
            )

        messages.append({"role": "user", "content": user_prompt})

        model = settings.long_context_model
        logger.info(
            "STREAM REFINEMENT: docs=%d | ctx_len=%d | model=%s",
            len(document_ids), len(context_text), model,
        )

        # Step 4: Stream response with auto-continuation
        full_response = ""

        for _ in range(settings.max_continuations):
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )

            chunk_response = ""
            finish_reason = None

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    chunk_response += token
                    full_response += token
                    yield _format_sse("token", {"token": token})

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            if finish_reason != "length":
                break

            messages.append({"role": "assistant", "content": chunk_response})
            messages.append({"role": "user", "content": settings.continuation_prompt})
            logger.info("STREAM REFINEMENT: auto-continuation triggered")

        # Step 5: Extract cited sources
        cited_sources = extract_citations(full_response, sources_info)

        logger.info(
            "STREAM REFINEMENT DONE | response_len=%d | sources=%d",
            len(full_response), len(cited_sources),
        )
        yield _format_sse("done", {
            "full_response": full_response,
            "sources": cited_sources,
            "sources_count": len(cited_sources),
        })

    except Exception as e:
        logger.error("STREAM REFINEMENT ERROR: %s", str(e), exc_info=True)
        yield _format_sse("error", {"message": str(e)})
