"""
Chat service - handles RAG pipeline for generating responses.

Shared utilities (_fuse_results, _fetch_full_document_chunks,
extract_citations, _generate_multi_queries) live in chat_utils.py
to avoid circular imports and code duplication.
"""
import logging
import time
from openai import AsyncOpenAI

from app.config import get_settings
from app.services.context_compressor import compress_context, estimate_tokens
from app.services.query_router import classify_query, QueryType

# --- Shared utilities (imported from chat_utils) ---
from app.services.chat_utils import (
    _fetch_full_document_chunks,
    extract_citations,
    _generate_multi_queries,
    perform_hybrid_search_async,
    resolve_parent_context,
    format_context_xml,
    get_formatted_notes_context,
    MAX_FULL_DOCUMENT_CHUNKS,
)

logger = logging.getLogger(__name__)


def get_openai_client() -> AsyncOpenAI:
    """Get AsyncOpenAI client."""
    return AsyncOpenAI(api_key=get_settings().openai_api_key)


async def generate_response(
    query: str,
    notebook_id: str,
    document_ids: list[str] | None = None,
    conversation_history: list[dict] | None = None,
    top_k: int = None
) -> dict:
    """
    Generate a response using RAG pipeline.
    
    Args:
        query: User's question
        notebook_id: ID of the notebook to search in
        document_ids: Optional list of specific document IDs to search
        conversation_history: Optional conversation history for context
        top_k: Number of results to retrieve (default from settings)
        
    Returns:
        dict with answer, sources, and metadata
    """
    start_time = time.time()
    settings = get_settings()
    top_k = top_k or settings.default_top_k
    
    try:
        # -----------------------------------------------------------------
        # Step 1: Multi-Query expansion — generate alternative queries
        # to improve recall across vector and BM25 searches.
        # -----------------------------------------------------------------
        queries = await _generate_multi_queries(query)

        # -----------------------------------------------------------------
        # Step 2: Hybrid search (vector + BM25, RRF fusion, reranking)
        # -----------------------------------------------------------------
        search_results = await perform_hybrid_search_async(
            queries=queries,
            n_results=top_k,
            notebook_id=notebook_id,
            document_ids=document_ids,
        )

        # Step 3: Query routing
        top_score = search_results[0]["relevance_score"] if search_results else 0.0
        routing_decision = classify_query(
            query=query,
            top_results=search_results,
            top_score=top_score
        )
        
        logger.info(f"Query routing: type={routing_decision.query_type}, confidence={routing_decision.confidence}")
        
        # Step 4: FULL_DOCUMENT intercept — fetch ALL chunks instead of top_k
        if routing_decision.query_type == QueryType.FULL_DOCUMENT:
            logger.info("FULL_DOCUMENT detected — fetching entire document scope")
            search_results = _fetch_full_document_chunks(
                notebook_id=notebook_id,
                document_ids=document_ids
            )
            logger.info(f"FULL_DOCUMENT: fetched {len(search_results)} chunks (cap={MAX_FULL_DOCUMENT_CHUNKS})")
        
        # Step 5: Resolve parent chunks if available
        sources_info = []

        if routing_decision.query_type == QueryType.FULL_DOCUMENT:
            # FULL_DOCUMENT: content was already fetched by _fetch_full_document_chunks.
            resolved_contexts_data = [
                {
                    "context_text": result.get("text", ""),
                    "citation_text": result.get("text", ""),
                    "metadata": result.get("metadata", {}),
                    "relevance_score": result.get("relevance_score", 1.0),
                }
                for result in search_results if result.get("text")
            ]
            for ctx in resolved_contexts_data:
                meta = ctx["metadata"]
                sources_info.append({
                    "document_id": meta.get("document_id", ""),
                    "chunk_id": "",
                    "score": ctx["relevance_score"],
                    "metadata": meta,
                })
        else:
            # --- Parent-context resolution via centralized function ---
            resolved_contexts_data = resolve_parent_context(search_results)

            for ctx in resolved_contexts_data:
                meta = ctx["metadata"]
                sources_info.append({
                    "document_id": meta.get("document_id", ""),
                    "chunk_id": meta.get("chunk_index", ""),
                    "score": ctx["relevance_score"],
                    "metadata": meta,
                })
        
        if not resolved_contexts_data:
            return {
                "answer": settings.no_results_message,
                "sources": [],
                "metadata": {
                    "query_type": routing_decision.query_type.value,
                    "confidence": routing_decision.confidence,
                    "processing_time": time.time() - start_time,
                    "chunks_retrieved": 0
                }
            }
        
        # Step 6: Context assembly and optional compression
        original_context = format_context_xml(resolved_contexts_data)

        # Inject saved notes as priority context
        notes_context = get_formatted_notes_context(notebook_id)
        if notes_context:
            original_context = f"{notes_context}\n\n{original_context}"
        
        # Determine model based on routing
        if routing_decision.query_type == QueryType.FULL_DOCUMENT:
            model = settings.long_context_model
            # FULL_DOCUMENT: skip compression — the whole point is to send everything
            final_context = original_context
            was_compressed = False
            logger.info(f"Using long-context model ({model}), bypassing compression")
        elif routing_decision.query_type == QueryType.DIRECT_ANSWER:
            model = settings.chat_model
            final_context = original_context
            was_compressed = False
        else:
            model = settings.chat_model
            # Check if compression is needed
            estimated = estimate_tokens(original_context)
            if estimated <= settings.max_context_tokens:
                final_context = original_context
                was_compressed = False
            else:
                # Build chunk dicts for compress_context
                compression_chunks = [
                    {"text": r.get("text", ""), "metadata": r.get("metadata", {})}
                    for r in search_results if r.get("text")
                ]
                final_context = compress_context(
                    compression_chunks, query, settings.max_context_tokens
                )
                was_compressed = True
        
        # Step 7: Build messages and call LLM
        messages = _build_messages(
            query=query,
            context=final_context,
            conversation_history=conversation_history,
            query_type=routing_decision.query_type
        )

        # --- Auto-Continuation for long responses (aligned with streaming) ---
        client = get_openai_client()
        answer = ""

        for _ in range(settings.max_continuations):
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1,
                max_completion_tokens=4096,
            )

            chunk_text = response.choices[0].message.content or ""
            answer += chunk_text
            finish_reason = response.choices[0].finish_reason

            if finish_reason != "length":
                break

            # Response was truncated — append continuation prompt and retry
            messages.append({"role": "assistant", "content": chunk_text})
            messages.append({"role": "user", "content": settings.continuation_prompt})
            logger.info("Auto-continuation triggered (finish_reason=length)")

        processing_time = time.time() - start_time

        # --- Clean Citation Extraction (aligned with streaming) ---
        # Build sources_info in the same shape expected by extract_citations
        sources_for_citation = []
        for si in sources_info:
            meta = si.get("metadata", {})
            sources_for_citation.append({
                "document": meta.get("document_name", ""),
                "page": meta.get("page", 0),
                "chunk_index": meta.get("chunk_index", si.get("chunk_id", "")),
                "chunk_text": meta.get("text", ""),
                "relevance_score": si.get("score", 0.0),
                # Preserve original keys for downstream consumers
                "document_id": si.get("document_id", ""),
                "chunk_id": si.get("chunk_id", ""),
                "metadata": meta,
            })

        cited_sources = extract_citations(answer, sources_for_citation)

        return {
            "answer": answer,
            "sources": cited_sources,
            "metadata": {
                "query_type": routing_decision.query_type.value,
                "confidence": routing_decision.confidence,
                "model_used": model,
                "processing_time": processing_time,
                "chunks_retrieved": len(resolved_contexts_data),
                "context_compressed": was_compressed,
                "routing_explanation": routing_decision.reason,
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        raise



def _build_messages(
    query: str,
    context: str,
    conversation_history: list[dict] | None = None,
    query_type: QueryType = QueryType.SYNTHESIS
) -> list[dict]:
    """Build the message list for the LLM call."""
    
    settings = get_settings()
    system_prompt = settings.chat_system_prompt or settings.fallback_system_prompt
    
    if query_type == QueryType.FULL_DOCUMENT:
        system_prompt += settings.full_document_suffix
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history
    if conversation_history:
        for msg in conversation_history[-6:]:  # Last 6 messages for context
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
    
    user_prompt = settings.chat_user_prompt_template.format(
        context_text=context, query=query
    )
    
    messages.append({
        "role": "user",
        "content": user_prompt
    })
    
    return messages


async def generate_refinement_response(
    query: str,
    document_ids: list[str],
    conversation_history: list[dict] | None = None
) -> dict:
    """
    Generate a refinement response for conversations tied to specific documents.

    Args:
        query: User's refinement request
        document_ids: List of document IDs to use as context
        conversation_history: Previous conversation messages

    Returns:
        dict with answer, sources, and metadata
    """
    start_time = time.time()
    settings = get_settings()

    try:
        # Fetch all chunks for the specified documents
        search_results = _fetch_full_document_chunks(
            notebook_id="",  # Not needed when document_ids are provided
            document_ids=document_ids
        )

        # Costruisci il contesto usando XML
        resolved_contexts_data = []
        sources_info = []
        for result in search_results:
            content = result.get("text", "")
            if not content:
                continue
            metadata = result.get("metadata", {})
            
            resolved_contexts_data.append({
                "context_text": content,
                "metadata": metadata
            })
            
            sources_info.append({
                "document_id": metadata.get("document_id", ""),
                "chunk_id": result.get("id", ""),
                "score": result.get("relevance_score", 1.0),
                "metadata": metadata
            })

        if not resolved_contexts_data:
            return {
                "answer": settings.no_refinement_results_message,
                "sources": [],
                "metadata": {
                    "query_type": "refinement",
                    "processing_time": time.time() - start_time,
                    "chunks_retrieved": 0
                }
            }


        context_text = format_context_xml(resolved_contexts_data)

        # Find the last assistant response for refinement context
        previous_response = None
        if conversation_history:
            for msg in reversed(conversation_history):
                if msg.get("role") == "assistant":
                    previous_response = msg.get("content", "")
                    break

        # Build refinement messages
        system_prompt = settings.refinement_system_prompt
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-6:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

        # Choose the appropriate user prompt template
        if previous_response:
            user_prompt = settings.refinement_user_prompt_template.format(
                previous_response=previous_response,
                context_text=context_text,
                query=query
            )
        else:
            user_prompt = settings.refinement_user_prompt_no_history_template.format(
                context_text=context_text,
                query=query
            )

        messages.append({"role": "user", "content": user_prompt})

        # --- Auto-Continuation for long refinement responses ---
        model = settings.long_context_model
        client = get_openai_client()
        answer = ""

        for _ in range(settings.max_continuations):
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1,
                max_completion_tokens=4096,
            )

            chunk_text = response.choices[0].message.content or ""
            answer += chunk_text
            finish_reason = response.choices[0].finish_reason

            if finish_reason != "length":
                break

            messages.append({"role": "assistant", "content": chunk_text})
            messages.append({"role": "user", "content": settings.continuation_prompt})
            logger.info("Refinement auto-continuation triggered (finish_reason=length)")

        # --- Clean Citation Extraction ---
        sources_for_citation = []
        for si in sources_info:
            meta = si.get("metadata", {})
            sources_for_citation.append({
                "document": meta.get("document_name", ""),
                "page": meta.get("page", 0),
                "chunk_index": meta.get("chunk_index", si.get("chunk_id", "")),
                "chunk_text": meta.get("text", ""),
                "relevance_score": si.get("score", 0.0),
                "document_id": si.get("document_id", ""),
                "chunk_id": si.get("chunk_id", ""),
                "metadata": meta,
            })

        cited_sources = extract_citations(answer, sources_for_citation)

        return {
            "answer": answer,
            "sources": cited_sources,
            "metadata": {
                "query_type": "refinement",
                "model_used": model,
                "processing_time": time.time() - start_time,
                "chunks_retrieved": len(resolved_contexts_data),
            }
        }

    except Exception as e:
        logger.error(f"Error generating refinement response: {e}", exc_info=True)
        raise