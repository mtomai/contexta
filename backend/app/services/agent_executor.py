import re
import json
import logging
from typing import Dict, List, Any, AsyncGenerator
from openai import AsyncOpenAI

from app.config import get_settings
from app.services.conversation_db import get_conversation_db
from app.services.chat_utils import (
    _fetch_full_document_chunks,
    format_context_xml,
    extract_citations,
    get_formatted_notes_context,
)

logger = logging.getLogger(__name__)

settings = get_settings()


def _format_sse(event: str, data: dict) -> str:
    """Format data as Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def build_prompt_with_variables(template: str, variables: Dict[str, Any]) -> str:
    """
    Replace {{variable_name}} placeholders with actual values.

    Supports:
    - {{variable_name}} - simple replacement
    - {{variable_name|default_value}} - with fallback

    Args:
        template: Template string with {{placeholders}}
        variables: Dictionary of variable values

    Returns:
        Processed string with placeholders replaced
    """
    def replace_var(match):
        var_expr = match.group(1)
        if '|' in var_expr:
            var_name, default = var_expr.split('|', 1)
            return str(variables.get(var_name.strip(), default.strip()))
        var_name = var_expr.strip()
        # Keep placeholder if variable not found (for debugging)
        return str(variables.get(var_name, f'{{{{{var_name}}}}}'))

    return re.sub(r'\{\{([^}]+)\}\}', replace_var, template)


async def execute_agent_prompt(
    agent_prompt: Dict[str, Any],
    document_ids: List[str],
    notebook_id: str,
    variable_values: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute an agent prompt with the three-layer system.

    Layers:
    1. system_prompt - Sets AI behavior and rules
    2. user_prompt - Main instruction with {{context}} and variables
    3. template_prompt - Optional document template for output structure

    Args:
        agent_prompt: Agent prompt configuration from database
        document_ids: List of document IDs to process
        notebook_id: Notebook UUID
        variable_values: Values for dynamic variables

    Returns:
        Dictionary with conversation_id and title
    """
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    # Get document chunks using shared full-document fetcher
    chunks = _fetch_full_document_chunks(notebook_id=notebook_id, document_ids=document_ids)

    if not chunks:
        # Create conversation with error message
        conv_db = get_conversation_db()
        conversation_title = f"[Agent] {agent_prompt['name']}"
        conversation_id = conv_db.create_conversation(
            title=conversation_title,
            notebook_id=notebook_id,
            document_ids=document_ids
        )
        conv_db.add_message(
            conversation_id=conversation_id,
            role="user",
            content=f"[Agent: {agent_prompt['name']}]"
        )
        conv_db.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=settings.agent_no_content_message
        )
        return {
            "conversation_id": conversation_id,
            "title": conversation_title
        }

    # Build sources_info from standardised chunk dicts
    sources_info = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        sources_info.append({
            "document": meta.get("document_name", ""),
            "page": meta.get("page", 0),
            "chunk_index": meta.get("parent_index", meta.get("chunk_index", 0)),
            "chunk_text": chunk.get("text", ""),
            "relevance_score": chunk.get("relevance_score", 1.0),
        })

    # Format context as XML
    context_text = format_context_xml([
        {"context_text": c["text"], "metadata": c["metadata"]}
        for c in chunks
    ])

    # Inject saved notes as priority context
    notes_context = get_formatted_notes_context(notebook_id)
    if notes_context:
        context_text = f"{notes_context}\n\n{context_text}"

    # Prepare all variables including context
    all_variables = {
        **variable_values,
        "context": context_text,
    }

    # Build prompts from templates
    system_prompt = build_prompt_with_variables(
        agent_prompt["system_prompt"],
        all_variables
    )

    user_prompt = build_prompt_with_variables(
        agent_prompt["user_prompt"],
        all_variables
    )

    # Add template if present
    if agent_prompt.get("template_prompt"):
        template_content = build_prompt_with_variables(
            agent_prompt["template_prompt"],
            all_variables
        )
        user_prompt += settings.agent_template_prefix.format(
            template_content=template_content
        )

    # Add citation instruction
    user_prompt += "\n\n" + settings.agent_citation_instruction

    # Execute LLM call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    answer = ""

    try:
        for _ in range(settings.max_continuations):
            response = await client.chat.completions.create(
                model=settings.chat_model,
                messages=messages
            )

            chunk_content = response.choices[0].message.content or ""
            answer += chunk_content

            if response.choices[0].finish_reason != "length":
                break

            messages.append({"role": "assistant", "content": chunk_content})
            messages.append({"role": "user", "content": settings.continuation_prompt})

    except Exception as e:
        raise RuntimeError(f"Error calling chat model: {str(e)}")

    # Extract citations
    cited_sources = extract_citations(answer, sources_info)

    # Create conversation with result
    conv_db = get_conversation_db()
    conversation_title = f"[Agent] {agent_prompt['name']}"
    conversation_id = conv_db.create_conversation(
        title=conversation_title,
        notebook_id=notebook_id,
        document_ids=document_ids
    )

    # Format variable values for display
    vars_display = ", ".join([f"{k}={v}" for k, v in variable_values.items()]) if variable_values else "none"

    # Store messages
    conv_db.add_message(
        conversation_id=conversation_id,
        role="user",
        content=f"[Agent: {agent_prompt['name']}]\nVariabili: {vars_display}"
    )

    sources_for_db = [
        {
            "document": s.get("document", ""),
            "page": s.get("page", 0),
            "chunk_text": s.get("chunk_text", ""),
            "relevance_score": s.get("relevance_score", 1.0)
        }
        for s in cited_sources[:10]  # Limit sources stored
    ]

    conv_db.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        sources=sources_for_db
    )

    return {
        "conversation_id": conversation_id,
        "title": conversation_title
    }


async def execute_agent_prompt_stream(
    agent_prompt: Dict[str, Any],
    document_ids: List[str],
    notebook_id: str,
    variable_values: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """
    Execute an agent prompt with streaming SSE output.

    Same logic as execute_agent_prompt but yields SSE events:
    - thought: progress messages
    - token: individual response tokens
    - done: completion with conversation_id, title, sources
    - error: error message
    """
    try:
        yield _format_sse("thought", {"message": "Inizializzazione agente e lettura documenti..."})

        client = AsyncOpenAI(api_key=settings.openai_api_key)

        # Get document chunks using shared full-document fetcher
        chunks = _fetch_full_document_chunks(notebook_id=notebook_id, document_ids=document_ids)

        if not chunks:
            # Create conversation with error message
            conv_db = get_conversation_db()
            conversation_title = f"[Agent] {agent_prompt['name']}"
            conversation_id = conv_db.create_conversation(
                title=conversation_title,
                notebook_id=notebook_id,
                document_ids=document_ids
            )
            conv_db.add_message(
                conversation_id=conversation_id,
                role="user",
                content=f"[Agent: {agent_prompt['name']}]"
            )
            conv_db.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=settings.agent_no_content_message
            )
            yield _format_sse("token", {"content": settings.agent_no_content_message})
            yield _format_sse("done", {
                "conversation_id": conversation_id,
                "title": conversation_title,
                "sources": []
            })
            return

        # Build sources_info from standardised chunk dicts
        sources_info = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            sources_info.append({
                "document": meta.get("document_name", ""),
                "page": meta.get("page", 0),
                "chunk_index": meta.get("parent_index", meta.get("chunk_index", 0)),
                "chunk_text": chunk.get("text", ""),
                "relevance_score": chunk.get("relevance_score", 1.0),
            })

        # Format context as XML
        context_text = format_context_xml([
            {"context_text": c["text"], "metadata": c["metadata"]}
            for c in chunks
        ])

        # Inject saved notes as priority context
        notes_context = get_formatted_notes_context(notebook_id)
        if notes_context:
            context_text = f"{notes_context}\n\n{context_text}"

        # Prepare all variables including context
        all_variables = {
            **variable_values,
            "context": context_text,
        }

        # Build prompts from templates
        system_prompt = build_prompt_with_variables(
            agent_prompt["system_prompt"],
            all_variables
        )

        user_prompt = build_prompt_with_variables(
            agent_prompt["user_prompt"],
            all_variables
        )

        # Add template if present
        if agent_prompt.get("template_prompt"):
            template_content = build_prompt_with_variables(
                agent_prompt["template_prompt"],
                all_variables
            )
            user_prompt += settings.agent_template_prefix.format(
                template_content=template_content
            )

        # Add citation instruction
        user_prompt += "\n\n" + settings.agent_citation_instruction

        yield _format_sse("thought", {"message": "Analisi dei contenuti e generazione del report in corso..."})

        # Execute LLM call with streaming
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        full_response = ""

        for continuation_i in range(settings.max_continuations):
            stream = await client.chat.completions.create(
                model=settings.chat_model,
                messages=messages,
                stream=True
            )

            chunk_response = ""
            finish_reason = None

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    chunk_response += token
                    full_response += token
                    yield _format_sse("token", {"content": token})

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            if finish_reason != "length":
                break

            # Auto-continuation
            messages.append({"role": "assistant", "content": chunk_response})
            messages.append({"role": "user", "content": settings.continuation_prompt})
            yield _format_sse("thought", {"message": "Estensione del report..."})
            logger.info("AGENT STREAM: auto-continuation %d triggered", continuation_i + 1)

        # Extract citations
        cited_sources = extract_citations(full_response, sources_info)

        # Create conversation with result
        conv_db = get_conversation_db()
        conversation_title = f"[Agent] {agent_prompt['name']}"
        conversation_id = conv_db.create_conversation(
            title=conversation_title,
            notebook_id=notebook_id,
            document_ids=document_ids
        )

        # Format variable values for display
        vars_display = ", ".join([f"{k}={v}" for k, v in variable_values.items()]) if variable_values else "none"

        # Store messages
        conv_db.add_message(
            conversation_id=conversation_id,
            role="user",
            content=f"[Agent: {agent_prompt['name']}]\nVariabili: {vars_display}"
        )

        sources_for_db = [
            {
                "document": s.get("document", ""),
                "page": s.get("page", 0),
                "chunk_text": s.get("chunk_text", ""),
                "relevance_score": s.get("relevance_score", 1.0)
            }
            for s in cited_sources[:10]
        ]

        conv_db.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=full_response,
            sources=sources_for_db
        )

        logger.info(
            "AGENT STREAM DONE | response_len=%d | sources=%d",
            len(full_response), len(cited_sources),
        )

        yield _format_sse("done", {
            "conversation_id": conversation_id,
            "title": conversation_title,
            "sources": cited_sources
        })

    except Exception as e:
        logger.error("AGENT STREAM ERROR: %s", str(e), exc_info=True)
        yield _format_sse("error", {"message": str(e)})
