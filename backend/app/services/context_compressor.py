"""
Context Compressor Module

Compresses retrieved chunks to reduce token usage while preserving relevant information.
Uses a smaller/cheaper reasoning model (like o4-mini) for compression when context exceeds token limits.
Supports a "Long Context" bypass to feed entire documents to models like gpt-4.1.
"""

from typing import List, Dict, Any
from openai import OpenAI

from app.config import get_settings

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    Approximate: 1 token ≈ 4 characters for Italian/English text.
    """
    return len(text) // 4


def format_chunk_with_source(chunk: Dict[str, Any]) -> str:
    """
    Format a chunk with its source information.
    """
    metadata = chunk.get("metadata", {})
    filename = metadata.get("document_name", metadata.get("filename", "documento"))
    page = metadata.get("page", "?")
    text = chunk.get("text", chunk.get("chunk_text", ""))
    
    return f"[{filename}, pagina {page}]\n{text}"


def compress_context(
    chunks: List[Dict[str, Any]], 
    query: str, 
    max_tokens: int = None,
    force_long_context: bool = False
) -> str:
    """
    Compress chunks to fit within token limit while preserving relevance.
    
    Args:
        chunks: List of chunk dictionaries
        query: User query for relevance filtering
        max_tokens: Maximum tokens for output context
        force_long_context: If True, bypasses compression entirely to use with gpt-4.1
        
    Returns:
        Compressed (or full) context string
    """
    if not chunks:
        return ""
    
    if max_tokens is None:
        max_tokens = settings.max_context_tokens

    # Format all chunks
    formatted_chunks = [format_chunk_with_source(c) for c in chunks]
    full_context = "\n\n---\n\n".join(formatted_chunks)
    
    # 1. BYPASS PER IL LONG CONTEXT (Es. per gpt-4.1)
    if force_long_context:
        return full_context
        
    # 2. Check if compression is even needed
    estimated = estimate_tokens(full_context)
    if estimated <= max_tokens:
        return full_context
    
    # 3. Compression needed - use the light model (e.g., o4-mini)
    try:
        # Prepara la chiamata API. I modelli serie "o" (reasoning) hanno regole diverse
        is_reasoning_model = settings.light_model.startswith(tuple(settings.reasoning_model_prefixes))
        
        api_kwargs = {
            "model": settings.light_model,
            "messages": [{
                "role": "user" if is_reasoning_model else "system",
                "content": settings.compression_prompt_template.format(
                    query=query,
                    full_context=full_context
                )
            }]
        }
        
        # Gestione parametri specifici per i modelli "o" vs "gpt"
        if is_reasoning_model:
            api_kwargs["max_completion_tokens"] = max_tokens
            # I modelli o4 non supportano la temperature standard nella stessa API call
        else:
            api_kwargs["max_completion_tokens"] = max_tokens
            api_kwargs["temperature"] = settings.compression_temperature

        response = client.chat.completions.create(**api_kwargs)
        compressed = response.choices[0].message.content
        return compressed
        
    except Exception as e:
        print(f"Errore durante la compressione con {settings.light_model}: {e}")
        # Fallback meccanico: truncate to fit (keep most relevant chunks first)
        truncated_chunks = []
        current_tokens = 0
        
        for chunk in formatted_chunks:
            chunk_tokens = estimate_tokens(chunk)
            if current_tokens + chunk_tokens <= max_tokens:
                truncated_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                break
        
        return "\n\n---\n\n".join(truncated_chunks)


def get_compression_stats(original: str, compressed: str) -> Dict[str, Any]:
    """
    Calculate compression statistics.
    """
    original_tokens = estimate_tokens(original)
    compressed_tokens = estimate_tokens(compressed)
    
    reduction = original_tokens - compressed_tokens
    reduction_percent = (reduction / original_tokens * 100) if original_tokens > 0 else 0
    
    return {
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "tokens_saved": reduction,
        "reduction_percent": round(reduction_percent, 2)
    }