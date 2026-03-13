import asyncio
from typing import List
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import tiktoken

from app.config import get_settings

settings = get_settings()

# Initialize OpenAI client
client = OpenAI(api_key=settings.openai_api_key)

# Thread pool for parallel embedding operations
_executor = ThreadPoolExecutor(max_workers=4)


# Cache the tokenizer
_tokenizer = None


def get_tokenizer():
    """Get or create the tiktoken tokenizer for embedding models."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))


def truncate_text_for_embedding(text: str, max_tokens: int = None) -> str:
    """
    Truncate text to fit within embedding model token limits.

    Args:
        text: Text to potentially truncate
        max_tokens: Maximum token count (default from settings)

    Returns:
        Truncated text if needed, otherwise original text
    """
    if max_tokens is None:
        max_tokens = settings.max_embedding_tokens
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)

    if len(tokens) <= max_tokens:
        return text

    # Truncate to max_tokens and decode back to text
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens)

    # Try to break at a clean boundary (sentence or word)
    last_period = truncated_text.rfind('.')
    last_newline = truncated_text.rfind('\n')
    break_point = max(last_period, last_newline)

    if break_point > len(truncated_text) * 0.8:  # Only use if we keep at least 80%
        return truncated_text[:break_point + 1]

    # Otherwise just break at word boundary
    last_space = truncated_text.rfind(' ')
    if last_space > len(truncated_text) * 0.9:
        return truncated_text[:last_space]

    return truncated_text


def create_embeddings(texts: List[str], model: str = None) -> List[List[float]]:
    """
    Create embeddings for a list of texts using OpenAI API.

    Args:
        texts: List of text strings to embed
        model: Embedding model to use (default from settings)

    Returns:
        List of embedding vectors
    """
    if model is None:
        model = settings.embedding_model

    try:
        # Truncate texts that are too long for the embedding model
        # Also filter out empty strings and None values (OpenAI API returns 400 for empty input)
        truncated_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                truncated = truncate_text_for_embedding(text)
                if truncated and truncated.strip():
                    truncated_texts.append(truncated)
                    valid_indices.append(i)

        if not truncated_texts:
            raise ValueError("No valid texts to embed - all texts are empty or whitespace")

        # OpenAI API can handle batch requests
        response = client.embeddings.create(
            input=truncated_texts,
            model=model
        )

        # Extract embeddings from response
        valid_embeddings = [item.embedding for item in response.data]

        # Reconstruct full list with None for invalid entries
        embeddings = [None] * len(texts)
        for idx, emb in zip(valid_indices, valid_embeddings):
            embeddings[idx] = emb

        return embeddings

    except Exception as e:
        raise RuntimeError(f"Error creating embeddings: {str(e)}")


def create_embedding(text: str, model: str = None) -> List[float]:
    """
    Create embedding for a single text using OpenAI API.

    Args:
        text: Text string to embed
        model: Embedding model to use (default from settings)

    Returns:
        Embedding vector
    """
    embeddings = create_embeddings([text], model)
    return embeddings[0]


def embed_document_chunks(chunks: List[dict]) -> List[dict]:
    """
    Create embeddings for document chunks.

    Args:
        chunks: List of chunk dictionaries with 'text' and 'metadata' fields

    Returns:
        List of chunks with added 'embedding' field (only chunks with valid embeddings)
    """
    # Extract texts to embed
    texts = [chunk["text"] for chunk in chunks]

    # Create embeddings in batch
    embeddings = create_embeddings(texts)

    # Add embeddings to chunks, filtering out chunks with None embeddings
    result = []
    for chunk, embedding in zip(chunks, embeddings):
        if embedding is not None:
            chunk["embedding"] = embedding
            result.append(chunk)

    return result


async def create_embeddings_parallel(
    texts: List[str],
    batch_size: int = 1000
) -> List[List[float]]:
    """
    Create embeddings in parallel batches for large documents.

    Splits texts into batches and processes them concurrently
    using a thread pool for improved throughput.

    Args:
        texts: List of texts to embed
        batch_size: Number of texts per batch (OpenAI supports up to 2048, using 1000 for safety)

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    # Single batch - no parallelization needed
    if len(texts) <= batch_size:
        return create_embeddings(texts)
    
    loop = asyncio.get_event_loop()
    
    # Split into batches
    batches = [
        texts[i:i + batch_size] 
        for i in range(0, len(texts), batch_size)
    ]
    
    # Process batches in parallel
    tasks = [
        loop.run_in_executor(_executor, create_embeddings, batch)
        for batch in batches
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Flatten results
    return [emb for batch_result in results for emb in batch_result]


async def embed_document_chunks_parallel(
    chunks: List[dict],
    batch_size: int = 100
) -> List[dict]:
    """
    Create embeddings for document chunks using parallel processing.

    Useful for large documents with many chunks.

    Args:
        chunks: List of chunk dictionaries with 'text' and 'metadata' fields
        batch_size: Number of chunks per batch (increased from 20 for better throughput)

    Returns:
        List of chunks with added 'embedding' field (only chunks with valid embeddings)
    """
    texts = [chunk["text"] for chunk in chunks]

    embeddings = await create_embeddings_parallel(texts, batch_size)

    # Add embeddings to chunks, filtering out chunks with None embeddings
    result = []
    for chunk, embedding in zip(chunks, embeddings):
        if embedding is not None:
            chunk["embedding"] = embedding
            result.append(chunk)

    return result

