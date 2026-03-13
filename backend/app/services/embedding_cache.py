"""
Embedding Cache Module

Provides caching for query embeddings to reduce API calls and costs.
Query embeddings are cached in memory with an LRU eviction policy.
"""

import hashlib
from typing import List, Optional
from collections import OrderedDict
import threading


from app.config import get_settings


class EmbeddingCache:
    """Thread-safe LRU cache for embeddings."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize cache with maximum size.
        
        Args:
            max_size: Maximum number of embeddings to cache
        """
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text using MD5 hash."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        Retrieve embedding from cache if exists.
        
        Args:
            text: Text to look up
            
        Returns:
            Cached embedding or None if not found
        """
        key = self._get_cache_key(text)
        
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            
            self._misses += 1
            return None
    
    def set(self, text: str, embedding: List[float]) -> None:
        """
        Store embedding in cache.
        
        Args:
            text: Original text
            embedding: Embedding vector to cache
        """
        key = self._get_cache_key(text)
        
        with self._lock:
            # If key exists, move to end
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = embedding
                return
            
            # Add new entry
            self._cache[key] = embedding
            
            # Evict oldest if over capacity
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 2)
            }
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


# Global cache instance
_settings = get_settings()
_embedding_cache = EmbeddingCache(max_size=_settings.embedding_cache_size)


def get_embedding_cache() -> EmbeddingCache:
    """Get the global embedding cache instance."""
    return _embedding_cache


def create_embedding_cached(text: str) -> List[float]:
    """
    Create embedding with caching support.
    
    First checks cache, if miss then calls OpenAI API and caches result.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector
    """
    cache = get_embedding_cache()
    
    # Check cache first
    cached = cache.get(text)
    if cached is not None:
        return cached
    
    # Cache miss - call API
    from app.services.embeddings import create_embedding
    embedding = create_embedding(text)
    
    # Store in cache
    cache.set(text, embedding)
    
    return embedding
