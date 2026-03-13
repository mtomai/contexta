"""Tests for EmbeddingCache service."""
import pytest

from app.services.embedding_cache import EmbeddingCache


@pytest.fixture()
def cache():
    return EmbeddingCache(max_size=5)


class TestEmbeddingCache:
    def test_set_and_get(self, cache):
        cache.set("hello", [0.1, 0.2, 0.3])
        result = cache.get("hello")
        assert result == [0.1, 0.2, 0.3]

    def test_get_miss(self, cache):
        assert cache.get("nonexistent") is None

    def test_cache_key_deterministic(self, cache):
        key1 = cache._get_cache_key("same text")
        key2 = cache._get_cache_key("same text")
        assert key1 == key2

    def test_cache_key_different_texts(self, cache):
        key1 = cache._get_cache_key("text a")
        key2 = cache._get_cache_key("text b")
        assert key1 != key2

    def test_lru_eviction(self, cache):
        # Fill cache to max_size=5
        for i in range(5):
            cache.set(f"item{i}", [float(i)])

        # Add one more, which should evict the oldest (item0)
        cache.set("item5", [5.0])
        assert cache.get("item0") is None
        assert cache.get("item5") == [5.0]

    def test_lru_access_refreshes(self, cache):
        for i in range(5):
            cache.set(f"item{i}", [float(i)])

        # Access item0 to make it most recently used
        cache.get("item0")

        # Add new item, should evict item1 (next oldest)
        cache.set("item5", [5.0])
        assert cache.get("item0") is not None  # Still there
        assert cache.get("item1") is None       # Evicted

    def test_overwrite_existing_key(self, cache):
        cache.set("key", [1.0])
        cache.set("key", [2.0])
        assert cache.get("key") == [2.0]

    def test_get_stats(self, cache):
        cache.set("a", [1.0])
        cache.get("a")       # hit
        cache.get("b")       # miss

        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 5
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate_percent"] == 50.0

    def test_get_stats_empty(self, cache):
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate_percent"] == 0

    def test_clear(self, cache):
        cache.set("a", [1.0])
        cache.set("b", [2.0])
        cache.clear()
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        # After clear, gets are misses
        assert cache.get("a") is None
        assert cache.get("b") is None
