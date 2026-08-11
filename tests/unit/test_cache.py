"""
Tests for the TTL cache behind provider quota protection.

This is a small piece of code carrying a disproportionate product constraint:
JSearch's free tier is a few hundred requests per *month*, so the difference
between a working cache and a broken one is whether a multi-user app is viable
on a hobby tier at all.
"""
import time

import pytest

from backend.core.cache import (
    CacheBackend,
    InMemoryTTLCache,
    get_cache,
    reset_cache,
)


@pytest.fixture
def cache():
    return InMemoryTTLCache(max_entries=4, default_ttl=60.0)


def test_conforms_to_the_backend_protocol(cache):
    """
    Phase 6 swaps in Redis. The protocol is what makes that an implementation
    rather than a refactor of every call site.
    """
    assert isinstance(cache, CacheBackend)


def test_stores_and_returns_a_value(cache):
    cache.set("k", ["job"])
    assert cache.get("k") == ["job"]


def test_missing_key_returns_none(cache):
    assert cache.get("absent") is None


def test_expired_entry_is_not_returned(cache):
    cache.set("k", ["job"], ttl=0.01)
    time.sleep(0.03)
    assert cache.get("k") is None


def test_expired_entry_is_dropped_on_read(cache):
    """
    Expiry is handled on read rather than by a sweeper task: with a bounded
    cache there is no unbounded-growth risk, and a background task would be one
    more thing needing clean shutdown.
    """
    cache.set("k", ["job"], ttl=0.01)
    time.sleep(0.03)
    cache.get("k")
    assert cache.size == 0


def test_invalidate_removes_one_entry(cache):
    cache.set("a", 1)
    cache.set("b", 2)
    cache.invalidate("a")
    assert cache.get("a") is None
    assert cache.get("b") == 2


def test_invalidating_an_absent_key_is_not_an_error(cache):
    cache.invalidate("never-existed")


def test_clear_empties_everything(cache):
    cache.set("a", 1)
    cache.clear()
    assert cache.size == 0
    assert cache.get("a") is None


# ---------------------------------------------------------------------------
# Bounding
# ---------------------------------------------------------------------------


def test_size_is_bounded(cache):
    """A long-running process must not grow without limit."""
    for i in range(10):
        cache.set(f"k{i}", i)
    assert cache.size <= 4


def test_expired_entries_are_evicted_before_live_ones(cache):
    """
    Discarding something already dead is strictly better than evicting an entry
    that would still have served a request.
    """
    cache.set("stale", "old", ttl=0.01)
    for i in range(3):
        cache.set(f"live{i}", i, ttl=60.0)
    time.sleep(0.03)

    cache.set("newcomer", "new", ttl=60.0)

    assert cache.get("stale") is None
    assert cache.get("newcomer") == "new"
    assert all(cache.get(f"live{i}") == i for i in range(3))


def test_overwriting_an_existing_key_does_not_evict(cache):
    for i in range(4):
        cache.set(f"k{i}", i)
    cache.set("k0", "updated")

    assert cache.size == 4
    assert cache.get("k0") == "updated"


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------


def test_hit_and_miss_counts_are_tracked(cache):
    """
    Hit rate is the number that says whether the quota protection is actually
    working — Phase 6 exports it, and a rate near zero means the cache key is
    varying when it should not.
    """
    cache.set("k", 1)
    cache.get("k")
    cache.get("k")
    cache.get("absent")

    assert cache.hits == 2
    assert cache.misses == 1
    assert cache.hit_rate == pytest.approx(2 / 3)


def test_hit_rate_is_zero_before_any_reads(cache):
    assert cache.hit_rate == 0.0


def test_expiry_counts_as_a_miss(cache):
    cache.set("k", 1, ttl=0.01)
    time.sleep(0.03)
    cache.get("k")
    assert cache.misses == 1


# ---------------------------------------------------------------------------
# Process-wide instance
# ---------------------------------------------------------------------------


def test_get_cache_returns_the_same_instance():
    """A cache that does not outlive the request it serves is not a cache."""
    assert get_cache() is get_cache()


def test_reset_cache_empties_the_shared_instance():
    get_cache().set("k", 1)
    reset_cache()
    assert get_cache().get("k") is None
