"""
Cache used to keep provider quotas survivable.

WHY THIS MATTERS MORE THAN IT LOOKS
-----------------------------------
JSearch's free tier is a few hundred requests per *month*. Without caching, two
users running the same search, or one user refreshing, spends real budget for
an identical answer. The cache is what makes a multi-user app viable on a hobby
tier at all — it is a correctness constraint on the product, not a latency
optimization.

WHY NOT cachetools
------------------
The roadmap suggested `cachetools.TTLCache`. Defining a small `CacheBackend`
protocol instead costs about fifty lines and buys the thing the roadmap
actually wants from this module: Phase 6 swaps in Redis "without changing
callers". With an explicit interface that swap is implementing a protocol; with
a concrete third-party class it is a refactor of every call site. It also
avoids adding a dependency for behaviour this simple.

CONCURRENCY
-----------
The in-memory implementation is safe under asyncio because every method is
synchronous and contains no await points, so no other task can interleave
mid-operation. It is NOT safe across processes or threads — which is exactly
why it is behind an interface.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Provider responses age slowly. Job boards do not turn over in minutes, and a
# stale-by-an-hour listing is a far smaller problem than an exhausted quota.
DEFAULT_TTL_SECONDS = 6 * 60 * 60  # 6 hours

# Bounded so a long-running process cannot grow without limit. Each entry is a
# provider's decoded response for one criteria set.
DEFAULT_MAX_ENTRIES = 512


@runtime_checkable
class CacheBackend(Protocol):
    """Minimal cache contract. Phase 6 adds a Redis implementation."""

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value, or None if absent or expired."""
        ...

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store a value under `key`."""
        ...

    def invalidate(self, key: str) -> None:
        """Drop a single entry. No error if it is not present."""
        ...

    def clear(self) -> None:
        """Drop everything. Primarily for tests and admin actions."""
        ...


class InMemoryTTLCache:
    """
    Process-local TTL cache with a bounded size.

    Eviction is oldest-insertion-first rather than true LRU. For this workload
    the two are nearly equivalent — entries expire on a fixed TTL and are
    rarely re-read many times — and insertion order comes free with `dict`,
    where real LRU would need extra bookkeeping on every read.
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        default_ttl: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        # key -> (expires_at_monotonic, value)
        self._entries: dict[str, tuple[float, Any]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None

        expires_at, value = entry
        if expires_at <= time.monotonic():
            # Expired entries are removed on read rather than by a sweeper.
            # With a bounded dict there is no unbounded-growth risk, and a
            # background task would be one more thing to shut down cleanly.
            del self._entries[key]
            self.misses += 1
            return None

        self.hits += 1
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        if len(self._entries) >= self._max_entries and key not in self._entries:
            self._evict_oldest()
        expires_at = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
        self._entries[key] = (expires_at, value)

    def invalidate(self, key: str) -> None:
        self._entries.pop(key, None)

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def _evict_oldest(self) -> None:
        # Prefer discarding something already expired before evicting a live
        # entry — cheaper than it looks, and only runs when the cache is full.
        now = time.monotonic()
        for key, (expires_at, _) in self._entries.items():
            if expires_at <= now:
                del self._entries[key]
                return
        oldest = next(iter(self._entries), None)
        if oldest is not None:
            del self._entries[oldest]

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        """Hit rate since construction, or 0.0 with no reads yet."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


_default_cache: Optional[InMemoryTTLCache] = None


def get_cache() -> InMemoryTTLCache:
    """
    Process-wide cache instance.

    A module-level singleton rather than a per-request object on purpose: a
    cache that does not outlive the request it serves is not a cache.
    """
    global _default_cache
    if _default_cache is None:
        _default_cache = InMemoryTTLCache()
    return _default_cache


def reset_cache() -> None:
    """Drop the process-wide cache. For tests and admin use."""
    if _default_cache is not None:
        _default_cache.clear()
