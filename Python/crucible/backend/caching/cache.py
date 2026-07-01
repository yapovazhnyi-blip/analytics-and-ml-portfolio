"""
Result Cache — LRU cache with TTL for expensive, repeatable computations.

WHY THIS EXISTS
----------------
Profiling and SHAP computation are CPU-bound and can take 5-30 seconds on
medium datasets. Without caching:
  - Two users profiling the same dataset trigger two full computations
  - Refreshing a browser tab re-runs the same profiling job
  - The advisor endpoint, which calls profiling internally, re-profiles
    on every call even if nothing changed

CACHE KEY DESIGN
------------------
Keys are (operation, content_hash) tuples, not (operation, dataset_id).
Using content_hash instead of dataset_id means:
  - If a dataset's underlying file changes but keeps the same ID, the cache
    correctly misses (content_hash changes) rather than serving stale results
  - Two different datasets with identical content (e.g. a re-upload) share
    a cache entry, avoiding duplicate computation

BACKEND CHOICE
---------------
This is an in-memory cache (a single Python dict with TTL eviction), not Redis.
For a single-process deployment (current Crucible architecture — one Uvicorn
worker due to SQLite), in-memory is simpler and has zero additional infra.

For multi-worker or multi-instance deployments, swap CacheBackend's
implementation for a Redis-backed one — the interface (get/set/invalidate)
stays identical, so callers don't change.

TTL DEFAULTS
------------
Profiling results: 1 hour (datasets rarely change mid-session)
SHAP values: 24 hours (tied to a specific completed experiment, immutable)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Optional


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class LRUTTLCache:
    """
    Thread-safe in-memory cache with LRU eviction and per-entry TTL.

    Usage:
        cache = LRUTTLCache(max_size=100)
        cache.set("key", expensive_result, ttl_secs=3600)
        result = cache.get("key")   # None if missing or expired
    """

    def __init__(self, max_size: int = 200):
        self._max_size = max_size
        self._store: dict[str, _CacheEntry] = {}
        self._access_order: list[str] = []   # most-recently-used at the end
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if time.monotonic() > entry.expires_at:
                # Expired — evict and treat as a miss
                del self._store[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self._misses += 1
                return None

            self._hits += 1
            self._touch(key)
            return entry.value

    def set(self, key: str, value: Any, ttl_secs: float = 3600) -> None:
        with self._lock:
            self._store[key] = _CacheEntry(
                value=value,
                expires_at=time.monotonic() + ttl_secs,
            )
            self._touch(key)
            self._evict_if_needed()

    def invalidate(self, key: str) -> bool:
        """Removes a specific key. Returns True if it existed."""
        with self._lock:
            existed = key in self._store
            self._store.pop(key, None)
            if key in self._access_order:
                self._access_order.remove(key)
            return existed

    def invalidate_prefix(self, prefix: str) -> int:
        """Removes all keys starting with prefix. Returns count removed."""
        with self._lock:
            to_remove = [k for k in self._store if k.startswith(prefix)]
            for k in to_remove:
                del self._store[k]
                if k in self._access_order:
                    self._access_order.remove(k)
            return len(to_remove)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._access_order.clear()

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size":      len(self._store),
                "max_size":  self._max_size,
                "hits":      self._hits,
                "misses":    self._misses,
                "hit_rate":  round(self._hits / total, 4) if total else 0.0,
            }

    def _touch(self, key: str) -> None:
        """Marks key as most-recently-used. Caller must hold the lock."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _evict_if_needed(self) -> None:
        """Evicts least-recently-used entries until under max_size. Caller must hold the lock."""
        while len(self._store) > self._max_size and self._access_order:
            oldest = self._access_order.pop(0)
            self._store.pop(oldest, None)


# ── Module-level singletons ────────────────────────────────────────────────────
# One cache per operation type — keeps profiling and SHAP entries from
# evicting each other under memory pressure, and allows different TTLs.

_profiling_cache = LRUTTLCache(max_size=100)
_shap_cache       = LRUTTLCache(max_size=100)

PROFILING_TTL_SECS = 3600        # 1 hour
SHAP_TTL_SECS       = 86400       # 24 hours — tied to an immutable completed experiment


def get_profiling_cache() -> LRUTTLCache:
    return _profiling_cache


def get_shap_cache() -> LRUTTLCache:
    return _shap_cache


def cache_key(*parts: str) -> str:
    """Builds a deterministic cache key from string parts."""
    return ":".join(str(p) for p in parts)
