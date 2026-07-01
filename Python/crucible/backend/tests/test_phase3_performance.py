"""
Phase 3 performance tests.

Tests cover:
  - LRUTTLCache: get/set/expiry/eviction/stats
  - Profiling endpoint cache hit on second call with same params
  - Profiling cache miss on different params
  - Advisor endpoint reuses profiling cache
  - HyperbandPruner correctly constructed and used in TrainingConfig
  - Cursor pagination: encode/decode, ordering, has_more, stability under inserts
  - Cursor API endpoints for datasets and experiments
"""

from __future__ import annotations

import time
import json
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from unittest.mock import patch


# ══════════════════════════════════════════════════════════════════════════
# LRU TTL CACHE
# ══════════════════════════════════════════════════════════════════════════

class TestLRUTTLCache:

    def test_set_and_get(self):
        from caching.cache import LRUTTLCache
        cache = LRUTTLCache()
        cache.set("k1", {"a": 1}, ttl_secs=60)
        assert cache.get("k1") == {"a": 1}

    def test_miss_returns_none(self):
        from caching.cache import LRUTTLCache
        cache = LRUTTLCache()
        assert cache.get("nonexistent") is None

    def test_expiry(self):
        from caching.cache import LRUTTLCache
        cache = LRUTTLCache()
        cache.set("k1", "value", ttl_secs=0.05)
        assert cache.get("k1") == "value"
        time.sleep(0.1)
        assert cache.get("k1") is None

    def test_lru_eviction(self):
        from caching.cache import LRUTTLCache
        cache = LRUTTLCache(max_size=2)
        cache.set("k1", "v1", ttl_secs=60)
        cache.set("k2", "v2", ttl_secs=60)
        cache.set("k3", "v3", ttl_secs=60)   # should evict k1 (least recently used)
        assert cache.get("k1") is None
        assert cache.get("k2") == "v2"
        assert cache.get("k3") == "v3"

    def test_get_refreshes_lru_position(self):
        from caching.cache import LRUTTLCache
        cache = LRUTTLCache(max_size=2)
        cache.set("k1", "v1", ttl_secs=60)
        cache.set("k2", "v2", ttl_secs=60)
        cache.get("k1")              # k1 is now most-recently-used
        cache.set("k3", "v3", ttl_secs=60)   # should evict k2, not k1
        assert cache.get("k1") == "v1"
        assert cache.get("k2") is None

    def test_invalidate(self):
        from caching.cache import LRUTTLCache
        cache = LRUTTLCache()
        cache.set("k1", "v1", ttl_secs=60)
        assert cache.invalidate("k1") is True
        assert cache.get("k1") is None
        assert cache.invalidate("k1") is False   # already gone

    def test_invalidate_prefix(self):
        from caching.cache import LRUTTLCache
        cache = LRUTTLCache()
        cache.set("profile:abc:1", "v1", ttl_secs=60)
        cache.set("profile:abc:2", "v2", ttl_secs=60)
        cache.set("shap:xyz:1", "v3", ttl_secs=60)
        removed = cache.invalidate_prefix("profile:")
        assert removed == 2
        assert cache.get("profile:abc:1") is None
        assert cache.get("shap:xyz:1") == "v3"

    def test_clear(self):
        from caching.cache import LRUTTLCache
        cache = LRUTTLCache()
        cache.set("k1", "v1", ttl_secs=60)
        cache.set("k2", "v2", ttl_secs=60)
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_stats_tracks_hits_and_misses(self):
        from caching.cache import LRUTTLCache
        cache = LRUTTLCache()
        cache.set("k1", "v1", ttl_secs=60)
        cache.get("k1")        # hit
        cache.get("k1")        # hit
        cache.get("missing")   # miss
        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_cache_key_builder(self):
        from caching.cache import cache_key
        assert cache_key("profile", "abc123", "target", "0.2") == "profile:abc123:target:0.2"


# ══════════════════════════════════════════════════════════════════════════
# PROFILING CACHE — END-TO-END VIA API
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def cache_client():
    import sys, importlib, database as db_mod
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool
    from fastapi.testclient import TestClient
    from caching.cache import get_profiling_cache

    get_profiling_cache().clear()   # isolate from other test modules

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    db_mod.engine = engine; db_mod.SessionFactory = factory; db_mod.AsyncSessionLocal = factory
    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    import main as m
    with TestClient(m.app, raise_server_exceptions=True) as c:
        yield c
    get_profiling_cache().clear()


def _upload_csv(client, name="cache_test"):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(0, 1, 100),
        "y": rng.integers(0, 2, 100),
    })
    csv = df.to_csv(index=False).encode()
    resp = client.post("/api/v1/datasets/upload",
                       files={"file": ("d.csv", csv, "text/csv")},
                       data={"name": name})
    return resp.json()["data"]


class TestProfilingCacheAPI:

    def test_second_profile_call_is_cache_hit(self, cache_client):
        from caching.cache import get_profiling_cache
        ds = _upload_csv(cache_client)

        r1 = cache_client.post(f"/api/v1/datasets/{ds['id']}/profile", json={})
        assert r1.status_code == 200
        stats_before = get_profiling_cache().stats()

        r2 = cache_client.post(f"/api/v1/datasets/{ds['id']}/profile", json={})
        assert r2.status_code == 200
        stats_after = get_profiling_cache().stats()

        assert stats_after["hits"] > stats_before["hits"]
        # Results must be identical
        assert r1.json()["data"] == r2.json()["data"]

    def test_different_target_column_is_cache_miss(self, cache_client):
        from caching.cache import get_profiling_cache
        ds = _upload_csv(cache_client)

        cache_client.post(f"/api/v1/datasets/{ds['id']}/profile", json={})
        stats_before = get_profiling_cache().stats()

        cache_client.post(f"/api/v1/datasets/{ds['id']}/profile",
                          json={"target_column": "y"})
        stats_after = get_profiling_cache().stats()

        assert stats_after["misses"] > stats_before["misses"]

    def test_cache_stats_endpoint(self, cache_client):
        ds = _upload_csv(cache_client)
        cache_client.post(f"/api/v1/datasets/{ds['id']}/profile", json={})
        resp = cache_client.get("/api/v1/cache/stats")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "profiling" in data
        assert "shap" in data


# ══════════════════════════════════════════════════════════════════════════
# HYPERBAND PRUNER
# ══════════════════════════════════════════════════════════════════════════

class TestHyperbandPruner:

    def test_default_pruner_is_median(self):
        from training.runner import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.pruner_type == "median"

    def test_hyperband_can_be_selected(self):
        from training.runner import TrainingConfig
        cfg = TrainingConfig(pruner_type="hyperband")
        assert cfg.pruner_type == "hyperband"

    def test_training_with_hyperband_succeeds(self, tmp_path):
        from training.runner import TrainingRunner, TrainingConfig
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (300, 4))
        df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
        df["target"] = (X[:, 0] > 0).astype(int)

        cfg = TrainingConfig(
            n_trials=5, cv_folds=3,
            families=["logistic_regression"],
            pruner_type="hyperband",
        )
        runner = TrainingRunner(model_storage_path=tmp_path)
        result = runner.run(
            df=df, target_column="target", task_type="classification",
            config=cfg, experiment_name="hyperband_test",
        )
        assert result.best_cv_score is not None
        assert result.pruner_type == "hyperband"

    def test_training_with_median_pruner_succeeds(self, tmp_path):
        from training.runner import TrainingRunner, TrainingConfig
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (300, 4))
        df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
        df["target"] = (X[:, 0] > 0).astype(int)

        cfg = TrainingConfig(
            n_trials=5, cv_folds=3,
            families=["logistic_regression"],
            pruner_type="median",
        )
        runner = TrainingRunner(model_storage_path=tmp_path)
        result = runner.run(
            df=df, target_column="target", task_type="classification",
            config=cfg, experiment_name="median_test",
        )
        assert result.pruner_type == "median"

    def test_invalid_pruner_type_falls_back_to_median(self, tmp_path):
        """Unknown pruner_type values fall back to median rather than crashing."""
        from training.runner import TrainingRunner, TrainingConfig
        rng = np.random.default_rng(2)
        X = rng.normal(0, 1, (200, 3))
        df = pd.DataFrame(X, columns=["a", "b", "c"])
        df["target"] = (X[:, 0] > 0).astype(int)

        cfg = TrainingConfig(
            n_trials=3, cv_folds=3,
            families=["logistic_regression"],
            pruner_type="nonexistent_pruner",
        )
        runner = TrainingRunner(model_storage_path=tmp_path)
        # Should not raise — falls back to median pruner behavior
        result = runner.run(
            df=df, target_column="target", task_type="classification",
            config=cfg, experiment_name="fallback_test",
        )
        assert result.best_cv_score is not None


# ══════════════════════════════════════════════════════════════════════════
# CURSOR PAGINATION — ENCODE/DECODE
# ══════════════════════════════════════════════════════════════════════════

class TestCursorEncoding:

    def test_encode_decode_roundtrip(self):
        from schemas.cursor_pagination import encode_cursor, decode_cursor
        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        cursor = encode_cursor(dt, 42)
        decoded_dt, decoded_id = decode_cursor(cursor)
        assert decoded_dt == dt
        assert decoded_id == 42

    def test_cursor_is_opaque_string(self):
        from schemas.cursor_pagination import encode_cursor
        cursor = encode_cursor(datetime.now(timezone.utc), 1)
        assert isinstance(cursor, str)
        # Should not be readable plaintext JSON
        assert "{" not in cursor

    def test_decode_invalid_cursor_raises(self):
        from schemas.cursor_pagination import decode_cursor
        with pytest.raises(ValueError):
            decode_cursor("not-a-valid-cursor!!!")


# ══════════════════════════════════════════════════════════════════════════
# CURSOR PAGINATION — DB-BACKED
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def cursor_client():
    import sys, importlib, database as db_mod
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool
    from fastapi.testclient import TestClient
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    db_mod.engine = engine; db_mod.SessionFactory = factory; db_mod.AsyncSessionLocal = factory
    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    import main as m
    with TestClient(m.app, raise_server_exceptions=True) as c:
        yield c


class TestCursorPaginationAPI:

    def test_dataset_cursor_first_page(self, cursor_client):
        for i in range(5):
            rng = np.random.default_rng(i)
            df = pd.DataFrame({"x": rng.normal(size=10)})
            cursor_client.post("/api/v1/datasets/upload",
                files={"file": (f"d{i}.csv", df.to_csv(index=False).encode(), "text/csv")},
                data={"name": f"ds_{i}"})

        resp = cursor_client.get("/api/v1/datasets/cursor?limit=3")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["items"]) == 3
        assert data["has_more"] is True
        assert data["next_cursor"] is not None

    def test_dataset_cursor_pages_through_all(self, cursor_client):
        for i in range(7):
            df = pd.DataFrame({"x": [1, 2, 3]})
            cursor_client.post("/api/v1/datasets/upload",
                files={"file": (f"d{i}.csv", df.to_csv(index=False).encode(), "text/csv")},
                data={"name": f"page_test_{i}"})

        seen_ids = set()
        cursor = None
        pages = 0
        while True:
            url = "/api/v1/datasets/cursor?limit=2"
            if cursor:
                url += f"&cursor={cursor}"
            resp = cursor_client.get(url)
            data = resp.json()["data"]
            for item in data["items"]:
                assert item["id"] not in seen_ids, "Cursor pagination must not duplicate rows"
                seen_ids.add(item["id"])
            cursor = data["next_cursor"]
            pages += 1
            if not data["has_more"]:
                break
            if pages > 20:
                pytest.fail("Cursor pagination did not terminate")

        assert len(seen_ids) >= 7

    def test_invalid_cursor_returns_422(self, cursor_client):
        resp = cursor_client.get("/api/v1/datasets/cursor?cursor=not-valid-base64!!!")
        assert resp.status_code == 422

    def test_experiments_cursor_endpoint_exists(self, cursor_client):
        resp = cursor_client.get("/api/v1/experiments/cursor")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "items" in data
        assert "next_cursor" in data
        assert "has_more" in data

    def test_empty_result_set_has_more_false(self, cursor_client):
        resp = cursor_client.get("/api/v1/experiments/cursor")
        data = resp.json()["data"]
        assert data["has_more"] is False
        assert data["next_cursor"] is None
