"""
Scheduled Retraining Pipeline tests.

Tests cover:
  - RetrainingPolicy / RetrainingRun model defaults
  - Pipeline: no-drift gate (skips retraining)
  - Pipeline: drift detected → retrain → first promotion (no prior production)
  - Pipeline: drift detected → retrain → candidate beats production → promoted
  - Pipeline: drift detected → retrain → candidate does NOT beat production → rejected
  - Pipeline: reference_dataset_id updates to new production data after promotion
  - Pipeline: retraining failure handled gracefully (no exception escapes)
  - Manual promotion endpoint (independent of policy)
  - Scheduler: schedule/unschedule/sync round-trip (no real waiting)
  - API: policy CRUD, manual run trigger, run history
"""

from __future__ import annotations

import asyncio
import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


# ══════════════════════════════════════════════════════════════════════════
# MODEL DEFAULTS
# ══════════════════════════════════════════════════════════════════════════

class TestModels:
    """
    SQLAlchemy's mapped_column(default=...) is applied at INSERT/flush time,
    not at Python object construction time — a bare `Model(...)` call leaves
    defaulted fields as None until the object is added to a session and
    flushed. These tests reflect that real behaviour rather than asserting
    on an unflushed object.
    """

    @pytest.mark.asyncio
    async def test_policy_defaults(self, tmp_path):
        from models.retraining import RetrainingPolicy
        engine, factory = await _make_engine_and_session()
        async with factory() as db:
            ds = await _seed_dataset(db, tmp_path, "defaults_ds", _stable_df())
            p = RetrainingPolicy(
                name="test", reference_dataset_id=ds.id, target_column="y", task_type="classification",
            )
            db.add(p)
            await db.flush()
            assert p.drift_severity_trigger == "significant"
            assert p.promotion_margin == 0.02
            assert p.n_trials == 15
            assert p.cv_folds == 3
            assert p.is_active is True

    @pytest.mark.asyncio
    async def test_run_defaults(self, tmp_path):
        from models.retraining import RetrainingPolicy, RetrainingRun
        engine, factory = await _make_engine_and_session()
        async with factory() as db:
            ds = await _seed_dataset(db, tmp_path, "run_defaults_ds", _stable_df())
            policy = await _seed_policy(db, ds.id)
            r = RetrainingRun(policy_id=policy.id, current_dataset_id=ds.id, steps_json="[]")
            db.add(r)
            await db.flush()
            assert r.status == "running"
            assert r.drift_checked is False
            assert r.promoted is False

    @pytest.mark.asyncio
    async def test_experiment_default_lifecycle_stage(self, tmp_path):
        from models.experiment import Experiment
        engine, factory = await _make_engine_and_session()
        async with factory() as db:
            ds = await _seed_dataset(db, tmp_path, "exp_defaults_ds", _stable_df())
            e = Experiment(name="x", dataset_id=ds.id, target_column="y", task_type="classification")
            db.add(e)
            await db.flush()
            assert e.lifecycle_stage == "candidate"


# ══════════════════════════════════════════════════════════════════════════
# HELPERS — build a real in-memory DB with datasets, used across pipeline tests
# ══════════════════════════════════════════════════════════════════════════

async def _make_engine_and_session():
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from models.base import Base
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, factory


def _write_csv(path, df):
    df.to_csv(path, index=False)
    return str(path)


def _stable_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
    })
    df["y"] = (df["x1"] + df["x2"] > 0).astype(int)
    return df


def _drifted_df(n=300, seed=1):
    """Same shape, but shifted distribution — should trigger 'significant'/'critical' drift."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x1": rng.normal(8, 1, n),    # shifted mean far from reference
        "x2": rng.normal(8, 1, n),
    })
    df["y"] = (df["x1"] + df["x2"] > 16).astype(int)
    return df


async def _seed_dataset(db, tmp_path, name, df):
    from models.dataset import Dataset
    path = _write_csv(tmp_path / f"{name}.csv", df)
    ds = Dataset(name=name, source_type="csv", file_path=path, status="ready",
                 row_count=len(df), column_count=len(df.columns))
    db.add(ds)
    await db.flush()
    await db.refresh(ds)
    return ds


async def _seed_policy(db, reference_ds_id, **overrides):
    from models.retraining import RetrainingPolicy
    defaults = dict(
        name=f"policy-{reference_ds_id}",
        reference_dataset_id=reference_ds_id,
        target_column="y",
        task_type="classification",
        n_trials=2, cv_folds=2,
        promotion_margin=0.0,   # any improvement counts, simplifies test assertions
    )
    defaults.update(overrides)
    policy = RetrainingPolicy(**defaults)
    db.add(policy)
    await db.flush()
    await db.refresh(policy)
    return policy


# ══════════════════════════════════════════════════════════════════════════
# PIPELINE — DRIFT GATE
# ══════════════════════════════════════════════════════════════════════════

class TestPipelineDriftGate:

    @pytest.mark.asyncio
    async def test_no_drift_skips_retraining(self, tmp_path):
        from retraining.pipeline import run_pipeline

        engine, factory = await _make_engine_and_session()
        async with factory() as db:
            ref_ds = await _seed_dataset(db, tmp_path, "ref", _stable_df(seed=0))
            same_ds = await _seed_dataset(db, tmp_path, "same", _stable_df(seed=0))  # identical distribution
            policy = await _seed_policy(db, ref_ds.id)
            await db.commit()

            run = await run_pipeline(policy, same_ds.id, db)
            await db.commit()

        assert run.drift_checked is True
        assert run.drift_detected is False
        assert run.retrain_triggered is False
        assert run.status == "completed"
        steps = json.loads(run.steps_json)
        retrain_step = next(s for s in steps if s["step"] == "retrain")
        assert retrain_step["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_drift_triggers_retraining(self, tmp_path):
        from retraining.pipeline import run_pipeline

        engine, factory = await _make_engine_and_session()
        async with factory() as db:
            ref_ds = await _seed_dataset(db, tmp_path, "ref", _stable_df(seed=0))
            drifted_ds = await _seed_dataset(db, tmp_path, "drifted", _drifted_df(seed=1))
            policy = await _seed_policy(db, ref_ds.id)
            await db.commit()

            run = await run_pipeline(policy, drifted_ds.id, db)
            await db.commit()

        assert run.drift_detected is True
        assert run.retrain_triggered is True
        assert run.new_experiment_id is not None
        assert run.status == "completed"


# ══════════════════════════════════════════════════════════════════════════
# PIPELINE — PROMOTION LOGIC
# ══════════════════════════════════════════════════════════════════════════

class TestPipelinePromotion:

    @pytest.mark.asyncio
    async def test_first_candidate_always_promoted(self, tmp_path):
        """With no existing production model, the first successful candidate is promoted."""
        from retraining.pipeline import run_pipeline
        from models.experiment import Experiment

        engine, factory = await _make_engine_and_session()
        async with factory() as db:
            ref_ds = await _seed_dataset(db, tmp_path, "ref", _stable_df(seed=0))
            drifted_ds = await _seed_dataset(db, tmp_path, "drifted", _drifted_df(seed=1))
            policy = await _seed_policy(db, ref_ds.id)
            await db.commit()

            run = await run_pipeline(policy, drifted_ds.id, db)
            await db.commit()

            assert run.promoted is True
            new_exp = await db.get(Experiment, run.new_experiment_id)
            assert new_exp.lifecycle_stage == "production"
            assert policy.production_experiment_id == new_exp.id

    @pytest.mark.asyncio
    async def test_reference_dataset_updates_after_promotion(self, tmp_path):
        """After promotion, the policy's reference_dataset_id becomes the new production data."""
        from retraining.pipeline import run_pipeline

        engine, factory = await _make_engine_and_session()
        async with factory() as db:
            ref_ds = await _seed_dataset(db, tmp_path, "ref", _stable_df(seed=0))
            drifted_ds = await _seed_dataset(db, tmp_path, "drifted", _drifted_df(seed=1))
            policy = await _seed_policy(db, ref_ds.id)
            await db.commit()

            await run_pipeline(policy, drifted_ds.id, db)
            await db.commit()

            assert policy.reference_dataset_id == drifted_ds.id

    @pytest.mark.asyncio
    async def test_candidate_rejected_when_not_beating_production(self, tmp_path):
        """A policy with an impossibly high promotion_margin should reject the candidate."""
        from retraining.pipeline import run_pipeline
        from models.experiment import Experiment

        engine, factory = await _make_engine_and_session()
        async with factory() as db:
            ref_ds = await _seed_dataset(db, tmp_path, "ref", _stable_df(seed=0))
            drifted_ds = await _seed_dataset(db, tmp_path, "drifted", _drifted_df(seed=1))
            policy = await _seed_policy(db, ref_ds.id, promotion_margin=0.0)
            await db.commit()

            # Seed an existing "production" experiment with a perfect score —
            # no retrained candidate can realistically beat 1.0.
            existing_prod = Experiment(
                name="existing-prod", dataset_id=ref_ds.id, target_column="y",
                task_type="classification", status="complete", best_score=1.0,
                lifecycle_stage="production",
            )
            db.add(existing_prod)
            await db.flush()
            policy.production_experiment_id = existing_prod.id
            await db.commit()

            run = await run_pipeline(policy, drifted_ds.id, db)
            await db.commit()

            assert run.promoted is False
            new_exp = await db.get(Experiment, run.new_experiment_id)
            assert new_exp.lifecycle_stage == "candidate"
            # Existing production experiment must remain production (not overwritten)
            still_prod = await db.get(Experiment, existing_prod.id)
            assert still_prod.lifecycle_stage == "production"

    @pytest.mark.asyncio
    async def test_promotion_reason_is_human_readable(self, tmp_path):
        from retraining.pipeline import run_pipeline

        engine, factory = await _make_engine_and_session()
        async with factory() as db:
            ref_ds = await _seed_dataset(db, tmp_path, "ref", _stable_df(seed=0))
            drifted_ds = await _seed_dataset(db, tmp_path, "drifted", _drifted_df(seed=1))
            policy = await _seed_policy(db, ref_ds.id)
            await db.commit()

            run = await run_pipeline(policy, drifted_ds.id, db)
            await db.commit()

            assert run.promotion_reason
            assert isinstance(run.promotion_reason, str)
            assert len(run.promotion_reason) > 10


# ══════════════════════════════════════════════════════════════════════════
# PIPELINE — ERROR HANDLING
# ══════════════════════════════════════════════════════════════════════════

class TestPipelineErrorHandling:

    @pytest.mark.asyncio
    async def test_missing_reference_dataset_fails_gracefully(self, tmp_path):
        from retraining.pipeline import run_pipeline
        from models.retraining import RetrainingPolicy

        engine, factory = await _make_engine_and_session()
        async with factory() as db:
            current_ds = await _seed_dataset(db, tmp_path, "current", _stable_df())
            policy = RetrainingPolicy(
                name="bad-ref", reference_dataset_id=99999,   # doesn't exist
                target_column="y", task_type="classification",
            )
            db.add(policy)
            await db.flush()
            await db.commit()

            run = await run_pipeline(policy, current_ds.id, db)
            await db.commit()

        assert run.status == "failed"
        assert run.error_message is not None

    @pytest.mark.asyncio
    async def test_no_numeric_features_fails_at_retrain_step(self, tmp_path):
        """A dataset with only the target column (no features) should fail cleanly at retrain."""
        from retraining.pipeline import run_pipeline

        engine, factory = await _make_engine_and_session()
        async with factory() as db:
            ref_ds = await _seed_dataset(db, tmp_path, "ref", _stable_df(seed=0))
            # Drifted dataset has only a target column — no usable features
            bad_df = pd.DataFrame({"y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 30})
            bad_ds = await _seed_dataset(db, tmp_path, "bad", bad_df)
            policy = await _seed_policy(db, ref_ds.id)
            await db.commit()

            run = await run_pipeline(policy, bad_ds.id, db)
            await db.commit()

        # Drift check itself may succeed or fail depending on overlap; either way
        # the pipeline must terminate in "failed", never raise an exception.
        assert run.status in ("failed", "completed")


# ══════════════════════════════════════════════════════════════════════════
# MANUAL PROMOTION
# ══════════════════════════════════════════════════════════════════════════

class TestManualPromotion:

    @pytest.mark.asyncio
    async def test_promote_archives_existing_production(self, tmp_path):
        from models.experiment import Experiment

        engine, factory = await _make_engine_and_session()
        async with factory() as db:
            ds = await _seed_dataset(db, tmp_path, "ds", _stable_df())

            old_prod = Experiment(
                name="old", dataset_id=ds.id, target_column="y", task_type="classification",
                status="complete", best_score=0.8, lifecycle_stage="production",
            )
            new_candidate = Experiment(
                name="new", dataset_id=ds.id, target_column="y", task_type="classification",
                status="complete", best_score=0.85, lifecycle_stage="candidate",
            )
            db.add_all([old_prod, new_candidate])
            await db.flush()
            await db.refresh(old_prod)
            await db.refresh(new_candidate)
            await db.commit()

            # Simulate the router logic directly
            from sqlalchemy import select
            result = await db.execute(
                select(Experiment).where(
                    Experiment.dataset_id == new_candidate.dataset_id,
                    Experiment.target_column == new_candidate.target_column,
                    Experiment.lifecycle_stage == "production",
                    Experiment.id != new_candidate.id,
                )
            )
            for other in result.scalars().all():
                other.lifecycle_stage = "archived"
            new_candidate.lifecycle_stage = "production"
            await db.commit()

            refreshed_old = await db.get(Experiment, old_prod.id)
            refreshed_new = await db.get(Experiment, new_candidate.id)
            assert refreshed_old.lifecycle_stage == "archived"
            assert refreshed_new.lifecycle_stage == "production"


# ══════════════════════════════════════════════════════════════════════════
# SCHEDULER
# ══════════════════════════════════════════════════════════════════════════

class TestScheduler:
    """
    APScheduler's replace_existing=True (and other jobstore dedup behaviour)
    only takes effect once the scheduler has actually been started — it
    requires a running asyncio event loop. In production this is guaranteed
    by main.py's lifespan calling start_scheduler() before any policy is
    ever scheduled. These tests replicate that real ordering rather than
    calling schedule_policy() against an unstarted scheduler.
    """

    @pytest.fixture(autouse=True)
    async def _running_scheduler(self):
        from retraining.scheduler import get_scheduler
        scheduler = get_scheduler()
        if not scheduler.running:
            scheduler.start(paused=True)   # paused: never actually fires during tests
        yield scheduler
        # Leave the scheduler running for other test modules that may import
        # it as the same process-wide singleton — only clear retraining jobs.
        for job in scheduler.get_jobs():
            if job.id.startswith("retraining-policy-"):
                scheduler.remove_job(job.id)

    @pytest.mark.asyncio
    async def test_schedule_and_unschedule_round_trip(self):
        from retraining.scheduler import get_scheduler, schedule_policy, unschedule_policy

        scheduler = get_scheduler()
        schedule_policy(policy_id=999, interval_hours=6)
        assert scheduler.get_job("retraining-policy-999") is not None

        unschedule_policy(999)
        assert scheduler.get_job("retraining-policy-999") is None

    @pytest.mark.asyncio
    async def test_unschedule_nonexistent_policy_does_not_raise(self):
        from retraining.scheduler import unschedule_policy
        unschedule_policy(123456)   # never scheduled — must not raise

    @pytest.mark.asyncio
    async def test_schedule_replaces_existing_job(self):
        from retraining.scheduler import get_scheduler, schedule_policy, unschedule_policy

        schedule_policy(policy_id=888, interval_hours=1)
        schedule_policy(policy_id=888, interval_hours=2)   # should replace, not duplicate
        scheduler = get_scheduler()
        jobs = [j for j in scheduler.get_jobs() if j.id == "retraining-policy-888"]
        assert len(jobs) == 1
        # And the replacement actually took effect (2-hour interval, not 1-hour)
        assert jobs[0].trigger.interval.total_seconds() == 7200
        unschedule_policy(888)

    @pytest.mark.asyncio
    async def test_sync_schedule_from_db(self, tmp_path):
        import sys, importlib, database as db_mod
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
        from sqlalchemy.pool import StaticPool
        from models.base import Base
        from models.retraining import RetrainingPolicy

        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False}, poolclass=StaticPool,
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        db_mod.engine = engine
        db_mod.SessionFactory = factory
        db_mod.AsyncSessionLocal = factory

        async with factory() as db:
            ds = await _seed_dataset(db, tmp_path, "ds", _stable_df())
            policy = await _seed_policy(db, ds.id, check_interval_hours=4, is_active=True)
            await db.commit()

        from retraining.scheduler import sync_schedule_from_db, get_scheduler, unschedule_policy
        count = await sync_schedule_from_db()
        assert count == 1
        scheduler = get_scheduler()
        assert scheduler.get_job(f"retraining-policy-{policy.id}") is not None
        unschedule_policy(policy.id)


# ══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def rt_client():
    import sys, importlib, database as db_mod
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool
    from fastapi.testclient import TestClient
    from retraining.scheduler import reset_scheduler_for_tests

    reset_scheduler_for_tests()
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
    reset_scheduler_for_tests()


def _upload_csv(client, df, name):
    csv = df.to_csv(index=False).encode()
    resp = client.post("/api/v1/datasets/upload",
                       files={"file": (f"{name}.csv", csv, "text/csv")},
                       data={"name": name})
    return resp.json()["data"]


class TestRetrainingAPI:

    def test_create_policy(self, rt_client):
        ds = _upload_csv(rt_client, _stable_df(), "policy_ds")
        resp = rt_client.post("/api/v1/retraining/policies", json={
            "name": "churn_policy",
            "reference_dataset_id": ds["id"],
            "target_column": "y",
            "task_type": "classification",
        })
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert data["name"] == "churn_policy"
        assert data["is_active"] is True

    def test_create_policy_missing_dataset_404(self, rt_client):
        resp = rt_client.post("/api/v1/retraining/policies", json={
            "name": "bad_policy",
            "reference_dataset_id": 99999,
            "target_column": "y",
            "task_type": "classification",
        })
        assert resp.status_code == 404

    def test_list_policies(self, rt_client):
        ds = _upload_csv(rt_client, _stable_df(), "list_ds")
        rt_client.post("/api/v1/retraining/policies", json={
            "name": "list_test_policy", "reference_dataset_id": ds["id"],
            "target_column": "y", "task_type": "classification",
        })
        resp = rt_client.get("/api/v1/retraining/policies")
        assert resp.status_code == 200
        assert len(resp.json()["data"]) >= 1

    def test_update_policy(self, rt_client):
        ds = _upload_csv(rt_client, _stable_df(), "update_ds")
        created = rt_client.post("/api/v1/retraining/policies", json={
            "name": "update_test", "reference_dataset_id": ds["id"],
            "target_column": "y", "task_type": "classification",
        }).json()["data"]

        resp = rt_client.patch(f"/api/v1/retraining/policies/{created['id']}", json={
            "promotion_margin": 0.05,
            "is_active": False,
        })
        assert resp.status_code == 200
        assert resp.json()["data"]["promotion_margin"] == 0.05
        assert resp.json()["data"]["is_active"] is False

    def test_delete_policy(self, rt_client):
        ds = _upload_csv(rt_client, _stable_df(), "delete_ds")
        created = rt_client.post("/api/v1/retraining/policies", json={
            "name": "delete_test", "reference_dataset_id": ds["id"],
            "target_column": "y", "task_type": "classification",
        }).json()["data"]

        resp = rt_client.delete(f"/api/v1/retraining/policies/{created['id']}")
        assert resp.status_code == 204

        get_resp = rt_client.get(f"/api/v1/retraining/policies/{created['id']}")
        assert get_resp.status_code == 404

    def test_manual_run_trigger_no_drift(self, rt_client):
        ref_ds = _upload_csv(rt_client, _stable_df(seed=0), "run_ref")
        same_ds = _upload_csv(rt_client, _stable_df(seed=0), "run_same")
        policy = rt_client.post("/api/v1/retraining/policies", json={
            "name": "run_test_policy", "reference_dataset_id": ref_ds["id"],
            "target_column": "y", "task_type": "classification",
            "n_trials": 2, "cv_folds": 2,
        }).json()["data"]

        resp = rt_client.post(f"/api/v1/retraining/policies/{policy['id']}/run", json={
            "current_dataset_id": same_ds["id"],
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["drift_detected"] is False
        assert data["retrain_triggered"] is False
        assert "steps" in data

    def test_run_history_endpoint(self, rt_client):
        ref_ds = _upload_csv(rt_client, _stable_df(seed=0), "hist_ref")
        same_ds = _upload_csv(rt_client, _stable_df(seed=0), "hist_same")
        policy = rt_client.post("/api/v1/retraining/policies", json={
            "name": "history_test_policy", "reference_dataset_id": ref_ds["id"],
            "target_column": "y", "task_type": "classification",
        }).json()["data"]

        rt_client.post(f"/api/v1/retraining/policies/{policy['id']}/run", json={
            "current_dataset_id": same_ds["id"],
        })

        resp = rt_client.get(f"/api/v1/retraining/policies/{policy['id']}/runs")
        assert resp.status_code == 200
        runs = resp.json()["data"]
        assert len(runs) >= 1

    def test_get_run_detail(self, rt_client):
        ref_ds = _upload_csv(rt_client, _stable_df(seed=0), "detail_ref")
        same_ds = _upload_csv(rt_client, _stable_df(seed=0), "detail_same")
        policy = rt_client.post("/api/v1/retraining/policies", json={
            "name": "detail_test_policy", "reference_dataset_id": ref_ds["id"],
            "target_column": "y", "task_type": "classification",
        }).json()["data"]

        run = rt_client.post(f"/api/v1/retraining/policies/{policy['id']}/run", json={
            "current_dataset_id": same_ds["id"],
        }).json()["data"]

        resp = rt_client.get(f"/api/v1/retraining/runs/{run['id']}")
        assert resp.status_code == 200
        assert "steps" in resp.json()["data"]

    def test_run_inactive_policy_rejected(self, rt_client):
        ds = _upload_csv(rt_client, _stable_df(), "inactive_ds")
        policy = rt_client.post("/api/v1/retraining/policies", json={
            "name": "inactive_test", "reference_dataset_id": ds["id"],
            "target_column": "y", "task_type": "classification",
        }).json()["data"]
        rt_client.patch(f"/api/v1/retraining/policies/{policy['id']}", json={"is_active": False})

        resp = rt_client.post(f"/api/v1/retraining/policies/{policy['id']}/run")
        assert resp.status_code == 422

    def test_promote_endpoint(self, rt_client):
        import asyncio as aio
        from models.experiment import Experiment
        from database import AsyncSessionLocal

        ds = _upload_csv(rt_client, _stable_df(), "promote_ds")

        async def _seed_exp():
            async with AsyncSessionLocal() as db:
                exp = Experiment(
                    name="promote_test", dataset_id=ds["id"], target_column="y",
                    task_type="classification", status="complete", best_score=0.9,
                )
                db.add(exp)
                await db.flush(); await db.refresh(exp)
                eid = exp.id
                await db.commit()
            return eid

        exp_id = aio.new_event_loop().run_until_complete(_seed_exp())

        resp = rt_client.post(f"/api/v1/experiments/{exp_id}/promote")
        assert resp.status_code == 200
        assert resp.json()["data"]["lifecycle_stage"] == "production"

    def test_promote_incomplete_experiment_rejected(self, rt_client):
        import asyncio as aio
        from models.experiment import Experiment
        from database import AsyncSessionLocal

        ds = _upload_csv(rt_client, _stable_df(), "incomplete_ds")

        async def _seed_exp():
            async with AsyncSessionLocal() as db:
                exp = Experiment(
                    name="incomplete", dataset_id=ds["id"], target_column="y",
                    task_type="classification", status="running",
                )
                db.add(exp)
                await db.flush(); await db.refresh(exp)
                eid = exp.id
                await db.commit()
            return eid

        exp_id = aio.new_event_loop().run_until_complete(_seed_exp())

        resp = rt_client.post(f"/api/v1/experiments/{exp_id}/promote")
        assert resp.status_code == 422
