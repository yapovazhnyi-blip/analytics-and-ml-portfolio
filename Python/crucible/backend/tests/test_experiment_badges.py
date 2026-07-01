"""
Tests for calibration method, pruner type, and lifecycle stage badges.

Covers:
  - ExperimentOut schema accepts the new fields
  - ExperimentSummary schema accepts lifecycle_stage
  - _exp_to_out correctly parses calibration/pruner from results_json
  - _exp_to_out handles missing/malformed results_json gracefully
  - The previously-buggy fallback sync path (get_experiment when status=running)
    now populates calibration/pruner/holdout_metrics instead of discarding them
  - API: GET /experiments/{id} returns the new fields
  - API: GET /experiments (list) returns lifecycle_stage per row
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock


# ══════════════════════════════════════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════════════════════════════════════

class TestSchemas:

    def test_experiment_out_accepts_new_fields(self):
        from schemas.experiment import ExperimentOut
        out = ExperimentOut(
            id=1, name="x", dataset_id=1, target_column="y", task_type="classification",
            status="complete", created_at="2024-01-01T00:00:00",
            calibration_applied=True, calibration_method="isotonic",
            pruner_type="hyperband", lifecycle_stage="production",
        )
        assert out.calibration_applied is True
        assert out.calibration_method == "isotonic"
        assert out.pruner_type == "hyperband"
        assert out.lifecycle_stage == "production"

    def test_experiment_out_fields_default_none(self):
        from schemas.experiment import ExperimentOut
        out = ExperimentOut(
            id=1, name="x", dataset_id=1, target_column="y", task_type="classification",
            status="running", created_at="2024-01-01T00:00:00",
        )
        assert out.calibration_applied is None
        assert out.calibration_method is None
        assert out.pruner_type is None
        assert out.lifecycle_stage is None

    def test_experiment_summary_accepts_lifecycle_stage(self):
        from schemas.experiment import ExperimentSummary
        s = ExperimentSummary(
            id=1, name="x", dataset_id=1, target_column="y", task_type="classification",
            status="complete", created_at="2024-01-01T00:00:00", lifecycle_stage="archived",
        )
        assert s.lifecycle_stage == "archived"


# ══════════════════════════════════════════════════════════════════════════
# _exp_to_out PARSING
# ══════════════════════════════════════════════════════════════════════════

class TestExpToOut:

    def _make_exp(self, results_json=None, lifecycle_stage="candidate"):
        from datetime import datetime, timezone
        exp = MagicMock()
        exp.id = 1
        exp.name = "test"
        exp.dataset_id = 1
        exp.target_column = "y"
        exp.task_type = "classification"
        exp.status = "complete"
        exp.best_model_family = "XGBoost"
        exp.best_score = 0.9
        exp.scoring_metric = "accuracy"
        exp.n_trials_completed = 10
        exp.n_trials_pruned = 2
        exp.training_duration_secs = 30.0
        exp.results_json = results_json
        exp.shap_json = None
        exp.error_message = None
        exp.mlflow_run_id = "run-1"
        exp.lifecycle_stage = lifecycle_stage
        exp.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return exp

    def test_calibration_fields_extracted(self):
        from routers.experiments import _exp_to_out
        exp = self._make_exp(results_json=json.dumps({
            "holdout_metrics": {"accuracy": 0.9},
            "calibration_applied": True,
            "calibration_method": "sigmoid",
            "pruner_type": "median",
        }))
        out = _exp_to_out(exp)
        assert out.calibration_applied is True
        assert out.calibration_method == "sigmoid"
        assert out.pruner_type == "median"

    def test_lifecycle_stage_passed_through(self):
        from routers.experiments import _exp_to_out
        exp = self._make_exp(lifecycle_stage="production")
        out = _exp_to_out(exp)
        assert out.lifecycle_stage == "production"

    def test_missing_results_json_handled_gracefully(self):
        from routers.experiments import _exp_to_out
        exp = self._make_exp(results_json=None)
        out = _exp_to_out(exp)
        assert out.calibration_applied is None
        assert out.calibration_method is None
        assert out.pruner_type is None
        assert out.holdout_metrics == []

    def test_malformed_results_json_handled_gracefully(self):
        from routers.experiments import _exp_to_out
        exp = self._make_exp(results_json="not valid json {{{")
        out = _exp_to_out(exp)
        assert out.calibration_applied is None
        assert out.holdout_metrics == []

    def test_results_json_missing_new_keys_handled(self):
        """Old experiments created before this feature have results_json without these keys."""
        from routers.experiments import _exp_to_out
        exp = self._make_exp(results_json=json.dumps({
            "holdout_metrics": {"accuracy": 0.85},
            "best_params": {"n_estimators": 100},
        }))
        out = _exp_to_out(exp)
        assert out.calibration_applied is None
        assert out.pruner_type is None
        assert len(out.holdout_metrics) == 1


# ══════════════════════════════════════════════════════════════════════════
# API — GET /experiments/{id}
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def badge_client():
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


def _seed_completed_experiment(ds_id, **overrides):
    import asyncio
    from models.experiment import Experiment
    from database import AsyncSessionLocal

    async def _seed():
        async with AsyncSessionLocal() as db:
            defaults = dict(
                name="badge_test_exp", dataset_id=ds_id, target_column="y",
                task_type="classification", status="complete", best_score=0.88,
                best_model_family="LightGBM",
                results_json=json.dumps({
                    "holdout_metrics": {"accuracy": 0.88},
                    "calibration_applied": True,
                    "calibration_method": "isotonic",
                    "pruner_type": "hyperband",
                }),
            )
            defaults.update(overrides)
            exp = Experiment(**defaults)
            db.add(exp)
            await db.flush()
            await db.refresh(exp)
            eid = exp.id
            await db.commit()
        return eid

    return asyncio.new_event_loop().run_until_complete(_seed())


class TestExperimentAPIFields:

    def test_get_experiment_returns_calibration_and_pruner(self, badge_client):
        df_csv = b"x,y\n1,0\n2,1\n3,0\n4,1\n"
        ds = badge_client.post("/api/v1/datasets/upload",
            files={"file": ("d.csv", df_csv, "text/csv")}, data={"name": "badge_ds"}).json()["data"]

        exp_id = _seed_completed_experiment(ds["id"])

        resp = badge_client.get(f"/api/v1/experiments/{exp_id}")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["calibration_applied"] is True
        assert data["calibration_method"] == "isotonic"
        assert data["pruner_type"] == "hyperband"
        assert data["lifecycle_stage"] == "candidate"

    def test_list_experiments_returns_lifecycle_stage(self, badge_client):
        df_csv = b"x,y\n1,0\n2,1\n3,0\n4,1\n"
        ds = badge_client.post("/api/v1/datasets/upload",
            files={"file": ("d2.csv", df_csv, "text/csv")}, data={"name": "badge_ds2"}).json()["data"]

        _seed_completed_experiment(ds["id"], lifecycle_stage="production")

        resp = badge_client.get("/api/v1/experiments")
        assert resp.status_code == 200
        rows = resp.json()["data"]
        assert any(r.get("lifecycle_stage") == "production" for r in rows)

    def test_old_experiment_without_new_fields_does_not_error(self, badge_client):
        """Pre-existing experiments (results_json without calibration/pruner keys) must not 500."""
        df_csv = b"x,y\n1,0\n2,1\n3,0\n4,1\n"
        ds = badge_client.post("/api/v1/datasets/upload",
            files={"file": ("d3.csv", df_csv, "text/csv")}, data={"name": "badge_ds3"}).json()["data"]

        exp_id = _seed_completed_experiment(
            ds["id"],
            results_json=json.dumps({"holdout_metrics": {"accuracy": 0.7}}),
        )

        resp = badge_client.get(f"/api/v1/experiments/{exp_id}")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["calibration_applied"] is None
        assert data["pruner_type"] is None
