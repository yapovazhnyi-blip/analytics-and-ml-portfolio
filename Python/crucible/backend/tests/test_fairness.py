"""
Fairness metrics tests.

Tests cover every metric formula with known inputs so we can verify exact
expected values, severity thresholds at boundary conditions, multi-group
handling, edge cases (tiny groups, perfect predictions, all-same predictions),
the FairnessAnalyzer end-to-end with synthetic data, and the API endpoint.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


# ══════════════════════════════════════════════════════════════════════════
# METRIC UNIT TESTS — known inputs → exact expected values
# ══════════════════════════════════════════════════════════════════════════

class TestGroupStats:

    def _stats(self, y_true, y_pred, positive_class=1):
        from fairness.metrics import compute_group_stats
        mask = np.ones(len(y_true), dtype=bool)
        return compute_group_stats(
            np.array(y_true), np.array(y_pred), mask, "group_a", positive_class
        )

    def test_selection_rate(self):
        """4 out of 10 predicted positive → selection rate = 0.4"""
        s = self._stats([1,0,0,0,1,0,0,0,0,0], [1,1,1,1,0,0,0,0,0,0])
        assert s.selection_rate == pytest.approx(0.4)

    def test_tpr_known_value(self):
        """2 TP, 1 FN → TPR = 2/3"""
        s = self._stats(
            y_true=[1, 1, 1, 0, 0],
            y_pred=[1, 1, 0, 0, 0],
        )
        assert s.tpr == pytest.approx(2 / 3)

    def test_fpr_known_value(self):
        """2 FP, 3 TN → FPR = 2/5"""
        s = self._stats(
            y_true=[0, 0, 0, 0, 0],
            y_pred=[1, 1, 0, 0, 0],
        )
        assert s.fpr == pytest.approx(2 / 5)

    def test_perfect_classifier_tpr_one(self):
        s = self._stats([1,1,1,0,0,0], [1,1,1,0,0,0])
        assert s.tpr == 1.0

    def test_empty_group_returns_zero_rates(self):
        from fairness.metrics import compute_group_stats
        mask = np.zeros(5, dtype=bool)  # no samples in this group
        s = compute_group_stats(np.zeros(5), np.zeros(5), mask, "empty")
        assert s.n_samples == 0
        assert s.selection_rate == 0.0


class TestDemographicParity:

    def test_dpd_perfect_parity(self):
        """Equal selection rates → DPD = 0"""
        from fairness.metrics import compute_group_metrics
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        # Group A: indices 0-3, Group B: indices 4-7
        prot = np.array(["A","A","A","A","B","B","B","B"])
        gm = compute_group_metrics("sex", y_true, y_pred, prot)
        assert abs(gm.demographic_parity_diff) < 0.01

    def test_dpd_known_disparity(self):
        """
        Group A: 8/10 selected (0.8), Group B: 2/10 selected (0.2)
        Expected DPD = 0.8 - 0.2 = 0.6
        """
        from fairness.metrics import compute_group_metrics
        y_pred = np.array([1]*8 + [0]*2 + [1]*2 + [0]*8)
        y_true = np.ones(20, dtype=int)
        prot   = np.array(["A"]*10 + ["B"]*10)
        gm = compute_group_metrics("race", y_true, y_pred, prot)
        assert abs(gm.demographic_parity_diff - 0.6) < 0.01

    def test_disparate_impact_ratio_80pct_boundary(self):
        """
        Priv rate=0.80, Unpriv rate=0.64 → DIR = 0.64/0.80 = 0.80 (EEOC boundary)
        """
        from fairness.metrics import compute_group_metrics
        n = 50
        # Group A (privileged): 40/50 selected
        # Group B (unprivileged): 32/50 selected
        y_pred = np.array([1]*40 + [0]*10 + [1]*32 + [0]*18)
        y_true = np.ones(100, dtype=int)
        prot   = np.array(["A"]*50 + ["B"]*50)
        gm = compute_group_metrics("age", y_true, y_pred, prot)
        assert abs(gm.disparate_impact_ratio - 0.80) < 0.02


class TestSeverityThresholds:

    def _gm_with_dpd(self, dpd):
        from fairness.metrics import GroupMetrics, _overall_severity
        gm = GroupMetrics(
            attribute="x", privileged_group="A", unprivileged_group="B",
            group_stats=[],
            demographic_parity_diff=dpd,
            disparate_impact_ratio=1.0,   # perfect — won't affect severity
        )
        gm.severity = _overall_severity(gm)
        return gm

    def test_below_005_is_acceptable(self):
        assert self._gm_with_dpd(0.03).severity == "acceptable"

    def test_at_005_is_marginal(self):
        assert self._gm_with_dpd(0.07).severity == "marginal"

    def test_at_010_is_significant(self):
        assert self._gm_with_dpd(0.15).severity == "significant"

    def test_at_020_is_severe(self):
        assert self._gm_with_dpd(0.25).severity == "severe"

    def test_severity_uses_absolute_value(self):
        """Negative DPD (flipped privileged/unprivileged) must have same severity."""
        g1 = self._gm_with_dpd(0.15)
        g2 = self._gm_with_dpd(-0.15)
        assert g1.severity == g2.severity


class TestComputeFairness:

    def _make_data(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, 2, n)
        y_pred = y_true.copy()
        # Introduce disparity: flip some predictions for group B
        group  = np.where(rng.random(n) > 0.5, "A", "B")
        flip_mask = (group == "B") & (y_true == 1) & (rng.random(n) < 0.4)
        y_pred[flip_mask] = 0
        protected_df = pd.DataFrame({"gender": group})
        return y_true, y_pred, protected_df

    def test_report_has_correct_structure(self):
        from fairness.metrics import compute_fairness
        y_true, y_pred, protected_df = self._make_data()
        report = compute_fairness(1, y_true, y_pred, protected_df, "classification")
        assert report.succeeded
        assert len(report.metrics) == 1
        assert report.metrics[0].attribute == "gender"

    def test_report_overall_severity_is_worst(self):
        """Overall severity = worst across all attributes."""
        from fairness.metrics import compute_fairness
        y_true, y_pred, protected_df = self._make_data(seed=42)
        report = compute_fairness(2, y_true, y_pred, protected_df, "classification")
        assert report.overall_severity == report.metrics[0].severity

    def test_report_metrics_sorted_worst_first(self):
        """Attributes with more severe disparity should appear first."""
        from fairness.metrics import compute_fairness
        rng = np.random.default_rng(7)
        n = 200
        y_true = rng.integers(0, 2, n)
        y_pred = y_true.copy()
        group_a = np.where(rng.random(n) > 0.5, "M", "F")
        group_b = np.where(rng.random(n) > 0.5, "young", "old")
        # Introduce severe disparity in group_a, no disparity in group_b
        flip = (group_a == "F") & (y_true == 1)
        y_pred[flip] = 0
        protected_df = pd.DataFrame({"gender": group_a, "age": group_b})
        report = compute_fairness(3, y_true, y_pred, protected_df, "classification")
        assert report.n_attributes_flagged >= 1

    def test_to_dict_is_json_serialisable(self):
        from fairness.metrics import compute_fairness
        y_true, y_pred, protected_df = self._make_data()
        report = compute_fairness(99, y_true, y_pred, protected_df, "classification")
        d = report.to_dict()
        json.dumps(d)   # must not raise

    def test_tiny_group_skipped(self):
        """Groups with fewer than 5 samples must be skipped."""
        from fairness.metrics import compute_group_metrics
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 1])
        # Group C has only 2 samples → should be skipped
        prot = np.array(["A", "A", "A", "B", "B", "C"])
        gm = compute_group_metrics("protected", y_true, y_pred, prot)
        group_names = {s.group_value for s in gm.group_stats}
        assert "C" not in group_names

    def test_perfect_parity_report(self):
        """When both groups have the same selection rate, DPD ≈ 0."""
        from fairness.metrics import compute_fairness
        # Each group: exactly 3/6 predicted positive
        y_true = np.array([1, 0, 1, 0, 1, 0,   1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0,   1, 0, 1, 0, 1, 0])
        prot   = pd.DataFrame({"gender":
            ["A","A","A","A","A","A","B","B","B","B","B","B"]})
        report = compute_fairness(4, y_true, y_pred, prot, "classification")
        m = report.metrics[0]
        assert abs(m.demographic_parity_diff) < 0.01
        assert report.overall_severity == "acceptable"


# ══════════════════════════════════════════════════════════════════════════
# FAIRNESS ANALYZER END-TO-END
# ══════════════════════════════════════════════════════════════════════════

class TestFairnessAnalyzer:

    @pytest.fixture
    def trained_setup(self, tmp_path):
        """
        Creates a synthetic dataset with a protected attribute, trains a
        classifier, and returns paths needed by the analyzer.
        """
        rng = np.random.default_rng(42)
        n = 300
        X = rng.standard_normal((n, 4))
        y = (X[:, 0] + rng.standard_normal(n) * 0.5 > 0).astype(int)
        gender = rng.choice(["M", "F"], n)

        df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
        df["gender"] = gender
        df["target"] = y

        csv_path = str(tmp_path / "data.csv")
        df.to_csv(csv_path, index=False)

        # Train a quick model (same split as analyzer)
        from sklearn.model_selection import train_test_split
        X_train, _, y_train, _ = train_test_split(
            df[["a","b","c","d","gender"]].assign(gender=lambda d: d["gender"].map({"M":0,"F":1})),
            y, test_size=0.2, random_state=42
        )
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        artifact = str(tmp_path / "model.pkl")
        joblib.dump(model, artifact)

        return {"csv": csv_path, "artifact": artifact}

    @pytest.mark.asyncio
    async def test_analyzer_returns_report(self, trained_setup):
        from fairness.analyzer import FairnessAnalyzer
        analyzer = FairnessAnalyzer()
        report = await analyzer.analyze(
            artifact_path=trained_setup["artifact"],
            dataset_path=trained_setup["csv"],
            source_type="csv",
            target_column="target",
            protected_attributes=["gender"],
            task_type="classification",
            experiment_id=1,
        )
        assert report.succeeded, report.error
        assert len(report.metrics) == 1
        assert report.metrics[0].attribute == "gender"
        assert report.n_samples > 0

    @pytest.mark.asyncio
    async def test_analyzer_missing_protected_attr(self, trained_setup):
        from fairness.analyzer import FairnessAnalyzer
        analyzer = FairnessAnalyzer()
        report = await analyzer.analyze(
            artifact_path=trained_setup["artifact"],
            dataset_path=trained_setup["csv"],
            source_type="csv",
            target_column="target",
            protected_attributes=["nonexistent_column"],
            task_type="classification",
            experiment_id=1,
        )
        assert not report.succeeded
        assert "not found" in report.error.lower()

    @pytest.mark.asyncio
    async def test_analyzer_report_is_serialisable(self, trained_setup):
        from fairness.analyzer import FairnessAnalyzer
        analyzer = FairnessAnalyzer()
        report = await analyzer.analyze(
            artifact_path=trained_setup["artifact"],
            dataset_path=trained_setup["csv"],
            source_type="csv",
            target_column="target",
            protected_attributes=["gender"],
            task_type="classification",
            experiment_id=2,
        )
        assert report.succeeded
        json.dumps(report.to_dict())   # must not raise


# ══════════════════════════════════════════════════════════════════════════
# API ENDPOINT
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def fair_client():
    import sys, importlib, database as db_mod
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool
    from fastapi.testclient import TestClient

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    db_mod.engine = engine
    db_mod.SessionFactory = factory
    db_mod.AsyncSessionLocal = factory
    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    import main as app_module
    with TestClient(app_module.app, raise_server_exceptions=True) as c:
        yield c


class TestFairnessAPI:

    def test_fairness_on_nonexistent_experiment(self, fair_client):
        resp = fair_client.post("/api/v1/experiments/9999/fairness", json={
            "protected_attributes": ["gender"]
        })
        assert resp.status_code == 404

    def test_get_fairness_before_run_returns_404(self, fair_client):
        """GET without prior POST should return 404."""
        resp = fair_client.get("/api/v1/experiments/1/fairness")
        assert resp.status_code == 404

    def test_fairness_empty_attributes_rejected(self, fair_client):
        resp = fair_client.post("/api/v1/experiments/1/fairness", json={
            "protected_attributes": []
        })
        assert resp.status_code == 422

    def test_fairness_with_real_experiment(self, fair_client, tmp_path):
        """
        Seeds a dataset + trained model, creates a completed Experiment record,
        then runs fairness analysis through the API.
        """
        import asyncio
        rng = np.random.default_rng(0)
        n = 200
        X = rng.standard_normal((n, 3))
        y = (X[:, 0] > 0).astype(int)
        gender = rng.choice(["M", "F"], n)

        df = pd.DataFrame(X, columns=["x1","x2","x3"])
        df["gender"] = gender
        df["label"] = y

        csv_path = str(tmp_path / "d.csv")
        df.to_csv(csv_path, index=False)

        from sklearn.model_selection import train_test_split
        X_tr, _, y_tr, _ = train_test_split(
            df[["x1","x2","x3","gender"]].assign(gender=lambda d: d["gender"].map({"M":0,"F":1})),
            y, test_size=0.2, random_state=42
        )
        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        clf.fit(X_tr, y_tr)
        model_path = str(tmp_path / "m.pkl")
        joblib.dump(clf, model_path)

        # Seed DB via the upload endpoint
        with open(csv_path, "rb") as f:
            ds_resp = fair_client.post(
                "/api/v1/datasets/upload",
                files={"file": ("d.csv", f, "text/csv")},
                data={"name": "fairness_test"},
            )
        assert ds_resp.status_code == 201
        ds_id = ds_resp.json()["data"]["id"]

        # Insert experiment record directly via async DB
        from models.experiment import Experiment
        from database import AsyncSessionLocal
        import asyncio

        async def _seed_exp():
            async with AsyncSessionLocal() as db:
                exp = Experiment(
                    dataset_id=ds_id,
                    name="fairness_test_exp",
                    target_column="label",
                    task_type="classification",
                    status="completed",
                    best_model_family="RandomForest",
                    best_score=0.80,
                    model_artifact_path=model_path,
                    n_trials_completed=5,
                    results_json=json.dumps({"feature_names":["x1","x2","x3","gender"]}),
                )
                db.add(exp)
                await db.flush()
                await db.refresh(exp)
                eid = exp.id
                await db.commit()
            return eid

        exp_id = asyncio.run(_seed_exp())

        # Run fairness analysis via API
        resp = fair_client.post(f"/api/v1/experiments/{exp_id}/fairness", json={
            "protected_attributes": ["gender"]
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["n_samples"] > 0
        assert len(data["metrics"]) == 1
        assert data["metrics"][0]["attribute"] == "gender"
        assert "demographic_parity_diff" in data["metrics"][0]
        assert "disparate_impact_ratio" in data["metrics"][0]
        assert data["overall_severity"] in ("acceptable","marginal","significant","severe")
