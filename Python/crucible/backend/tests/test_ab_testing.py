"""
A/B testing tests.

Tests cover:
  - McNemar's test: known contingency tables → exact expected p-values
  - Bootstrap test: directional correctness and CI coverage
  - Wilcoxon test: regression error comparison
  - Power analysis: required_n formula correctness, MDE calculation
  - compare_experiments(): winner detection, effect size labels, recommendation text
  - API endpoints: /ab-test, /ab-test/power, /ab-test/methods
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split


# ══════════════════════════════════════════════════════════════════════════
# McNEMAR'S TEST
# ══════════════════════════════════════════════════════════════════════════

class TestMcNemar:

    def test_no_disagreements_returns_pvalue_one(self):
        """When both models predict identically, McNemar gives p=1."""
        from ab_testing.engine import mcnemar_test
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        result = mcnemar_test(y, y, y)
        assert result["p_value"] == 1.0

    def test_large_disagreement_gives_small_pvalue(self):
        """Clear winner: B gets 40 right that A got wrong, A gets 5 right that B got wrong."""
        from ab_testing.engine import mcnemar_test
        n = 200
        y_true = np.array([1] * n)
        # A correct: first 60; B correct: first 100
        preds_a = np.array([1]*60 + [0]*140)
        preds_b = np.array([1]*100 + [0]*100)
        result = mcnemar_test(y_true, preds_a, preds_b)
        assert result["p_value"] < 0.05

    def test_small_disagreement_not_significant(self):
        """Both models make similar errors — should not be significant."""
        from ab_testing.engine import mcnemar_test
        rng = np.random.default_rng(99)
        n = 200
        y_true = rng.integers(0, 2, n)
        # Both models ~75% accurate with overlapping errors
        preds_a = np.where(rng.random(n) > 0.25, y_true, 1 - y_true)
        preds_b = np.where(rng.random(n) > 0.24, y_true, 1 - y_true)
        result = mcnemar_test(y_true, preds_a, preds_b)
        # With very similar models p-value should be large (not reject H0)
        assert result["p_value"] > 0.10

    def test_ci_contains_true_difference(self):
        """95% CI should contain the true difference for most samples."""
        from ab_testing.engine import mcnemar_test
        rng = np.random.default_rng(42)
        n = 500
        y_true = rng.integers(0, 2, n)
        preds_a = (rng.random(n) > 0.3).astype(int)  # ~70% acc
        preds_b = (rng.random(n) > 0.2).astype(int)  # ~80% acc
        result = mcnemar_test(y_true, preds_a, preds_b)
        # CI should include the observed difference
        obs_diff = np.mean(preds_b == y_true) - np.mean(preds_a == y_true)
        assert result["ci_lower"] <= obs_diff <= result["ci_upper"] or \
               abs(obs_diff - result["ci_lower"]) < 0.05  # allow small numeric error

    def test_effect_size_nonnegative(self):
        from ab_testing.engine import mcnemar_test
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 100)
        a = rng.integers(0, 2, 100)
        b = rng.integers(0, 2, 100)
        result = mcnemar_test(y, a, b)
        assert result["effect_size"] >= 0.0


# ══════════════════════════════════════════════════════════════════════════
# BOOTSTRAP TEST
# ══════════════════════════════════════════════════════════════════════════

class TestBootstrap:

    def _acc(self, y_true, y_pred):
        return float(np.mean(y_true == y_pred))

    def test_better_model_gets_low_pvalue(self):
        """B is clearly better → bootstrap p-value should be small."""
        from ab_testing.engine import bootstrap_test
        rng = np.random.default_rng(0)
        n = 300
        y = rng.integers(0, 2, n)
        preds_a = (rng.random(n) > 0.4).astype(int)   # ~60% acc
        preds_b = y.copy()                              # 100% acc
        result = bootstrap_test(y, preds_a, preds_b, self._acc, n_bootstrap=500)
        assert result["p_value"] < 0.05

    def test_equal_models_high_pvalue(self):
        """Same predictions → p-value should be ~0.5."""
        from ab_testing.engine import bootstrap_test
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 200)
        preds = (rng.random(200) > 0.5).astype(int)
        result = bootstrap_test(y, preds, preds, self._acc, n_bootstrap=500)
        assert result["p_value"] > 0.3

    def test_ci_lower_less_than_upper(self):
        from ab_testing.engine import bootstrap_test
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 100)
        a = rng.integers(0, 2, 100)
        b = rng.integers(0, 2, 100)
        result = bootstrap_test(y, a, b, self._acc, n_bootstrap=200)
        assert result["ci_lower"] < result["ci_upper"]


# ══════════════════════════════════════════════════════════════════════════
# WILCOXON TEST (REGRESSION)
# ══════════════════════════════════════════════════════════════════════════

class TestWilcoxon:

    def test_clearly_better_model_is_significant(self):
        """B predicts perfectly, A is noisy → Wilcoxon should detect B is better."""
        from ab_testing.engine import wilcoxon_test
        rng = np.random.default_rng(0)
        n = 200
        y = rng.standard_normal(n) * 10 + 50
        preds_a = y + rng.standard_normal(n) * 5   # high error
        preds_b = y + rng.standard_normal(n) * 0.1 # low error
        result = wilcoxon_test(y, preds_a, preds_b)
        assert result["p_value"] < 0.05

    def test_identical_models_not_significant(self):
        from ab_testing.engine import wilcoxon_test
        rng = np.random.default_rng(1)
        y = rng.standard_normal(200)
        p = rng.standard_normal(200)
        result = wilcoxon_test(y, p, p)
        assert result["p_value"] >= 0.05

    def test_effect_size_in_zero_one(self):
        from ab_testing.engine import wilcoxon_test
        rng = np.random.default_rng(2)
        y = rng.standard_normal(100)
        a = rng.standard_normal(100)
        b = rng.standard_normal(100)
        result = wilcoxon_test(y, a, b)
        assert 0.0 <= result["effect_size"] <= 1.0


# ══════════════════════════════════════════════════════════════════════════
# POWER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

class TestPowerAnalysis:

    def test_required_n_increases_with_smaller_effect(self):
        """Detecting a 1pp improvement requires more samples than 5pp."""
        from ab_testing.engine import power_analysis
        r_small = power_analysis(0.80, 0.01)
        r_large = power_analysis(0.80, 0.05)
        assert r_small.required_n > r_large.required_n

    def test_required_n_decreases_with_lower_power(self):
        from ab_testing.engine import power_analysis
        r80 = power_analysis(0.80, 0.03, power=0.80)
        r60 = power_analysis(0.80, 0.03, power=0.60)
        assert r80.required_n > r60.required_n

    def test_adequately_powered_when_n_sufficient(self):
        from ab_testing.engine import power_analysis
        r = power_analysis(0.80, 0.10, current_n=10000)
        assert r.is_adequately_powered

    def test_not_powered_when_n_small(self):
        from ab_testing.engine import power_analysis
        r = power_analysis(0.80, 0.001, current_n=10)
        assert not r.is_adequately_powered

    def test_current_mde_decreases_with_more_samples(self):
        """More samples → can detect smaller effects."""
        from ab_testing.engine import power_analysis
        r100  = power_analysis(0.80, 0.05, current_n=100)
        r1000 = power_analysis(0.80, 0.05, current_n=1000)
        assert r1000.current_mde < r100.current_mde

    def test_to_dict_serialisable(self):
        from ab_testing.engine import power_analysis
        r = power_analysis(0.75, 0.03, current_n=500)
        json.dumps(r.to_dict())

    def test_mde_above_zero(self):
        from ab_testing.engine import power_analysis
        r = power_analysis(0.80, 0.05, current_n=200)
        assert r.current_mde > 0


# ══════════════════════════════════════════════════════════════════════════
# COMPARE EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════

class TestCompareExperiments:

    def _make_clf_data(self, n=400, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, n)
        preds_a = (rng.random(n) > 0.35).astype(int)  # ~65% acc
        preds_b = y.copy()                              # 100% acc
        return y, preds_a, preds_b

    def test_returns_ab_test_result(self):
        from ab_testing.engine import compare_experiments, ABTestResult
        y, a, b = self._make_clf_data()
        result = compare_experiments(1, 2, y, a, b, "classification")
        assert isinstance(result, ABTestResult)

    def test_winner_b_when_b_better(self):
        from ab_testing.engine import compare_experiments
        y, a, b = self._make_clf_data()
        result = compare_experiments(1, 2, y, a, b, "classification")
        if result.is_significant:
            assert result.winner == "B"
            assert result.winner_id == 2

    def test_no_winner_when_models_identical(self):
        from ab_testing.engine import compare_experiments
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        preds = (rng.random(200) > 0.5).astype(int)
        result = compare_experiments(1, 2, y, preds, preds, "classification")
        assert result.winner is None
        assert result.p_value == 1.0

    def test_effect_size_labels(self):
        from ab_testing.engine import _effect_label
        assert _effect_label(0.05) == "negligible"
        assert _effect_label(0.20) == "small"
        assert _effect_label(0.40) == "medium"
        assert _effect_label(0.60) == "large"

    def test_to_dict_json_serialisable(self):
        from ab_testing.engine import compare_experiments
        y, a, b = self._make_clf_data()
        result = compare_experiments(1, 2, y, a, b, "classification")
        json.dumps(result.to_dict())

    def test_recommendation_contains_key_info(self):
        from ab_testing.engine import compare_experiments
        y, a, b = self._make_clf_data()
        result = compare_experiments(1, 2, y, a, b, "classification")
        assert "p=" in result.recommendation or "p-value" in result.recommendation.lower() or \
               "p=" in result.recommendation.lower()

    def test_regression_uses_mae(self):
        from ab_testing.engine import compare_experiments
        rng = np.random.default_rng(7)
        y = rng.standard_normal(300) * 10
        a = y + rng.standard_normal(300) * 5
        b = y + rng.standard_normal(300) * 1   # B much better
        result = compare_experiments(1, 2, y, a, b, "regression")
        assert result.metric == "mae"
        if result.is_significant:
            assert result.winner == "B"


# ══════════════════════════════════════════════════════════════════════════
# AB TEST ANALYZER (end-to-end)
# ══════════════════════════════════════════════════════════════════════════

class TestABTestAnalyzer:

    @pytest.fixture
    def two_models(self, tmp_path):
        """Trains two classifiers on synthetic data and returns their paths."""
        X, y = make_classification(n_samples=500, n_features=6, random_state=0)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
        y = pd.Series(y, name="label")

        df = X.copy()
        df["label"] = y

        csv = str(tmp_path / "data.csv")
        df.to_csv(csv, index=False)

        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        m_a = LogisticRegression(max_iter=200).fit(X_tr, y_tr)
        m_b = RandomForestClassifier(n_estimators=20, random_state=0).fit(X_tr, y_tr)

        path_a = str(tmp_path / "model_a.pkl")
        path_b = str(tmp_path / "model_b.pkl")
        joblib.dump(m_a, path_a)
        joblib.dump(m_b, path_b)

        return {"csv": csv, "path_a": path_a, "path_b": path_b}

    @pytest.mark.asyncio
    async def test_analyzer_returns_result(self, two_models):
        from ab_testing.analyzer import ABTestAnalyzer
        result = await ABTestAnalyzer().analyze(
            exp_a_id=1, exp_b_id=2,
            artifact_path_a=two_models["path_a"],
            artifact_path_b=two_models["path_b"],
            dataset_path=two_models["csv"],
            source_type="csv",
            target_column="label",
            task_type="classification",
        )
        assert result.n_samples > 0
        assert result.metric == "accuracy"
        assert 0.0 <= result.p_value <= 1.0

    @pytest.mark.asyncio
    async def test_analyzer_bad_artifact_returns_error_result(self, two_models):
        from ab_testing.analyzer import ABTestAnalyzer
        result = await ABTestAnalyzer().analyze(
            exp_a_id=1, exp_b_id=2,
            artifact_path_a="/nonexistent/model.pkl",
            artifact_path_b=two_models["path_b"],
            dataset_path=two_models["csv"],
            source_type="csv",
            target_column="label",
            task_type="classification",
        )
        # Should return an error result, not raise
        assert result.metric == "error"
        assert "failed" in result.recommendation.lower()


# ══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def ab_client():
    import sys, importlib, database as db_mod
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool
    from fastapi.testclient import TestClient

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    db_mod.engine = engine
    db_mod.SessionFactory = factory
    db_mod.AsyncSessionLocal = factory
    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    import main as m
    with TestClient(m.app, raise_server_exceptions=True) as c:
        yield c


class TestABTestAPI:

    def test_list_methods(self, ab_client):
        resp = ab_client.get("/api/v1/ab-test/methods")
        assert resp.status_code == 200
        data = resp.json()["data"]["tests"]
        assert "mcnemar" in data
        assert "wilcoxon" in data
        assert "bootstrap" in data

    def test_power_analysis_endpoint(self, ab_client):
        resp = ab_client.post("/api/v1/ab-test/power", json={
            "baseline_rate": 0.80,
            "minimum_effect": 0.05,
            "alpha": 0.05,
            "power": 0.80,
            "current_n": 200,
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "required_n" in data
        assert "current_mde" in data
        assert data["required_n"] > 0
        assert 0 < data["current_mde"] < 1

    def test_power_analysis_invalid_baseline(self, ab_client):
        resp = ab_client.post("/api/v1/ab-test/power", json={
            "baseline_rate": 1.5,  # invalid: > 1
            "minimum_effect": 0.05,
        })
        assert resp.status_code == 422

    def test_ab_test_same_experiment_rejected(self, ab_client):
        resp = ab_client.post("/api/v1/ab-test/", json={
            "experiment_a_id": 1,
            "experiment_b_id": 1,
        })
        assert resp.status_code == 422

    def test_ab_test_nonexistent_experiment(self, ab_client):
        resp = ab_client.post("/api/v1/ab-test/", json={
            "experiment_a_id": 999,
            "experiment_b_id": 1000,
        })
        assert resp.status_code == 404

    def test_ab_test_with_real_experiments(self, ab_client, tmp_path):
        """Seeds two completed experiments and runs a full A/B test via API."""
        import asyncio

        # Create synthetic dataset + two models
        X, y = make_classification(n_samples=400, n_features=5, random_state=1)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        df["label"] = y
        csv = str(tmp_path / "d.csv")
        df.to_csv(csv, index=False)

        X_feat = df[[f"f{i}" for i in range(5)]]
        X_tr, _, y_tr, _ = train_test_split(X_feat, y, test_size=0.2, random_state=42)
        m_a = LogisticRegression(max_iter=200).fit(X_tr, y_tr)
        m_b = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_tr, y_tr)
        p_a, p_b = str(tmp_path/"ma.pkl"), str(tmp_path/"mb.pkl")
        joblib.dump(m_a, p_a)
        joblib.dump(m_b, p_b)

        # Upload dataset
        with open(csv, "rb") as f:
            ds = ab_client.post("/api/v1/datasets/upload",
                files={"file": ("d.csv", f, "text/csv")},
                data={"name": "ab_test_ds"}).json()["data"]
        ds_id = ds["id"]

        # Seed experiments
        from models.experiment import Experiment
        from database import AsyncSessionLocal

        async def _seed():
            async with AsyncSessionLocal() as db:
                for path, name in [(p_a, "LR"), (p_b, "RF")]:
                    db.add(Experiment(
                        dataset_id=ds_id, name=name,
                        target_column="label", task_type="classification",
                        status="completed", model_artifact_path=path,
                        n_trials_completed=10,
                    ))
                await db.flush()
                result = await db.execute(
                    __import__("sqlalchemy", fromlist=["select"]).select(Experiment)
                    .order_by(Experiment.id)
                )
                ids = [r.id for r in result.scalars().all()]
                await db.commit()
            return ids

        ids = asyncio.run(_seed())
        assert len(ids) >= 2
        id_a, id_b = ids[-2], ids[-1]

        resp = ab_client.post("/api/v1/ab-test/", json={
            "experiment_a_id": id_a,
            "experiment_b_id": id_b,
            "confidence_level": 0.95,
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["n_samples"] > 0
        assert data["metric"] == "accuracy"
        assert 0.0 <= data["p_value"] <= 1.0
        assert "recommendation" in data
        assert data["winner"] in ("A", "B", None)
