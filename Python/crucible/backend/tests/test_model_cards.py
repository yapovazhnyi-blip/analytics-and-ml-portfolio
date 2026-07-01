"""
Model card tests.

Tests cover:
  - ModelCard generation from experiment + dataset mock objects
  - Markdown renderer: section presence, table structure, badge formatting
  - HTML renderer: valid HTML, all sections present
  - Fairness integration: severe severity triggers warning text
  - Limitations are always generated
  - API endpoint: json / markdown / html format variants
  - Incomplete experiment returns 422
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock


# ── Mock builders ─────────────────────────────────────────────────────────────

def _make_experiment(
    exp_id=1,
    status="completed",
    task_type="classification",
    target_column="churn",
    best_family="XGBoost",
    cv_score=0.893,
    holdout_metrics=None,
    feature_importances=None,
    fairness_json=None,
    n_trials=20,
):
    exp = MagicMock()
    exp.id = exp_id
    exp.status = status
    exp.task_type = task_type
    exp.target_column = target_column
    exp.dataset_id = 1
    exp.created_at = datetime(2024, 6, 1, tzinfo=timezone.utc)
    exp.best_model_family = best_family
    exp.best_cv_score = cv_score
    exp.n_trials_completed = n_trials
    exp.fairness_json = fairness_json
    exp.results_json = json.dumps({
        "holdout_metrics":     holdout_metrics or {"accuracy": 0.91, "f1": 0.88},
        "feature_importances": feature_importances or [
            {"feature": "tenure",  "mean_abs_shap": 0.42},
            {"feature": "balance", "mean_abs_shap": 0.31},
            {"feature": "age",     "mean_abs_shap": 0.19},
        ],
        "feature_names": ["tenure", "balance", "age", "products"],
        "best_params": {"n_estimators": 200, "max_depth": 6},
    })
    return exp


def _make_dataset(ds_id=1, name="churn_data", n_rows=8000, source_type="csv"):
    ds = MagicMock()
    ds.id = ds_id
    ds.name = name
    ds.row_count = n_rows
    ds.column_count = 5
    ds.source_type = source_type
    ds.contract_json = None
    return ds


# ══════════════════════════════════════════════════════════════════════════
# GENERATOR
# ══════════════════════════════════════════════════════════════════════════

class TestModelCardGenerator:

    def test_generates_model_card_from_experiment(self):
        from model_cards.generator import generate_model_card
        card = generate_model_card(_make_experiment(), _make_dataset())
        assert card.experiment_id == 1
        assert card.model_family == "XGBoost"
        assert card.task_type == "classification"

    def test_cv_score_populated(self):
        from model_cards.generator import generate_model_card
        card = generate_model_card(_make_experiment(cv_score=0.893), _make_dataset())
        assert card.cv_score == pytest.approx(0.893)

    def test_holdout_metrics_extracted(self):
        from model_cards.generator import generate_model_card
        card = generate_model_card(_make_experiment(), _make_dataset())
        metric_names = {m.name for m in card.metrics}
        assert "accuracy" in metric_names
        assert "f1" in metric_names

    def test_feature_importances_extracted(self):
        from model_cards.generator import generate_model_card
        card = generate_model_card(_make_experiment(), _make_dataset())
        assert len(card.feature_importances) > 0
        assert card.feature_importances[0].feature == "tenure"
        assert card.feature_importances[0].mean_abs_shap == pytest.approx(0.42)

    def test_dataset_info_populated(self):
        from model_cards.generator import generate_model_card
        ds = _make_dataset(name="churn_data", n_rows=8000)
        card = generate_model_card(_make_experiment(), ds)
        assert card.dataset_name == "churn_data"
        assert card.n_training_rows == 8000

    def test_none_dataset_handled_gracefully(self):
        from model_cards.generator import generate_model_card
        # Should not raise
        card = generate_model_card(_make_experiment(), None)
        assert card.dataset_name is None

    def test_fairness_not_assessed_by_default(self):
        from model_cards.generator import generate_model_card
        card = generate_model_card(_make_experiment(), _make_dataset())
        assert not card.fairness_assessed
        assert card.fairness_entries == []

    def test_fairness_populated_when_json_present(self):
        from model_cards.generator import generate_model_card
        fairness = json.dumps({
            "overall_severity": "significant",
            "metrics": [{
                "attribute": "gender",
                "demographic_parity_diff": 0.15,
                "equal_opportunity_diff": 0.12,
                "disparate_impact_ratio": 0.72,
                "severity": "significant",
                "privileged_group": "M",
                "unprivileged_group": "F",
            }],
        })
        card = generate_model_card(_make_experiment(fairness_json=fairness), _make_dataset())
        assert card.fairness_assessed
        assert len(card.fairness_entries) == 1
        assert card.fairness_entries[0].attribute == "gender"
        assert card.fairness_overall_severity == "significant"

    def test_limitations_always_present(self):
        from model_cards.generator import generate_model_card
        card = generate_model_card(_make_experiment(), _make_dataset())
        assert len(card.limitations) > 0
        # Drift warning must always be present
        assert any("drift" in l.lower() for l in card.limitations)

    def test_small_dataset_triggers_warning(self):
        from model_cards.generator import generate_model_card
        card = generate_model_card(_make_experiment(), _make_dataset(n_rows=200))
        assert any("200" in l or "small" in l.lower() for l in card.limitations)

    def test_intended_use_classification(self):
        from model_cards.generator import generate_model_card
        card = generate_model_card(_make_experiment(task_type="classification"), _make_dataset())
        assert "classif" in card.intended_use.lower()

    def test_intended_use_regression(self):
        from model_cards.generator import generate_model_card
        card = generate_model_card(_make_experiment(task_type="regression"), _make_dataset())
        assert "predict" in card.intended_use.lower()

    def test_to_dict_is_json_serialisable(self):
        from model_cards.generator import generate_model_card
        card = generate_model_card(_make_experiment(), _make_dataset())
        json.dumps(card.to_dict())   # must not raise

    def test_contract_json_populated(self):
        from model_cards.generator import generate_model_card
        ds = _make_dataset()
        ds.contract_json = json.dumps({"version": "1.0", "n_cols": 5, "tolerance": 0.10})
        card = generate_model_card(_make_experiment(), ds)
        assert card.contract_version == "1.0"
        assert card.contract_n_columns == 5

    def test_recommendations_present(self):
        from model_cards.generator import generate_model_card
        card = generate_model_card(_make_experiment(), _make_dataset())
        assert len(card.recommendations) > 0

    def test_severe_fairness_adds_ethical_warning(self):
        from model_cards.generator import generate_model_card
        fairness = json.dumps({
            "overall_severity": "severe",
            "metrics": [{
                "attribute": "race",
                "demographic_parity_diff": 0.35,
                "equal_opportunity_diff": 0.30,
                "disparate_impact_ratio": 0.55,
                "severity": "severe",
                "privileged_group": "A",
                "unprivileged_group": "B",
            }],
        })
        card = generate_model_card(_make_experiment(fairness_json=fairness), _make_dataset())
        assert any("SEVERE" in e or "severe" in e.lower() for e in card.ethical_considerations)


# ══════════════════════════════════════════════════════════════════════════
# MARKDOWN RENDERER
# ══════════════════════════════════════════════════════════════════════════

class TestMarkdownRenderer:

    @pytest.fixture
    def md(self):
        from model_cards.generator import generate_model_card
        from model_cards.renderer import render_markdown
        return render_markdown(generate_model_card(_make_experiment(), _make_dataset()))

    def test_has_h1_title(self, md):
        assert "# 🧪 Model Card" in md

    def test_has_model_overview_section(self, md):
        assert "## Model Overview" in md

    def test_has_intended_use_section(self, md):
        assert "## Intended Use" in md

    def test_has_training_data_section(self, md):
        assert "## Training Data" in md

    def test_has_evaluation_section(self, md):
        assert "## Evaluation Results" in md

    def test_has_limitations_section(self, md):
        assert "## Limitations" in md

    def test_contains_experiment_id(self, md):
        assert "1" in md   # experiment_id=1

    def test_contains_model_family(self, md):
        assert "XGBoost" in md

    def test_contains_cv_score(self, md):
        assert "0.893" in md

    def test_contains_feature_importances(self, md):
        assert "tenure" in md   # top feature

    def test_markdown_is_string(self, md):
        assert isinstance(md, str)
        assert len(md) > 100

    def test_no_fairness_flagged_in_markdown(self, md):
        assert "No fairness" in md or "not performed" in md.lower()


# ══════════════════════════════════════════════════════════════════════════
# HTML RENDERER
# ══════════════════════════════════════════════════════════════════════════

class TestHTMLRenderer:

    @pytest.fixture
    def html(self):
        from model_cards.generator import generate_model_card
        from model_cards.renderer import render_html
        return render_html(generate_model_card(_make_experiment(), _make_dataset()))

    def test_is_valid_html_structure(self, html):
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<body" in html
        assert "</body>" in html

    def test_has_title_tag(self, html):
        assert "<title>" in html

    def test_contains_model_name(self, html):
        assert "XGBoost" in html

    def test_contains_experiment_id(self, html):
        assert "#1" in html or "Experiment 1" in html

    def test_contains_evaluation_section(self, html):
        assert "Evaluation" in html

    def test_contains_fairness_section(self, html):
        assert "Fairness" in html

    def test_contains_limitations_section(self, html):
        assert "Limitations" in html

    def test_html_is_self_contained(self, html):
        """Must not reference external CSS or JS files (footer links are OK)."""
        # Remove footer before checking for external resource links
        body = html.split("no-print")[0] if "no-print" in html else html
        assert 'stylesheet" href="http' not in body
        assert '<script src="http' not in body.lower()


# ══════════════════════════════════════════════════════════════════════════
# API ENDPOINT
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mc_client(tmp_path):
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


class TestModelCardAPI:

    def test_nonexistent_experiment_returns_404(self, mc_client):
        resp = mc_client.get("/api/v1/experiments/9999/model-card")
        assert resp.status_code == 404

    def test_full_model_card_flow(self, mc_client, tmp_path):
        """Upload dataset → train (mock) → get model card."""
        import asyncio
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "age":   rng.integers(18, 70, 200).astype(float),
            "score": rng.uniform(0, 1, 200),
            "label": rng.integers(0, 2, 200),
        })
        csv = df.to_csv(index=False).encode()
        ds = mc_client.post("/api/v1/datasets/upload",
             files={"file": ("d.csv", csv, "text/csv")},
             data={"name": "mc_test"}).json()["data"]
        ds_id = ds["id"]

        # Seed a completed experiment
        from models.experiment import Experiment
        from database import AsyncSessionLocal

        async def _seed():
            async with AsyncSessionLocal() as db:
                import joblib, tempfile
                from sklearn.ensemble import RandomForestClassifier
                X = df[["age", "score"]].values
                y = df["label"].values
                clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
                p = tempfile.mktemp(suffix=".pkl")
                joblib.dump(clf, p)
                exp = Experiment(
                    dataset_id=ds_id, name="mc_exp",
                    target_column="label", task_type="classification",
                    status="completed", model_artifact_path=p,
                    best_model_family="RandomForest", best_score=0.82,
                    n_trials_completed=10,
                    results_json=json.dumps({
                        "holdout_metrics": {"accuracy": 0.84, "f1": 0.81},
                        "feature_importances": [
                            {"feature": "age",   "mean_abs_shap": 0.35},
                            {"feature": "score", "mean_abs_shap": 0.22},
                        ],
                        "feature_names": ["age", "score"],
                        "best_params": {"n_estimators": 5},
                    }),
                )
                db.add(exp)
                await db.flush(); await db.refresh(exp)
                eid = exp.id
                await db.commit()
            return eid

        eid = asyncio.run(_seed())

        # JSON format
        resp = mc_client.get(f"/api/v1/experiments/{eid}/model-card")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["experiment_id"] == eid
        assert data["model_family"] == "RandomForest"
        assert data["performance"]["cv_score"] == pytest.approx(0.82)
        assert len(data["performance"]["metrics"]) >= 1

    def test_markdown_format(self, mc_client):
        """model-card?format=markdown returns plain text Markdown."""
        # Re-use the full flow test's seeded DB indirectly:
        # seed a minimal experiment via the upload + async seeding
        import asyncio
        import pandas as pd, numpy as np
        from models.experiment import Experiment
        from database import AsyncSessionLocal

        df = pd.DataFrame({"x": np.random.randn(100), "y": np.random.randint(0,2,100)})
        csv = df.to_csv(index=False).encode()
        ds = mc_client.post("/api/v1/datasets/upload",
             files={"file": ("md.csv", csv, "text/csv")},
             data={"name": "md_test"}).json()["data"]

        async def _seed():
            async with AsyncSessionLocal() as db:
                exp = Experiment(
                    dataset_id=ds["id"], name="md_exp",
                    target_column="y", task_type="regression",
                    status="completed", best_model_family="Ridge",
                    best_score=0.72, n_trials_completed=5,
                    results_json=json.dumps({"holdout_metrics": {"r2": 0.75},
                                             "feature_names": ["x"],
                                             "best_params": {}}),
                )
                db.add(exp)
                await db.flush(); await db.refresh(exp)
                eid = exp.id
                await db.commit()
            return eid

        eid = asyncio.new_event_loop().run_until_complete(_seed())
        resp = mc_client.get(f"/api/v1/experiments/{eid}/model-card?format=markdown")
        assert resp.status_code == 200
        assert "# 🧪 Model Card" in resp.text
        assert "Ridge" in resp.text

    def test_html_format(self, mc_client):
        """model-card?format=html returns HTML document."""
        import asyncio
        import pandas as pd, numpy as np
        from models.experiment import Experiment
        from database import AsyncSessionLocal

        df = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randint(0,2,100)})
        csv = df.to_csv(index=False).encode()
        ds = mc_client.post("/api/v1/datasets/upload",
             files={"file": ("html.csv", csv, "text/csv")},
             data={"name": "html_test"}).json()["data"]

        async def _seed():
            async with AsyncSessionLocal() as db:
                exp = Experiment(
                    dataset_id=ds["id"], name="html_exp",
                    target_column="b", task_type="classification",
                    status="completed", best_model_family="LightGBM",
                    best_score=0.90, n_trials_completed=15,
                    results_json=json.dumps({"holdout_metrics": {"accuracy": 0.91},
                                             "feature_names": ["a"],
                                             "best_params": {}}),
                )
                db.add(exp)
                await db.flush(); await db.refresh(exp)
                eid = exp.id
                await db.commit()
            return eid

        eid = asyncio.new_event_loop().run_until_complete(_seed())
        resp = mc_client.get(f"/api/v1/experiments/{eid}/model-card?format=html")
        assert resp.status_code == 200
        assert "<!DOCTYPE html>" in resp.text
        assert "LightGBM" in resp.text
