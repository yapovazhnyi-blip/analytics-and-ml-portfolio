"""
Phase 3 tests — lineage DAG, deployment generator, advisor response parsing.

API endpoint tests use small in-memory datasets and completed experiment
fixtures to validate the full request/response cycle.
"""

import ast
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lineage.dag import (
    LineageGraph,
    NodeType,
    build_lineage,
    build_multi_experiment_lineage,
)
from deployment.generator import (
    FeatureSpec,
    ModelPackage,
    build_deployment_package,
    _generate_dockerfile,
    _generate_fastapi_app,
    _generate_k8s_deployment,
    _generate_openapi,
)
from advisor.claude import AdvisorSuggestion, _parse_suggestions


# ── Shared fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def base_lineage_kwargs():
    return dict(
        experiment_id=1,
        dataset_name="titanic",
        dataset_row_count=891,
        dataset_column_count=12,
        content_hash="abc123def456",
        source_type="csv",
        feature_columns=["age", "fare", "pclass"],
        target_column="survived",
        task_type="classification",
        training_config={"n_trials": 20, "cv_folds": 3},
        best_family="random_forest",
        best_score=0.834,
        scoring_metric="roc_auc",
        holdout_metrics={"holdout_accuracy": 0.821, "holdout_roc_auc": 0.887},
        n_trials=18,
        n_pruned=2,
    )


@pytest.fixture
def sample_pkg():
    return ModelPackage(
        model_name="titanic_rf",
        model_family="random_forest",
        feature_specs=[
            FeatureSpec("age", "float"),
            FeatureSpec("fare", "float"),
            FeatureSpec("pclass", "int"),
        ],
        target_name="survived",
        task_type="classification",
        best_score=0.834,
        scoring_metric="roc_auc",
    )


@pytest.fixture
def trained_model(tmp_path):
    """A real joblib model artifact for deployment tests."""
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (100, 3))
    y = (X[:, 0] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    return str(path)


# ══════════════════════════════════════════════════════════════════════════
# LINEAGE DAG
# ══════════════════════════════════════════════════════════════════════════

class TestLineageDAG:

    def test_build_returns_four_nodes(self, base_lineage_kwargs):
        lg = build_lineage(**base_lineage_kwargs)
        assert len(lg.nodes) == 4

    def test_node_types_are_correct(self, base_lineage_kwargs):
        lg = build_lineage(**base_lineage_kwargs)
        types = {n.node_type for n in lg.nodes}
        assert types == {
            NodeType.DATA_VERSION,
            NodeType.PREPROCESSING,
            NodeType.MODEL_CONFIG,
            NodeType.EVALUATION,
        }

    def test_edges_form_linear_chain(self, base_lineage_kwargs):
        lg = build_lineage(**base_lineage_kwargs)
        assert len(lg.edges) == 3  # dv→pp, pp→mc, mc→ev

    def test_data_version_node_contains_hash(self, base_lineage_kwargs):
        lg = build_lineage(**base_lineage_kwargs)
        dv = next(n for n in lg.nodes if n.node_type == NodeType.DATA_VERSION)
        assert "abc123" in dv.node_id

    def test_same_dataset_produces_same_dv_node_id(self, base_lineage_kwargs):
        """Two experiments from the same dataset must share the data version node."""
        lg1 = build_lineage(**{**base_lineage_kwargs, "experiment_id": 1})
        lg2 = build_lineage(**{**base_lineage_kwargs, "experiment_id": 2})
        dv1 = next(n for n in lg1.nodes if n.node_type == NodeType.DATA_VERSION)
        dv2 = next(n for n in lg2.nodes if n.node_type == NodeType.DATA_VERSION)
        assert dv1.node_id == dv2.node_id

    def test_different_datasets_produce_different_dv_ids(self, base_lineage_kwargs):
        lg1 = build_lineage(**base_lineage_kwargs)
        lg2 = build_lineage(**{**base_lineage_kwargs, "content_hash": "zzz999yyy888"})
        dv1 = next(n for n in lg1.nodes if n.node_type == NodeType.DATA_VERSION)
        dv2 = next(n for n in lg2.nodes if n.node_type == NodeType.DATA_VERSION)
        assert dv1.node_id != dv2.node_id

    def test_to_react_flow_shape(self, base_lineage_kwargs):
        lg = build_lineage(**base_lineage_kwargs)
        rf = lg.to_react_flow()
        assert "nodes" in rf and "edges" in rf
        assert all("id" in n and "type" in n and "data" in n for n in rf["nodes"])
        assert all("source" in e and "target" in e for e in rf["edges"])

    def test_multi_experiment_lineage_merges_shared_nodes(self, base_lineage_kwargs):
        """Two experiments from the same dataset should share the data_version node."""
        exp1 = {**base_lineage_kwargs, "experiment_id": 1}
        exp2 = {**base_lineage_kwargs, "experiment_id": 2,
                "best_family": "gradient_boosting", "best_score": 0.851}
        result = build_multi_experiment_lineage([exp1, exp2])

        # 2 experiments × 4 nodes, but dv and pp nodes are shared → 6 total
        assert len(result["nodes"]) == 6
        assert result["is_dag"] is True
        assert result["n_experiments"] == 2

    def test_evaluation_node_contains_score(self, base_lineage_kwargs):
        lg = build_lineage(**base_lineage_kwargs)
        ev = next(n for n in lg.nodes if n.node_type == NodeType.EVALUATION)
        assert "0.834" in ev.subtitle or "0.834" in str(ev.metadata)

    def test_root_node_id_is_data_version(self, base_lineage_kwargs):
        lg = build_lineage(**base_lineage_kwargs)
        dv = next(n for n in lg.nodes if n.node_type == NodeType.DATA_VERSION)
        assert lg.root_node_id == dv.node_id


# ══════════════════════════════════════════════════════════════════════════
# DEPLOYMENT GENERATOR
# ══════════════════════════════════════════════════════════════════════════

class TestDeploymentGenerator:

    def test_zip_contains_all_required_files(self, trained_model, sample_pkg, tmp_path):
        zip_path = build_deployment_package(trained_model, sample_pkg, tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            names = {Path(n).name for n in zf.namelist()}
        for required in ["Dockerfile", "requirements.txt", "main.py", "model.joblib",
                         "openapi.json", "deployment.yaml", "README.md"]:
            assert required in names, f"Missing {required}"

    def test_fastapi_app_is_valid_python(self, sample_pkg):
        code = _generate_fastapi_app(sample_pkg)
        ast.parse(code)  # raises SyntaxError if invalid

    def test_fastapi_app_includes_all_features(self, sample_pkg):
        code = _generate_fastapi_app(sample_pkg)
        for spec in sample_pkg.feature_specs:
            assert spec.name in code

    def test_fastapi_app_has_proba_for_classifier(self, sample_pkg):
        code = _generate_fastapi_app(sample_pkg)
        assert "predict_proba" in code

    def test_fastapi_app_no_proba_for_regressor(self):
        pkg = ModelPackage(
            model_name="price_model", model_family="ridge",
            feature_specs=[FeatureSpec("sqft", "float")],
            target_name="price", task_type="regression", best_score=0.91,
        )
        code = _generate_fastapi_app(pkg)
        assert "predict_proba" not in code

    def test_dockerfile_uses_slim_base(self, sample_pkg):
        df = _generate_dockerfile(sample_pkg)
        assert "python:3.11-slim" in df
        assert "EXPOSE 8000" in df
        assert "uvicorn" in df

    def test_dockerfile_copies_requirements_before_app(self, sample_pkg):
        df = _generate_dockerfile(sample_pkg)
        assert df.index("requirements.txt") < df.index("COPY app/")

    def test_k8s_manifest_has_deployment_and_service(self, sample_pkg):
        yaml = _generate_k8s_deployment(sample_pkg)
        assert "kind: Deployment" in yaml
        assert "kind: Service" in yaml

    def test_k8s_manifest_has_health_probes(self, sample_pkg):
        yaml = _generate_k8s_deployment(sample_pkg)
        assert "readinessProbe" in yaml
        assert "livenessProbe" in yaml
        assert "/health" in yaml

    def test_k8s_manifest_has_resource_limits(self, sample_pkg):
        yaml = _generate_k8s_deployment(sample_pkg)
        assert "resources:" in yaml
        assert "limits:" in yaml
        assert "requests:" in yaml

    def test_k8s_replicas_configurable(self):
        pkg = ModelPackage(
            model_name="m", model_family="rf",
            feature_specs=[FeatureSpec("x", "float")],
            target_name="y", task_type="classification",
            best_score=0.9, replicas=5,
        )
        yaml = _generate_k8s_deployment(pkg)
        assert "replicas: 5" in yaml

    def test_openapi_has_all_feature_properties(self, sample_pkg):
        spec = _generate_openapi(sample_pkg)
        props = spec["paths"]["/predict"]["post"]["requestBody"]["content"][
            "application/json"]["schema"]["properties"]
        for feat in sample_pkg.feature_specs:
            assert feat.name in props

    def test_model_survives_roundtrip(self, trained_model, sample_pkg, tmp_path):
        import joblib
        zip_path = build_deployment_package(trained_model, sample_pkg, tmp_path)
        # Extract model from zip and load it
        with zipfile.ZipFile(zip_path) as zf:
            model_member = next(n for n in zf.namelist() if n.endswith("model.joblib"))
            zf.extract(model_member, tmp_path / "extracted")
        loaded = joblib.load(tmp_path / "extracted" / model_member)
        assert hasattr(loaded, "predict")
        preds = loaded.predict(np.random.default_rng(0).normal(0, 1, (3, 3)))
        assert len(preds) == 3


# ══════════════════════════════════════════════════════════════════════════
# CLAUDE ADVISOR — parsing only (no API key needed)
# ══════════════════════════════════════════════════════════════════════════

class TestAdvisorParsing:

    def test_parses_valid_json_array(self):
        raw = json.dumps([{
            "category": "leakage",
            "severity": "high",
            "title": "Feature encodes target",
            "explanation": "Column score has r=0.99 with target.",
            "action": "Drop score column before training.",
            "column": "score",
        }])
        suggestions = _parse_suggestions(raw)
        assert len(suggestions) == 1
        assert suggestions[0].category == "leakage"
        assert suggestions[0].severity == "high"
        assert suggestions[0].column == "score"

    def test_strips_markdown_fences(self):
        raw = '```json\n[{"category":"general","severity":"info","title":"Clean","explanation":"Looks good.","action":"Proceed.","column":null}]\n```'
        suggestions = _parse_suggestions(raw)
        assert len(suggestions) == 1

    def test_caps_at_six_suggestions(self):
        items = [
            {"category": "general", "severity": "info", "title": f"S{i}",
             "explanation": "x", "action": "y", "column": None}
            for i in range(10)
        ]
        suggestions = _parse_suggestions(json.dumps(items))
        assert len(suggestions) <= 6

    def test_invalid_json_returns_empty(self):
        suggestions = _parse_suggestions("not json at all")
        assert suggestions == []

    def test_partial_item_skipped(self):
        raw = json.dumps([
            {"category": "leakage", "severity": "high", "title": "ok",
             "explanation": "e", "action": "a", "column": None},
            "not a dict",
        ])
        suggestions = _parse_suggestions(raw)
        assert len(suggestions) == 1

    def test_null_column_is_none(self):
        raw = json.dumps([{
            "category": "general", "severity": "info", "title": "t",
            "explanation": "e", "action": "a", "column": None,
        }])
        suggestions = _parse_suggestions(raw)
        assert suggestions[0].column is None
