"""
Phase 3 router — lineage DAG, deployment generation, Claude advisor.

Endpoints:
  GET  /experiments/{id}/lineage          — lineage DAG for single experiment
  GET  /datasets/{id}/lineage             — merged lineage for all dataset experiments
  POST /experiments/{id}/deploy           — generate deployment package (returns zip download)
  POST /datasets/{id}/profile/advise      — run profile then call Claude advisor
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from auth.dependencies import get_current_user
from fastapi import Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_db
from models import Dataset, Experiment
from schemas.common import DataResponse
from lineage.dag import build_lineage, build_multi_experiment_lineage
from deployment.generator import (
    ModelPackage, FeatureSpec, build_deployment_package,
)
from advisor.claude import get_advisor_suggestions
from profiling.runner import ProfileRunner
from config import settings

router = APIRouter(tags=["phase3"], dependencies=[Depends(get_current_user)])


# ── Lineage — single experiment ────────────────────────────────────────────

@router.get("/experiments/{exp_id}/lineage")
async def get_experiment_lineage(exp_id: int, db: AsyncSession = Depends(get_db)):
    exp = await db.get(Experiment, exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")

    ds = await db.get(Dataset, exp.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    holdout = {}
    feature_cols = []
    training_conf = {}

    if exp.results_json:
        try:
            r = json.loads(exp.results_json)
            holdout = r.get("holdout_metrics", {})
        except Exception:
            pass

    if exp.preprocessing_config:
        try:
            pc = json.loads(exp.preprocessing_config)
            feature_cols = pc.get("feature_cols", [])
        except Exception:
            pass

    if exp.training_config:
        try:
            training_conf = json.loads(exp.training_config)
        except Exception:
            pass

    lineage = build_lineage(
        experiment_id=exp.id,
        dataset_name=ds.name,
        dataset_row_count=ds.row_count,
        dataset_column_count=ds.column_count,
        content_hash=ds.content_hash,
        source_type=ds.source_type,
        feature_columns=feature_cols,
        target_column=exp.target_column,
        task_type=exp.task_type,
        training_config=training_conf,
        best_family=exp.best_model_family,
        best_score=exp.best_score,
        scoring_metric=exp.scoring_metric,
        holdout_metrics=holdout,
        n_trials=exp.n_trials_completed,
        n_pruned=exp.n_trials_pruned,
    )

    return DataResponse(data=lineage.to_react_flow())


# ── Lineage — all experiments for a dataset ────────────────────────────────

@router.get("/datasets/{dataset_id}/lineage")
async def get_dataset_lineage(dataset_id: int, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    rows = await db.scalars(
        select(Experiment)
        .where(Experiment.dataset_id == dataset_id)
        .where(Experiment.status == "complete")
        .order_by(Experiment.created_at)
    )
    experiments = rows.all()

    if not experiments:
        return DataResponse(data={"nodes": [], "edges": [], "n_experiments": 0, "is_dag": True})

    exp_dicts = []
    for exp in experiments:
        holdout = {}
        feature_cols = []
        training_conf = {}

        if exp.results_json:
            try:
                r = json.loads(exp.results_json)
                holdout = r.get("holdout_metrics", {})
            except Exception:
                pass
        if exp.preprocessing_config:
            try:
                feature_cols = json.loads(exp.preprocessing_config).get("feature_cols", [])
            except Exception:
                pass
        if exp.training_config:
            try:
                training_conf = json.loads(exp.training_config)
            except Exception:
                pass

        exp_dicts.append(dict(
            experiment_id=exp.id,
            dataset_name=ds.name,
            dataset_row_count=ds.row_count,
            dataset_column_count=ds.column_count,
            content_hash=ds.content_hash,
            source_type=ds.source_type,
            feature_columns=feature_cols,
            target_column=exp.target_column,
            task_type=exp.task_type,
            training_config=training_conf,
            best_family=exp.best_model_family,
            best_score=exp.best_score,
            scoring_metric=exp.scoring_metric,
            holdout_metrics=holdout,
            n_trials=exp.n_trials_completed,
            n_pruned=exp.n_trials_pruned,
        ))

    return DataResponse(data=build_multi_experiment_lineage(exp_dicts))


# ── Deployment package ─────────────────────────────────────────────────────

class DeployRequest(BaseModel):
    model_name: Optional[str] = None
    replicas: int = Field(default=2, ge=1, le=20)
    memory_limit: str = "512Mi"
    cpu_limit: str = "500m"


@router.post("/experiments/{exp_id}/deploy")
async def generate_deployment(
    exp_id: int,
    body: DeployRequest = DeployRequest(),
    db: AsyncSession = Depends(get_db),
):
    """
    Generates a full deployment package for a trained experiment.
    Returns a zip file download.
    """
    exp = await db.get(Experiment, exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")
    if exp.status != "complete":
        raise HTTPException(status_code=422, detail="Experiment must be complete to deploy")
    if not exp.model_artifact_path or not os.path.exists(exp.model_artifact_path):
        raise HTTPException(status_code=422, detail="Model artifact not found")

    ds = await db.get(Dataset, exp.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Build feature specs from dataset schema
    feature_specs = []
    feature_cols = []
    if exp.preprocessing_config:
        try:
            feature_cols = json.loads(exp.preprocessing_config).get("feature_cols", [])
        except Exception:
            pass

    schema_cols = {}
    if ds.schema_json:
        try:
            for col in json.loads(ds.schema_json):
                schema_cols[col["name"]] = col
        except Exception:
            pass

    for col in feature_cols:
        col_info = schema_cols.get(col, {})
        dtype_str = col_info.get("dtype", "float64")
        if "int" in dtype_str:
            dtype = "int"
        elif "float" in dtype_str or "num" in dtype_str:
            dtype = "float"
        else:
            dtype = "float"
        feature_specs.append(FeatureSpec(
            name=col,
            dtype=dtype,
            nullable=col_info.get("nullable", False),
        ))

    if not feature_specs:
        raise HTTPException(status_code=422, detail="No feature columns found in experiment")

    model_name = (body.model_name or f"{ds.name}_{exp.best_model_family or 'model'}").replace(" ", "_")

    pkg = ModelPackage(
        model_name=model_name,
        model_family=exp.best_model_family or "unknown",
        feature_specs=feature_specs,
        target_name=exp.target_column,
        task_type=exp.task_type,
        best_score=exp.best_score or 0.0,
        scoring_metric=exp.scoring_metric or "score",
        experiment_id=exp.id,
        replicas=body.replicas,
        memory_limit=body.memory_limit,
        cpu_limit=body.cpu_limit,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = build_deployment_package(
            model_path=exp.model_artifact_path,
            pkg=pkg,
            output_dir=Path(tmp_dir),
        )
        # Copy to a stable location before tmp_dir is cleaned up
        stable_path = Path(settings.model_storage_path) / zip_path.name
        import shutil
        shutil.copy2(zip_path, stable_path)

    return FileResponse(
        path=str(stable_path),
        media_type="application/zip",
        filename=zip_path.name,
    )


# ── Claude advisor ─────────────────────────────────────────────────────────

class AdvisorRequest(BaseModel):
    target_column: Optional[str] = None
    time_column: Optional[str] = None


class AdvisorSuggestionOut(BaseModel):
    category: str
    severity: str
    title: str
    explanation: str
    action: str
    column: Optional[str] = None


class AdvisorOut(BaseModel):
    suggestions: list[AdvisorSuggestionOut]
    model: str
    used_tokens: int
    error: Optional[str] = None


@router.post("/datasets/{dataset_id}/profile/advise", response_model=DataResponse[AdvisorOut])
async def advise_dataset(
    dataset_id: int,
    body: AdvisorRequest = AdvisorRequest(),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Runs the profiling suite on a dataset, then calls the Claude advisor
    to generate actionable suggestions based on the findings.

    Uses the authenticated user's API key if stored, otherwise falls back
    to the server-side ANTHROPIC_API_KEY.
    """
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    if ds.status != "ready":
        raise HTTPException(status_code=422, detail=f"Dataset not ready (status: {ds.status})")
    if not ds.file_path:
        raise HTTPException(status_code=422, detail="Dataset has no local file")

    from caching.cache import get_profiling_cache, cache_key, PROFILING_TTL_SECS

    content_key = ds.content_hash or f"id-{dataset_id}"
    key = cache_key(
        "profile", content_key,
        body.target_column or "_", body.time_column or "_", "0.2",   # default test_fraction
    )
    cache = get_profiling_cache()
    report = cache.get(key)

    if report is None:
        try:
            df = ProfileRunner.load_dataframe(ds.file_path, ds.source_type)
            runner = ProfileRunner()
            report = await runner.run(
                df=df,
                dataset_id=dataset_id,
                target_column=body.target_column,
                time_column=body.time_column,
            )
            cache.set(key, report, ttl_secs=PROFILING_TTL_SECS)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Profiling failed: {exc}")

    from auth.key_manager import get_anthropic_key
    api_key = await get_anthropic_key(user, require=False) or ""
    advisor_resp = await get_advisor_suggestions(report.to_advisor_prompt(), api_key=api_key)

    return DataResponse(data=AdvisorOut(
        suggestions=[
            AdvisorSuggestionOut(
                category=s.category,
                severity=s.severity,
                title=s.title,
                explanation=s.explanation,
                action=s.action,
                column=s.column,
            )
            for s in advisor_resp.suggestions
        ],
        model=advisor_resp.model,
        used_tokens=advisor_resp.used_tokens,
        error=advisor_resp.error,
    ))


# ── ONNX Export ───────────────────────────────────────────────────────────────

class ONNXExportOut(BaseModel):
    onnx_path: str
    input_name: str
    output_names: list[str]
    n_features: int
    opset_version: int
    model_size_kb: float
    download_url: str
    error: Optional[str] = None


@router.post("/experiments/{experiment_id}/export/onnx",
             response_model=DataResponse[ONNXExportOut])
async def export_onnx(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Converts the best-trained model for an experiment to ONNX format.

    ONNX Runtime is 3-10× faster than sklearn/XGBoost on CPU inference
    because it applies graph-level optimisations (operator fusion,
    SIMD vectorisation) that framework-specific runtimes do not.

    Returns the path to the .onnx file and metadata for the ONNX server.
    The ONNX file is included in the deployment package (POST /experiments/{id}/deploy).

    Supported families: all sklearn families, XGBoost, LightGBM.
    Keras/TF requires tf2onnx (see error message for instructions).
    """
    from deployment.onnx_exporter import export_to_onnx, ONNXExportResult

    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(404, f"Experiment {experiment_id} not found")
    if exp.status != "completed":
        raise HTTPException(422, f"Experiment not completed (status: {exp.status})")
    if not exp.artifact_path:
        raise HTTPException(422, "Experiment has no model artifact")

    import json
    results_raw = exp.results_json or "{}"
    results = json.loads(results_raw) if isinstance(results_raw, str) else results_raw
    feature_names = results.get("feature_names", [])
    task_type = exp.task_type or "classification"

    if not feature_names:
        raise HTTPException(422, "No feature names found in experiment results. Re-run the experiment.")

    output_dir = str(Path(settings.model_storage_path) / "onnx")

    import asyncio
    loop = asyncio.get_event_loop()
    result: ONNXExportResult = await loop.run_in_executor(
        None,
        export_to_onnx,
        exp.artifact_path,
        feature_names,
        task_type,
        output_dir,
        experiment_id,
    )

    if not result.succeeded:
        raise HTTPException(500, f"ONNX export failed: {result.error}")

    return DataResponse(data=ONNXExportOut(
        onnx_path=result.onnx_path,
        input_name=result.input_name,
        output_names=result.output_names,
        n_features=result.n_features,
        opset_version=result.opset_version,
        model_size_kb=result.model_size_kb,
        download_url=f"/api/v1/experiments/{experiment_id}/export/onnx/download",
    ))


@router.get("/experiments/{experiment_id}/export/onnx/download")
async def download_onnx(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Downloads the ONNX model file for an experiment."""
    from fastapi.responses import FileResponse
    output_dir = Path(settings.model_storage_path) / "onnx"
    onnx_path = output_dir / f"experiment_{experiment_id}_model.onnx"

    if not onnx_path.exists():
        raise HTTPException(
            404,
            "ONNX file not found. Call POST /experiments/{id}/export/onnx first."
        )

    return FileResponse(
        path=str(onnx_path),
        filename=f"experiment_{experiment_id}_model.onnx",
        media_type="application/octet-stream",
    )
