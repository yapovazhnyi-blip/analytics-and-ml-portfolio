"""
Deployment generator for Crucible Phase 3.

Promoted and hardened from spike/docker_gen_spike.py with one addition:
Kubernetes manifest generation (resolved in the gap analysis — companies
deploy models to K8s clusters, not bare docker run commands).

Output package structure:
  {model_name}/
    Dockerfile
    requirements.txt
    app/
      main.py          ← FastAPI endpoint, auto-generated
      model.joblib     ← serialised model
    openapi.json       ← OpenAPI 3.0 spec
    k8s/
      deployment.yaml  ← Kubernetes Deployment + Service
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Model package spec ─────────────────────────────────────────────────────

@dataclass
class FeatureSpec:
    name: str
    dtype: str        # "float" | "int" | "str"
    nullable: bool = False


@dataclass
class ModelPackage:
    model_name: str
    model_family: str
    feature_specs: list[FeatureSpec]
    target_name: str
    task_type: str              # "classification" | "regression"
    best_score: float
    scoring_metric: str = "roc_auc"
    experiment_id: Optional[int] = None
    python_version: str = "3.11"
    base_image: str = "python:3.11-slim"
    replicas: int = 2           # K8s deployment replicas
    memory_request: str = "256Mi"
    memory_limit: str = "512Mi"
    cpu_request: str = "250m"
    cpu_limit: str = "500m"


# ── Code generators ────────────────────────────────────────────────────────

def _generate_fastapi_app(pkg: ModelPackage) -> str:
    fields = "\n".join(
        f"    {s.name}: {'Optional[' + _py_type(s.dtype) + ']' if s.nullable else _py_type(s.dtype)}"
        for s in pkg.feature_specs
    )
    feature_list = ", ".join(f"body.{s.name}" for s in pkg.feature_specs)

    proba_block = ""
    proba_return = "None"
    if pkg.task_type == "classification":
        proba_block = (
            "\n    proba = None\n"
            "    if hasattr(model, 'predict_proba'):\n"
            "        proba = model.predict_proba(X).tolist()[0]"
        )
        proba_return = "proba"

    return (
        f'"""\n'
        f'Crucible auto-generated endpoint.\n'
        f'Model: {pkg.model_name} ({pkg.model_family})\n'
        f'Task: {pkg.task_type} | {pkg.scoring_metric}: {pkg.best_score:.4f}\n'
        f'"""\n'
        f'from contextlib import asynccontextmanager\n'
        f'from typing import Optional\n'
        f'import joblib\n'
        f'import numpy as np\n'
        f'from fastapi import FastAPI\n'
        f'from pydantic import BaseModel\n\n'
        f'model = None\n\n'
        f'@asynccontextmanager\n'
        f'async def lifespan(app: FastAPI):\n'
        f'    global model\n'
        f'    model = joblib.load("model.joblib")\n'
        f'    yield\n\n'
        f'app = FastAPI(\n'
        f'    title="{pkg.model_name}",\n'
        f'    description="Crucible | {pkg.model_family} | {pkg.task_type}",\n'
        f'    version="1.0.0",\n'
        f'    lifespan=lifespan,\n'
        f')\n\n'
        f'class PredictRequest(BaseModel):\n'
        f'{fields}\n\n'
        f'class PredictResponse(BaseModel):\n'
        f'    prediction: float\n'
        f'    probabilities: Optional[list[float]] = None\n\n'
        f'@app.post("/predict", response_model=PredictResponse)\n'
        f'async def predict(body: PredictRequest):\n'
        f'    X = np.array([[{feature_list}]])\n'
        f'    prediction = float(model.predict(X)[0])\n'
        f'{proba_block}\n'
        f'    return PredictResponse(prediction=prediction, probabilities={proba_return})\n\n'
        f'@app.get("/health")\n'
        f'async def health():\n'
        f'    return {{"status": "ok", "model": "{pkg.model_name}"}}\n'
    )


def _generate_dockerfile(pkg: ModelPackage) -> str:
    return (
        f"FROM {pkg.base_image}\n\n"
        f"WORKDIR /app\n\n"
        f"RUN apt-get update && apt-get install -y --no-install-recommends curl \\\n"
        f"    && rm -rf /var/lib/apt/lists/*\n\n"
        f"COPY requirements.txt .\n"
        f"RUN pip install --no-cache-dir -r requirements.txt\n\n"
        f"COPY app/ .\n\n"
        f"EXPOSE 8000\n\n"
        f'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]\n'
    )


def _generate_requirements() -> str:
    packages = ["fastapi", "uvicorn", "pydantic", "scikit-learn", "numpy", "joblib"]
    lines = []
    for pkg in packages:
        try:
            version = importlib.metadata.version(pkg)
            lines.append(f"{pkg}=={version}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(pkg)
    return "\n".join(lines) + "\n"


def _generate_openapi(pkg: ModelPackage) -> dict:
    properties = {
        s.name: {"type": _openapi_type(s.dtype)}
        for s in pkg.feature_specs
    }
    required = [s.name for s in pkg.feature_specs if not s.nullable]

    return {
        "openapi": "3.0.0",
        "info": {
            "title": pkg.model_name,
            "description": f"Crucible | {pkg.model_family} | {pkg.task_type}",
            "version": "1.0.0",
        },
        "paths": {
            "/predict": {
                "post": {
                    "summary": "Run prediction",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": properties,
                                    "required": required,
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Prediction",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "prediction": {"type": "number"},
                                            "probabilities": {
                                                "type": "array",
                                                "items": {"type": "number"},
                                                "nullable": True,
                                            },
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/health": {
                "get": {
                    "summary": "Health check",
                    "responses": {"200": {"description": "OK"}},
                }
            },
        },
    }


def _generate_k8s_deployment(pkg: ModelPackage) -> str:
    """
    Kubernetes Deployment + Service manifest.

    Design decisions (from gap analysis):
    - Deployment with configurable replicas (default 2 — one rolling-update replica available)
    - Resource requests and limits (prevents OOM kills in shared clusters)
    - Readiness probe on /health (ensures traffic only goes to ready pods)
    - Liveness probe on /health (restarts pods that hang)
    - Service of type ClusterIP (internal cluster routing; use Ingress for external)
    """
    safe_name = pkg.model_name.lower().replace("_", "-")
    return (
        f"# Crucible auto-generated Kubernetes manifest\n"
        f"# Apply with: kubectl apply -f k8s/deployment.yaml\n"
        f"---\n"
        f"apiVersion: apps/v1\n"
        f"kind: Deployment\n"
        f"metadata:\n"
        f"  name: {safe_name}\n"
        f"  labels:\n"
        f"    app: {safe_name}\n"
        f"    generated-by: crucible\n"
        f"spec:\n"
        f"  replicas: {pkg.replicas}\n"
        f"  selector:\n"
        f"    matchLabels:\n"
        f"      app: {safe_name}\n"
        f"  strategy:\n"
        f"    type: RollingUpdate\n"
        f"    rollingUpdate:\n"
        f"      maxSurge: 1\n"
        f"      maxUnavailable: 0\n"
        f"  template:\n"
        f"    metadata:\n"
        f"      labels:\n"
        f"        app: {safe_name}\n"
        f"    spec:\n"
        f"      containers:\n"
        f"        - name: {safe_name}\n"
        f"          image: {safe_name}:latest\n"
        f"          ports:\n"
        f"            - containerPort: 8000\n"
        f"          resources:\n"
        f"            requests:\n"
        f"              memory: \"{pkg.memory_request}\"\n"
        f"              cpu: \"{pkg.cpu_request}\"\n"
        f"            limits:\n"
        f"              memory: \"{pkg.memory_limit}\"\n"
        f"              cpu: \"{pkg.cpu_limit}\"\n"
        f"          readinessProbe:\n"
        f"            httpGet:\n"
        f"              path: /health\n"
        f"              port: 8000\n"
        f"            initialDelaySeconds: 5\n"
        f"            periodSeconds: 10\n"
        f"          livenessProbe:\n"
        f"            httpGet:\n"
        f"              path: /health\n"
        f"              port: 8000\n"
        f"            initialDelaySeconds: 15\n"
        f"            periodSeconds: 20\n"
        f"---\n"
        f"apiVersion: v1\n"
        f"kind: Service\n"
        f"metadata:\n"
        f"  name: {safe_name}\n"
        f"spec:\n"
        f"  selector:\n"
        f"    app: {safe_name}\n"
        f"  ports:\n"
        f"    - protocol: TCP\n"
        f"      port: 80\n"
        f"      targetPort: 8000\n"
        f"  type: ClusterIP\n"
    )


def build_deployment_package(
    model_path: str,
    pkg: ModelPackage,
    output_dir: Path,
) -> Path:
    """
    Writes the full deployment package and returns the path to the zip.

    Security: model_name is sanitised before being used as a directory name
    and a Kubernetes resource name. Kubernetes names must be lowercase
    alphanumeric with hyphens, max 63 chars (RFC 1123 label). Directory
    names must not contain path separators or control characters.
    """
    import re as _re

    # Sanitise: keep only alphanumerics, hyphens, underscores.
    # Reject anything that looks like a path traversal attempt.
    safe_model_name = _re.sub(r"[^a-zA-Z0-9_\-]", "_", pkg.model_name)
    safe_model_name = safe_model_name[:63]  # K8s label max length
    if not safe_model_name or safe_model_name.startswith(("-", "_")):
        safe_model_name = "model_" + safe_model_name.lstrip("-_")

    # Replace pkg.model_name with the sanitised version for all file operations
    pkg = ModelPackage(
        model_name=safe_model_name,
        model_family=pkg.model_family,
        feature_specs=pkg.feature_specs,
        target_name=pkg.target_name,
        task_type=pkg.task_type,
        best_score=pkg.best_score,
        scoring_metric=pkg.scoring_metric,
        experiment_id=pkg.experiment_id,
        python_version=pkg.python_version,
        base_image=pkg.base_image,
        replicas=pkg.replicas,
        memory_request=pkg.memory_request,
        memory_limit=pkg.memory_limit,
        cpu_request=pkg.cpu_request,
        cpu_limit=pkg.cpu_limit,
    )
    root = output_dir / pkg.model_name
    app_dir = root / "app"
    k8s_dir = root / "k8s"
    app_dir.mkdir(parents=True, exist_ok=True)
    k8s_dir.mkdir(parents=True, exist_ok=True)

    # Copy model artifact
    import shutil
    shutil.copy2(model_path, app_dir / "model.joblib")

    (app_dir / "main.py").write_text(_generate_fastapi_app(pkg))
    (root / "Dockerfile").write_text(_generate_dockerfile(pkg))
    (root / "requirements.txt").write_text(_generate_requirements())
    (root / "openapi.json").write_text(json.dumps(_generate_openapi(pkg), indent=2))
    (k8s_dir / "deployment.yaml").write_text(_generate_k8s_deployment(pkg))

    # README with quick-start instructions
    safe_name = pkg.model_name.lower().replace("_", "-")
    feature_json_fields = ", ".join(f'"{s.name}": 0' for s in pkg.feature_specs)
    readme = (
        f"# {pkg.model_name}\n\n"
        f"Auto-generated by Crucible. Model: {pkg.model_family} | "
        f"{pkg.task_type} | {pkg.scoring_metric}: {pkg.best_score:.4f}\n\n"
        f"## Docker\n"
        f"```bash\n"
        f"docker build -t {safe_name} .\n"
        f"docker run -p 8000:8000 {safe_name}\n"
        f"```\n\n"
        f"## Kubernetes\n"
        f"```bash\n"
        f"docker build -t {safe_name}:latest .\n"
        f"kubectl apply -f k8s/deployment.yaml\n"
        f"```\n\n"
        f"## Predict\n"
        f"```bash\n"
        f"curl -X POST http://localhost:8000/predict \\\n"
        f"  -H 'Content-Type: application/json' \\\n"
        f"  -d '{{{feature_json_fields}}}'\n"
        f"```\n"
    )
    (root / "README.md").write_text(readme)

    # Zip
    zip_path = output_dir / f"{pkg.model_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in root.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(output_dir))

    return zip_path


# ── Type helpers ───────────────────────────────────────────────────────────

def _py_type(dtype: str) -> str:
    return {"float": "float", "int": "int", "str": "str"}.get(dtype, "float")


def _openapi_type(dtype: str) -> str:
    return {"float": "number", "int": "integer", "str": "string"}.get(dtype, "number")
