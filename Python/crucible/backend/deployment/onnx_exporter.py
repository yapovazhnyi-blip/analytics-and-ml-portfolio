"""
ONNX Exporter — converts trained Crucible models to ONNX interchange format.

WHY ONNX
--------
Crucible trains models with joblib-serialised sklearn / XGBoost / LightGBM.
These are runtime-specific: a joblib file requires Python + scikit-learn +
whichever library trained the model to be installed everywhere you deploy.

ONNX (Open Neural Network Exchange) is a vendor-neutral exchange format that
separates the model architecture from the runtime:

  Trained with sklearn  →  ONNX  →  Run with ONNX Runtime (C++)
                                  →  Run with ONNX Runtime (Go / Java / C#)
                                  →  Run with DirectML (Windows GPU)
                                  →  Run with CoreML (Apple Silicon)

Speed improvement on CPU:
  sklearn predict()    ≈  100ms/1000 samples
  ONNX Runtime         ≈  10–30ms/1000 samples   (3–10× faster)

The improvement comes from ONNX Runtime's graph-level optimisations (operator
fusion, memory planning, SIMD vectorisation) that scikit-learn's pure-Python
predict() does not perform.

SUPPORTED FAMILIES
------------------
sklearn families via skl2onnx:
  random_forest, gradient_boosting — RandomForestClassifier/Regressor,
  GradientBoostingClassifier/Regressor → TreeEnsembleClassifier/Regressor

  logistic_regression, ridge — LinearClassifier / LinearRegressor

  svm — SVMClassifier / SVMRegressor (LinearSVC / LinearSVR only;
        kernel SVM ONNX conversion requires onnxmltools)

  knn — KNNClassifier / KNNRegressor (skl2onnx 1.15+)

XGBoost and LightGBM via their native ONNX converters:
  xgboost — XGBClassifier/Regressor has a built-in save_model() that can
            write ONNX directly when onnxruntime is installed.

  lightgbm — via skl2onnx's LightGBM operator (skl2onnx 1.12+).

Pipeline handling:
  Models trained in a Pipeline (e.g. StandardScaler → LogisticRegression)
  are converted as a whole pipeline — the ONNX graph includes preprocessing.

LIMITATIONS
-----------
- Keras/TF models: tf2onnx conversion requires a separate install and is
  not included here. The caller receives a clear error message.
- Kernel SVM (RBF, poly): ONNX conversion requires onnxmltools which has
  heavier dependencies. Skipped with a clear message.
- Models with custom transformers in their Pipeline: only standard sklearn
  transformers are in the skl2onnx operator map.

GENERATED FILES
---------------
  {experiment_id}_model.onnx        — the ONNX graph
  {experiment_id}_onnx_server.py    — FastAPI inference server using ONNX Runtime
  {experiment_id}_onnx_readme.md    — how to run the ONNX server
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ONNXExportResult:
    onnx_path: str
    input_name: str
    output_names: list[str]
    n_features: int
    opset_version: int
    model_size_bytes: int
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def model_size_kb(self) -> float:
        return round(self.model_size_bytes / 1024, 1)


# ── Main exporter ─────────────────────────────────────────────────────────────

def export_to_onnx(
    artifact_path: str,
    feature_names: list[str],
    task_type: str,
    output_dir: str,
    experiment_id: int,
    opset: int = 17,
) -> ONNXExportResult:
    """
    Loads a Crucible model artifact and exports it to ONNX.

    Args:
        artifact_path:  Path to the joblib model file saved by TrainingRunner.
        feature_names:  List of feature column names (defines input shape).
        task_type:      "classification" or "regression".
        output_dir:     Directory to write the .onnx file into.
        experiment_id:  Used to name the output file.
        opset:          ONNX opset version. 17 is widely supported.

    Returns:
        ONNXExportResult with path and metadata, or error message.
    """
    import joblib

    try:
        model = joblib.load(artifact_path)
    except Exception as exc:
        return _error(f"Failed to load model artifact: {exc}")

    n_features = len(feature_names)
    onnx_filename = f"experiment_{experiment_id}_model.onnx"
    onnx_path = str(Path(output_dir) / onnx_filename)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    family_name = type(model).__name__.lower()

    # ── Dispatch by model type ────────────────────────────────────────────────

    if "xgb" in family_name:
        return _export_xgboost(model, onnx_path, n_features, task_type, opset)

    if "lgbm" in family_name:
        return _export_lightgbm(model, onnx_path, n_features, task_type, opset)

    if "keras" in family_name or "sequential" in family_name:
        return _error(
            "Keras/TensorFlow models require tf2onnx: pip install tf2onnx. "
            "After installing, run: python -m tf2onnx.convert --keras model.h5 --output model.onnx"
        )

    # Default: sklearn Pipeline or estimator via skl2onnx
    return _export_sklearn(model, onnx_path, n_features, task_type, opset)


# ── sklearn / Pipeline ────────────────────────────────────────────────────────

def _export_sklearn(model, onnx_path, n_features, task_type, opset) -> ONNXExportResult:
    """Converts any sklearn-compatible model to ONNX via skl2onnx."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        return _error("skl2onnx not installed: pip install skl2onnx")

    # The ONNX input type: N samples × n_features floats
    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    # Kernel SVM requires onnxmltools — detect early and give clear message
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    est_name = type(estimator).__name__
    if "SVC" in est_name or "SVR" in est_name:
        kernel = getattr(estimator, "kernel", "rbf")
        if kernel != "linear":
            return _error(
                f"Kernel SVM (kernel='{kernel}') requires onnxmltools for ONNX conversion. "
                "LinearSVC/LinearSVR work without onnxmltools. Retrain with kernel='linear' "
                "or install onnxmltools."
            )

    try:
        onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=opset)
    except Exception as exc:
        return _error(f"skl2onnx conversion failed: {exc}")

    return _write(onnx_model, onnx_path, n_features, opset)


# ── XGBoost ───────────────────────────────────────────────────────────────────

def _export_xgboost(model, onnx_path, n_features, task_type, opset) -> ONNXExportResult:
    """
    XGBoost native ONNX export.

    XGBoost's sklearn wrapper (XGBClassifier/XGBRegressor) has a built-in
    get_booster() that can export to ONNX when onnxmltools or the xgboost
    native ONNX support is available.

    We use skl2onnx's XGBoost operator (added in skl2onnx 1.12) as it's the
    most reliable path without requiring onnxmltools.
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType as OMLFloatTensor
        # onnxmltools path
        initial_type = [("float_input", OMLFloatTensor([None, n_features]))]
        onnx_model = convert_xgboost(model, initial_types=initial_type)
        return _write(onnx_model, onnx_path, n_features, opset)
    except ImportError:
        pass

    # Fallback: try skl2onnx directly (works for sklearn-compatible XGB wrapper)
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=opset)
        return _write(onnx_model, onnx_path, n_features, opset)
    except Exception as exc:
        return _error(
            f"XGBoost ONNX export failed: {exc}. "
            "Install onnxmltools for full XGBoost ONNX support: pip install onnxmltools"
        )


# ── LightGBM ──────────────────────────────────────────────────────────────────

def _export_lightgbm(model, onnx_path, n_features, task_type, opset) -> ONNXExportResult:
    """LightGBM to ONNX via skl2onnx (supported since skl2onnx 1.12)."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=opset)
        return _write(onnx_model, onnx_path, n_features, opset)
    except Exception as exc:
        return _error(
            f"LightGBM ONNX export failed: {exc}. "
            "Try: pip install onnxmltools lightgbm"
        )


# ── Write + validate ──────────────────────────────────────────────────────────

def _write(onnx_model, onnx_path, n_features, opset) -> ONNXExportResult:
    """Writes the ONNX model to disk and validates it with ONNX Runtime."""
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    model_size = os.path.getsize(onnx_path)

    # Validate: load with ONNX Runtime and run a dummy inference
    try:
        import onnxruntime as rt
        sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name  = sess.get_inputs()[0].name
        output_names = [o.name for o in sess.get_outputs()]

        dummy = np.zeros((1, n_features), dtype=np.float32)
        sess.run(output_names, {input_name: dummy})

        return ONNXExportResult(
            onnx_path=onnx_path,
            input_name=input_name,
            output_names=output_names,
            n_features=n_features,
            opset_version=opset,
            model_size_bytes=model_size,
        )
    except Exception as exc:
        return _error(f"ONNX Runtime validation failed: {exc}")


def _error(msg: str) -> ONNXExportResult:
    return ONNXExportResult(
        onnx_path="", input_name="", output_names=[],
        n_features=0, opset_version=0, model_size_bytes=0, error=msg,
    )


# ── ONNX Runtime inference server template ────────────────────────────────────

def generate_onnx_server(
    onnx_filename: str,
    feature_names: list[str],
    task_type: str,
    input_name: str,
    output_names: list[str],
) -> str:
    """
    Generates a self-contained FastAPI inference server that uses ONNX Runtime.
    No sklearn or XGBoost required at runtime — just onnxruntime.
    """
    predict_logic = (
        "predictions = outputs[0].tolist()"
        if task_type == "regression"
        else "predictions = outputs[0].tolist()\n    probabilities = outputs[1].tolist() if len(outputs) > 1 else None"
    )

    return f'''"""
ONNX Runtime inference server — generated by Crucible.

Requirements: pip install fastapi uvicorn onnxruntime numpy
Run: uvicorn server:app --host 0.0.0.0 --port 8080

Why ONNX Runtime instead of sklearn/XGBoost?
  - No framework-specific dependency at inference time
  - 3-10x faster on CPU via graph-level optimisations
  - Works in environments where the training framework is not installed
"""
import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ONNX_PATH    = "{onnx_filename}"
FEATURE_NAMES = {feature_names!r}
INPUT_NAME    = "{input_name}"
OUTPUT_NAMES  = {output_names!r}

app = FastAPI(title="Crucible ONNX Model", version="1.0.0")

# Load the ONNX model once at startup
_session = rt.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])


class PredictRequest(BaseModel):
    # Dict mapping feature name → value for a single sample
    features: dict[str, float]


class PredictResponse(BaseModel):
    prediction: object
    probabilities: list[float] | None = None


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    missing = set(FEATURE_NAMES) - set(body.features)
    if missing:
        raise HTTPException(422, detail=f"Missing features: {{missing}}")

    X = np.array([[body.features[f] for f in FEATURE_NAMES]], dtype=np.float32)
    outputs = _session.run(OUTPUT_NAMES, {{INPUT_NAME: X}})
    {predict_logic}

    return PredictResponse(prediction=predictions, probabilities=probabilities if "{task_type}" != "regression" else None)


@app.get("/health")
def health():
    return {{"status": "ok", "model": ONNX_PATH}}
'''


def generate_onnx_readme(
    experiment_id: int,
    onnx_filename: str,
    n_features: int,
    model_size_kb: float,
    family_name: str,
    opset: int,
) -> str:
    return f"""# ONNX Deployment — Experiment {experiment_id}

## Model
- **Original family**: {family_name}
- **ONNX file**: `{onnx_filename}` ({model_size_kb} KB)
- **Opset version**: {opset}
- **Input features**: {n_features}

## Install
```bash
pip install fastapi uvicorn onnxruntime numpy
```

## Run
```bash
uvicorn onnx_server:app --host 0.0.0.0 --port 8080
```

## Predict
```bash
curl -X POST http://localhost:8080/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"features": {{"feature_1": 1.0, "feature_2": 2.5}}}}'
```

## Why ONNX Runtime vs sklearn/joblib?
ONNX Runtime uses fused kernels and SIMD vectorisation that sklearn's
pure-Python predict() does not. On CPU, expect **3–10× lower latency**
especially for tree ensemble models (RandomForest, GradientBoosting, XGBoost, LightGBM).

## INT8 Quantisation (optional)
```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("{onnx_filename}", "model_int8.onnx", weight_type=QuantType.QInt8)
```
INT8 quantisation reduces the model file size by ~4× and further improves
throughput on CPUs with AVX-512 / VNNI support (Intel Ice Lake+, AMD Zen 4+).
"""
