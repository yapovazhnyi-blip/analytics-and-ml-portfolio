"""
Agent Tools — Crucible's API surface as callable tools for the ReAct agent.

WHAT TOOLS ARE
--------------
Tools are the bridge between the LLM's reasoning and real-world actions.
Each tool has:
  - A name (snake_case, unique)
  - A description (what it does and when to use it — the LLM reads this)
  - An input schema (JSON Schema defining what parameters it accepts)
  - An executor function (the actual Python code that runs)

The Anthropic API's native tool_use feature handles the protocol:
  1. Claude receives the list of available tools
  2. Claude decides which tool to call and with what inputs
  3. The API returns a tool_use block (tool name + inputs)
  4. We execute the tool and return a tool_result block
  5. Claude reads the result and decides what to do next

WHY DIRECT FUNCTION CALLS OVER HTTP
-------------------------------------
The agent runs inside the same FastAPI process. Tools call the database
and service functions directly rather than making HTTP requests to
localhost. This is faster (no TCP round-trip) and avoids auth token
management (the agent session already has the DB connection).

AVAILABLE TOOLS
---------------
list_datasets           — See what data is available for analysis
get_dataset_info        — Schema, row count, column names and types
run_profiling           — Deep analysis: missingness, leakage, distributions
start_experiment        — Run AutoML training on a dataset
get_experiment_status   — Poll training progress (running/completed/failed)
get_experiment_results  — Best model, metrics, top feature importances
generate_deployment     — Create FastAPI + Docker + K8s deployment package
list_rag_documents      — See indexed documents for Q&A
query_rag               — Ask a question about indexed documents
export_onnx             — Convert the best model to ONNX for faster inference
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional


# ── Tool schema type ──────────────────────────────────────────────────────────

@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict


@dataclass
class ToolResult:
    tool_name: str
    content: str       # always string — Claude reads this as text
    is_error: bool = False


# ── Tool definitions ──────────────────────────────────────────────────────────

LIST_DATASETS = Tool(
    name="list_datasets",
    description=(
        "Lists all datasets that are ready for analysis in Crucible. "
        "Call this first to discover what data is available. "
        "Returns dataset IDs, names, row counts, column counts, and status. "
        "Use the dataset ID from this response in other tools."
    ),
    input_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
)

GET_DATASET_INFO = Tool(
    name="get_dataset_info",
    description=(
        "Returns detailed schema information for a specific dataset: "
        "column names, data types, and a sample of the first few rows. "
        "Use this to understand the data structure before choosing target columns."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "dataset_id": {
                "type": "integer",
                "description": "The dataset ID from list_datasets.",
            }
        },
        "required": ["dataset_id"],
    },
)

RUN_PROFILING = Tool(
    name="run_profiling",
    description=(
        "Runs deep statistical profiling on a dataset. "
        "Returns: missingness analysis, multicollinearity (VIF), "
        "data leakage detection, and target distribution. "
        "Always run this before starting an experiment to catch data quality issues. "
        "Requires a target_column to analyse the relationship between features and target."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "target_column": {
                "type": "string",
                "description": "Column to predict. Must be a column name in the dataset.",
            },
        },
        "required": ["dataset_id", "target_column"],
    },
)

START_EXPERIMENT = Tool(
    name="start_experiment",
    description=(
        "Starts an AutoML experiment: trains multiple model families "
        "(Random Forest, XGBoost, LightGBM, Logistic Regression, SVM, k-NN, Neural Network) "
        "using Optuna hyperparameter search and cross-validation. "
        "Returns a job_id immediately — the training runs in the background. "
        "Use get_experiment_status to poll for completion."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "target_column": {
                "type": "string",
                "description": "Column to predict.",
            },
            "task_type": {
                "type": "string",
                "enum": ["classification", "regression"],
                "description": "classification for categorical targets, regression for numeric targets.",
            },
            "n_trials": {
                "type": "integer",
                "description": "Number of Optuna hyperparameter search trials. Default 20. More = slower but better.",
                "default": 20,
            },
        },
        "required": ["dataset_id", "target_column", "task_type"],
    },
)

GET_EXPERIMENT_STATUS = Tool(
    name="get_experiment_status",
    description=(
        "Checks the status of a running or completed experiment. "
        "Status values: 'running' (still training), 'completed' (done), 'failed' (error). "
        "When status is 'running', wait a moment and check again. "
        "When 'completed', call get_experiment_results to see the metrics."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "experiment_id": {
                "type": "integer",
                "description": "The experiment ID returned by start_experiment.",
            }
        },
        "required": ["experiment_id"],
    },
)

GET_EXPERIMENT_RESULTS = Tool(
    name="get_experiment_results",
    description=(
        "Returns the full results of a completed experiment: "
        "best model family, cross-validation score, holdout metrics (accuracy/F1/R²), "
        "and top 10 feature importances (SHAP values). "
        "Only call this when get_experiment_status returns 'completed'."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "experiment_id": {"type": "integer"},
        },
        "required": ["experiment_id"],
    },
)

GENERATE_DEPLOYMENT = Tool(
    name="generate_deployment",
    description=(
        "Generates a deployment package for a completed experiment: "
        "a FastAPI prediction endpoint, Dockerfile, Kubernetes manifests, "
        "and an OpenAPI spec. Returns a download URL. "
        "Use this to get the model production-ready after training."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "experiment_id": {"type": "integer"},
            "model_name": {
                "type": "string",
                "description": "Name for the deployed model (used in K8s resource names). "
                               "Alphanumeric and hyphens only.",
                "default": "crucible-model",
            },
        },
        "required": ["experiment_id"],
    },
)

LIST_RAG_DOCUMENTS = Tool(
    name="list_rag_documents",
    description=(
        "Lists all documents indexed in the RAG (Retrieval-Augmented Generation) pipeline. "
        "Returns document IDs, names, chunk counts, and indexing status. "
        "Use document IDs with query_rag to answer questions about specific documents."
    ),
    input_schema={"type": "object", "properties": {}, "required": []},
)

QUERY_RAG = Tool(
    name="query_rag",
    description=(
        "Answers a question by searching through indexed documents. "
        "Uses hybrid BM25 + semantic search to find relevant passages, "
        "then generates a grounded answer with source citations. "
        "Use this to answer questions about documentation, reports, or knowledge bases."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to answer from the indexed documents.",
            },
            "document_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: restrict search to specific document IDs. "
                               "Leave empty to search all documents.",
            },
        },
        "required": ["question"],
    },
)

# ── Registry ──────────────────────────────────────────────────────────────────

ALL_TOOLS: list[Tool] = [
    LIST_DATASETS,
    GET_DATASET_INFO,
    RUN_PROFILING,
    START_EXPERIMENT,
    GET_EXPERIMENT_STATUS,
    GET_EXPERIMENT_RESULTS,
    GENERATE_DEPLOYMENT,
    LIST_RAG_DOCUMENTS,
    QUERY_RAG,
]

TOOL_MAP: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}


def tools_for_api() -> list[dict]:
    """Returns tool definitions in Anthropic API format."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
        }
        for t in ALL_TOOLS
    ]


# ── Tool executors ────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Executes agent tool calls against the Crucible database and services.

    Uses direct function/service calls rather than HTTP requests for speed.
    Requires an active AsyncSession for database operations.
    """

    def __init__(self, db):
        self.db = db

    async def execute(self, tool_name: str, tool_input: dict) -> ToolResult:
        """Dispatches to the correct tool executor, wrapped in an OTel span."""
        from observability.tracing import start_span

        with start_span(f"agent.tool.{tool_name}", {"tool_input_keys": ",".join(tool_input.keys())}) as span:
            try:
                fn = getattr(self, f"_run_{tool_name}", None)
                if fn is None:
                    span.set_attribute("error", True)
                    return ToolResult(
                        tool_name=tool_name,
                        content=f"Unknown tool: {tool_name!r}",
                        is_error=True,
                    )
                result = await fn(**tool_input)
                span.set_attribute("result_length", len(result))
                return ToolResult(tool_name=tool_name, content=result)
            except Exception as exc:
                span.set_attribute("error", True)
                return ToolResult(
                    tool_name=tool_name,
                    content=f"Tool error: {exc}",
                    is_error=True,
                )

    # ── list_datasets ─────────────────────────────────────────────────────────

    async def _run_list_datasets(self) -> str:
        from sqlalchemy import select
        from models.dataset import Dataset
        rows = await self.db.scalars(
            select(Dataset).where(Dataset.status == "ready").order_by(Dataset.created_at.desc()).limit(20)
        )
        datasets = rows.all()
        if not datasets:
            return "No ready datasets found. Upload a dataset first via POST /api/v1/datasets/upload."
        items = [
            f"ID {d.id}: '{d.name}' — {d.row_count or '?'} rows, "
            f"{d.source_type} format, created {d.created_at.date()}"
            for d in datasets
        ]
        return "Available datasets:\n" + "\n".join(items)

    # ── get_dataset_info ──────────────────────────────────────────────────────

    async def _run_get_dataset_info(self, dataset_id: int) -> str:
        from models.dataset import Dataset
        ds = await self.db.get(Dataset, dataset_id)
        if not ds:
            return f"Dataset {dataset_id} not found."
        schema = json.loads(ds.schema_json) if ds.schema_json else {}
        cols = schema.get("columns", {})
        col_summary = "\n".join(
            f"  - {name}: {info.get('dtype', 'unknown')}"
            for name, info in list(cols.items())[:30]
        )
        return (
            f"Dataset '{ds.name}' (ID {ds.id}):\n"
            f"  Rows: {ds.row_count or 'unknown'}\n"
            f"  Columns ({len(cols)}):\n{col_summary}"
        )

    # ── run_profiling ─────────────────────────────────────────────────────────

    async def _run_run_profiling(self, dataset_id: int, target_column: str) -> str:
        from models.dataset import Dataset
        from profiling.runner import ProfileRunner
        import asyncio

        ds = await self.db.get(Dataset, dataset_id)
        if not ds:
            return f"Dataset {dataset_id} not found."
        if not ds.file_path:
            return "Dataset has no file path."

        loop = asyncio.get_event_loop()
        def _profile():
            df = ProfileRunner.load_dataframe(ds.file_path, ds.source_type)
            runner = ProfileRunner()
            import asyncio as _asyncio
            return _asyncio.run(runner.run(df, dataset_id, target_column=target_column))

        report = await loop.run_in_executor(None, lambda: _sync_profile(ds, target_column))
        return _format_profile_summary(report)

    # ── start_experiment ──────────────────────────────────────────────────────

    async def _run_start_experiment(
        self,
        dataset_id: int,
        target_column: str,
        task_type: str,
        n_trials: int = 20,
    ) -> str:
        from models.dataset import Dataset
        from models.experiment import Experiment
        from training.runner import TrainingConfig
        from jobs.manager import start_job
        import uuid, asyncio

        ds = await self.db.get(Dataset, dataset_id)
        if not ds:
            return f"Dataset {dataset_id} not found."
        if ds.status != "ready":
            return f"Dataset not ready (status: {ds.status})."

        exp = Experiment(
            dataset_id=dataset_id,
            target_column=target_column,
            task_type=task_type,
            status="running",
            job_id=str(uuid.uuid4()),
            n_trials=n_trials,
        )
        self.db.add(exp)
        await self.db.flush()
        await self.db.refresh(exp)

        from profiling.runner import ProfileRunner
        from training.runner import TrainingRunner, TrainingConfig as TC
        from config import settings
        import asyncio

        exp_id = exp.id
        job_id = exp.job_id

        async def _train():
            from database import AsyncSessionLocal
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None, ProfileRunner.load_dataframe, ds.file_path, ds.source_type
            )
            config = TC(n_trials=n_trials, cv_folds=3)
            runner = TrainingRunner(settings.model_storage_path)

            from jobs.manager import get_progress_queue
            queue = get_progress_queue(job_id)

            result = await runner.run_async(df, target_column, task_type, config, queue)
            async with AsyncSessionLocal() as session:
                e = await session.get(Experiment, exp_id)
                if e:
                    if hasattr(result, "error") and result.error:
                        e.status = "failed"
                        e.error_message = str(result.error)
                    else:
                        e.status = "completed"
                        e.best_family = result.best_family
                        e.best_cv_score = result.best_cv_score
                        e.artifact_path = result.artifact_path
                        import json as _json
                        e.results_json = _json.dumps({
                            "feature_names": result.feature_names,
                            "holdout_metrics": result.holdout_metrics,
                            "best_params": result.best_params,
                        })
                    await session.commit()

        asyncio.create_task(_train())

        return (
            f"Experiment started successfully!\n"
            f"  Experiment ID: {exp.id}\n"
            f"  Dataset: '{ds.name}'\n"
            f"  Target: {target_column}\n"
            f"  Task: {task_type}\n"
            f"  Trials: {n_trials}\n"
            f"Use get_experiment_status(experiment_id={exp.id}) to check progress."
        )

    # ── get_experiment_status ─────────────────────────────────────────────────

    async def _run_get_experiment_status(self, experiment_id: int) -> str:
        from models.experiment import Experiment
        exp = await self.db.get(Experiment, experiment_id)
        if not exp:
            return f"Experiment {experiment_id} not found."
        if exp.status == "running":
            return f"Experiment {experiment_id} is still running. Check again in a few seconds."
        if exp.status == "failed":
            return f"Experiment {experiment_id} failed: {exp.error_message}"
        if exp.status == "completed":
            return (
                f"Experiment {experiment_id} completed!\n"
                f"  Best model: {exp.best_family}\n"
                f"  CV score: {exp.best_cv_score:.4f}\n"
                f"Call get_experiment_results(experiment_id={experiment_id}) for full metrics."
            )
        return f"Experiment {experiment_id} status: {exp.status}"

    # ── get_experiment_results ────────────────────────────────────────────────

    async def _run_get_experiment_results(self, experiment_id: int) -> str:
        from models.experiment import Experiment
        exp = await self.db.get(Experiment, experiment_id)
        if not exp:
            return f"Experiment {experiment_id} not found."
        if exp.status != "completed":
            return f"Experiment {experiment_id} is not completed (status: {exp.status})."

        results = json.loads(exp.results_json) if exp.results_json else {}
        holdout = results.get("holdout_metrics", {})
        features = results.get("feature_names", [])

        metrics_str = "\n".join(f"  {k}: {v:.4f}" for k, v in holdout.items()) if holdout else "  (none)"
        top_features = features[:10] if features else []
        feat_str = ", ".join(top_features) if top_features else "(none)"

        return (
            f"Experiment {experiment_id} results:\n"
            f"  Best model: {exp.best_family}\n"
            f"  CV score: {exp.best_cv_score:.4f}\n"
            f"  Holdout metrics:\n{metrics_str}\n"
            f"  Top features: {feat_str}"
        )

    # ── generate_deployment ───────────────────────────────────────────────────

    async def _run_generate_deployment(
        self, experiment_id: int, model_name: str = "crucible-model"
    ) -> str:
        from models.experiment import Experiment
        exp = await self.db.get(Experiment, experiment_id)
        if not exp:
            return f"Experiment {experiment_id} not found."
        if exp.status != "completed":
            return f"Experiment {experiment_id} is not completed."
        return (
            f"Deployment package ready.\n"
            f"  Download URL: /api/v1/experiments/{experiment_id}/deploy\n"
            f"  Contents: FastAPI endpoint, Dockerfile, K8s manifests, OpenAPI spec\n"
            f"  Model: {exp.best_family} (experiment {experiment_id})\n"
            f"  To deploy: docker build -t {model_name} . && docker run -p 8080:8080 {model_name}"
        )

    # ── list_rag_documents ────────────────────────────────────────────────────

    async def _run_list_rag_documents(self) -> str:
        from sqlalchemy import select
        from models.rag_document import RAGDocument
        rows = await self.db.scalars(
            select(RAGDocument).where(RAGDocument.status == "ready").order_by(RAGDocument.created_at.desc()).limit(20)
        )
        docs = rows.all()
        if not docs:
            return "No RAG documents indexed. Upload documents via POST /api/v1/rag/documents."
        items = [
            f"ID {d.document_id}: '{d.name}' — {d.chunk_count} chunks, {d.chunk_strategy} strategy"
            for d in docs
        ]
        return "Indexed RAG documents:\n" + "\n".join(items)

    # ── query_rag ─────────────────────────────────────────────────────────────

    async def _run_query_rag(
        self, question: str, document_ids: Optional[list[str]] = None
    ) -> str:
        from rag.pipeline import RAGPipeline, RAGConfig
        from config import settings
        import os

        pipeline = RAGPipeline(
            vector_store_dir=os.path.join(settings.dataset_storage_path, "..", "rag"),
            config=RAGConfig(),
        )
        result = await pipeline.query(
            question=question,
            k=5,
            document_ids=document_ids or None,
        )
        if not result.succeeded:
            return f"RAG query failed: {result.error}"
        citations = ", ".join(c.source_name for c in result.citations) if result.citations else "none"
        return f"Answer: {result.answer}\n\nSources: {citations}"


# ── Profile helper (sync wrapper) ─────────────────────────────────────────────

def _sync_profile(ds, target_column: str):
    """Synchronous profile runner for use in run_in_executor."""
    import asyncio
    from profiling.runner import ProfileRunner

    df = ProfileRunner.load_dataframe(ds.file_path, ds.source_type)
    runner = ProfileRunner()
    return asyncio.run(runner.run(df, ds.id, target_column=target_column))


def _format_profile_summary(report) -> str:
    """Formats a profiling report into a concise text summary for the agent."""
    if report is None:
        return "Profiling failed or returned no data."

    lines = ["Profiling summary:"]

    # Missing data
    missing = getattr(report, "missing", None) or {}
    if missing:
        high_missing = {k: v for k, v in missing.items() if v > 0.1}
        if high_missing:
            lines.append(f"  ⚠ High missing data: {', '.join(f'{k} ({v:.0%})' for k, v in list(high_missing.items())[:5])}")
        else:
            lines.append("  ✓ No significant missing data")

    # Leakage
    leakage = getattr(report, "leakage_warnings", []) or []
    if leakage:
        lines.append(f"  ⚠ Leakage warnings ({len(leakage)}): {', '.join(str(w) for w in leakage[:3])}")

    # Target
    target = getattr(report, "target_summary", None)
    if target:
        lines.append(f"  Target distribution: {target}")

    return "\n".join(lines) if len(lines) > 1 else "Profiling completed. No major issues found."
