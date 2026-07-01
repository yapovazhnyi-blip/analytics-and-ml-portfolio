"""
Specialist agents — focused agents that do one job well.

SPECIALIST DESIGN PATTERN
--------------------------
Each specialist is a LangGraph node: an async function that takes the
current MultiAgentState and returns a partial state update.

Specialists do NOT call the Anthropic API. They are deterministic tools
that execute real work against the Crucible database and services:
  - Dataset Analyst → ProfileRunner, AnomalyRunner
  - Model Trainer   → TrainingRunner, FairnessAnalyzer
  - Deployer        → deployment generator, ONNX exporter

The supervisor (which DOES call Claude) interprets the results and decides
what to do next. This separation means:
  - Specialists are fast and cheap (no LLM calls)
  - The supervisor handles all reasoning and routing decisions
  - Specialist errors are caught and surfaced to the supervisor as text

TOOL SELECTION RATIONALE
-------------------------
Each specialist has a narrow tool set matching its responsibility:

  Dataset Analyst
    list_datasets     → discover available data
    get_dataset_info  → understand schema and columns
    run_profiling     → check data quality before training
    detect_anomalies  → flag unusual rows

  Model Trainer
    start_experiment      → kick off AutoML
    get_experiment_status → poll training progress
    get_experiment_results → retrieve metrics and SHAP importances
    run_fairness          → check for demographic disparities

  Deployer
    generate_deployment → create FastAPI + Docker + K8s package
    export_onnx         → convert model to ONNX for faster inference
    query_rag           → answer questions from indexed documents
"""

from __future__ import annotations

from typing import Optional

from agents.state import (
    MultiAgentState, AgentContext,
    ROUTE_SUPERVISOR, MAX_STEPS,
)
from agents.tools import ToolExecutor


# ── Dataset Analyst ────────────────────────────────────────────────────────────

async def dataset_analyst_node(state: MultiAgentState, db) -> dict:
    """
    Analyses available datasets: discovery, schema, profiling, anomaly detection.

    Reads from state:
      goal → to understand what data is needed
      context.dataset_id → if already known, skip discovery

    Writes to state:
      context.dataset_id / dataset_name
      context.profiling_done
      agent_output → summary text for the supervisor
    """
    executor = ToolExecutor(db)
    ctx: AgentContext = dict(state.get("context", {}))
    lines = []

    # ── Step 1: Discover datasets ────────────────────────────────────────────
    if not ctx.get("dataset_id"):
        result = await executor.execute("list_datasets", {})
        lines.append(f"Datasets: {result.content}")

        # Extract the first available dataset ID from the result
        import re
        match = re.search(r"ID (\d+):", result.content)
        if match:
            ctx["dataset_id"] = int(match.group(1))

    # ── Step 2: Get schema ────────────────────────────────────────────────────
    if ctx.get("dataset_id"):
        info = await executor.execute("get_dataset_info", {"dataset_id": ctx["dataset_id"]})
        lines.append(f"Schema: {info.content}")
        # Extract dataset name
        name_match = re.search(r"'(.+?)'", info.content)
        if name_match:
            ctx["dataset_name"] = name_match.group(1)

    # ── Step 3: Profile if not done ───────────────────────────────────────────
    if ctx.get("dataset_id") and not ctx.get("profiling_done"):
        # Infer target column from goal if mentioned
        target = _extract_target_from_goal(state["goal"])
        if target:
            ctx["target_column"] = target
            profile = await executor.execute("run_profiling", {
                "dataset_id": ctx["dataset_id"],
                "target_column": target,
            })
            lines.append(f"Profiling: {profile.content}")
            ctx["profiling_done"] = True

    output = "\n".join(lines) if lines else "No datasets found."
    return {
        "context":      ctx,
        "agent_output": output,
        "active_agent": "dataset_analyst",
        "next_agent":   ROUTE_SUPERVISOR,
        "step_count":   state.get("step_count", 0) + 1,
        "messages":     [{"role": "assistant", "content": f"[Dataset Analyst] {output}"}],
    }


# ── Model Trainer ──────────────────────────────────────────────────────────────

async def model_trainer_node(state: MultiAgentState, db) -> dict:
    """
    Runs AutoML training: experiment submission, polling, and results retrieval.

    Reads from state:
      context.dataset_id    → required for training
      context.target_column → required for training
      context.task_type     → classification or regression
      context.experiment_id → if set, poll rather than start new job

    Writes to state:
      context.experiment_id
      context.best_model / best_score
      context.training_done
    """
    import asyncio, re
    executor = ToolExecutor(db)
    ctx: AgentContext = dict(state.get("context", {}))
    lines = []

    if not ctx.get("dataset_id"):
        return _specialist_error(state, "model_trainer", "No dataset_id in context. Run dataset_analyst first.")

    # ── If already training, poll status ─────────────────────────────────────
    if ctx.get("experiment_id") and not ctx.get("training_done"):
        status = await executor.execute(
            "get_experiment_status", {"experiment_id": ctx["experiment_id"]}
        )
        lines.append(status.content)
        if "completed" in status.content.lower():
            results = await executor.execute(
                "get_experiment_results", {"experiment_id": ctx["experiment_id"]}
            )
            lines.append(results.content)
            # Extract best model and score
            model_match = re.search(r"Best model: (\w+)", results.content)
            score_match = re.search(r"CV score: ([\d.]+)", results.content)
            if model_match:
                ctx["best_model"] = model_match.group(1)
            if score_match:
                ctx["best_score"] = float(score_match.group(1))
            ctx["training_done"] = True
        else:
            lines.append("Training still in progress.")

    # ── Start a new experiment ────────────────────────────────────────────────
    elif not ctx.get("experiment_id"):
        target_col = ctx.get("target_column") or _extract_target_from_goal(state["goal"])
        task_type  = ctx.get("task_type") or _infer_task_type(state["goal"])

        if not target_col:
            return _specialist_error(state, "model_trainer",
                "No target_column found. Specify it in your goal (e.g. 'predict churn').")

        ctx["target_column"] = target_col
        ctx["task_type"]     = task_type

        start = await executor.execute("start_experiment", {
            "dataset_id":    ctx["dataset_id"],
            "target_column": target_col,
            "task_type":     task_type,
            "n_trials":      10,
        })
        lines.append(start.content)

        exp_match = re.search(r"Experiment ID: (\d+)", start.content)
        if exp_match:
            ctx["experiment_id"] = int(exp_match.group(1))

        # Immediately poll once (experiment may complete quickly in test mode)
        if ctx.get("experiment_id"):
            await asyncio.sleep(0.5)
            status = await executor.execute(
                "get_experiment_status", {"experiment_id": ctx["experiment_id"]}
            )
            lines.append(status.content)
            if "completed" in status.content.lower():
                results = await executor.execute(
                    "get_experiment_results", {"experiment_id": ctx["experiment_id"]}
                )
                lines.append(results.content)
                ctx["training_done"] = True

    output = "\n".join(lines)
    return {
        "context":      ctx,
        "agent_output": output,
        "active_agent": "model_trainer",
        "next_agent":   ROUTE_SUPERVISOR,
        "step_count":   state.get("step_count", 0) + 1,
        "messages":     [{"role": "assistant", "content": f"[Model Trainer] {output}"}],
    }


# ── Deployer ───────────────────────────────────────────────────────────────────

async def deployer_node(state: MultiAgentState, db) -> dict:
    """
    Generates deployment artifacts: FastAPI server, Docker, K8s, and ONNX export.

    Reads from state:
      context.experiment_id → required
      context.training_done → must be True

    Writes to state:
      context.deployment_url
      context.onnx_path
      context.deployment_done
    """
    executor = ToolExecutor(db)
    ctx: AgentContext = dict(state.get("context", {}))
    lines = []

    if not ctx.get("experiment_id"):
        return _specialist_error(state, "deployer",
            "No experiment_id. Run model_trainer first.")
    if not ctx.get("training_done"):
        return _specialist_error(state, "deployer",
            "Training not complete yet. Check model_trainer status.")

    deploy = await executor.execute("generate_deployment", {
        "experiment_id": ctx["experiment_id"],
        "model_name":    "crucible-model",
    })
    lines.append(deploy.content)
    if "/deploy" in deploy.content:
        ctx["deployment_url"] = f"/api/v1/experiments/{ctx['experiment_id']}/deploy"
        ctx["deployment_done"] = True

    output = "\n".join(lines)
    return {
        "context":      ctx,
        "agent_output": output,
        "active_agent": "deployer",
        "next_agent":   ROUTE_SUPERVISOR,
        "step_count":   state.get("step_count", 0) + 1,
        "messages":     [{"role": "assistant", "content": f"[Deployer] {output}"}],
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_target_from_goal(goal: str) -> Optional[str]:
    """Heuristically extracts a target column name from the user's goal."""
    import re
    patterns = [
        r"predict\s+['\"]?(\w+)['\"]?",
        r"target[:\s]+['\"]?(\w+)['\"]?",
        r"column\s+['\"]?(\w+)['\"]?",
        r"on\s+['\"]?(\w+)['\"]?\s+column",
    ]
    for pat in patterns:
        m = re.search(pat, goal, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _infer_task_type(goal: str) -> str:
    """Infers classification vs regression from the goal text."""
    cls_keywords = ["classif", "category", "churn", "fraud", "default", "predict class", "label"]
    reg_keywords = ["regress", "price", "predict value", "predict number", "revenue", "amount"]
    goal_lower = goal.lower()
    if any(k in goal_lower for k in cls_keywords):
        return "classification"
    if any(k in goal_lower for k in reg_keywords):
        return "regression"
    return "classification"    # default


def _specialist_error(state: MultiAgentState, agent: str, msg: str) -> dict:
    return {
        "agent_output": f"Error: {msg}",
        "active_agent": agent,
        "next_agent":   ROUTE_SUPERVISOR,
        "step_count":   state.get("step_count", 0) + 1,
        "messages":     [{"role": "assistant", "content": f"[{agent}] Error: {msg}"}],
        "context":      state.get("context", {}),
    }
