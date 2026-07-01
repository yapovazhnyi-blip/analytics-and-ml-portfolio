"""
Retraining Pipeline Runner — executes one RetrainingPolicy end-to-end.

This is the actual DAG execution engine: drift check → gate → retrain →
promotion decision. Each step's outcome is appended to RetrainingRun.steps_json
as it happens, so a run in progress (or a failed run) shows exactly how far
it got and why — the same transparency an Airflow task-instance log gives you,
without needing a separate Airflow deployment.

Reuses Crucible's existing infrastructure rather than reimplementing it:
  - drift/detector.py:compare_datasets() for the drift check
  - jobs/manager.py:start_job() for retraining — the SAME code path
    POST /experiments uses, so a pipeline-triggered retrain produces a
    perfectly normal Experiment row, visible in the regular experiments list
  - training/runner.py's "higher score = better" invariant for promotion
    (see models/retraining.py docstring for why this works universally)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from drift.detector import compare_datasets
from jobs.manager import start_job, get_job, JobStatus
from models.dataset import Dataset
from models.experiment import Experiment
from models.retraining import RetrainingPolicy, RetrainingRun
from profiling.runner import ProfileRunner
from training.runner import TrainingConfig

SEVERITY_ORDER = {"stable": 0, "slight": 1, "significant": 2, "critical": 3}
POLL_INTERVAL_SECS = 2


def _step(steps: list[dict], name: str, status: str, detail: str = "") -> None:
    """Appends a step record. Mutates the passed-in list in place."""
    steps.append({
        "step": name,
        "status": status,     # "running" | "completed" | "skipped" | "failed"
        "detail": detail,
        "timestamp": time.time(),
    })


async def run_pipeline(
    policy: RetrainingPolicy,
    current_dataset_id: int,
    db,
) -> RetrainingRun:
    """
    Executes the full retraining pipeline for one policy against one
    "current data" dataset, and persists a RetrainingRun audit record.

    This function is the single source of truth for the pipeline logic —
    called identically whether triggered manually (POST .../run) or by
    the APScheduler-driven recurring check (retraining/scheduler.py).
    """
    start = time.monotonic()
    steps: list[dict] = []

    run = RetrainingRun(
        policy_id=policy.id,
        current_dataset_id=current_dataset_id,
        status="running",
        steps_json="[]",
    )
    db.add(run)
    await db.flush()
    await db.refresh(run)

    def _save_steps():
        run.steps_json = json.dumps(steps)

    # ── Step 1: Drift check ──────────────────────────────────────────────────
    _step(steps, "drift_check", "running")
    _save_steps()
    await db.flush()

    reference_ds = await db.get(Dataset, policy.reference_dataset_id)
    current_ds = await db.get(Dataset, current_dataset_id)

    if not reference_ds or not current_ds:
        _step(steps, "drift_check", "failed", "Reference or current dataset not found")
        run.status = "failed"
        run.error_message = "Reference or current dataset not found"
        _save_steps()
        run.elapsed_secs = round(time.monotonic() - start, 2)
        await db.flush()
        return run

    try:
        reference_df = ProfileRunner.load_dataframe(reference_ds.file_path, reference_ds.source_type)
        current_df = ProfileRunner.load_dataframe(current_ds.file_path, current_ds.source_type)
        drift_report = compare_datasets(
            reference_df, current_df,
            reference_id=policy.reference_dataset_id,
            current_id=current_dataset_id,
            target_col=policy.target_column,
        )
    except Exception as exc:
        _step(steps, "drift_check", "failed", str(exc))
        run.status = "failed"
        run.error_message = f"Drift check failed: {exc}"
        _save_steps()
        run.elapsed_secs = round(time.monotonic() - start, 2)
        await db.flush()
        return run

    run.drift_checked = True
    run.drift_report_json = json.dumps(drift_report.to_dict())

    trigger_rank = SEVERITY_ORDER.get(policy.drift_severity_trigger, 2)
    observed_rank = SEVERITY_ORDER.get(drift_report.severity, 0)
    run.drift_detected = observed_rank >= trigger_rank

    _step(steps, "drift_check", "completed",
          f"severity={drift_report.severity} ({drift_report.n_features_drifted}/{drift_report.n_features_checked} features drifted)")
    _save_steps()
    await db.flush()

    # ── Gate: stop here if drift doesn't meet the trigger threshold ─────────
    if not run.drift_detected:
        _step(steps, "retrain", "skipped",
              f"Drift severity '{drift_report.severity}' below trigger threshold '{policy.drift_severity_trigger}'")
        run.status = "completed"
        run.retrain_triggered = False
        _save_steps()
        run.elapsed_secs = round(time.monotonic() - start, 2)
        await db.flush()
        return run

    # ── Step 2: Retrain ───────────────────────────────────────────────────────
    _step(steps, "retrain", "running")
    _save_steps()
    await db.flush()
    run.retrain_triggered = True

    try:
        feature_cols = [
            c for c in current_df.columns
            if c != policy.target_column and current_df[c].dtype.kind in "iufcb"
        ]
        if not feature_cols:
            raise ValueError("No numeric feature columns found in current dataset")

        new_exp = Experiment(
            name=f"{policy.name} — auto-retrain {time.strftime('%Y-%m-%d %H:%M')}",
            dataset_id=current_dataset_id,
            target_column=policy.target_column,
            task_type=policy.task_type,
            training_config=json.dumps({
                "n_trials": policy.n_trials, "cv_folds": policy.cv_folds,
                "triggered_by": "retraining_pipeline", "policy_id": policy.id,
            }),
            status="running",
            lifecycle_stage="candidate",
        )
        db.add(new_exp)
        await db.flush()
        await db.refresh(new_exp)
        run.new_experiment_id = new_exp.id

        config = TrainingConfig(n_trials=policy.n_trials, cv_folds=policy.cv_folds)
        job_id = await start_job(
            experiment_id=new_exp.id,
            df=current_df,
            target_column=policy.target_column,
            task_type=policy.task_type,
            config=config,
            feature_names=feature_cols,
            run_shap=False,   # pipeline runs prioritise speed; SHAP can be requested separately later
        )
        new_exp.preprocessing_config = json.dumps({"job_id": job_id, "feature_cols": feature_cols})
        await db.flush()

        # Poll until the job reaches a terminal state — same pattern as
        # routers/experiments.py:_update_experiment_on_complete
        job = get_job(job_id)
        while job and job.status not in (JobStatus.COMPLETE, JobStatus.ERROR):
            await asyncio.sleep(POLL_INTERVAL_SECS)
            job = get_job(job_id)

        if not job or job.status != JobStatus.COMPLETE or not job.result:
            error_msg = job.error if job and job.error else "Training job failed or disappeared"
            new_exp.status = "error"
            new_exp.error_message = error_msg
            _step(steps, "retrain", "failed", error_msg)
            run.status = "failed"
            run.error_message = error_msg
            _save_steps()
            run.elapsed_secs = round(time.monotonic() - start, 2)
            await db.flush()
            return run

        r = job.result
        new_exp.status = "complete"
        new_exp.best_model_family = r.best_family
        new_exp.best_score = r.best_cv_score
        new_exp.scoring_metric = r.scoring_metric
        new_exp.n_trials_completed = r.n_trials_completed
        new_exp.n_trials_pruned = r.n_trials_pruned
        new_exp.training_duration_secs = r.elapsed_secs
        new_exp.model_artifact_path = r.artifact_path
        new_exp.mlflow_run_id = r.mlflow_run_id
        new_exp.results_json = json.dumps({
            "holdout_metrics":     r.holdout_metrics,
            "best_params":         r.best_params,
            "calibration_applied": r.calibration_applied,
            "calibration_method":  r.calibration_method,
            "pruner_type":         r.pruner_type,
        })
        await db.flush()

        _step(steps, "retrain", "completed",
              f"new candidate: {r.best_family} scored {r.best_cv_score:.4f}")
        _save_steps()
        await db.flush()

    except Exception as exc:
        _step(steps, "retrain", "failed", str(exc))
        run.status = "failed"
        run.error_message = f"Retraining failed: {exc}"
        _save_steps()
        run.elapsed_secs = round(time.monotonic() - start, 2)
        await db.flush()
        return run

    # ── Step 3: Promotion decision ───────────────────────────────────────────
    _step(steps, "promotion_check", "running")
    _save_steps()
    await db.flush()

    current_production: Optional[Experiment] = None
    if policy.production_experiment_id:
        current_production = await db.get(Experiment, policy.production_experiment_id)

    new_score = new_exp.best_score or 0.0

    if current_production is None:
        promote = True
        reason = f"No existing production model for this policy — promoting first candidate (score {new_score:.4f})."
    else:
        old_score = current_production.best_score or 0.0
        promote = new_score >= (old_score + policy.promotion_margin)
        reason = (
            f"New candidate scored {new_score:.4f} vs. production's {old_score:.4f} "
            f"(required margin: +{policy.promotion_margin:.4f}). "
            f"{'Promoted.' if promote else 'Not promoted — production model retained.'}"
        )

    run.promotion_reason = reason
    run.promoted = promote

    if promote:
        if current_production:
            current_production.lifecycle_stage = "archived"
        new_exp.lifecycle_stage = "production"
        policy.production_experiment_id = new_exp.id
        # The newly-promoted model's training data becomes the new drift baseline
        policy.reference_dataset_id = current_dataset_id
        _step(steps, "promotion_check", "completed", "promoted to production")
    else:
        new_exp.lifecycle_stage = "candidate"
        _step(steps, "promotion_check", "completed", "candidate rejected, production unchanged")

    run.status = "completed"
    _save_steps()
    run.elapsed_secs = round(time.monotonic() - start, 2)
    await db.flush()

    return run
