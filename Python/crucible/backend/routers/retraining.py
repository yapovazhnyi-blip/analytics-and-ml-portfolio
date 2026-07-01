"""
Retraining Pipeline router — /api/v1/retraining/*

Endpoints:
  POST   /retraining/policies              — create a policy
  GET    /retraining/policies              — list policies
  GET    /retraining/policies/{id}         — get one policy
  PATCH  /retraining/policies/{id}         — update a policy (re-syncs the schedule)
  DELETE /retraining/policies/{id}         — delete a policy
  POST   /retraining/policies/{id}/run     — manually trigger a pipeline run now
  GET    /retraining/policies/{id}/runs    — run history for a policy
  GET    /retraining/runs/{id}             — get one run's full detail (steps, drift report)
  POST   /experiments/{id}/promote         — manually promote an experiment outside any policy
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from database import get_db
from models.dataset import Dataset
from models.experiment import Experiment
from models.retraining import RetrainingPolicy, RetrainingRun
from schemas.common import DataResponse

router = APIRouter(prefix="/retraining", tags=["retraining"], dependencies=[Depends(get_current_user)])

VALID_SEVERITIES = ("slight", "significant", "critical")


# ══════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════

class PolicyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    reference_dataset_id: int
    target_column: str
    task_type: str = Field(..., pattern="^(classification|regression)$")
    drift_severity_trigger: str = Field(default="significant", pattern="^(slight|significant|critical)$")
    promotion_margin: float = Field(default=0.02, ge=0.0, le=1.0)
    n_trials: int = Field(default=15, ge=1, le=200)
    cv_folds: int = Field(default=3, ge=2, le=10)
    check_interval_hours: Optional[float] = Field(default=None, gt=0)


class PolicyUpdate(BaseModel):
    description: Optional[str] = None
    latest_dataset_id: Optional[int] = None
    drift_severity_trigger: Optional[str] = Field(default=None, pattern="^(slight|significant|critical)$")
    promotion_margin: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    n_trials: Optional[int] = Field(default=None, ge=1, le=200)
    cv_folds: Optional[int] = Field(default=None, ge=2, le=10)
    check_interval_hours: Optional[float] = Field(default=None, gt=0)
    is_active: Optional[bool] = None


class RunTriggerRequest(BaseModel):
    current_dataset_id: Optional[int] = Field(
        default=None,
        description="Dataset representing the latest data batch to check for drift. "
                    "Defaults to the policy's latest_dataset_id, or its reference_dataset_id if unset.",
    )


def _policy_out(p: RetrainingPolicy) -> dict:
    return {
        "id":                       p.id,
        "name":                     p.name,
        "description":              p.description,
        "reference_dataset_id":     p.reference_dataset_id,
        "latest_dataset_id":        p.latest_dataset_id,
        "target_column":            p.target_column,
        "task_type":                p.task_type,
        "drift_severity_trigger":   p.drift_severity_trigger,
        "promotion_margin":         p.promotion_margin,
        "n_trials":                 p.n_trials,
        "cv_folds":                 p.cv_folds,
        "check_interval_hours":     p.check_interval_hours,
        "production_experiment_id": p.production_experiment_id,
        "is_active":                p.is_active,
        "created_at":               p.created_at.isoformat() if p.created_at else None,
    }


def _run_out(r: RetrainingRun, include_steps: bool = False) -> dict:
    out = {
        "id":                  r.id,
        "policy_id":           r.policy_id,
        "current_dataset_id":  r.current_dataset_id,
        "status":              r.status,
        "drift_checked":       r.drift_checked,
        "drift_detected":      r.drift_detected,
        "retrain_triggered":   r.retrain_triggered,
        "new_experiment_id":   r.new_experiment_id,
        "promoted":            r.promoted,
        "promotion_reason":    r.promotion_reason,
        "error_message":       r.error_message,
        "elapsed_secs":        r.elapsed_secs,
        "created_at":          r.created_at.isoformat() if r.created_at else None,
    }
    if include_steps:
        out["steps"] = json.loads(r.steps_json) if r.steps_json else []
        out["drift_report"] = json.loads(r.drift_report_json) if r.drift_report_json else None
    return out


# ══════════════════════════════════════════════════════════════════════════
# POLICY CRUD
# ══════════════════════════════════════════════════════════════════════════

@router.post("/policies", status_code=201)
async def create_policy(body: PolicyCreate, db: AsyncSession = Depends(get_db)):
    """
    Creates a retraining policy: a named, reusable rule for "watch this
    dataset/target for drift, and if it drifts enough, retrain and
    conditionally promote."

    If check_interval_hours is set, the policy is automatically scheduled
    to run on that interval (see retraining/scheduler.py). Otherwise it's
    manual-trigger only via POST /retraining/policies/{id}/run.
    """
    ds = await db.get(Dataset, body.reference_dataset_id)
    if not ds:
        raise HTTPException(404, f"Reference dataset {body.reference_dataset_id} not found")

    policy = RetrainingPolicy(
        name=body.name,
        description=body.description,
        reference_dataset_id=body.reference_dataset_id,
        target_column=body.target_column,
        task_type=body.task_type,
        drift_severity_trigger=body.drift_severity_trigger,
        promotion_margin=body.promotion_margin,
        n_trials=body.n_trials,
        cv_folds=body.cv_folds,
        check_interval_hours=body.check_interval_hours,
        is_active=True,
    )
    db.add(policy)
    await db.flush()
    await db.refresh(policy)

    if policy.check_interval_hours:
        from retraining.scheduler import schedule_policy
        schedule_policy(policy.id, policy.check_interval_hours)

    return DataResponse(data=_policy_out(policy))


@router.get("/policies")
async def list_policies(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(RetrainingPolicy).order_by(RetrainingPolicy.created_at.desc()))
    return DataResponse(data=[_policy_out(p) for p in result.scalars().all()])


@router.get("/policies/{policy_id}")
async def get_policy(policy_id: int, db: AsyncSession = Depends(get_db)):
    policy = await db.get(RetrainingPolicy, policy_id)
    if not policy:
        raise HTTPException(404, f"Policy {policy_id} not found")
    return DataResponse(data=_policy_out(policy))


@router.patch("/policies/{policy_id}")
async def update_policy(policy_id: int, body: PolicyUpdate, db: AsyncSession = Depends(get_db)):
    """
    Updates a policy. If check_interval_hours or is_active changes, the
    schedule is re-synced immediately — no restart required.
    """
    policy = await db.get(RetrainingPolicy, policy_id)
    if not policy:
        raise HTTPException(404, f"Policy {policy_id} not found")

    updates = body.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(policy, field, value)
    await db.flush()

    from retraining.scheduler import schedule_policy, unschedule_policy
    if policy.is_active and policy.check_interval_hours:
        schedule_policy(policy.id, policy.check_interval_hours)
    else:
        unschedule_policy(policy.id)

    return DataResponse(data=_policy_out(policy))


@router.delete("/policies/{policy_id}", status_code=204)
async def delete_policy(policy_id: int, db: AsyncSession = Depends(get_db)):
    policy = await db.get(RetrainingPolicy, policy_id)
    if not policy:
        raise HTTPException(404, f"Policy {policy_id} not found")

    from retraining.scheduler import unschedule_policy
    unschedule_policy(policy_id)

    await db.delete(policy)


# ══════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════

@router.post("/policies/{policy_id}/run")
async def trigger_run(
    policy_id: int,
    body: RunTriggerRequest = RunTriggerRequest(),
    db: AsyncSession = Depends(get_db),
):
    """
    Manually triggers a pipeline run for this policy right now.

    Runs synchronously within the request (drift check is fast; retraining
    only happens if drift is detected, and uses the same training config
    the policy specifies — keep n_trials modest for interactive use).

    For demos: this is the endpoint to call directly rather than waiting
    for a scheduled tick.
    """
    policy = await db.get(RetrainingPolicy, policy_id)
    if not policy:
        raise HTTPException(404, f"Policy {policy_id} not found")
    if not policy.is_active:
        raise HTTPException(422, "Policy is not active")

    current_dataset_id = body.current_dataset_id or policy.latest_dataset_id or policy.reference_dataset_id
    current_ds = await db.get(Dataset, current_dataset_id)
    if not current_ds:
        raise HTTPException(404, f"Current dataset {current_dataset_id} not found")

    from retraining.pipeline import run_pipeline
    run = await run_pipeline(policy, current_dataset_id, db)

    return DataResponse(data=_run_out(run, include_steps=True))


@router.get("/policies/{policy_id}/runs")
async def list_runs(policy_id: int, limit: int = 20, db: AsyncSession = Depends(get_db)):
    policy = await db.get(RetrainingPolicy, policy_id)
    if not policy:
        raise HTTPException(404, f"Policy {policy_id} not found")

    result = await db.execute(
        select(RetrainingRun).where(RetrainingRun.policy_id == policy_id)
        .order_by(RetrainingRun.created_at.desc()).limit(limit)
    )
    return DataResponse(data=[_run_out(r) for r in result.scalars().all()])


@router.get("/runs/{run_id}")
async def get_run(run_id: int, db: AsyncSession = Depends(get_db)):
    run = await db.get(RetrainingRun, run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    return DataResponse(data=_run_out(run, include_steps=True))


# ══════════════════════════════════════════════════════════════════════════
# MANUAL PROMOTION (outside any policy)
# ══════════════════════════════════════════════════════════════════════════

manual_router = APIRouter(tags=["retraining"], dependencies=[Depends(get_current_user)])


@manual_router.post("/experiments/{experiment_id}/promote")
async def promote_experiment(experiment_id: int, db: AsyncSession = Depends(get_db)):
    """
    Manually promotes an experiment to lifecycle_stage='production', outside
    of any automated policy. If a policy's production_experiment_id matches
    the dataset/target/task of this experiment, that policy is NOT
    automatically updated — manual promotion is independent of the
    automated pipeline by design (it's an override, not a policy edit).

    Any other experiment currently in lifecycle_stage='production' for the
    SAME dataset_id + target_column is archived.
    """
    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(404, f"Experiment {experiment_id} not found")
    if exp.status != "complete":
        raise HTTPException(422, f"Cannot promote an experiment with status={exp.status!r}")

    result = await db.execute(
        select(Experiment).where(
            Experiment.dataset_id == exp.dataset_id,
            Experiment.target_column == exp.target_column,
            Experiment.lifecycle_stage == "production",
            Experiment.id != exp.id,
        )
    )
    for other in result.scalars().all():
        other.lifecycle_stage = "archived"

    exp.lifecycle_stage = "production"
    await db.flush()

    return DataResponse(data={"id": exp.id, "lifecycle_stage": exp.lifecycle_stage})
