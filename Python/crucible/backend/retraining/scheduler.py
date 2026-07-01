"""
Retraining Scheduler — in-process recurring policy execution via APScheduler.

WHY APSCHEDULER, NOT A SEPARATE AIRFLOW DEPLOYMENT
------------------------------------------------------
Real Airflow is a multi-component system (webserver, scheduler, executor,
metadata DB) — disproportionate infrastructure for "run this check every
N hours" inside a single-tenant ML platform. APScheduler's AsyncIOScheduler
runs an in-process cron-like trigger loop inside the same FastAPI event loop,
giving genuine periodic execution with zero extra infrastructure.

This is a real architectural trade-off, not a toy: APScheduler is a widely-used
production pattern for exactly this scale of need (a handful of recurring jobs
inside one service), while Airflow is the right choice once you have dozens
of interdependent pipelines across multiple teams needing a shared scheduling
UI, backfill support, and cross-pipeline dependency management.

CRITICAL CONSTRAINT — SINGLE WORKER PROCESS ONLY
----------------------------------------------------
APScheduler's in-memory job store is per-process. If Crucible runs with
multiple Uvicorn workers (--workers > 1), EACH worker would independently
schedule and fire the same policy, causing duplicate retraining runs. This
is the same constraint documented on the Dockerfile's `--workers 1` CMD
(driven by SQLite's single-writer limitation) — the retraining scheduler
adds a second, independent reason to keep that constraint until Crucible
moves to PostgreSQL + a distributed scheduler lock.

WHAT GETS SCHEDULED
-----------------------
On startup, and whenever a policy is created/updated/deleted, sync_schedule()
rebuilds the scheduler's job list from the current set of active policies
with check_interval_hours set. Each scheduled tick calls run_pipeline()
against the policy's current latest_dataset_id (or its own reference
dataset if no fresher data has been pointed at it yet — a safe no-op).
"""

from __future__ import annotations

import logging
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger("crucible.retraining.scheduler")

_scheduler: Optional[AsyncIOScheduler] = None


def get_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler()
    return _scheduler


def start_scheduler() -> None:
    """Starts the scheduler. Call once at application startup."""
    scheduler = get_scheduler()
    if not scheduler.running:
        scheduler.start()
        logger.info("crucible.retraining.scheduler_started")


def shutdown_scheduler() -> None:
    """Stops the scheduler. Call at application shutdown."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("crucible.retraining.scheduler_stopped")


def reset_scheduler_for_tests() -> None:
    """
    Resets the singleton — used by tests that spin up a fresh TestClient
    (and therefore a fresh event loop) per test. Without this, the scheduler
    started against one test's event loop survives into the next test as a
    'running' scheduler bound to an already-closed loop, causing
    'RuntimeError: Event loop is closed' on any subsequent scheduler
    interaction. Mirrors jobs.queue.reset_job_queue() for the same reason.
    """
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        try:
            _scheduler.shutdown(wait=False)
        except Exception:
            pass   # event loop may already be closed — nothing to clean up
    _scheduler = None


def _job_id_for_policy(policy_id: int) -> str:
    return f"retraining-policy-{policy_id}"


async def _scheduled_tick(policy_id: int) -> None:
    """
    The function APScheduler actually calls on each tick. Resolves a fresh
    DB session and policy state at call time (not at schedule time), so
    policy edits between ticks are always respected.
    """
    from database import AsyncSessionLocal
    from models.retraining import RetrainingPolicy
    from retraining.pipeline import run_pipeline

    async with AsyncSessionLocal() as db:
        policy = await db.get(RetrainingPolicy, policy_id)
        if not policy or not policy.is_active:
            return   # policy was deleted or deactivated since scheduling — skip silently

        current_dataset_id = policy.latest_dataset_id or policy.reference_dataset_id
        try:
            await run_pipeline(policy, current_dataset_id, db)
            await db.commit()
        except Exception:
            logger.exception("crucible.retraining.scheduled_run_failed", extra={"policy_id": policy_id})
            await db.rollback()


def schedule_policy(policy_id: int, interval_hours: float) -> None:
    """
    Adds or replaces the recurring job for one policy.

    IMPORTANT: start_scheduler() must have been called first. APScheduler's
    replace_existing=True deduplication only takes effect once the scheduler
    has an active event loop — calling this against an unstarted scheduler
    silently creates duplicate jobs on repeated calls instead of replacing.
    main.py's lifespan guarantees correct ordering (start_scheduler() runs
    before sync_schedule_from_db()), so this is safe everywhere the app
    actually runs; it only matters for tests that call this directly.
    """
    scheduler = get_scheduler()
    scheduler.add_job(
        _scheduled_tick,
        trigger=IntervalTrigger(hours=interval_hours),
        args=[policy_id],
        id=_job_id_for_policy(policy_id),
        replace_existing=True,
        max_instances=1,   # never run two ticks of the same policy concurrently
    )


def unschedule_policy(policy_id: int) -> None:
    """Removes the recurring job for one policy, if it exists."""
    scheduler = get_scheduler()
    job_id = _job_id_for_policy(policy_id)
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)


async def sync_schedule_from_db() -> int:
    """
    Rebuilds the full schedule from the current DB state of active policies.
    Call on startup, and after any policy create/update/delete.

    Returns the number of policies now scheduled. Returns 0 (logging a
    warning, never raising) if the retraining_policies table doesn't exist
    yet — this can legitimately happen on a brand-new deployment before
    migrations finish, and a missing-table race here must never prevent
    the rest of the application from starting up.
    """
    from sqlalchemy import select
    from database import AsyncSessionLocal
    from models.retraining import RetrainingPolicy

    scheduler = get_scheduler()
    # Clear all existing retraining jobs, then re-add from current DB state.
    for job in scheduler.get_jobs():
        if job.id.startswith("retraining-policy-"):
            scheduler.remove_job(job.id)

    count = 0
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(RetrainingPolicy).where(
                    RetrainingPolicy.is_active == True,             # noqa: E712
                    RetrainingPolicy.check_interval_hours.is_not(None),
                )
            )
            for policy in result.scalars().all():
                schedule_policy(policy.id, policy.check_interval_hours)
                count += 1
    except Exception as exc:
        logger.warning(
            "crucible.retraining.sync_schedule_failed",
            extra={"error": str(exc)},
        )
        return 0

    return count
