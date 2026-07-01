"""
Job Queue Abstraction — survives process restarts, supports retries, real status.

WHY THIS REPLACES asyncio.create_task
----------------------------------------
Crucible's training endpoint currently does:

    asyncio.create_task(_train_background(job_id, ...))

This has three production problems:

  1. NO SURVIVAL ACROSS RESTARTS
     If Uvicorn restarts (deploy, crash, OOM-kill, autoscaling event), every
     in-flight asyncio.Task is silently destroyed. The job_id exists in the
     database with status="running" forever — there is no process left that
     will ever update it. The user sees a permanently "running" job.

  2. NO RETRIES
     If training raises an unhandled exception, the task dies. There is no
     mechanism to retry with backoff. A transient failure (e.g. a brief S3
     blip) permanently fails the job.

  3. NO BACKPRESSURE / CONCURRENCY CONTROL
     asyncio.create_task() spawns an unbounded number of concurrent tasks.
     If 50 users submit training jobs simultaneously, 50 CPU-bound training
     runs compete for the same event loop and CPU cores — there's no queue
     depth limit or worker pool sizing.

WHAT ARQ PROVIDES
-------------------
ARQ (Async Redis Queue) is a lightweight async job queue backed by Redis:
  - Jobs persist in Redis — a worker process restart resumes from the queue,
    it does not lose in-flight jobs (jobs not yet started are simply re-picked
    up by the next available worker; jobs that were running when a worker
    died are detected via Redis job heartbeat expiry and can be retried)
  - Built-in retry with configurable max_tries and backoff
  - Configurable max_jobs (concurrency limit) per worker process
  - Separate worker process(es) — the API server stays responsive even
    under heavy training load, because training never runs inside the
    request-handling event loop

ARCHITECTURE
------------
  FastAPI process            Redis              ARQ worker process(es)
  ┌────────────┐         ┌──────────┐         ┌──────────────────────┐
  │ POST /train│ enqueue │  job queue│  pick up │ run_training_job()   │
  │            │────────▶│           │◀─────────│  (separate process,  │
  │ returns    │         │           │          │   can scale          │
  │ job_id     │         └──────────┘          │   independently)      │
  └────────────┘                                └──────────────────────┘

DEFAULT: IN-MEMORY (NO REDIS REQUIRED)
-----------------------------------------
For local development and the default Crucible deployment (single SQLite
process, no separate infra), InMemoryJobQueue provides the SAME interface
without requiring Redis. It still fixes problem #3 (bounded concurrency via
a worker pool) but not #1 or #2 (no cross-restart survival, no retries) —
those require the real ARQ backend.

Switch via settings.job_queue_backend = "memory" (default) | "arq"
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional


class JobStatus(str, Enum):
    QUEUED    = "queued"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    RETRYING  = "retrying"


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    function_name: str
    enqueued_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    attempt: int = 1
    max_attempts: int = 3
    result: Any = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "job_id":        self.job_id,
            "status":        self.status.value,
            "function_name": self.function_name,
            "enqueued_at":   self.enqueued_at,
            "started_at":    self.started_at,
            "finished_at":   self.finished_at,
            "attempt":       self.attempt,
            "max_attempts":  self.max_attempts,
            "error":         self.error,
            "elapsed_secs":  (
                round((self.finished_at or time.monotonic()) - self.started_at, 2)
                if self.started_at else None
            ),
        }


class JobQueueBackend(ABC):
    """Abstract interface for job queues. InMemoryJobQueue and ArqJobQueue implement this."""

    @abstractmethod
    async def enqueue(
        self,
        func: Callable[..., Coroutine],
        *args,
        max_attempts: int = 3,
        **kwargs,
    ) -> str:
        """Enqueues a job and returns its job_id immediately (non-blocking)."""
        ...

    @abstractmethod
    async def get_status(self, job_id: str) -> Optional[JobRecord]:
        """Returns the current status of a job, or None if unknown."""
        ...

    @abstractmethod
    async def list_jobs(self, limit: int = 50) -> list[JobRecord]:
        """Returns the most recent jobs (for monitoring/debugging)."""
        ...


# ══════════════════════════════════════════════════════════════════════════
# IN-MEMORY BACKEND (default — no Redis required)
# ══════════════════════════════════════════════════════════════════════════

class InMemoryJobQueue(JobQueueBackend):
    """
    Async in-process job queue with bounded concurrency and retry support.

    Unlike bare asyncio.create_task(), this:
      - Tracks job status (queued/running/completed/failed) queryable by job_id
      - Bounds concurrency via a semaphore (max_concurrent workers)
      - Retries failed jobs up to max_attempts with exponential backoff
      - Records timing and error information

    Does NOT survive process restart — jobs in flight when the process dies
    are lost. For that guarantee, use ArqJobQueue with Redis.
    """

    def __init__(self, max_concurrent: int = 4):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._jobs: dict[str, JobRecord] = {}
        self._job_order: list[str] = []

    async def enqueue(
        self,
        func: Callable[..., Coroutine],
        *args,
        max_attempts: int = 3,
        **kwargs,
    ) -> str:
        job_id = uuid.uuid4().hex[:16]
        record = JobRecord(
            job_id=job_id,
            status=JobStatus.QUEUED,
            function_name=getattr(func, "__name__", "unknown"),
            enqueued_at=time.monotonic(),
            max_attempts=max_attempts,
        )
        self._jobs[job_id] = record
        self._job_order.append(job_id)

        asyncio.create_task(self._run_with_retry(job_id, func, args, kwargs, max_attempts))
        return job_id

    async def _run_with_retry(
        self, job_id: str, func, args, kwargs, max_attempts: int
    ) -> None:
        record = self._jobs[job_id]
        backoff = 1.0

        for attempt in range(1, max_attempts + 1):
            record.attempt = attempt
            record.status = JobStatus.RUNNING if attempt == 1 else JobStatus.RETRYING
            record.started_at = time.monotonic()

            async with self._semaphore:
                try:
                    result = await func(*args, **kwargs)
                    record.status = JobStatus.COMPLETED
                    record.result = result
                    record.finished_at = time.monotonic()
                    return
                except Exception as exc:
                    record.error = str(exc)
                    if attempt >= max_attempts:
                        record.status = JobStatus.FAILED
                        record.finished_at = time.monotonic()
                        return
                    # Exponential backoff before retry
                    await asyncio.sleep(backoff)
                    backoff *= 2

    async def get_status(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    async def list_jobs(self, limit: int = 50) -> list[JobRecord]:
        recent_ids = self._job_order[-limit:]
        return [self._jobs[jid] for jid in reversed(recent_ids) if jid in self._jobs]


# ══════════════════════════════════════════════════════════════════════════
# ARQ BACKEND (production — requires Redis)
# ══════════════════════════════════════════════════════════════════════════

class ArqJobQueue(JobQueueBackend):
    """
    Redis-backed job queue using ARQ.

    Requires a separate worker process running:
        arq jobs.worker_settings.WorkerSettings

    Jobs enqueued here are picked up by any available worker process —
    including a worker that started after this job was enqueued. This is
    what provides restart-survival: if the API process restarts, jobs
    already in the Redis queue are untouched and will still be processed
    by a worker process (which can run as a separate, independently-scaled
    deployment).
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self._redis_url = redis_url
        self._pool = None   # lazily created on first use

    async def _get_pool(self):
        if self._pool is None:
            from arq import create_pool
            from arq.connections import RedisSettings
            settings = RedisSettings.from_dsn(self._redis_url)
            self._pool = await create_pool(settings)
        return self._pool

    async def enqueue(
        self,
        func: Callable[..., Coroutine],
        *args,
        max_attempts: int = 3,
        **kwargs,
    ) -> str:
        pool = await self._get_pool()
        func_name = getattr(func, "__name__", "unknown")
        job = await pool.enqueue_job(func_name, *args, _max_tries=max_attempts, **kwargs)
        return job.job_id

    async def get_status(self, job_id: str) -> Optional[JobRecord]:
        pool = await self._get_pool()
        from arq.jobs import Job as ArqJob
        job = ArqJob(job_id, pool)
        info = await job.info()
        if info is None:
            return None

        status_map = {
            "deferred": JobStatus.QUEUED,
            "queued":   JobStatus.QUEUED,
            "in_progress": JobStatus.RUNNING,
            "complete": JobStatus.COMPLETED,
            "not_found": JobStatus.FAILED,
        }
        return JobRecord(
            job_id=job_id,
            status=status_map.get(str(info.status), JobStatus.QUEUED),
            function_name=info.function or "unknown",
            enqueued_at=info.enqueue_time.timestamp() if info.enqueue_time else 0,
            result=info.result,
        )

    async def list_jobs(self, limit: int = 50) -> list[JobRecord]:
        # ARQ does not provide a built-in "list all jobs" API — this would
        # require maintaining a separate index (e.g. a Redis sorted set of
        # job IDs updated on enqueue). Left as a known limitation; the
        # in-memory backend's list_jobs is the one actually used by the
        # /jobs/recent monitoring endpoint in the default configuration.
        return []


# ── Factory ────────────────────────────────────────────────────────────────────

_default_queue: Optional[JobQueueBackend] = None


def get_job_queue() -> JobQueueBackend:
    """
    Returns the configured job queue backend (singleton).

    settings.job_queue_backend = "memory" (default) | "arq"
    settings.redis_url required when using "arq"
    """
    global _default_queue
    if _default_queue is not None:
        return _default_queue

    from config import settings
    backend = getattr(settings, "job_queue_backend", "memory")

    if backend == "arq":
        redis_url = getattr(settings, "redis_url", "redis://localhost:6379")
        _default_queue = ArqJobQueue(redis_url=redis_url)
    else:
        _default_queue = InMemoryJobQueue(
            max_concurrent=getattr(settings, "job_queue_max_concurrent", 4)
        )
    return _default_queue


def reset_job_queue() -> None:
    """Resets the singleton — used by tests to get a fresh queue per test."""
    global _default_queue
    _default_queue = None
