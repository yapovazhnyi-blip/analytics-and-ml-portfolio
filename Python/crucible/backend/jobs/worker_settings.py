"""
ARQ Worker Entrypoint.

Run this as a separate process to consume jobs from Redis:

    arq jobs.worker_settings.WorkerSettings

This process is independent of the FastAPI server — it can be scaled
horizontally (more worker processes = more concurrent training jobs)
without touching the API server's resource allocation.

DEPLOYMENT
----------
In docker-compose.yml, this runs as a separate service:

    worker:
      build: ./backend
      command: arq jobs.worker_settings.WorkerSettings
      depends_on: [redis]

In Kubernetes, this would be a separate Deployment with its own replica count,
independent from the API server's Deployment.
"""

from __future__ import annotations

import os
from arq.connections import RedisSettings

async def run_training_job(ctx, dataset_id: int, target_column: str, task_type: str, **kwargs):
    """
    ARQ task wrapper around Crucible's TrainingRunner.

    ctx is ARQ's job context, containing the Redis pool and job metadata.
    This function must be importable by name — ARQ enqueues jobs by function
    name string and looks them up in the functions list below.
    """
    from training.runner import TrainingRunner, TrainingConfig
    from database import AsyncSessionLocal
    from models.dataset import Dataset
    from profiling.runner import ProfileRunner

    async with AsyncSessionLocal() as db:
        ds = await db.get(Dataset, dataset_id)
        if not ds:
            raise ValueError(f"Dataset {dataset_id} not found")

        df = ProfileRunner.load_dataframe(ds.file_path, ds.source_type)
        config = TrainingConfig(**kwargs)
        runner = TrainingRunner()
        result = runner.run(
            df=df, target_column=target_column, task_type=task_type,
            config=config, experiment_name=f"arq-job-{ctx['job_id']}",
        )
        return result.to_dict() if hasattr(result, "to_dict") else result.__dict__


class WorkerSettings:
    """ARQ worker configuration — referenced by the `arq` CLI command."""
    functions = [run_training_job]
    redis_settings = RedisSettings.from_dsn(os.environ.get("REDIS_URL", "redis://localhost:6379"))   # reads REDIS_URL env var, or localhost default
    max_jobs = 4                        # concurrent jobs per worker process
    job_timeout = 3600                  # 1 hour max per training job
    keep_result = 3600                  # keep job results for 1 hour after completion
