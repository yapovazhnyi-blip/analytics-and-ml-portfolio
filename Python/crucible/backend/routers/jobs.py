"""Job queue monitoring router — /api/v1/jobs/*"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from auth.dependencies import get_current_user
from schemas.common import DataResponse

router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=[Depends(get_current_user)])


@router.get("/{job_id}")
async def get_job_status(job_id: str):
    """
    Returns the status of a background job (training, etc.) submitted
    through the job queue.

    Status values: queued | running | retrying | completed | failed
    """
    from jobs.queue import get_job_queue

    queue = get_job_queue()
    record = await queue.get_status(job_id)
    if record is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return DataResponse(data=record.to_dict())


@router.get("")
async def list_recent_jobs(limit: int = 50):
    """
    Returns the most recently enqueued jobs, for monitoring and debugging.

    Note: when job_queue_backend="arq", this returns an empty list — ARQ
    does not provide a built-in job-listing API. Use individual job status
    lookups (GET /jobs/{job_id}) instead, or check Redis directly.
    """
    from jobs.queue import get_job_queue

    queue = get_job_queue()
    records = await queue.list_jobs(limit=limit)
    return DataResponse(data=[r.to_dict() for r in records])
