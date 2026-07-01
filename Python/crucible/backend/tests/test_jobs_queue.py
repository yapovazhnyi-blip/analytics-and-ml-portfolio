"""
Job queue tests.

InMemoryJobQueue: tested fully against real asyncio behavior (no mocking needed).
ArqJobQueue: tested with mocked Redis pool (no real Redis server required),
following the same mocking pattern used for BigQuery/SageMaker in this codebase.
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ══════════════════════════════════════════════════════════════════════════
# IN-MEMORY JOB QUEUE
# ══════════════════════════════════════════════════════════════════════════

class TestInMemoryJobQueue:

    @pytest.mark.asyncio
    async def test_enqueue_returns_job_id(self):
        from jobs.queue import InMemoryJobQueue

        async def task():
            return "done"

        queue = InMemoryJobQueue()
        job_id = await queue.enqueue(task)
        assert isinstance(job_id, str)
        assert len(job_id) == 16

    @pytest.mark.asyncio
    async def test_successful_job_reaches_completed(self):
        from jobs.queue import InMemoryJobQueue, JobStatus

        async def task():
            return {"value": 42}

        queue = InMemoryJobQueue()
        job_id = await queue.enqueue(task)

        # Poll until terminal state (fast in tests since the task is trivial)
        for _ in range(50):
            record = await queue.get_status(job_id)
            if record.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                break
            await asyncio.sleep(0.01)

        assert record.status == JobStatus.COMPLETED
        assert record.result == {"value": 42}

    @pytest.mark.asyncio
    async def test_failed_job_retries_then_fails(self):
        from jobs.queue import InMemoryJobQueue, JobStatus

        call_count = {"n": 0}

        async def always_fails():
            call_count["n"] += 1
            raise ValueError("boom")

        queue = InMemoryJobQueue()
        job_id = await queue.enqueue(always_fails, max_attempts=2)

        for _ in range(200):
            record = await queue.get_status(job_id)
            if record.status == JobStatus.FAILED:
                break
            await asyncio.sleep(0.02)

        assert record.status == JobStatus.FAILED
        assert call_count["n"] == 2          # retried once, then gave up
        assert "boom" in record.error

    @pytest.mark.asyncio
    async def test_job_succeeds_after_retry(self):
        from jobs.queue import InMemoryJobQueue, JobStatus

        call_count = {"n": 0}

        async def fails_once_then_succeeds():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValueError("transient error")
            return "recovered"

        queue = InMemoryJobQueue()
        job_id = await queue.enqueue(fails_once_then_succeeds, max_attempts=3)

        for _ in range(200):
            record = await queue.get_status(job_id)
            if record.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                break
            await asyncio.sleep(0.02)

        assert record.status == JobStatus.COMPLETED
        assert record.result == "recovered"
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_unknown_job_id_returns_none(self):
        from jobs.queue import InMemoryJobQueue
        queue = InMemoryJobQueue()
        result = await queue.get_status("nonexistent-job-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_jobs_returns_recent_first(self):
        from jobs.queue import InMemoryJobQueue

        async def task():
            return "ok"

        queue = InMemoryJobQueue()
        ids = [await queue.enqueue(task) for _ in range(3)]

        for _ in range(50):
            jobs = await queue.list_jobs(limit=10)
            if all(j.result is not None for j in jobs):
                break
            await asyncio.sleep(0.02)

        jobs = await queue.list_jobs(limit=10)
        assert len(jobs) == 3
        # Most recent first
        assert jobs[0].job_id == ids[-1]

    @pytest.mark.asyncio
    async def test_list_jobs_respects_limit(self):
        from jobs.queue import InMemoryJobQueue

        async def task():
            return "ok"

        queue = InMemoryJobQueue()
        for _ in range(10):
            await queue.enqueue(task)
        await asyncio.sleep(0.1)

        jobs = await queue.list_jobs(limit=3)
        assert len(jobs) == 3

    @pytest.mark.asyncio
    async def test_concurrency_bounded_by_semaphore(self):
        """At most max_concurrent jobs should run simultaneously."""
        from jobs.queue import InMemoryJobQueue

        active = {"count": 0, "max_seen": 0}
        lock = asyncio.Lock()

        async def slow_task():
            async with lock:
                active["count"] += 1
                active["max_seen"] = max(active["max_seen"], active["count"])
            await asyncio.sleep(0.05)
            async with lock:
                active["count"] -= 1
            return "done"

        queue = InMemoryJobQueue(max_concurrent=2)
        for _ in range(6):
            await queue.enqueue(slow_task)

        await asyncio.sleep(0.3)   # let all jobs complete
        assert active["max_seen"] <= 2

    @pytest.mark.asyncio
    async def test_job_record_to_dict_serialisable(self):
        import json
        from jobs.queue import InMemoryJobQueue, JobStatus

        async def task():
            return "ok"

        queue = InMemoryJobQueue()
        job_id = await queue.enqueue(task)
        await asyncio.sleep(0.1)
        record = await queue.get_status(job_id)
        json.dumps(record.to_dict())   # must not raise

    @pytest.mark.asyncio
    async def test_args_and_kwargs_passed_through(self):
        from jobs.queue import InMemoryJobQueue, JobStatus

        async def add(a, b, multiplier=1):
            return (a + b) * multiplier

        queue = InMemoryJobQueue()
        job_id = await queue.enqueue(add, 3, 4, multiplier=2)

        for _ in range(50):
            record = await queue.get_status(job_id)
            if record.status == JobStatus.COMPLETED:
                break
            await asyncio.sleep(0.01)

        assert record.result == 14


# ══════════════════════════════════════════════════════════════════════════
# ARQ JOB QUEUE (mocked Redis pool)
# ══════════════════════════════════════════════════════════════════════════

class TestArqJobQueue:

    @pytest.mark.asyncio
    async def test_enqueue_calls_pool_enqueue_job(self):
        from jobs.queue import ArqJobQueue

        async def my_task():
            pass
        my_task.__name__ = "my_task"

        queue = ArqJobQueue(redis_url="redis://fake:6379")
        mock_pool = AsyncMock()
        mock_job = MagicMock()
        mock_job.job_id = "arq-job-123"
        mock_pool.enqueue_job = AsyncMock(return_value=mock_job)

        with patch.object(queue, "_get_pool", AsyncMock(return_value=mock_pool)):
            job_id = await queue.enqueue(my_task, max_attempts=5)

        assert job_id == "arq-job-123"
        mock_pool.enqueue_job.assert_called_once()
        call_kwargs = mock_pool.enqueue_job.call_args
        assert call_kwargs.kwargs.get("_max_tries") == 5

    @pytest.mark.asyncio
    async def test_get_status_returns_none_for_unknown_job(self):
        from jobs.queue import ArqJobQueue

        queue = ArqJobQueue(redis_url="redis://fake:6379")
        mock_pool = AsyncMock()

        with patch.object(queue, "_get_pool", AsyncMock(return_value=mock_pool)), \
             patch("arq.jobs.Job") as MockJob:
            mock_job_instance = MagicMock()
            mock_job_instance.info = AsyncMock(return_value=None)
            MockJob.return_value = mock_job_instance

            result = await queue.get_status("unknown-job")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_status_maps_complete_status(self):
        from jobs.queue import ArqJobQueue, JobStatus

        queue = ArqJobQueue(redis_url="redis://fake:6379")
        mock_pool = AsyncMock()

        with patch.object(queue, "_get_pool", AsyncMock(return_value=mock_pool)), \
             patch("arq.jobs.Job") as MockJob:
            mock_info = MagicMock()
            mock_info.status = "complete"
            mock_info.function = "run_training_job"
            mock_info.enqueue_time = None
            mock_info.result = {"score": 0.9}

            mock_job_instance = MagicMock()
            mock_job_instance.info = AsyncMock(return_value=mock_info)
            MockJob.return_value = mock_job_instance

            record = await queue.get_status("job-1")

        assert record.status == JobStatus.COMPLETED
        assert record.result == {"score": 0.9}

    @pytest.mark.asyncio
    async def test_list_jobs_returns_empty(self):
        """ARQ backend does not support listing all jobs — documented limitation."""
        from jobs.queue import ArqJobQueue
        queue = ArqJobQueue()
        result = await queue.list_jobs()
        assert result == []

    @pytest.mark.asyncio
    async def test_pool_created_lazily_and_cached(self):
        from jobs.queue import ArqJobQueue

        queue = ArqJobQueue(redis_url="redis://fake:6379")
        assert queue._pool is None

        with patch("arq.create_pool", AsyncMock(return_value=MagicMock())) as mock_create:
            pool1 = await queue._get_pool()
            pool2 = await queue._get_pool()

        assert pool1 is pool2
        mock_create.assert_called_once()   # only created once, then cached


# ══════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════

class TestJobQueueFactory:

    def test_default_backend_is_memory(self):
        from jobs.queue import get_job_queue, reset_job_queue, InMemoryJobQueue
        from config import settings
        reset_job_queue()
        original = settings.job_queue_backend
        try:
            settings.job_queue_backend = "memory"
            queue = get_job_queue()
        finally:
            settings.job_queue_backend = original
            reset_job_queue()
        assert isinstance(queue, InMemoryJobQueue)

    def test_arq_backend_selected_via_settings(self):
        from jobs.queue import get_job_queue, reset_job_queue, ArqJobQueue
        from config import settings
        reset_job_queue()
        original = settings.job_queue_backend
        try:
            settings.job_queue_backend = "arq"
            queue = get_job_queue()
        finally:
            settings.job_queue_backend = original
            reset_job_queue()
        assert isinstance(queue, ArqJobQueue)

    def test_factory_returns_singleton(self):
        from jobs.queue import get_job_queue, reset_job_queue
        reset_job_queue()
        q1 = get_job_queue()
        q2 = get_job_queue()
        assert q1 is q2
        reset_job_queue()


# ══════════════════════════════════════════════════════════════════════════
# WORKER SETTINGS MODULE
# ══════════════════════════════════════════════════════════════════════════

class TestWorkerSettings:

    def test_worker_settings_has_functions(self):
        from jobs.worker_settings import WorkerSettings
        assert len(WorkerSettings.functions) >= 1

    def test_run_training_job_is_registered(self):
        from jobs.worker_settings import WorkerSettings, run_training_job
        assert run_training_job in WorkerSettings.functions

    def test_worker_settings_has_timeout(self):
        from jobs.worker_settings import WorkerSettings
        assert WorkerSettings.job_timeout > 0

    def test_worker_settings_has_max_jobs(self):
        from jobs.worker_settings import WorkerSettings
        assert WorkerSettings.max_jobs >= 1


# ══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def jobs_client():
    import sys, importlib, database as db_mod
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool
    from fastapi.testclient import TestClient
    from jobs.queue import reset_job_queue

    reset_job_queue()
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    db_mod.engine = engine; db_mod.SessionFactory = factory; db_mod.AsyncSessionLocal = factory
    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    import main as m
    with TestClient(m.app, raise_server_exceptions=True) as c:
        yield c
    reset_job_queue()


class TestJobsAPI:

    def test_unknown_job_returns_404(self, jobs_client):
        resp = jobs_client.get("/api/v1/jobs/nonexistent-job-id")
        assert resp.status_code == 404

    def test_list_recent_jobs_empty_initially(self, jobs_client):
        resp = jobs_client.get("/api/v1/jobs")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    @pytest.mark.asyncio
    async def test_status_endpoint_reflects_completed_job(self, jobs_client):
        from jobs.queue import get_job_queue, JobStatus

        async def quick_task():
            return {"answer": 42}

        queue = get_job_queue()
        job_id = await queue.enqueue(quick_task)

        for _ in range(50):
            record = await queue.get_status(job_id)
            if record.status == JobStatus.COMPLETED:
                break
            await asyncio.sleep(0.01)

        resp = jobs_client.get(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "completed"
