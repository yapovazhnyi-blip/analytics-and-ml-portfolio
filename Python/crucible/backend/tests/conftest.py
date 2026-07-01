"""
Shared pytest fixtures and configuration.

WHY THE AUTOUSE SCHEDULER RESET EXISTS
------------------------------------------
retraining/scheduler.py holds a module-level AsyncIOScheduler singleton,
started during FastAPI's lifespan. Many test files across this suite each
create their own fresh TestClient(app) instance — and each one runs the
full lifespan, including start_scheduler(). APScheduler binds to whichever
asyncio event loop is running when .start() is called; when a TestClient's
`with` block exits, pytest-asyncio's event loop for that test is torn down.

Without a reset between tests, the SECOND test file to boot a TestClient
in the same pytest session finds the scheduler already "running" (from the
first test's now-closed loop) and skips starting it again — any subsequent
scheduler interaction then raises "RuntimeError: Event loop is closed".

This autouse fixture guarantees every test starts with a clean, unstarted
scheduler singleton, regardless of which test file or fixture pattern is
used — a single fix here covers the whole suite rather than requiring every
TestClient-based fixture (20+ files) to remember to do this individually.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_retraining_scheduler():
    from retraining.scheduler import reset_scheduler_for_tests
    reset_scheduler_for_tests()
    yield
    reset_scheduler_for_tests()
