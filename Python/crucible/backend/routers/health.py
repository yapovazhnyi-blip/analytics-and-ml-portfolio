"""
Health check endpoints.

Two endpoints following Kubernetes/load-balancer conventions:
  GET /health/live  — is the process alive? (liveness probe)
  GET /health/ready — is the app ready to serve traffic? (readiness probe)

The readiness probe checks the database connection — if the DB is
unreachable, the app can't serve meaningful requests and should be
taken out of rotation. The liveness probe is intentionally minimal.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from database import get_db
from config import settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def liveness():
    """Process is alive. Returns immediately — no I/O."""
    return {"status": "ok", "app": settings.app_name}


@router.get("/ready")
async def readiness(db: AsyncSession = Depends(get_db)):
    """
    Process is ready to serve traffic.
    Checks DB connectivity — fails if the database is unreachable.
    """
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ok", "database": "reachable"}
    except Exception as exc:
        # Return 503 so load balancers remove this instance from rotation
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail={"status": "error", "database": str(exc)},
        )
