"""
Crucible FastAPI application.

This is the entry point — it wires together:
  - Lifespan: startup (DB init) and shutdown (connection pool cleanup)
  - CORS middleware (configured per environment)
  - Global exception handlers (consistent error envelope)
  - All routers under /api/v1/

The lifespan pattern (rather than @app.on_event) is the modern FastAPI
approach — it runs startup/shutdown as a single async context manager,
which means setup and teardown are co-located and always paired.
"""

import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from database import engine, init_db


def _mask_db_url(url: str) -> str:
    """
    Masks credentials in a database URL before logging.
    postgresql+asyncpg://user:secret@host/db → postgresql+asyncpg://user:***@host/db
    Prevents database passwords from appearing in log aggregation systems.
    """
    try:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url)
        if parsed.password:
            masked = parsed._replace(
                netloc=f"{parsed.username}:***@{parsed.hostname}"
                + (f":{parsed.port}" if parsed.port else "")
            )
            return urlunparse(masked)
    except Exception:
        pass
    return url

# ── Structured logging setup ───────────────────────────────────────────────
# structlog gives JSON-formatted logs in production, coloured in dev.
# This must happen before any logger is acquired.

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        logging.DEBUG if settings.debug else logging.INFO
    ),
)
logger = structlog.get_logger()


# ── Lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup, then yields to serve requests,
    then runs teardown when the process exits.

    Startup:
      - Initialise database tables (create_all for SQLite in Phase 1)
      - Log the API is ready

    Shutdown:
      - Dispose the async engine (flushes connection pool)
    """
    logger.info("crucible.startup", database_url=_mask_db_url(settings.database_url))
    await init_db()

    from observability.tracing import setup_tracing
    setup_tracing(app)
    logger.info("crucible.tracing_ready", exporter=getattr(settings, "otel_exporter", "console"))

    from retraining.scheduler import start_scheduler, sync_schedule_from_db
    start_scheduler()
    n_scheduled = await sync_schedule_from_db()
    logger.info("crucible.retraining_scheduler_ready", n_policies_scheduled=n_scheduled)

    logger.info("crucible.ready", app=settings.app_name, version=settings.api_version)

    yield  # ← requests are served here

    logger.info("crucible.shutdown")
    from retraining.scheduler import shutdown_scheduler
    shutdown_scheduler()
    await engine.dispose()


# ── App instantiation ──────────────────────────────────────────────────────

app = FastAPI(
    title="Crucible",
    description=(
        "A generalised ML experimentation platform. "
        "Ingest data from any source, profile it deeply, "
        "run AutoML training, explain results with SHAP, "
        "and deploy in one click."
    ),
    version="0.1.0",
    # Swagger UI and ReDoc are only available in debug mode.
    # In production (DEBUG=false) these are disabled — they expose your full
    # API surface to anyone who discovers the URL. API discovery should be
    # controlled, not freely browsable.
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)


# ── Rate limiting (slowapi) ────────────────────────────────────────────────
#
# Protects endpoints from brute-force, abuse, and runaway cost.
# Limits are keyed by IP address. In production behind a proxy, set
# X-Forwarded-For to get the real client IP (not the proxy IP).
#
# LIMITS BY ENDPOINT TYPE:
#   auth/login    5/minute   — brute-force password guessing
#   agent/run     10/minute  — each run costs Anthropic API credits
#   fine-tuning   5/hour     — GPU training jobs are expensive
#   experiments   30/minute  — training is CPU-bound but manageable
#   general reads 120/minute — generous limit for UI browsing

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


# ── CORS ───────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Security headers ───────────────────────────────────────────────────────
# These headers apply to every response. They are defence-in-depth and
# do not replace authentication — they limit the blast radius of XSS,
# clickjacking, and MIME-sniffing attacks.

from starlette.middleware.base import BaseHTTPMiddleware

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Generates a unique ID for every HTTP request and attaches it to:
      - request.state.request_id  (available in all route handlers and logs)
      - X-Request-ID response header (returned to callers for log correlation)

    If the caller supplies an X-Request-ID header, that value is reused —
    this enables distributed tracing where a gateway assigns an ID that
    propagates through all downstream services.

    WHY THIS MATTERS
    ----------------
    Without request IDs, correlating a FastAPI error log with the browser
    network tab or a client error report requires timestamp matching, which
    fails when multiple requests are in-flight simultaneously. With a request
    ID, you can grep a single ID across every log sink and reconstruct the
    full lifecycle of any request in seconds.
    """
    async def dispatch(self, request, call_next):
        import uuid
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        if not settings.debug:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)


# ── Global exception handlers ──────────────────────────────────────────────
# These ensure every error — even unhandled ones — returns the consistent
# {"error": {"code": ..., "message": ...}} envelope the frontend expects.

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "crucible.unhandled_exception",
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        headers={"X-Request-ID": request_id},
        content={
            "error": {
                "code":    "INTERNAL_ERROR",
                "message": "An unexpected error occurred.",
                "request_id": request_id,
                "details": [],
            }
        },
    )


# ── Routers ────────────────────────────────────────────────────────────────
# All routes live under /api/v1/ — versioned from day one so breaking
# changes can be introduced as /api/v2/ without disrupting existing clients.

from routers import health, datasets, connectors, profiling, experiments as exp_router, phase3, oauth_callback, rag, evaluation, auth, drift, fine_tuning as ft_router, forecasting as fc_router, agent as agent_router, fairness as fairness_router, ab_testing as ab_router, data_contracts as dc_router, anomaly as anomaly_router, model_cards as mc_router, api_keys as api_keys_router, cloud as cloud_router, jobs as jobs_router, agent_training as at_router, retraining as retraining_router  # noqa: E402

API_PREFIX = f"/api/{settings.api_version}"

app.include_router(health.router,           prefix=API_PREFIX)
app.include_router(api_keys_router.router,  prefix=API_PREFIX)
app.include_router(auth.router,             prefix=API_PREFIX)   # no auth required on auth endpoints
app.include_router(datasets.router,         prefix=API_PREFIX)
app.include_router(connectors.router,       prefix=API_PREFIX)
app.include_router(profiling.router,        prefix=API_PREFIX)
app.include_router(exp_router.router,       prefix=API_PREFIX)
app.include_router(phase3.router,           prefix=API_PREFIX)
app.include_router(oauth_callback.router,   prefix=API_PREFIX)
app.include_router(oauth_callback.router)
app.include_router(rag.router,              prefix=API_PREFIX)
app.include_router(evaluation.router,       prefix=API_PREFIX)
app.include_router(drift.router,            prefix=API_PREFIX)
app.include_router(fc_router.router,          prefix=API_PREFIX)
app.include_router(ft_router.router,          prefix=API_PREFIX)
app.include_router(ft_router.ws_router)
app.include_router(retraining_router.router,        prefix=API_PREFIX)
app.include_router(retraining_router.manual_router, prefix=API_PREFIX)
app.include_router(at_router.router,       prefix=API_PREFIX)
app.include_router(jobs_router.router,     prefix=API_PREFIX)
app.include_router(cloud_router.router,    prefix=API_PREFIX)
app.include_router(mc_router.router,        prefix=API_PREFIX)
app.include_router(anomaly_router.router,    prefix=API_PREFIX)
app.include_router(dc_router.router,         prefix=API_PREFIX)
app.include_router(ab_router.router,         prefix=API_PREFIX)
app.include_router(fairness_router.router,     prefix=API_PREFIX)
app.include_router(agent_router.router,       prefix=API_PREFIX)
app.include_router(agent_router.ws_router)
app.include_router(exp_router.ws_router)


# ── Root redirect ──────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Redirects browsers to the API docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")
