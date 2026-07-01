# ── Stage 1: dependency builder ──────────────────────────────────────────────
# Uses a full Python image to compile any native extensions.
# The compiled packages are then copied to the slim runtime stage.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build deps (needed for numpy, cryptography, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — this layer is only rebuilt when requirements.txt changes.
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Crucible ML Platform"
LABEL org.opencontainers.image.description="End-to-end ML experimentation platform"
LABEL org.opencontainers.image.source="https://github.com/yapovazhnyi-blip/analytics-and-ml-portfolio"

# Runtime-only system libs.
# libgomp1 provides libgomp.so.1 (GNU OpenMP runtime) — required by LightGBM's
# compiled C++ core for multi-threading. This is an OS-level shared library,
# not something `pip install lightgbm` can provide; the Linux wheel expects it
# to already be present on the system (unlike the Windows wheel, which bundles
# the equivalent DLL differently — this is why a native Windows venv works
# without this fix, but a Linux-based Docker image does not).
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — security best practice.
# If the container is compromised, the attacker does not get root access.
RUN groupadd --gid 1001 crucible && \
    useradd --uid 1001 --gid crucible --shell /bin/bash --create-home crucible

# Copy compiled packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code (owned by crucible, not root)
COPY --chown=crucible:crucible . .

# Create data directories
RUN mkdir -p \
        /app/data/storage \
        /app/data/datasets \
        /app/data/models \
        /app/data/rag \
    && chown -R crucible:crucible /app/data

USER crucible

EXPOSE 8000

# Health check — used by Docker, Kubernetes, and load balancers.
# --start-period=30s gives Alembic migrations time to run on first start.
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=30s \
    --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "65", \
     "--access-log"]