"""
Crucible application configuration.

All settings are read from environment variables (or a .env file).
Pydantic-settings validates types at startup — a missing required var
raises an error immediately rather than failing silently at runtime.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",        # silently ignore unknown env vars
    )

    app_name: str = "Crucible"
    debug: bool = False
    api_version: str = "v1"

    # ── Database ───────────────────────────────────────────────────────────
    # SQLite for development, Postgres for production.
    # Use the async driver prefix — database.py strips it for the sync mirror.
    database_url: str = "sqlite+aiosqlite:///./data/crucible.db"

    # ── Storage ────────────────────────────────────────────────────────────
    dataset_storage_path: str = "./data/datasets"
    model_storage_path: str = "./data/models"

    # ── MLflow (Phase 2) ───────────────────────────────────────────────────
    mlflow_tracking_uri: str = "sqlite:///./data/mlflow.db"

    # ── Anthropic Claude advisor ───────────────────────────────────────────
    anthropic_api_key: Optional[str] = None

    # ── Multi-provider LLM backend ──────────────────────────────────────────
    # llm_provider: "anthropic" (default) | "bedrock" | "openai_compat" |
    #               "ollama" | "groq" | "openrouter" | "together"
    # See llm/base.py:resolve_backend() for full resolution logic.
    llm_provider: str           = "anthropic"
    llm_model: Optional[str]    = None   # overrides provider default if set
    llm_base_url: Optional[str] = None   # required for "openai_compat"; presets exist for others
    llm_api_key: Optional[str]  = None   # for openai_compat providers (Groq, OpenRouter, Together)

    # ── Background job queue ────────────────────────────────────────────────
    # job_queue_backend: "memory" (default, no Redis required) | "arq" (production, Redis-backed)
    job_queue_backend: str = "memory"
    job_queue_max_concurrent: int = 4    # for "memory" backend
    redis_url: str = "redis://localhost:6379"   # for "arq" backend

    # ── Observability (OpenTelemetry tracing) ───────────────────────────────
    # otel_exporter: "console" (default, prints spans to stdout) | "otlp" | "none"
    otel_exporter: str = "console"
    otel_endpoint: str = "http://localhost:4318/v1/traces"   # for "otlp" exporter

    # ── Experiment tracking ──────────────────────────────────────────────────
    # tracking_backend: "mlflow" (default, self-hostable, already used for
    # mlflow_tracking_uri above) | "wandb" (cloud SaaS) | "none" (disabled)
    tracking_backend: str = "mlflow"
    wandb_api_key: Optional[str] = None
    wandb_project: str = "crucible"

    # ── Encryption key for sensitive fields (OAuth tokens, DB passwords) ───
    # Must be exactly 32 bytes (256-bit) when base64-decoded.
    # Generate with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    encryption_key: Optional[str] = None

    # ── CORS ───────────────────────────────────────────────────────────────
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── Authentication (Phase 11) ──────────────────────────────────────────
    # Generate a strong secret key:
    #   python -c "import secrets; print(secrets.token_hex(32))"
    # Never commit the real key to git — store in .env or environment.
    secret_key: str = "change-me-in-production-generate-with-secrets-token-hex-32"
    access_token_expire_minutes: int = 1440    # 24 hours
    refresh_token_expire_days: int = 30

    # Set to True in local development to skip token validation.
    # All protected endpoints return a synthetic "dev" admin user.
    # NEVER set to True in production.
    disable_auth: bool = True   # default True during build-out; flip to False before deploy

    # ── Storage backend ────────────────────────────────────────────────────
    # Controls where Crucible stores uploaded files and model artifacts.
    # "local"  → local filesystem (default for development)
    # "s3"     → AWS S3 (requires aws_bucket_name)
    storage_backend: str = "local"
    storage_local_root: str = "./data/storage"   # root dir for local backend

    # AWS / S3 configuration (used when storage_backend = "s3")
    aws_bucket_name: str   = ""
    aws_region: str        = "us-east-1"
    aws_storage_prefix: str = "crucible/"        # key prefix within the bucket
    aws_endpoint_url: str  = ""                  # override for MinIO / R2 / custom


settings = Settings()

# Ensure storage directories exist at config load time
os.makedirs(settings.dataset_storage_path, exist_ok=True)
os.makedirs(settings.model_storage_path, exist_ok=True)
os.makedirs("./data", exist_ok=True)
