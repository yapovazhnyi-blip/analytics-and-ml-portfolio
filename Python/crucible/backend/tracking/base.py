"""
Tracking Backend Abstraction — experiment tracking provider independence.

WHY ABSTRACT THIS, GIVEN MLFLOW ALREADY WORKS
-------------------------------------------------
Crucible already logs every training run to MLflow: params, holdout metrics,
and the model artifact, with the run_id stored on the Experiment record.
That's real, working code — this isn't filling a missing capability.

The reason to abstract it anyway is the same reason LLMBackend and
StorageBackend exist: provider choice isn't a technical question, it's an
organisational one. A team standardises on MLflow because it's self-hostable
with no per-seat cost. Another standardises on Weights & Biases because its
sweep visualisation and team collaboration UI are better for their workflow.
Neither choice is wrong — a platform that only supports one forces every
team to either migrate their tracking history or run a second, unintegrated
system alongside it.

THIS FOLLOWS THE SAME PATTERN AS THREE EXISTING ABSTRACTIONS
------------------------------------------------------------------
  StorageBackend  (storage/base.py)   — Local | S3
  LLMBackend      (llm/base.py)       — Anthropic | Bedrock | OpenAI-compatible
  JobQueueBackend (jobs/queue.py)     — In-memory | ARQ/Redis

TrackingBackend completes the set: one default that requires no external
account (MLflow, self-hostable, already integrated), one cloud SaaS
alternative (Weights & Biases) selected via settings.tracking_backend.

INTERFACE
---------
A tracking run has a simple, universal lifecycle across every provider:
  start a run → log params → log metrics → log an artifact → end the run

Both MLflow and W&B's Python SDKs follow this exact shape (mlflow.start_run /
wandb.init, log_params/log, log_artifact, end_run/finish) — the abstraction
adds essentially zero translation overhead, unlike the LLM backends which
had to bridge genuinely different message formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrackedRun:
    """Result of a completed tracking run — what gets stored on the Experiment record."""
    run_id: Optional[str]       # provider's run identifier, or None if tracking failed/disabled
    provider: str                # "mlflow" | "wandb" | "none"
    run_url: Optional[str] = None   # direct link to the run in the provider's UI, if available
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.run_id is not None and self.error is None


class TrackingBackend(ABC):
    """
    Abstract base for experiment tracking providers.

    log_run() is the single entry point — it covers the full
    start → log params → log metrics → log artifact → end lifecycle in
    one call, matching how training/runner.py actually uses tracking
    today (one shot at the end of a completed training run, not
    incremental logging mid-training). A future incremental-logging use
    case (e.g. live loss curves during DPO training) would extend this
    interface with start_run()/log_metrics()/end_run() as separate calls;
    today's single log_run() call is the right shape for what's needed.
    """

    @abstractmethod
    def log_run(
        self,
        run_name: str,
        params: dict,
        metrics: dict,
        artifact_path: Optional[str] = None,
        tags: Optional[dict] = None,
    ) -> TrackedRun:
        """
        Logs a complete training run in one call.

        Args:
            run_name:      Human-readable name for this run.
            params:        Hyperparameters / config values (stringified internally).
            metrics:       Numeric metrics (non-numeric values are silently dropped).
            artifact_path: Local file path to upload as the run's artifact (e.g. the
                           trained model). Optional — some callers may only want metrics.
            tags:          Optional key-value tags for filtering/search in the provider UI.

        Returns:
            TrackedRun with the provider's run_id, or with error set if logging
            failed. This method must NEVER raise — tracking is always best-effort;
            a tracking failure must not fail the training run it's attached to.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...


def get_tracking_backend() -> TrackingBackend:
    """
    Factory — returns the configured tracking backend based on settings.

    settings.tracking_backend = "mlflow" (default) | "wandb" | "none"
    """
    from config import settings
    backend = (getattr(settings, "tracking_backend", "") or "mlflow").lower()

    if backend == "wandb":
        from tracking.wandb_backend import WandBBackend
        return WandBBackend(
            api_key=getattr(settings, "wandb_api_key", "") or "",
            project=getattr(settings, "wandb_project", "crucible") or "crucible",
        )

    if backend == "none":
        from tracking.null_backend import NullTrackingBackend
        return NullTrackingBackend()

    from tracking.mlflow_backend import MLflowBackend
    return MLflowBackend()
