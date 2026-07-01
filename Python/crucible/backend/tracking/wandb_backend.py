"""
Weights & Biases tracking backend.

WHY A TEAM WOULD CHOOSE THIS OVER MLFLOW
--------------------------------------------
W&B's value proposition over self-hosted MLflow is almost entirely UI and
collaboration: richer run comparison dashboards, native hyperparameter
sweep visualisation, team-shared run history without standing up and
maintaining a tracking server. The cost is a required cloud account and
(for private projects beyond the free tier) a per-seat subscription.

This is consistent with BYOK throughout Crucible: settings.wandb_api_key
is the server-level fallback; nothing here prevents adding a per-user W&B
key later via the same auth.key_manager pattern already used for Anthropic
keys, if a future multi-tenant deployment needs that.

GRACEFUL DEGRADATION
------------------------
If wandb isn't installed, or no API key is configured, log_run() returns
a TrackedRun with error set rather than raising — exactly like MLflowBackend.
Tracking is always best-effort; a missing W&B account must never fail the
training run it's attached to.
"""

from __future__ import annotations

from typing import Optional

from tracking.base import TrackingBackend, TrackedRun


class WandBBackend(TrackingBackend):
    """
    Logs runs to Weights & Biases.

    Each call to log_run() is a complete, self-contained W&B run:
    wandb.init() → wandb.log() → wandb.log_artifact() → wandb.finish().
    W&B's SDK is synchronous and blocking by default (matching MLflow's
    behaviour here), which is fine for the existing one-shot-at-training-end
    usage in training/runner.py.
    """

    def __init__(self, api_key: str = "", project: str = "crucible"):
        self._api_key = api_key
        self._project = project

    @property
    def provider_name(self) -> str:
        return "wandb"

    def log_run(
        self,
        run_name: str,
        params: dict,
        metrics: dict,
        artifact_path: Optional[str] = None,
        tags: Optional[dict] = None,
    ) -> TrackedRun:
        try:
            import wandb
        except ImportError:
            return TrackedRun(run_id=None, provider="wandb", error="wandb not installed")

        if not self._api_key:
            return TrackedRun(
                run_id=None, provider="wandb",
                error="No W&B API key configured (settings.wandb_api_key)",
            )

        run = None
        try:
            wandb.login(key=self._api_key, verify=False)

            run = wandb.init(
                project=self._project,
                name=run_name,
                config={k: str(v)[:250] for k, v in (params or {}).items()},
                tags=list(tags.keys()) if tags else None,
                reinit=True,   # allow multiple runs within the same process
            )

            if metrics:
                numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
                if numeric_metrics:
                    wandb.log(numeric_metrics)

            if artifact_path:
                artifact = wandb.Artifact(name=f"{run_name}-model", type="model")
                artifact.add_file(artifact_path)
                run.log_artifact(artifact)

            run_id = run.id
            run_url = run.get_url()
            return TrackedRun(run_id=run_id, provider="wandb", run_url=run_url)

        except Exception as exc:
            return TrackedRun(run_id=None, provider="wandb", error=str(exc))

        finally:
            if run is not None:
                wandb.finish()
