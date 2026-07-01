"""
MLflow tracking backend — the default, requires no external account.

This wraps exactly the same logic that lived inline in training/runner.py's
_log_to_mlflow() before this abstraction existed. The behaviour is identical;
only the entry point moved, so existing MLflow run history is unaffected.
"""

from __future__ import annotations

from typing import Optional

from tracking.base import TrackingBackend, TrackedRun


class MLflowBackend(TrackingBackend):
    """
    Logs runs to MLflow. Self-hostable — settings.mlflow_tracking_uri points
    at either a local SQLite-backed store (the Crucible default) or a
    remote MLflow tracking server (see docker-compose.yml's mlflow service).
    """

    @property
    def provider_name(self) -> str:
        return "mlflow"

    def log_run(
        self,
        run_name: str,
        params: dict,
        metrics: dict,
        artifact_path: Optional[str] = None,
        tags: Optional[dict] = None,
    ) -> TrackedRun:
        try:
            import mlflow
        except ImportError:
            return TrackedRun(run_id=None, provider="mlflow", error="mlflow not installed")

        try:
            with mlflow.start_run(run_name=run_name) as run:
                if params:
                    mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
                if metrics:
                    mlflow.log_metrics({
                        k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))
                    })
                if tags:
                    mlflow.set_tags({k: str(v) for k, v in tags.items()})
                if artifact_path:
                    mlflow.log_artifact(artifact_path)

                run_id = run.info.run_id
                tracking_uri = mlflow.get_tracking_uri()
                # Only build a browsable URL for http(s) tracking servers —
                # a local sqlite:/// or file:// URI has no UI to link to.
                run_url = (
                    f"{tracking_uri}/#/experiments/0/runs/{run_id}"
                    if tracking_uri.startswith("http") else None
                )
                return TrackedRun(run_id=run_id, provider="mlflow", run_url=run_url)

        except Exception as exc:
            return TrackedRun(run_id=None, provider="mlflow", error=str(exc))
