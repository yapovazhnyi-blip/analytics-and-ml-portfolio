"""
Null tracking backend — explicit no-op, selected via settings.tracking_backend = "none".

Distinct from a misconfigured MLflow/W&B backend that fails silently: this
is a deliberate choice to disable tracking entirely (e.g. local development
where nobody wants an MLflow SQLite file accumulating in ./data/, or a
demo deployment where tracking provider setup is out of scope). Returning
a TrackedRun with provider="none" and no error makes that intent explicit
in logs and the Experiment record, rather than looking like a failure.
"""

from __future__ import annotations

from typing import Optional

from tracking.base import TrackingBackend, TrackedRun


class NullTrackingBackend(TrackingBackend):

    @property
    def provider_name(self) -> str:
        return "none"

    def log_run(
        self,
        run_name: str,
        params: dict,
        metrics: dict,
        artifact_path: Optional[str] = None,
        tags: Optional[dict] = None,
    ) -> TrackedRun:
        return TrackedRun(run_id=None, provider="none")
