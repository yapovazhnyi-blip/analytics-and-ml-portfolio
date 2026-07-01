"""
OpenTelemetry Tracing — distributed tracing for Crucible.

WHY TRACING, NOT JUST LOGGING
--------------------------------
structlog (already used throughout Crucible) gives you individual log lines.
Tracing gives you a CAUSAL TREE: which operations happened inside which other
operations, and how long each one took relative to its parent.

For a multi-agent run, a log file shows:
  "Supervisor routing to dataset_analyst"
  "Tool call: list_datasets"
  "Tool call: run_profiling"
  "Supervisor routing to model_trainer"
  "Tool call: start_experiment"

A trace shows the same information as a waterfall diagram with exact
timing: the multi-agent run took 4.2s total, of which run_profiling took
2.8s (66%) — immediately telling you where to optimise, without manually
correlating timestamps across log lines.

WHAT GETS INSTRUMENTED
-------------------------
1. AUTOMATIC: every HTTP request via FastAPIInstrumentor — method, path,
   status code, and duration, with zero code changes to route handlers.

2. MANUAL (via @traced / start_span): the operations that matter for
   understanding ML platform behaviour specifically:
     - Agent tool calls (which tool, how long, success/failure)
     - Training job phases (profiling, AutoML search, SHAP, calibration)
     - Multi-agent supervisor routing decisions

EXPORTERS
---------
  console  — prints spans to stdout (default, zero infra, good for dev/demo)
  otlp     — exports to any OpenTelemetry-compatible backend (Jaeger,
             Grafana Tempo, Honeycomb, AWS X-Ray via the OTLP-X-Ray adapter)
  none     — tracing disabled entirely (zero overhead)

Configured via settings.otel_exporter: "console" (default) | "otlp" | "none"
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any, Callable, Optional

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

_tracer: Optional[trace.Tracer] = None
_initialised = False


def setup_tracing(app=None) -> None:
    """
    Initialises the OpenTelemetry SDK and instruments the FastAPI app.

    Call once at application startup (see main.py lifespan).
    Idempotent — safe to call multiple times (subsequent calls no-op).
    """
    global _tracer, _initialised
    if _initialised:
        return

    from config import settings
    exporter_type = getattr(settings, "otel_exporter", "console")

    if exporter_type == "none":
        _initialised = True
        return

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor,
    )
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create({
        "service.name": "crucible-backend",
        "service.version": "1.0.0",
    })
    provider = TracerProvider(resource=resource)

    if exporter_type == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        otlp_endpoint = getattr(settings, "otel_endpoint", "http://localhost:4318/v1/traces")
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    else:   # console
        exporter = ConsoleSpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("crucible")

    if app is not None:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)

    _initialised = True


def get_tracer() -> trace.Tracer:
    """Returns the configured tracer, initialising a no-op one if setup_tracing() wasn't called."""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("crucible")
    return _tracer


@contextmanager
def start_span(name: str, attributes: Optional[dict[str, Any]] = None):
    """
    Context manager for a manual span. Use for any operation worth measuring
    that isn't already covered by FastAPIInstrumentor's automatic HTTP spans.

    Usage:
        with start_span("training.profiling", {"dataset_id": 42}):
            report = await runner.run(df, ...)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, _safe_attr(value))
        try:
            yield span
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise


def traced(name: Optional[str] = None):
    """
    Decorator version of start_span — wraps a function (sync or async) in a span.

    Usage:
        @traced("agent.tool_call")
        async def execute_tool(self, tool_name, tool_input):
            ...
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        import asyncio
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with start_span(span_name):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with start_span(span_name):
                    return func(*args, **kwargs)
            return sync_wrapper

    return decorator


def _safe_attr(value: Any) -> Any:
    """OTel span attributes must be str/int/float/bool or homogeneous sequences thereof."""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
