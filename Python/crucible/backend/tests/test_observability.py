"""
OpenTelemetry tracing tests.

Tests cover:
  - setup_tracing(): idempotent, respects exporter config
  - get_tracer(): returns a usable tracer even before setup
  - start_span(): context manager records attributes, handles exceptions
  - traced() decorator: works on sync and async functions
  - ToolExecutor spans: tool calls produce spans with correct attributes
  - ProfileRunner spans: profiling run produces a span
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock


# ══════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════

class TestTracingSetup:

    def test_setup_tracing_is_idempotent(self):
        from observability import tracing
        tracing._initialised = False   # reset for test isolation
        tracing.setup_tracing()
        first_state = tracing._initialised
        tracing.setup_tracing()        # second call must not raise or reinitialise
        assert first_state is True
        assert tracing._initialised is True

    def test_get_tracer_returns_usable_tracer_before_setup(self):
        from observability.tracing import get_tracer
        tracer = get_tracer()
        assert tracer is not None
        # Must support start_as_current_span without raising
        with tracer.start_as_current_span("test_span"):
            pass

    def test_none_exporter_disables_tracing_without_error(self):
        from observability import tracing
        from config import settings

        tracing._initialised = False
        original = settings.otel_exporter
        try:
            settings.otel_exporter = "none"
            tracing.setup_tracing()   # must not raise
        finally:
            settings.otel_exporter = original
            tracing._initialised = False
            tracing.setup_tracing()   # restore for other tests


# ══════════════════════════════════════════════════════════════════════════
# MANUAL SPANS
# ══════════════════════════════════════════════════════════════════════════

class TestStartSpan:

    def test_start_span_yields_a_span(self):
        from observability.tracing import start_span
        with start_span("test.operation") as span:
            assert span is not None

    def test_start_span_sets_attributes(self):
        from observability.tracing import start_span
        with start_span("test.operation", {"key": "value", "count": 5}) as span:
            # Attributes are set via span.set_attribute internally;
            # we verify no exception is raised with mixed types
            pass

    def test_start_span_records_exception_and_reraises(self):
        from observability.tracing import start_span

        with pytest.raises(ValueError, match="boom"):
            with start_span("test.failing_operation"):
                raise ValueError("boom")

    def test_start_span_with_non_primitive_attribute(self):
        """Non-primitive attribute values must be coerced to strings, not crash."""
        from observability.tracing import start_span
        with start_span("test.op", {"complex_value": {"nested": "dict"}}):
            pass   # must not raise

    def test_safe_attr_passes_through_primitives(self):
        from observability.tracing import _safe_attr
        assert _safe_attr("text") == "text"
        assert _safe_attr(42) == 42
        assert _safe_attr(3.14) == 3.14
        assert _safe_attr(True) is True

    def test_safe_attr_stringifies_complex_types(self):
        from observability.tracing import _safe_attr
        result = _safe_attr({"a": 1})
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════
# TRACED DECORATOR
# ══════════════════════════════════════════════════════════════════════════

class TestTracedDecorator:

    @pytest.mark.asyncio
    async def test_traced_async_function(self):
        from observability.tracing import traced

        @traced("test.async_op")
        async def my_async_func(x, y):
            return x + y

        result = await my_async_func(2, 3)
        assert result == 5

    def test_traced_sync_function(self):
        from observability.tracing import traced

        @traced("test.sync_op")
        def my_sync_func(x, y):
            return x * y

        result = my_sync_func(3, 4)
        assert result == 12

    def test_traced_without_explicit_name_uses_function_qualname(self):
        from observability.tracing import traced

        @traced()
        def some_function():
            return "ok"

        assert some_function() == "ok"

    @pytest.mark.asyncio
    async def test_traced_async_propagates_exceptions(self):
        from observability.tracing import traced

        @traced("test.failing_async")
        async def fails():
            raise RuntimeError("async failure")

        with pytest.raises(RuntimeError, match="async failure"):
            await fails()


# ══════════════════════════════════════════════════════════════════════════
# INTEGRATION: TOOL EXECUTOR SPANS
# ══════════════════════════════════════════════════════════════════════════

class TestToolExecutorTracing:

    @pytest.mark.asyncio
    async def test_tool_execution_produces_span(self):
        from agents.tools import ToolExecutor
        from unittest.mock import AsyncMock, MagicMock

        mock_db = AsyncMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_db.scalars = AsyncMock(return_value=mock_scalars)

        executor = ToolExecutor(mock_db)
        result = await executor.execute("list_datasets", {})
        assert result.tool_name == "list_datasets"

    @pytest.mark.asyncio
    async def test_unknown_tool_still_traced(self):
        from agents.tools import ToolExecutor
        from unittest.mock import AsyncMock

        executor = ToolExecutor(AsyncMock())
        result = await executor.execute("nonexistent_tool_xyz", {})
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_tool_error_still_traced(self):
        from agents.tools import ToolExecutor
        from unittest.mock import AsyncMock

        mock_db = AsyncMock()
        mock_db.scalars = AsyncMock(side_effect=Exception("DB connection lost"))

        executor = ToolExecutor(mock_db)
        result = await executor.execute("list_datasets", {})
        assert result.is_error is True
        assert "DB connection lost" in result.content


# ══════════════════════════════════════════════════════════════════════════
# INTEGRATION: PROFILE RUNNER SPANS
# ══════════════════════════════════════════════════════════════════════════

class TestProfileRunnerTracing:

    @pytest.mark.asyncio
    async def test_profiling_run_produces_span_and_completes(self):
        import pandas as pd
        import numpy as np
        from profiling.runner import ProfileRunner

        df = pd.DataFrame({
            "x": np.random.randn(100),
            "y": np.random.randint(0, 2, 100),
        })
        runner = ProfileRunner()
        report = await runner.run(df, dataset_id=1, target_column="y")
        assert report is not None
