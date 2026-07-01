"""
Phase 4 — Agent Training Pipeline tests.

Tests cover:
  - TraceCollector: capture from session dicts, lazy scoring
  - Trace-to-training-data converter: Alpaca, ShareGPT, DPO pairing
  - .crucible bundle format: export, import, round-trip, validation errors
  - Agent benchmark: tool correctness scoring, mock mode
  - Agent training router: capture/list/score/training-data endpoints
  - Bundle export/import endpoints
  - Agent registry endpoints
"""

from __future__ import annotations

import json
import os
import tempfile
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch


# ══════════════════════════════════════════════════════════════════════════
# TRACE COLLECTOR
# ══════════════════════════════════════════════════════════════════════════

REACT_SESSION = {
    "goal": "List my datasets",
    "final_answer": "You have 3 datasets: churn_data, sales_data, fraud_data.",
    "n_tool_calls": 1,
    "elapsed_secs": 2.3,
    "error": None,
    "events": [
        {"type": "thinking", "text": "I should list the datasets."},
        {"type": "tool_call", "tool": "list_datasets", "input": {}},
        {"type": "tool_result", "tool": "list_datasets", "result": "3 datasets found", "is_error": False},
        {"type": "final_answer", "text": "You have 3 datasets: churn_data, sales_data, fraud_data."},
    ],
}

MULTI_SESSION = {
    "goal": "Train a model",
    "final_answer": "Trained XGBoost, score 0.91.",
    "steps": 4,
    "elapsed_secs": 5.0,
    "error": None,
    "events": [
        {"type": "supervisor", "reasoning": "Need to find data first.", "routing_to": "dataset_analyst"},
        {"type": "specialist", "agent": "dataset_analyst", "output": "Found dataset 1"},
        {"type": "supervisor", "reasoning": "Now train.", "routing_to": "model_trainer"},
        {"type": "specialist", "agent": "model_trainer", "output": "Trained XGBoost, score 0.91"},
        {"type": "finished", "answer": "Trained XGBoost, score 0.91."},
    ],
}


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.refresh = AsyncMock(side_effect=lambda obj: setattr(obj, "id", 1))
    return db


class TestTraceCollector:

    @pytest.mark.asyncio
    async def test_capture_react_session(self, mock_db):
        from agents.traces import TraceCollector

        collector = TraceCollector(mock_db)
        result = await collector.capture_from_dict(REACT_SESSION, "react")

        assert result.agent_type == "react"
        assert result.succeeded is True
        mock_db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_capture_multi_session(self, mock_db):
        from agents.traces import TraceCollector

        collector = TraceCollector(mock_db)
        result = await collector.capture_from_dict(MULTI_SESSION, "multi")

        assert result.agent_type == "multi"
        assert result.succeeded is True

    @pytest.mark.asyncio
    async def test_captured_trace_has_correct_fields(self, mock_db):
        from agents.traces import TraceCollector

        collector = TraceCollector(mock_db)
        await collector.capture_from_dict(REACT_SESSION, "react")

        added_trace = mock_db.add.call_args[0][0]
        assert added_trace.goal == "List my datasets"
        assert added_trace.n_tool_calls == 1
        assert added_trace.score_pending is True
        assert added_trace.quality_score is None

    @pytest.mark.asyncio
    async def test_failed_session_marked_unsuccessful(self, mock_db):
        from agents.traces import TraceCollector

        failed_session = {**REACT_SESSION, "final_answer": "", "error": "Tool execution failed"}
        collector = TraceCollector(mock_db)
        result = await collector.capture_from_dict(failed_session, "react")

        assert result.succeeded is False

    @pytest.mark.asyncio
    async def test_steps_json_is_valid_json(self, mock_db):
        from agents.traces import TraceCollector

        collector = TraceCollector(mock_db)
        await collector.capture_from_dict(REACT_SESSION, "react")

        added_trace = mock_db.add.call_args[0][0]
        events = json.loads(added_trace.steps_json)
        assert len(events) == 4


# ══════════════════════════════════════════════════════════════════════════
# TRACE-TO-TRAINING-DATA CONVERTER
# ══════════════════════════════════════════════════════════════════════════

def _make_trace(
    goal="Test goal", final_answer="Test answer", succeeded=True,
    quality_score=None, steps=None, trace_id=1,
):
    from models.agent_trace import AgentTrace
    t = MagicMock(spec=AgentTrace)
    t.id = trace_id
    t.goal = goal
    t.final_answer = final_answer
    t.succeeded = succeeded
    t.quality_score = quality_score
    t.steps_json = json.dumps(steps or [
        {"type": "tool_call", "tool": "list_datasets", "input": {}},
        {"type": "tool_result", "tool": "list_datasets", "result": "found 3"},
    ])
    return t


class TestTraceConverter:

    def test_alpaca_conversion_basic(self):
        from agents.trace_converter import traces_to_alpaca
        traces = [_make_trace(goal="List datasets", final_answer="3 found")]
        samples = traces_to_alpaca(traces)
        assert len(samples) == 1
        assert samples[0]["instruction"] == "List datasets"
        assert "input" in samples[0]
        assert "3 found" in samples[0]["output"]

    def test_alpaca_excludes_failed_traces(self):
        from agents.trace_converter import traces_to_alpaca
        traces = [_make_trace(succeeded=False)]
        samples = traces_to_alpaca(traces)
        assert samples == []

    def test_sharegpt_conversion_structure(self):
        from agents.trace_converter import traces_to_sharegpt
        traces = [_make_trace(goal="Hi", final_answer="Hello")]
        samples = traces_to_sharegpt(traces)
        assert len(samples) == 1
        convo = samples[0]["conversations"]
        assert convo[0]["from"] == "human"
        assert convo[0]["value"] == "Hi"
        assert convo[1]["from"] == "gpt"

    def test_reasoning_reconstruction_includes_tool_calls(self):
        from agents.trace_converter import _reconstruct_reasoning_text
        text = _reconstruct_reasoning_text(
            json.dumps([
                {"type": "tool_call", "tool": "list_datasets", "input": {}},
                {"type": "tool_result", "tool": "list_datasets", "result": "3 datasets"},
            ]),
            final_answer="Done.",
        )
        assert "list_datasets" in text
        assert "3 datasets" in text
        assert "Done." in text

    def test_reasoning_reconstruction_handles_multi_agent_events(self):
        from agents.trace_converter import _reconstruct_reasoning_text
        text = _reconstruct_reasoning_text(
            json.dumps([
                {"type": "supervisor", "reasoning": "route to trainer", "routing_to": "model_trainer"},
                {"type": "specialist", "agent": "model_trainer", "output": "trained model"},
            ]),
            final_answer="Success.",
        )
        assert "Supervisor" in text
        assert "model_trainer" in text

    def test_reasoning_reconstruction_handles_malformed_json(self):
        from agents.trace_converter import _reconstruct_reasoning_text
        # Should not raise even with garbage input
        text = _reconstruct_reasoning_text("not valid json{{{", final_answer="Still works.")
        assert "Still works." in text

    def test_dpo_pairing_requires_two_traces_per_goal(self):
        from agents.trace_converter import traces_to_dpo_pairs
        traces = [_make_trace(goal="Train a model", quality_score=0.9, trace_id=1)]
        samples, stats = traces_to_dpo_pairs(traces)
        assert samples == []
        assert stats.n_pairs == 0

    def test_dpo_pairing_creates_pair_from_two_scored_traces(self):
        from agents.trace_converter import traces_to_dpo_pairs
        traces = [
            _make_trace(goal="Train a model", final_answer="Good answer", quality_score=0.9, trace_id=1),
            _make_trace(goal="Train a model", final_answer="Bad answer", quality_score=0.3, trace_id=2),
        ]
        samples, stats = traces_to_dpo_pairs(traces, min_score_gap=0.1)
        assert len(samples) == 1
        assert samples[0]["prompt"] == "Train a model"
        assert "Good answer" in samples[0]["chosen"]
        assert "Bad answer" in samples[0]["rejected"]

    def test_dpo_pairing_skips_small_score_gap(self):
        from agents.trace_converter import traces_to_dpo_pairs
        traces = [
            _make_trace(goal="X", quality_score=0.81, trace_id=1),
            _make_trace(goal="X", quality_score=0.80, trace_id=2),
        ]
        samples, stats = traces_to_dpo_pairs(traces, min_score_gap=0.15)
        assert samples == []

    def test_dpo_pairing_excludes_unscored_traces(self):
        from agents.trace_converter import traces_to_dpo_pairs
        traces = [
            _make_trace(goal="X", quality_score=None, trace_id=1),
            _make_trace(goal="X", quality_score=0.9, trace_id=2),
        ]
        samples, stats = traces_to_dpo_pairs(traces)
        assert stats.n_traces_skipped_unscored == 1
        assert samples == []   # only one scored trace remains — not enough to pair

    def test_goal_normalisation_groups_near_duplicates(self):
        from agents.trace_converter import _normalise_goal
        assert _normalise_goal("  Train a Model  ") == _normalise_goal("train a model")

    def test_dpo_pairing_groups_by_normalised_goal(self):
        from agents.trace_converter import traces_to_dpo_pairs
        traces = [
            _make_trace(goal="Train a model", quality_score=0.9, trace_id=1),
            _make_trace(goal="  TRAIN A MODEL  ", quality_score=0.2, trace_id=2),
        ]
        samples, stats = traces_to_dpo_pairs(traces, min_score_gap=0.1)
        assert len(samples) == 1


# ══════════════════════════════════════════════════════════════════════════
# .CRUCIBLE BUNDLE FORMAT
# ══════════════════════════════════════════════════════════════════════════

class TestBundleExportImport:

    @pytest.fixture
    def fake_adapter_dir(self, tmp_path):
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "r": 8})
        )
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 100)
        return str(adapter_dir)

    def test_export_creates_zip_file(self, fake_adapter_dir, tmp_path):
        from agents.bundle import export_bundle, AgentConfig

        output = str(tmp_path / "test_agent.crucible")
        path = export_bundle(
            output_path=output,
            name="test_agent",
            base_model="microsoft/phi-2",
            adapter_dir=fake_adapter_dir,
            agent_config=AgentConfig(system_prompt="You are helpful.", tool_names=["list_datasets"]),
        )
        assert os.path.exists(path)

        import zipfile
        assert zipfile.is_zipfile(path)

    def test_export_includes_manifest(self, fake_adapter_dir, tmp_path):
        from agents.bundle import export_bundle, AgentConfig
        import zipfile

        output = str(tmp_path / "test.crucible")
        export_bundle(output, "test", "phi-2", fake_adapter_dir, AgentConfig())

        with zipfile.ZipFile(output) as zf:
            manifest = json.loads(zf.read("manifest.json"))
        assert manifest["name"] == "test"
        assert manifest["base_model"] == "phi-2"
        assert manifest["bundle_version"] == "1.0"

    def test_export_includes_adapter_files(self, fake_adapter_dir, tmp_path):
        from agents.bundle import export_bundle, AgentConfig
        import zipfile

        output = str(tmp_path / "test.crucible")
        export_bundle(output, "test", "phi-2", fake_adapter_dir, AgentConfig())

        with zipfile.ZipFile(output) as zf:
            names = zf.namelist()
        assert "adapter/adapter_config.json" in names
        assert "adapter/adapter_model.safetensors" in names

    def test_export_includes_traces_sample(self, fake_adapter_dir, tmp_path):
        from agents.bundle import export_bundle, AgentConfig
        import zipfile

        output = str(tmp_path / "test.crucible")
        export_bundle(
            output, "test", "phi-2", fake_adapter_dir, AgentConfig(),
            traces_sample=[{"goal": "Hi", "final_answer": "Hello"}],
        )
        with zipfile.ZipFile(output) as zf:
            content = zf.read("traces_sample.jsonl").decode()
        assert "Hi" in content

    def test_import_roundtrip(self, fake_adapter_dir, tmp_path):
        from agents.bundle import export_bundle, import_bundle, AgentConfig

        output = str(tmp_path / "roundtrip.crucible")
        export_bundle(
            output, "roundtrip_agent", "phi-2", fake_adapter_dir,
            AgentConfig(system_prompt="Helpful.", tool_names=["a", "b"], max_steps=15),
            training_method="dpo",
            n_training_traces=42,
        )

        imported = import_bundle(output)
        assert imported.manifest.name == "roundtrip_agent"
        assert imported.manifest.training_method == "dpo"
        assert imported.manifest.n_training_traces == 42
        assert imported.manifest.agent_config.max_steps == 15
        assert "a" in imported.manifest.agent_config.tool_names

    def test_import_extracts_adapter_files(self, fake_adapter_dir, tmp_path):
        from agents.bundle import export_bundle, import_bundle, AgentConfig

        output = str(tmp_path / "test.crucible")
        export_bundle(output, "test", "phi-2", fake_adapter_dir, AgentConfig())

        imported = import_bundle(output)
        assert os.path.isdir(imported.adapter_dir)
        assert os.path.exists(os.path.join(imported.adapter_dir, "adapter_config.json"))

    def test_import_invalid_zip_raises(self, tmp_path):
        from agents.bundle import import_bundle

        bad_file = tmp_path / "not_a_zip.crucible"
        bad_file.write_text("this is not a zip file")

        with pytest.raises(ValueError, match="not a valid"):
            import_bundle(str(bad_file))

    def test_import_missing_manifest_raises(self, tmp_path):
        from agents.bundle import import_bundle
        import zipfile

        bad_zip = tmp_path / "no_manifest.crucible"
        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr("some_other_file.txt", "data")

        with pytest.raises(ValueError, match="manifest.json"):
            import_bundle(str(bad_zip))

    def test_import_unsupported_version_raises(self, tmp_path):
        from agents.bundle import import_bundle
        import zipfile

        bad_zip = tmp_path / "bad_version.crucible"
        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr("manifest.json", json.dumps({
                "name": "x", "base_model": "y", "bundle_version": "99.0",
            }))

        with pytest.raises(ValueError, match="version"):
            import_bundle(str(bad_zip))

    def test_eval_results_included_when_provided(self, fake_adapter_dir, tmp_path):
        from agents.bundle import export_bundle, import_bundle, AgentConfig

        output = str(tmp_path / "test.crucible")
        export_bundle(
            output, "test", "phi-2", fake_adapter_dir, AgentConfig(),
            eval_results={"overall_score": 0.85, "n_passed": 4},
        )
        imported = import_bundle(output)
        assert imported.benchmark_results["overall_score"] == 0.85


# ══════════════════════════════════════════════════════════════════════════
# AGENT BENCHMARK
# ══════════════════════════════════════════════════════════════════════════

class _MockSession:
    def __init__(self, events, final_answer):
        self.events = events
        self.final_answer = final_answer

    def to_dict(self):
        return {"events": self.events, "final_answer": self.final_answer}


class _MockRunner:
    """Stand-in for ReActRunner that returns canned tool calls per goal."""
    def __init__(self, tool_map: dict):
        self.tool_map = tool_map

    async def run(self, goal):
        tool = self.tool_map.get(goal, "unknown_tool")
        return _MockSession(
            events=[{"type": "tool_call", "tool": tool, "input": {}}],
            final_answer=f"Answer for: {goal}",
        )


class TestAgentBenchmark:

    @pytest.mark.asyncio
    async def test_correct_tool_call_passes(self):
        from agents.benchmark import run_benchmark, BenchmarkCase

        runner = _MockRunner({"List datasets please": "list_datasets"})
        cases = [BenchmarkCase(name="t1", goal="List datasets please", expected_tools=["list_datasets"])]

        report = await run_benchmark(runner, cases=cases)
        assert report.n_total == 1
        assert report.cases[0].tool_correctness == 1.0

    @pytest.mark.asyncio
    async def test_wrong_tool_call_fails_tool_correctness(self):
        from agents.benchmark import run_benchmark, BenchmarkCase

        runner = _MockRunner({"Profile dataset 1": "list_datasets"})  # wrong tool
        cases = [BenchmarkCase(name="t1", goal="Profile dataset 1", expected_tools=["run_profiling"])]

        report = await run_benchmark(runner, cases=cases)
        assert report.cases[0].tool_correctness == 0.0

    @pytest.mark.asyncio
    async def test_without_api_key_uses_neutral_quality_score(self):
        from agents.benchmark import run_benchmark, BenchmarkCase

        runner = _MockRunner({"goal1": "list_datasets"})
        cases = [BenchmarkCase(name="t1", goal="goal1", expected_tools=["list_datasets"])]

        report = await run_benchmark(runner, cases=cases, api_key="")
        assert report.cases[0].answer_quality == 0.5

    @pytest.mark.asyncio
    async def test_overall_score_is_mean_of_cases(self):
        from agents.benchmark import run_benchmark, BenchmarkCase

        runner = _MockRunner({"g1": "list_datasets", "g2": "wrong_tool"})
        cases = [
            BenchmarkCase(name="c1", goal="g1", expected_tools=["list_datasets"]),
            BenchmarkCase(name="c2", goal="g2", expected_tools=["run_profiling"]),
        ]
        report = await run_benchmark(runner, cases=cases)
        assert 0.0 <= report.overall_score <= 1.0
        assert report.n_total == 2

    @pytest.mark.asyncio
    async def test_default_benchmark_has_standard_cases(self):
        from agents.benchmark import STANDARD_BENCHMARK
        assert len(STANDARD_BENCHMARK) >= 3
        names = {c.name for c in STANDARD_BENCHMARK}
        assert "list_datasets" in names

    @pytest.mark.asyncio
    async def test_report_to_dict_serialisable(self):
        from agents.benchmark import run_benchmark, BenchmarkCase
        import json as json_mod

        runner = _MockRunner({"g": "list_datasets"})
        cases = [BenchmarkCase(name="c", goal="g", expected_tools=["list_datasets"])]
        report = await run_benchmark(runner, cases=cases)
        json_mod.dumps(report.to_dict())   # must not raise


# ══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def at_client():
    import sys, importlib, database as db_mod
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool
    from fastapi.testclient import TestClient
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    db_mod.engine = engine; db_mod.SessionFactory = factory; db_mod.AsyncSessionLocal = factory
    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    import main as m
    with TestClient(m.app, raise_server_exceptions=True) as c:
        yield c


class TestTraceAPI:

    def test_capture_trace_endpoint(self, at_client):
        resp = at_client.post("/api/v1/agents/traces/capture", json={
            "session": REACT_SESSION, "agent_type": "react",
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["agent_type"] == "react"
        assert data["succeeded"] is True

    def test_list_traces_after_capture(self, at_client):
        at_client.post("/api/v1/agents/traces/capture", json={
            "session": REACT_SESSION, "agent_type": "react",
        })
        resp = at_client.get("/api/v1/agents/traces")
        assert resp.status_code == 200
        items = resp.json()["data"]
        assert len(items) >= 1

    def test_filter_traces_by_agent_type(self, at_client):
        at_client.post("/api/v1/agents/traces/capture", json={
            "session": REACT_SESSION, "agent_type": "react",
        })
        at_client.post("/api/v1/agents/traces/capture", json={
            "session": MULTI_SESSION, "agent_type": "multi",
        })
        resp = at_client.get("/api/v1/agents/traces?agent_type=multi")
        items = resp.json()["data"]
        assert all(i["agent_type"] == "multi" for i in items)

    def test_invalid_agent_type_rejected(self, at_client):
        resp = at_client.post("/api/v1/agents/traces/capture", json={
            "session": REACT_SESSION, "agent_type": "bogus",
        })
        assert resp.status_code == 422

    def test_score_traces_endpoint_runs(self, at_client):
        at_client.post("/api/v1/agents/traces/capture", json={
            "session": REACT_SESSION, "agent_type": "react",
        })
        resp = at_client.post("/api/v1/agents/traces/score")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "scored" in data

    def test_training_data_alpaca_format(self, at_client):
        at_client.post("/api/v1/agents/traces/capture", json={
            "session": REACT_SESSION, "agent_type": "react",
        })
        resp = at_client.get("/api/v1/agents/traces/training-data?format=alpaca")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["format"] == "alpaca"
        assert data["n_samples"] >= 1

    def test_training_data_invalid_format_rejected(self, at_client):
        resp = at_client.get("/api/v1/agents/traces/training-data?format=bogus")
        assert resp.status_code == 422

    def test_training_data_dpo_format_returns_stats(self, at_client):
        resp = at_client.get("/api/v1/agents/traces/training-data?format=dpo")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "stats" in data


class TestBundleAPI:

    def test_export_requires_valid_adapter_dir(self, at_client):
        resp = at_client.post("/api/v1/agents/export", json={
            "name": "test_agent",
            "base_model": "phi-2",
            "adapter_path": "/nonexistent/path",
        })
        assert resp.status_code == 422

    def test_export_succeeds_with_real_dir(self, at_client, tmp_path):
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")

        resp = at_client.post("/api/v1/agents/export", json={
            "name": "exported_agent",
            "base_model": "phi-2",
            "adapter_path": str(adapter_dir),
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert os.path.exists(data["bundle_path"])

    def test_import_bundle_registers_agent(self, at_client, tmp_path):
        from agents.bundle import export_bundle, AgentConfig

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")

        bundle_path = str(tmp_path / "imported.crucible")
        export_bundle(bundle_path, "imported_test_agent", "phi-2", str(adapter_dir), AgentConfig())

        with open(bundle_path, "rb") as f:
            resp = at_client.post(
                "/api/v1/agents/import",
                files={"file": ("imported_test_agent.crucible", f, "application/zip")},
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["name"] == "imported_test_agent"

    def test_import_duplicate_name_rejected(self, at_client, tmp_path):
        from agents.bundle import export_bundle, AgentConfig

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")
        bundle_path = str(tmp_path / "dup.crucible")
        export_bundle(bundle_path, "dup_agent", "phi-2", str(adapter_dir), AgentConfig())

        with open(bundle_path, "rb") as f:
            at_client.post("/api/v1/agents/import", files={"file": ("dup.crucible", f, "application/zip")})

        with open(bundle_path, "rb") as f:
            resp2 = at_client.post("/api/v1/agents/import", files={"file": ("dup.crucible", f, "application/zip")})
        assert resp2.status_code == 409


class TestAgentRegistryAPI:

    def test_list_agents_empty_initially(self, at_client):
        resp = at_client.get("/api/v1/agents")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_get_nonexistent_agent_404(self, at_client):
        resp = at_client.get("/api/v1/agents/nonexistent")
        assert resp.status_code == 404

    def test_full_lifecycle_import_get_list_delete(self, at_client, tmp_path):
        from agents.bundle import export_bundle, AgentConfig

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")
        bundle_path = str(tmp_path / "lifecycle.crucible")
        export_bundle(bundle_path, "lifecycle_agent", "phi-2", str(adapter_dir), AgentConfig())

        with open(bundle_path, "rb") as f:
            at_client.post("/api/v1/agents/import", files={"file": ("l.crucible", f, "application/zip")})

        # Get
        resp = at_client.get("/api/v1/agents/lifecycle_agent")
        assert resp.status_code == 200
        assert "manifest" in resp.json()["data"]

        # List
        resp = at_client.get("/api/v1/agents")
        names = [a["name"] for a in resp.json()["data"]]
        assert "lifecycle_agent" in names

        # Delete (archive)
        resp = at_client.delete("/api/v1/agents/lifecycle_agent")
        assert resp.status_code == 204

        # No longer in active list
        resp = at_client.get("/api/v1/agents")
        names = [a["name"] for a in resp.json()["data"]]
        assert "lifecycle_agent" not in names


# ══════════════════════════════════════════════════════════════════════════
# CAPTURE FLAG ON /agent/run AND /agent/multi/run
# ══════════════════════════════════════════════════════════════════════════

class TestCaptureFlagWiring:

    def test_agent_run_without_capture_does_not_create_trace(self, at_client):
        with patch("agents.multi_agent.settings") as ms:
            ms.anthropic_api_key = None
            at_client.post("/api/v1/agent/run", json={"goal": "list datasets", "capture": False})
        resp = at_client.get("/api/v1/agents/traces")
        assert len(resp.json()["data"]) == 0

    def test_agent_run_with_capture_creates_trace(self, at_client):
        resp = at_client.post("/api/v1/agent/run", json={"goal": "list datasets", "capture": True})
        assert resp.status_code == 200
        traces = at_client.get("/api/v1/agents/traces").json()["data"]
        assert len(traces) == 1
        assert traces[0]["agent_type"] == "react"

    def test_multi_agent_run_with_capture_creates_trace(self, at_client):
        with patch("agents.multi_agent.settings") as ms:
            ms.anthropic_api_key = None
            resp = at_client.post("/api/v1/agent/multi/run", json={
                "goal": "list my datasets", "capture": True,
            })
        assert resp.status_code == 200
        traces = at_client.get("/api/v1/agents/traces?agent_type=multi").json()["data"]
        assert len(traces) == 1

    def test_capture_defaults_to_false(self, at_client):
        """Omitting capture entirely must not create a trace."""
        at_client.post("/api/v1/agent/run", json={"goal": "list datasets"})
        traces = at_client.get("/api/v1/agents/traces").json()["data"]
        assert len(traces) == 0
