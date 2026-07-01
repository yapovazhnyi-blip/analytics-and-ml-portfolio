"""
Agent tests.

All Claude API calls are mocked — tests run fully offline and cost nothing.
Coverage:
  - Tool schema validity (required fields, correct Anthropic format)
  - ToolExecutor: list_datasets, get_dataset_info, get_experiment_status,
    get_experiment_results
  - ReActRunner: single tool call → final answer, error handling,
    max_steps enforcement, no-API-key graceful error
  - API endpoints: GET /agent/tools, POST /agent/run, POST /agent/run/stream-id
"""

from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool


# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    import sys, importlib, database as db_mod
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    db_mod.engine = engine
    db_mod.SessionFactory = factory
    db_mod.AsyncSessionLocal = factory
    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    import main as app_module
    with TestClient(app_module.app, raise_server_exceptions=True) as c:
        yield c


# ══════════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════

class TestToolDefinitions:

    def test_all_tools_have_required_fields(self):
        from agents.tools import ALL_TOOLS
        for tool in ALL_TOOLS:
            assert tool.name, f"Tool missing name: {tool}"
            assert tool.description, f"Tool '{tool.name}' missing description"
            assert isinstance(tool.input_schema, dict)
            assert "type" in tool.input_schema
            assert "properties" in tool.input_schema

    def test_tools_for_api_format(self):
        from agents.tools import tools_for_api
        api_tools = tools_for_api()
        assert len(api_tools) >= 7
        for t in api_tools:
            assert "name" in t
            assert "description" in t
            assert "input_schema" in t

    def test_tool_map_contains_all_tools(self):
        from agents.tools import TOOL_MAP, ALL_TOOLS
        for tool in ALL_TOOLS:
            assert tool.name in TOOL_MAP

    def test_expected_tools_present(self):
        from agents.tools import TOOL_MAP
        expected = {
            "list_datasets", "get_dataset_info", "run_profiling",
            "start_experiment", "get_experiment_status",
            "get_experiment_results", "generate_deployment",
            "list_rag_documents", "query_rag",
        }
        for name in expected:
            assert name in TOOL_MAP, f"Missing tool: {name}"

    def test_input_schemas_are_valid_json_schema(self):
        """Each input_schema must have type=object and a properties dict."""
        from agents.tools import ALL_TOOLS
        for tool in ALL_TOOLS:
            schema = tool.input_schema
            assert schema.get("type") == "object", f"{tool.name}: schema type must be 'object'"
            assert isinstance(schema.get("properties"), dict), f"{tool.name}: missing properties"


# ══════════════════════════════════════════════════════════════════════════
# TOOL EXECUTOR
# ══════════════════════════════════════════════════════════════════════════

class TestToolExecutor:

    @pytest.fixture
    def mock_db(self):
        """Mock AsyncSession."""
        db = AsyncMock()
        return db

    @pytest.mark.asyncio
    async def test_list_datasets_empty(self, mock_db):
        from agents.tools import ToolExecutor
        from unittest.mock import AsyncMock, MagicMock

        # Mock scalars().all() returning empty list
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_db.scalars = AsyncMock(return_value=mock_result)

        executor = ToolExecutor(mock_db)
        result = await executor.execute("list_datasets", {})
        assert not result.is_error
        assert "No ready datasets" in result.content

    @pytest.mark.asyncio
    async def test_list_datasets_with_data(self, mock_db):
        from agents.tools import ToolExecutor
        from models.dataset import Dataset
        from datetime import datetime

        ds = MagicMock(spec=Dataset)
        ds.id = 1
        ds.name = "sales_data"
        ds.row_count = 5000
        ds.source_type = "csv"
        ds.created_at = datetime(2024, 1, 15)

        mock_result = MagicMock()
        mock_result.all.return_value = [ds]
        mock_db.scalars = AsyncMock(return_value=mock_result)

        executor = ToolExecutor(mock_db)
        result = await executor.execute("list_datasets", {})
        assert not result.is_error
        assert "sales_data" in result.content
        assert "5000" in result.content

    @pytest.mark.asyncio
    async def test_get_dataset_info_not_found(self, mock_db):
        from agents.tools import ToolExecutor
        mock_db.get = AsyncMock(return_value=None)

        executor = ToolExecutor(mock_db)
        result = await executor.execute("get_dataset_info", {"dataset_id": 999})
        assert not result.is_error
        assert "not found" in result.content.lower()

    @pytest.mark.asyncio
    async def test_get_dataset_info_with_schema(self, mock_db):
        from agents.tools import ToolExecutor
        from models.dataset import Dataset

        ds = MagicMock(spec=Dataset)
        ds.id = 1
        ds.name = "churn_data"
        ds.row_count = 10000
        ds.schema_json = json.dumps({
            "columns": {
                "age": {"dtype": "int64"},
                "tenure": {"dtype": "float64"},
                "churn": {"dtype": "int64"},
            }
        })
        mock_db.get = AsyncMock(return_value=ds)

        executor = ToolExecutor(mock_db)
        result = await executor.execute("get_dataset_info", {"dataset_id": 1})
        assert not result.is_error
        assert "churn_data" in result.content
        assert "age" in result.content

    @pytest.mark.asyncio
    async def test_get_experiment_status_running(self, mock_db):
        from agents.tools import ToolExecutor
        from models.experiment import Experiment

        exp = MagicMock(spec=Experiment)
        exp.id = 42
        exp.status = "running"
        mock_db.get = AsyncMock(return_value=exp)

        executor = ToolExecutor(mock_db)
        result = await executor.execute("get_experiment_status", {"experiment_id": 42})
        assert not result.is_error
        assert "running" in result.content.lower()

    @pytest.mark.asyncio
    async def test_get_experiment_status_completed(self, mock_db):
        from agents.tools import ToolExecutor
        from models.experiment import Experiment

        exp = MagicMock(spec=Experiment)
        exp.id = 5
        exp.status = "completed"
        exp.best_family = "XGBoost"
        exp.best_cv_score = 0.923
        mock_db.get = AsyncMock(return_value=exp)

        executor = ToolExecutor(mock_db)
        result = await executor.execute("get_experiment_status", {"experiment_id": 5})
        assert not result.is_error
        assert "completed" in result.content.lower()
        assert "XGBoost" in result.content
        assert "0.923" in result.content

    @pytest.mark.asyncio
    async def test_get_experiment_results_not_completed(self, mock_db):
        from agents.tools import ToolExecutor
        from models.experiment import Experiment

        exp = MagicMock(spec=Experiment)
        exp.id = 7
        exp.status = "running"
        mock_db.get = AsyncMock(return_value=exp)

        executor = ToolExecutor(mock_db)
        result = await executor.execute("get_experiment_results", {"experiment_id": 7})
        assert not result.is_error
        assert "not completed" in result.content.lower()

    @pytest.mark.asyncio
    async def test_get_experiment_results_completed(self, mock_db):
        from agents.tools import ToolExecutor
        from models.experiment import Experiment

        exp = MagicMock(spec=Experiment)
        exp.id = 3
        exp.status = "completed"
        exp.best_family = "RandomForest"
        exp.best_cv_score = 0.87
        exp.results_json = json.dumps({
            "feature_names": ["age", "income", "tenure"],
            "holdout_metrics": {"accuracy": 0.89, "f1": 0.85},
            "best_params": {"n_estimators": 200},
        })
        mock_db.get = AsyncMock(return_value=exp)

        executor = ToolExecutor(mock_db)
        result = await executor.execute("get_experiment_results", {"experiment_id": 3})
        assert not result.is_error
        assert "RandomForest" in result.content
        assert "0.87" in result.content
        assert "accuracy" in result.content.lower()

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, mock_db):
        from agents.tools import ToolExecutor
        executor = ToolExecutor(mock_db)
        result = await executor.execute("nonexistent_tool", {})
        assert result.is_error
        assert "Unknown tool" in result.content

    @pytest.mark.asyncio
    async def test_generate_deployment_not_completed(self, mock_db):
        from agents.tools import ToolExecutor
        from models.experiment import Experiment

        exp = MagicMock(spec=Experiment)
        exp.status = "running"
        mock_db.get = AsyncMock(return_value=exp)

        executor = ToolExecutor(mock_db)
        result = await executor.execute("generate_deployment", {"experiment_id": 1})
        assert not result.is_error
        assert "not completed" in result.content


# ══════════════════════════════════════════════════════════════════════════
# REACT RUNNER
# ══════════════════════════════════════════════════════════════════════════

class TestReActRunner:

    def _make_claude_response(self, text: str = None, tool_uses: list = None) -> dict:
        """Builds a mock Anthropic API response payload."""
        content = []
        if text:
            content.append({"type": "text", "text": text})
        if tool_uses:
            for tu in tool_uses:
                content.append({
                    "type":  "tool_use",
                    "id":    f"toolu_{tu['name']}",
                    "name":  tu["name"],
                    "input": tu.get("input", {}),
                })
        return {
            "content":     content,
            "stop_reason": "end_turn" if not tool_uses else "tool_use",
            "model":       "claude-haiku-mock",
            "usage":       {"input_tokens": 100, "output_tokens": 50},
        }

    def _make_mock_response(self, payload: dict):
        """
        Builds a mock httpx response.
        json() must be a plain MagicMock (sync) because httpx.Response.json() is not async.
        raise_for_status() is also sync.
        """
        resp = MagicMock()
        resp.json = MagicMock(return_value=payload)
        resp.raise_for_status = MagicMock()
        return resp

    def _make_mock_client(self, responses):
        """
        Builds an AsyncClient mock that returns the given responses in sequence.
        client.post() is async (returns a coroutine); the response itself is sync.
        """
        call_count = [0]

        async def _post(*args, **kwargs):
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            return responses[idx]

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = _post
        return mock_client

    @pytest.mark.asyncio
    async def test_runner_returns_session(self):
        """Runner returns an AgentSession with the expected structure."""
        from agents.runner import ReActRunner

        mock_db = AsyncMock()
        payload  = self._make_claude_response(
            text="I can see there are no datasets yet. Please upload a dataset first."
        )
        response = self._make_mock_response(payload)
        client   = self._make_mock_client([response])

        with patch("agents.runner.settings") as ms, \
             patch("agents.runner.httpx.AsyncClient", return_value=client):
            ms.anthropic_api_key = "test-key"
            runner  = ReActRunner(mock_db)
            session = await runner.run("List all datasets")

        assert session.succeeded
        assert "dataset" in session.final_answer.lower()
        assert session.n_tool_calls == 0

    @pytest.mark.asyncio
    async def test_runner_executes_tool_call(self):
        """Runner calls the tool executor when Claude returns a tool_use block."""
        from agents.runner import ReActRunner

        mock_db = AsyncMock()

        r1 = self._make_mock_response(self._make_claude_response(
            text="Let me list the datasets first.",
            tool_uses=[{"name": "list_datasets", "input": {}}],
        ))
        r2 = self._make_mock_response(self._make_claude_response(
            text="No datasets found in Crucible."
        ))
        client = self._make_mock_client([r1, r2])

        with patch("agents.runner.settings") as ms, \
             patch("agents.runner.httpx.AsyncClient", return_value=client), \
             patch.object(
                 __import__("agents.tools", fromlist=["ToolExecutor"]).ToolExecutor,
                 "_run_list_datasets",
                 new_callable=AsyncMock,
                 return_value="No ready datasets found."
             ):
            ms.anthropic_api_key = "test-key"
            session = await ReActRunner(mock_db).run("What datasets are available?")

        assert session.n_tool_calls == 1
        from agents.runner import ToolCallEvent
        tool_calls = [e for e in session.events if isinstance(e, ToolCallEvent)]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "list_datasets"

    @pytest.mark.asyncio
    async def test_runner_without_api_key_returns_error(self):
        """Runner returns an ErrorEvent when no API key is configured."""
        from agents.runner import ReActRunner, ErrorEvent

        mock_db = AsyncMock()

        with patch("agents.runner.settings") as mock_settings:
            mock_settings.anthropic_api_key = None
            runner = ReActRunner(mock_db)
            session = await runner.run("Do something")

        assert not session.succeeded
        error_events = [e for e in session.events if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert "ANTHROPIC_API_KEY" in error_events[0].message

    @pytest.mark.asyncio
    async def test_runner_enforces_max_steps(self):
        """Runner stops after MAX_STEPS tool calls."""
        from agents.runner import ReActRunner

        mock_db = AsyncMock()
        always_tool = self._make_mock_response(self._make_claude_response(
            tool_uses=[{"name": "list_datasets", "input": {}}]
        ))
        # Supply MAX_STEPS + 2 identical responses so we never run out
        client = self._make_mock_client([always_tool] * (ReActRunner.MAX_STEPS + 2))

        with patch("agents.runner.settings") as ms, \
             patch("agents.runner.httpx.AsyncClient", return_value=client), \
             patch.object(
                 __import__("agents.tools", fromlist=["ToolExecutor"]).ToolExecutor,
                 "_run_list_datasets",
                 new_callable=AsyncMock,
                 return_value="datasets listed"
             ):
            ms.anthropic_api_key = "test-key"
            session = await ReActRunner(mock_db).run("Keep listing datasets forever")

        assert session.n_tool_calls == ReActRunner.MAX_STEPS
        assert "maximum" in session.final_answer.lower()

    @pytest.mark.asyncio
    async def test_session_to_dict_structure(self):
        """AgentSession.to_dict() returns the expected structure."""
        from agents.runner import AgentSession, ThinkingEvent, FinalAnswerEvent

        session = AgentSession(goal="test goal")
        session.events = [
            ThinkingEvent("Thinking..."),
            FinalAnswerEvent("Done."),
        ]
        session.final_answer = "Done."
        session.n_tool_calls = 0
        session.elapsed_secs = 1.5

        d = session.to_dict()
        assert d["goal"] == "test goal"
        assert d["final_answer"] == "Done."
        assert d["n_tool_calls"] == 0
        assert len(d["events"]) == 2
        assert d["events"][0]["type"] == "thinking"
        assert d["events"][1]["type"] == "final_answer"

    @pytest.mark.asyncio
    async def test_stream_yields_events(self):
        """stream() yields individual events as they occur."""
        from agents.runner import ReActRunner, FinalAnswerEvent

        mock_db = AsyncMock()
        response = self._make_mock_response(self._make_claude_response(text="Done."))
        client   = self._make_mock_client([response])

        with patch("agents.runner.settings") as ms, \
             patch("agents.runner.httpx.AsyncClient", return_value=client):
            ms.anthropic_api_key = "test-key"
            events = []
            async for event in ReActRunner(mock_db).stream("test"):
                events.append(event)

        final = [e for e in events if isinstance(e, FinalAnswerEvent)]
        assert len(final) == 1
        assert final[0].text == "Done."


# ══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

class TestAgentAPI:

    def test_list_tools_endpoint(self, client):
        resp = client.get("/api/v1/agent/tools")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "tools" in data
        assert data["count"] >= 7
        # Each tool must have the required fields
        for tool in data["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

    def test_run_agent_without_api_key(self, client):
        """Without an API key the agent should return an error gracefully."""
        with patch("agents.runner.settings") as mock_settings, \
             patch("config.settings") as config_settings:
            mock_settings.anthropic_api_key = None
            config_settings.anthropic_api_key = None

            resp = client.post("/api/v1/agent/run", json={"goal": "List datasets"})

        # Should return 200 with an error in the session (not a 500)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "events" in data

    def test_run_agent_empty_goal_rejected(self, client):
        resp = client.post("/api/v1/agent/run", json={"goal": ""})
        assert resp.status_code == 422

    def test_create_stream_session(self, client):
        with patch("agents.runner.settings") as ms:
            ms.anthropic_api_key = None   # won't run, just checking session creation
            resp = client.post("/api/v1/agent/run/stream-id", json={"goal": "List datasets"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "session_id" in data
        assert data["session_id"].startswith("ag-")

    def test_run_agent_with_mocked_claude(self, client):
        """Full end-to-end: agent runs, gets a final answer."""
        final_text = "No datasets found. Please upload data first."
        payload  = {
            "content":     [{"type": "text", "text": final_text}],
            "stop_reason": "end_turn",
            "model":       "claude-haiku-mock",
            "usage":       {"input_tokens": 50, "output_tokens": 20},
        }
        response = MagicMock()
        response.json = MagicMock(return_value=payload)
        response.raise_for_status = MagicMock()

        async def _post(*args, **kwargs):
            return response

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = _post

        with patch("agents.runner.settings") as ms, \
             patch("agents.runner.httpx.AsyncClient", return_value=mock_client):
            ms.anthropic_api_key = "test-key"
            resp = client.post("/api/v1/agent/run",
                               json={"goal": "What datasets are available?"})

        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["final_answer"] == final_text
        assert data["n_tool_calls"] == 0
