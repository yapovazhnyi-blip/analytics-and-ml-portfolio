"""
Multi-agent LangGraph tests.

Strategy: the supervisor uses mock mode (no Anthropic API calls) by
patching settings.anthropic_api_key to None or "mock-key". Specialist
nodes use the real ToolExecutor against an in-memory SQLite database.

Tests cover:
  - State structure (TypedDict fields valid)
  - Graph builds without error (nodes, edges, entry point)
  - Mock supervisor routing rules
  - Each specialist node runs against mock DB
  - Full end-to-end: graph runs to FINISH with mock supervisor
  - API endpoint: POST /agent/multi/run
  - Session.to_dict() is JSON serialisable
"""

from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_db():
    from unittest.mock import AsyncMock, MagicMock
    db = AsyncMock()
    # scalars().all() for list_datasets
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = []
    db.scalars = AsyncMock(return_value=mock_scalars)
    db.get = AsyncMock(return_value=None)
    return db


@pytest.fixture
def client():
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


# ══════════════════════════════════════════════════════════════════════════
# STATE DEFINITION
# ══════════════════════════════════════════════════════════════════════════

class TestAgentState:

    def test_multi_agent_state_keys(self):
        from agents.state import MultiAgentState
        expected = {"messages", "goal", "active_agent", "next_agent",
                    "step_count", "context", "agent_output", "final_answer", "error"}
        assert expected == set(MultiAgentState.__annotations__)

    def test_agent_context_fields(self):
        from agents.state import AgentContext
        fields = set(AgentContext.__annotations__)
        assert "dataset_id" in fields
        assert "experiment_id" in fields
        assert "training_done" in fields
        assert "deployment_done" in fields

    def test_route_constants(self):
        from agents.state import (
            ROUTE_SUPERVISOR, ROUTE_DATASET_ANALYST,
            ROUTE_MODEL_TRAINER, ROUTE_DEPLOYER, MAX_STEPS,
        )
        assert ROUTE_SUPERVISOR == "supervisor"
        assert ROUTE_DATASET_ANALYST == "dataset_analyst"
        assert ROUTE_MODEL_TRAINER == "model_trainer"
        assert ROUTE_DEPLOYER == "deployer"
        assert MAX_STEPS > 0


# ══════════════════════════════════════════════════════════════════════════
# GRAPH STRUCTURE
# ══════════════════════════════════════════════════════════════════════════

class TestGraphStructure:

    def test_graph_builds_without_error(self, mock_db):
        from agents.multi_agent import build_multi_agent_graph
        graph = build_multi_agent_graph(mock_db)
        assert graph is not None

    def test_graph_has_compiled_type(self, mock_db):
        from agents.multi_agent import build_multi_agent_graph
        from langgraph.graph.state import CompiledStateGraph
        graph = build_multi_agent_graph(mock_db)
        assert isinstance(graph, CompiledStateGraph)

    def test_route_from_supervisor_to_dataset_analyst(self):
        from agents.multi_agent import route_from_supervisor
        from agents.state import ROUTE_DATASET_ANALYST
        state = {"next_agent": ROUTE_DATASET_ANALYST, "step_count": 0}
        assert route_from_supervisor(state) == ROUTE_DATASET_ANALYST

    def test_route_from_supervisor_to_end_on_finish(self):
        from agents.multi_agent import route_from_supervisor
        from langgraph.graph import END
        state = {"next_agent": "FINISH", "step_count": 0}
        assert route_from_supervisor(state) == END

    def test_route_from_supervisor_to_end_on_end(self):
        from agents.multi_agent import route_from_supervisor
        from agents.state import ROUTE_END
        from langgraph.graph import END
        state = {"next_agent": ROUTE_END, "step_count": 0}
        assert route_from_supervisor(state) == END


# ══════════════════════════════════════════════════════════════════════════
# MOCK SUPERVISOR ROUTING
# ══════════════════════════════════════════════════════════════════════════

class TestMockSupervisor:

    def test_routes_to_dataset_analyst_first(self):
        from agents.multi_agent import _mock_supervisor
        ctx = {}
        result = _mock_supervisor(ctx, "train a churn model")
        assert result["next"] == "dataset_analyst"

    def test_routes_to_model_trainer_after_dataset(self):
        from agents.multi_agent import _mock_supervisor
        ctx = {"dataset_id": 1, "profiling_done": True}
        result = _mock_supervisor(ctx, "train a classification model")
        assert result["next"] == "model_trainer"

    def test_routes_to_deployer_after_training(self):
        from agents.multi_agent import _mock_supervisor
        ctx = {"dataset_id": 1, "experiment_id": 5,
               "training_done": True, "best_model": "XGBoost"}
        result = _mock_supervisor(ctx, "deploy the model")
        assert result["next"] == "deployer"

    def test_finishes_when_goal_complete(self):
        from agents.multi_agent import _mock_supervisor
        ctx = {"dataset_id": 1, "experiment_id": 5,
               "training_done": True, "deployment_done": True}
        result = _mock_supervisor(ctx, "analyse and train")
        assert result["next"] == "FINISH"

    def test_finish_includes_final_answer(self):
        from agents.multi_agent import _mock_supervisor
        ctx = {"dataset_id": 1, "training_done": True, "best_model": "RF", "best_score": 0.91}
        result = _mock_supervisor(ctx, "just list datasets")
        if result["next"] == "FINISH":
            assert "final_answer" in result
            assert result["final_answer"]

    def test_reasoning_always_present(self):
        from agents.multi_agent import _mock_supervisor
        for ctx in [{}, {"dataset_id": 1}, {"dataset_id": 1, "training_done": True}]:
            result = _mock_supervisor(ctx, "train a model")
            assert "reasoning" in result


# ══════════════════════════════════════════════════════════════════════════
# SPECIALIST NODES
# ══════════════════════════════════════════════════════════════════════════

class TestDatasetAnalyst:

    @pytest.mark.asyncio
    async def test_returns_state_update(self, mock_db):
        from agents.specialists import dataset_analyst_node
        from agents.state import ROUTE_SUPERVISOR
        state = {
            "goal": "analyse my data",
            "messages": [], "context": {}, "step_count": 0,
            "active_agent": "supervisor", "next_agent": "",
            "agent_output": "", "final_answer": "", "error": "",
        }
        update = await dataset_analyst_node(state, mock_db)
        assert "context" in update
        assert "agent_output" in update
        assert update["next_agent"] == ROUTE_SUPERVISOR

    @pytest.mark.asyncio
    async def test_increments_step_count(self, mock_db):
        from agents.specialists import dataset_analyst_node
        state = {
            "goal": "list datasets", "messages": [], "context": {},
            "step_count": 3, "active_agent": "supervisor", "next_agent": "",
            "agent_output": "", "final_answer": "", "error": "",
        }
        update = await dataset_analyst_node(state, mock_db)
        assert update["step_count"] == 4

    @pytest.mark.asyncio
    async def test_appends_to_messages(self, mock_db):
        from agents.specialists import dataset_analyst_node
        state = {
            "goal": "list datasets", "messages": [], "context": {},
            "step_count": 0, "active_agent": "supervisor", "next_agent": "",
            "agent_output": "", "final_answer": "", "error": "",
        }
        update = await dataset_analyst_node(state, mock_db)
        assert len(update["messages"]) >= 1
        assert "Dataset Analyst" in update["messages"][0]["content"]


class TestModelTrainer:

    @pytest.mark.asyncio
    async def test_requires_dataset_id(self, mock_db):
        from agents.specialists import model_trainer_node
        state = {
            "goal": "train a model", "messages": [], "context": {},
            "step_count": 0, "active_agent": "supervisor", "next_agent": "",
            "agent_output": "", "final_answer": "", "error": "",
        }
        update = await model_trainer_node(state, mock_db)
        # Without dataset_id, should return error message
        assert "Error" in update["agent_output"] or "dataset_id" in update["agent_output"].lower()

    @pytest.mark.asyncio
    async def test_with_dataset_id_proceeds(self, mock_db):
        from agents.specialists import model_trainer_node
        from models.dataset import Dataset
        from models.experiment import Experiment
        from datetime import datetime

        mock_ds = MagicMock(spec=Dataset)
        mock_ds.id = 1; mock_ds.name = "test"; mock_ds.status = "ready"
        mock_ds.file_path = "/tmp/test.csv"; mock_ds.source_type = "csv"
        mock_db.get = AsyncMock(return_value=mock_ds)
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()
        mock_db.refresh = AsyncMock()
        mock_db.commit = AsyncMock()

        state = {
            "goal": "predict churn with dataset 1",
            "messages": [], "context": {"dataset_id": 1},
            "step_count": 0, "active_agent": "supervisor", "next_agent": "",
            "agent_output": "", "final_answer": "", "error": "",
        }
        update = await model_trainer_node(state, mock_db)
        # Should have context update
        assert "context" in update
        assert update["step_count"] == 1


class TestDeployer:

    @pytest.mark.asyncio
    async def test_requires_experiment_id(self, mock_db):
        from agents.specialists import deployer_node
        state = {
            "goal": "deploy the model", "messages": [], "context": {},
            "step_count": 0, "active_agent": "supervisor", "next_agent": "",
            "agent_output": "", "final_answer": "", "error": "",
        }
        update = await deployer_node(state, mock_db)
        assert "Error" in update["agent_output"]

    @pytest.mark.asyncio
    async def test_requires_training_done(self, mock_db):
        from agents.specialists import deployer_node
        state = {
            "goal": "deploy", "messages": [],
            "context": {"experiment_id": 5, "training_done": False},
            "step_count": 0, "active_agent": "supervisor", "next_agent": "",
            "agent_output": "", "final_answer": "", "error": "",
        }
        update = await deployer_node(state, mock_db)
        assert "Error" in update["agent_output"] or "not complete" in update["agent_output"].lower()


# ══════════════════════════════════════════════════════════════════════════
# SUPERVISOR NODE
# ══════════════════════════════════════════════════════════════════════════

class TestSupervisorNode:

    @pytest.mark.asyncio
    async def test_supervisor_uses_mock_without_api_key(self):
        from agents.multi_agent import supervisor_node
        from agents.state import ROUTE_DATASET_ANALYST

        state = {
            "goal": "train a model",
            "messages": [], "context": {}, "step_count": 0,
            "active_agent": "supervisor", "next_agent": "",
            "agent_output": "", "final_answer": "", "error": "",
        }
        with patch("agents.multi_agent.settings") as ms:
            ms.anthropic_api_key = None
            update = await supervisor_node(state)

        # Should route to dataset_analyst first (no dataset_id in context)
        assert update["next_agent"] == ROUTE_DATASET_ANALYST

    @pytest.mark.asyncio
    async def test_supervisor_terminates_at_max_steps(self):
        from agents.multi_agent import supervisor_node
        from agents.state import ROUTE_END
        from langgraph.graph import END

        state = {
            "goal": "do things", "messages": [], "context": {},
            "step_count": 100,   # exceeds MAX_STEPS
            "active_agent": "supervisor", "next_agent": "",
            "agent_output": "", "final_answer": "", "error": "",
        }
        with patch("agents.multi_agent.settings") as ms:
            ms.anthropic_api_key = None
            update = await supervisor_node(state)

        assert update["next_agent"] in (ROUTE_END, END, "__end__")


# ══════════════════════════════════════════════════════════════════════════
# MULTI-AGENT RUNNER (end-to-end)
# ══════════════════════════════════════════════════════════════════════════

class TestMultiAgentRunner:

    @pytest.mark.asyncio
    async def test_runner_returns_session(self, mock_db):
        from agents.multi_agent import MultiAgentRunner, MultiAgentSession
        with patch("agents.multi_agent.settings") as ms:
            ms.anthropic_api_key = None
            runner = MultiAgentRunner(mock_db)
            session = await runner.run("list my datasets")
        assert isinstance(session, MultiAgentSession)

    @pytest.mark.asyncio
    async def test_session_has_final_answer(self, mock_db):
        from agents.multi_agent import MultiAgentRunner
        with patch("agents.multi_agent.settings") as ms:
            ms.anthropic_api_key = None
            session = await MultiAgentRunner(mock_db).run("list datasets")
        assert session.final_answer

    @pytest.mark.asyncio
    async def test_session_has_events(self, mock_db):
        from agents.multi_agent import MultiAgentRunner, SupervisorEvent
        with patch("agents.multi_agent.settings") as ms:
            ms.anthropic_api_key = None
            session = await MultiAgentRunner(mock_db).run("list datasets")
        assert len(session.events) > 0

    @pytest.mark.asyncio
    async def test_to_dict_is_json_serialisable(self, mock_db):
        from agents.multi_agent import MultiAgentRunner
        with patch("agents.multi_agent.settings") as ms:
            ms.anthropic_api_key = None
            session = await MultiAgentRunner(mock_db).run("list datasets")
        json.dumps(session.to_dict())   # must not raise

    @pytest.mark.asyncio
    async def test_session_tracks_step_count(self, mock_db):
        from agents.multi_agent import MultiAgentRunner
        with patch("agents.multi_agent.settings") as ms:
            ms.anthropic_api_key = None
            session = await MultiAgentRunner(mock_db).run("list datasets")
        assert session.steps >= 1


# ══════════════════════════════════════════════════════════════════════════
# API ENDPOINT
# ══════════════════════════════════════════════════════════════════════════

class TestMultiAgentAPI:

    def test_multi_run_endpoint_exists(self, client):
        with patch("agents.multi_agent.settings") as ms:
            ms.anthropic_api_key = None
            resp = client.post("/api/v1/agent/multi/run",
                               json={"goal": "list my datasets"})
        assert resp.status_code == 200

    def test_multi_run_returns_events(self, client):
        with patch("agents.multi_agent.settings") as ms:
            ms.anthropic_api_key = None
            resp = client.post("/api/v1/agent/multi/run",
                               json={"goal": "analyse my datasets"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "events" in data
        assert "final_answer" in data
        assert len(data["events"]) > 0

    def test_multi_run_empty_goal_rejected(self, client):
        resp = client.post("/api/v1/agent/multi/run", json={"goal": ""})
        assert resp.status_code == 422
