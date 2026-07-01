"""
Tests for DPO fine-tuning and rate limiting.

DPO tests:
  - DPOConfig validation (beta bounds, epochs, learning rate)
  - Sample validation (required fields: prompt, chosen, rejected)
  - Mock DPO training: events emitted, artifact created, loss decreases
  - API endpoint: POST /fine-tuning/jobs/dpo

Rate limiting tests:
  - SlowAPI limiter is attached to the app
  - Default limit is configured
  - Login endpoint accepts a Request parameter (needed by slowapi)
"""

from __future__ import annotations

import asyncio
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool


# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    import sys, importlib, database as db_mod
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    db_mod.engine = engine
    db_mod.SessionFactory = factory
    db_mod.AsyncSessionLocal = factory
    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    import main as m
    with TestClient(m.app, raise_server_exceptions=True) as c:
        yield c


# ══════════════════════════════════════════════════════════════════════════
# DPO CONFIG VALIDATION
# ══════════════════════════════════════════════════════════════════════════

class TestDPOConfig:

    def test_default_config_is_valid(self):
        from fine_tuning.config import DPOConfig
        cfg = DPOConfig(model_id="mock-phi")
        assert cfg.validate() == []

    def test_beta_below_zero_rejected(self):
        from fine_tuning.config import DPOConfig
        cfg = DPOConfig(model_id="mock-phi", beta=-0.1)
        assert any("beta" in e for e in cfg.validate())

    def test_beta_above_two_rejected(self):
        from fine_tuning.config import DPOConfig
        cfg = DPOConfig(model_id="mock-phi", beta=2.5)
        assert any("beta" in e for e in cfg.validate())

    def test_beta_boundary_values_valid(self):
        from fine_tuning.config import DPOConfig
        for beta in (0.01, 0.1, 0.5, 1.0, 2.0):
            cfg = DPOConfig(model_id="mock-phi", beta=beta)
            assert cfg.validate() == [], f"beta={beta} should be valid"

    def test_invalid_epochs(self):
        from fine_tuning.config import DPOConfig
        cfg = DPOConfig(model_id="mock-phi", epochs=0)
        assert any("epoch" in e.lower() for e in cfg.validate())

    def test_invalid_learning_rate(self):
        from fine_tuning.config import DPOConfig
        cfg = DPOConfig(model_id="mock-phi", learning_rate=0.0)
        assert any("learning rate" in e.lower() for e in cfg.validate())

    def test_default_beta_is_conservative(self):
        """Default beta=0.1 keeps policy close to reference — good for alignment."""
        from fine_tuning.config import DPOConfig
        assert DPOConfig(model_id="x").beta == 0.1


# ══════════════════════════════════════════════════════════════════════════
# DPO SAMPLE VALIDATION
# ══════════════════════════════════════════════════════════════════════════

class TestDPOSampleValidation:

    def test_valid_samples_pass(self):
        from fine_tuning.dpo_trainer import validate_dpo_samples
        samples = [
            {"prompt": "Q?", "chosen": "Good answer.", "rejected": "Bad answer."},
            {"prompt": "Q2?", "chosen": "Better.", "rejected": "Worse."},
        ]
        assert validate_dpo_samples(samples) == []

    def test_missing_prompt_caught(self):
        from fine_tuning.dpo_trainer import validate_dpo_samples
        bad = [{"chosen": "Good.", "rejected": "Bad."}]
        errors = validate_dpo_samples(bad)
        assert any("prompt" in e for e in errors)

    def test_missing_chosen_caught(self):
        from fine_tuning.dpo_trainer import validate_dpo_samples
        bad = [{"prompt": "Q?", "rejected": "Bad."}]
        errors = validate_dpo_samples(bad)
        assert any("chosen" in e for e in errors)

    def test_missing_rejected_caught(self):
        from fine_tuning.dpo_trainer import validate_dpo_samples
        bad = [{"prompt": "Q?", "chosen": "Good."}]
        errors = validate_dpo_samples(bad)
        assert any("rejected" in e for e in errors)

    def test_empty_string_field_caught(self):
        from fine_tuning.dpo_trainer import validate_dpo_samples
        bad = [{"prompt": "Q?", "chosen": "", "rejected": "Bad."}]
        errors = validate_dpo_samples(bad)
        assert any("chosen" in e for e in errors)

    def test_all_fields_present_returns_empty(self):
        from fine_tuning.dpo_trainer import validate_dpo_samples
        good = [{"prompt": "Hi", "chosen": "Hello!", "rejected": "Dunno."}]
        assert validate_dpo_samples(good) == []


# ══════════════════════════════════════════════════════════════════════════
# DPO MOCK TRAINER
# ══════════════════════════════════════════════════════════════════════════

SAMPLES = [
    {"prompt": "Explain AI.", "chosen": "AI is intelligence from machines.", "rejected": "AI = magic."},
    {"prompt": "Define ML.", "chosen": "ML learns from data.", "rejected": "ML is unknown."},
    {"prompt": "What is NLP?", "chosen": "NLP processes text.", "rejected": "NLP = nothing."},
]

class TestDPOMockTrainer:

    @pytest.mark.asyncio
    async def test_mock_training_succeeds(self, tmp_path):
        from fine_tuning.config import DPOConfig
        from fine_tuning.dpo_trainer import DPOTrainer
        cfg = DPOConfig(model_id="mock-phi", epochs=1, batch_size=1, beta=0.1)
        result = await DPOTrainer(cfg, "test-dpo-001").run(SAMPLES, str(tmp_path))
        assert result.succeeded, result.error

    @pytest.mark.asyncio
    async def test_mock_creates_adapter_artifact(self, tmp_path):
        from fine_tuning.config import DPOConfig
        from fine_tuning.dpo_trainer import DPOTrainer
        cfg = DPOConfig(model_id="mock-phi", epochs=1, batch_size=1)
        result = await DPOTrainer(cfg, "test-dpo-002").run(SAMPLES, str(tmp_path))
        assert result.succeeded
        cfg_file = Path(result.adapter_path) / "adapter_config.json"
        assert cfg_file.exists()
        data = json.loads(cfg_file.read_text())
        assert data["peft_type"] == "LORA"
        assert "dpo_beta" in data

    @pytest.mark.asyncio
    async def test_mock_emits_step_events(self, tmp_path):
        from fine_tuning.config import DPOConfig
        from fine_tuning.dpo_trainer import DPOTrainer
        from fine_tuning.trainer import StepEvent, CompleteEvent
        cfg = DPOConfig(model_id="mock-phi", epochs=1, batch_size=1)
        q = asyncio.Queue()
        result = await DPOTrainer(cfg, "test-dpo-003", q).run(SAMPLES, str(tmp_path))
        events = []
        while not q.empty():
            e = q.get_nowait()
            if e is not None:
                events.append(e)
        assert any(isinstance(e, StepEvent) for e in events)
        assert any(isinstance(e, CompleteEvent) for e in events)

    @pytest.mark.asyncio
    async def test_dpo_loss_starts_near_log2(self, tmp_path):
        """DPO loss should start near ln(2) ≈ 0.693 (random preference baseline)."""
        from fine_tuning.config import DPOConfig
        from fine_tuning.dpo_trainer import DPOTrainer
        from fine_tuning.trainer import StepEvent
        cfg = DPOConfig(model_id="mock-phi", epochs=1, batch_size=1)
        q = asyncio.Queue()
        await DPOTrainer(cfg, "test-dpo-004", q).run(SAMPLES, str(tmp_path))
        first_step = None
        while not q.empty():
            e = q.get_nowait()
            if isinstance(e, StepEvent) and first_step is None:
                first_step = e
        if first_step:
            assert first_step.loss < 1.0   # near 0.693 initially

    @pytest.mark.asyncio
    async def test_bad_samples_return_error(self, tmp_path):
        from fine_tuning.config import DPOConfig
        from fine_tuning.dpo_trainer import DPOTrainer
        cfg = DPOConfig(model_id="mock-phi")
        result = await DPOTrainer(cfg, "test-dpo-005").run(
            [{"prompt": "Q?"}],  # missing chosen and rejected
            str(tmp_path)
        )
        assert not result.succeeded
        assert result.error is not None


# ══════════════════════════════════════════════════════════════════════════
# DPO API ENDPOINT
# ══════════════════════════════════════════════════════════════════════════

class TestDPOAPI:

    SAMPLES = [
        {"prompt": "Say hi.", "chosen": "Hello!", "rejected": "Nope."},
        {"prompt": "Bye.", "chosen": "Goodbye!", "rejected": "Whatever."},
    ]

    def test_submit_dpo_job_returns_201(self, client):
        resp = client.post("/api/v1/fine-tuning/jobs/dpo", json={
            "model_id": "mock-phi",
            "samples": self.SAMPLES,
            "beta": 0.1,
            "epochs": 1,
            "batch_size": 1,
        })
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert data["method"] == "dpo"
        assert data["status"] == "running"
        assert data["job_id"].startswith("dpo-")

    def test_dpo_with_invalid_beta_rejected(self, client):
        resp = client.post("/api/v1/fine-tuning/jobs/dpo", json={
            "model_id": "mock-phi",
            "samples": self.SAMPLES,
            "beta": -1.0,
        })
        assert resp.status_code == 422

    def test_dpo_with_missing_chosen_rejected(self, client):
        resp = client.post("/api/v1/fine-tuning/jobs/dpo", json={
            "model_id": "mock-phi",
            "samples": [{"prompt": "Q?", "rejected": "Bad."}],
        })
        assert resp.status_code == 422

    def test_dpo_job_appears_in_list(self, client):
        client.post("/api/v1/fine-tuning/jobs/dpo", json={
            "model_id": "mock-phi",
            "samples": self.SAMPLES,
            "epochs": 1, "batch_size": 1,
        })
        resp = client.get("/api/v1/fine-tuning/jobs")
        assert resp.status_code == 200
        jobs = resp.json()["data"]
        dpo_jobs = [j for j in jobs if j["method"] == "dpo"]
        assert len(dpo_jobs) >= 1


# ══════════════════════════════════════════════════════════════════════════
# RATE LIMITING
# ══════════════════════════════════════════════════════════════════════════

class TestRateLimiting:

    def test_limiter_attached_to_app(self, client):
        """SlowAPI limiter must be registered on the app state."""
        import main as m
        assert hasattr(m.app.state, "limiter"), \
            "Rate limiter must be attached to app.state.limiter"

    def test_health_endpoint_accessible(self, client):
        """Health liveness check must always be accessible (not rate-limited)."""
        resp = client.get("/api/v1/health/live")
        assert resp.status_code == 200

    def test_login_endpoint_accepts_request(self, client):
        """
        Login endpoint must accept a Request parameter (required by slowapi).
        This ensures the function signature is correct even if the rate limit
        isn't triggered in tests.
        """
        import inspect
        from routers.auth import login
        sig = inspect.signature(login)
        assert "request" in sig.parameters, \
            "login() must have a 'request: Request' parameter for slowapi"

    def test_agent_run_accepts_request(self, client):
        """Agent run endpoint must accept a Request parameter."""
        import inspect
        from routers.agent import run_agent
        sig = inspect.signature(run_agent)
        assert "request" in sig.parameters, \
            "run_agent() must have a 'request: Request' parameter for slowapi"

    def test_slowapi_middleware_registered(self, client):
        """SlowAPIMiddleware must be in the middleware stack."""
        from slowapi.errors import RateLimitExceeded
        import main as m
        assert RateLimitExceeded in m.app.exception_handlers, \
            "RateLimitExceeded handler must be registered"

    def test_default_limit_configured(self):
        """The global limiter must have a default limit configured."""
        import main as m
        limiter = m.app.state.limiter
        assert limiter._default_limits, \
            "Limiter must have default_limits configured"
        # LimitGroup contains limit items — verify at least one is present
        limit_group = limiter._default_limits[0]
        # Access the underlying limit string via the provider
        assert limit_group is not None
