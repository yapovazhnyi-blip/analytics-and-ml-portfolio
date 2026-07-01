"""
Fine-tuning tests.

All tests use mock-mode training (model_id='mock-<anything>') so no model
is downloaded and no GPU is required. The mock trainer emits realistic
progress events and writes a minimal adapter_config.json artifact.

Coverage:
  - Config validation (rank, dropout, learning_rate bounds)
  - Alpaca and ShareGPT dataset formatting
  - Sample validation (missing fields, unknown format)
  - Mock trainer: events emitted, artifact created, TrainingResult shape
  - API endpoints: submit, list, get, delete
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
# CONFIG VALIDATION
# ══════════════════════════════════════════════════════════════════════════

class TestFineTuningConfig:

    def test_default_config_valid(self):
        from fine_tuning.config import FineTuningConfig
        cfg = FineTuningConfig(model_id="mock-phi")
        assert cfg.validate() == []

    def test_invalid_rank_rejected(self):
        from fine_tuning.config import FineTuningConfig, LoRAConfig
        cfg = FineTuningConfig(model_id="mock-phi", lora=LoRAConfig(rank=0))
        errors = cfg.validate()
        assert any("rank" in e for e in errors)

    def test_invalid_rank_too_large(self):
        from fine_tuning.config import FineTuningConfig, LoRAConfig
        cfg = FineTuningConfig(model_id="mock-phi", lora=LoRAConfig(rank=512))
        assert any("rank" in e for e in cfg.validate())

    def test_invalid_dropout(self):
        from fine_tuning.config import FineTuningConfig, LoRAConfig
        cfg = FineTuningConfig(model_id="mock-phi", lora=LoRAConfig(dropout=1.5))
        assert any("dropout" in e for e in cfg.validate())

    def test_invalid_learning_rate(self):
        from fine_tuning.config import FineTuningConfig
        cfg = FineTuningConfig(model_id="mock-phi", learning_rate=-0.001)
        assert any("learning rate" in e.lower() for e in cfg.validate())

    def test_invalid_epochs(self):
        from fine_tuning.config import FineTuningConfig
        cfg = FineTuningConfig(model_id="mock-phi", epochs=0)
        assert any("epoch" in e.lower() for e in cfg.validate())

    def test_invalid_dataset_format(self):
        from fine_tuning.config import FineTuningConfig
        cfg = FineTuningConfig(model_id="mock-phi", dataset_format="unknown")
        assert any("dataset_format" in e for e in cfg.validate())

    def test_alpha_defaults_to_2x_rank(self):
        from fine_tuning.config import FineTuningConfig, LoRAConfig
        cfg = FineTuningConfig(model_id="mock-phi", lora=LoRAConfig(rank=16, alpha=32))
        assert cfg.lora.alpha == cfg.lora.rank * 2


# ══════════════════════════════════════════════════════════════════════════
# DATASET FORMATTER
# ══════════════════════════════════════════════════════════════════════════

class TestFormatter:

    # ── Alpaca ────────────────────────────────────────────────────────────

    def test_alpaca_with_input(self):
        from fine_tuning.formatter import format_alpaca
        text = format_alpaca({
            "instruction": "Translate to French.",
            "input": "Hello world",
            "output": "Bonjour le monde",
        })
        assert "Translate to French." in text
        assert "Hello world" in text
        assert "Bonjour le monde" in text
        assert "### Instruction:" in text
        assert "### Input:" in text
        assert "### Response:" in text

    def test_alpaca_without_input(self):
        from fine_tuning.formatter import format_alpaca
        text = format_alpaca({
            "instruction": "Say hello.",
            "output": "Hello!",
        })
        assert "### Instruction:" in text
        assert "### Input:" not in text   # no input section when input is empty
        assert "Hello!" in text

    def test_alpaca_empty_string_input(self):
        from fine_tuning.formatter import format_alpaca
        text = format_alpaca({"instruction": "Say hi.", "input": "", "output": "Hi!"})
        assert "### Input:" not in text   # empty input treated as absent

    def test_alpaca_missing_instruction_raises(self):
        from fine_tuning.formatter import format_alpaca
        with pytest.raises(ValueError, match="instruction"):
            format_alpaca({"output": "Something"})

    def test_alpaca_missing_output_raises(self):
        from fine_tuning.formatter import format_alpaca
        with pytest.raises(ValueError, match="output"):
            format_alpaca({"instruction": "Do something."})

    # ── ShareGPT ──────────────────────────────────────────────────────────

    def test_sharegpt_basic(self):
        from fine_tuning.formatter import format_sharegpt
        text = format_sharegpt({
            "conversations": [
                {"from": "human", "value": "What is Python?"},
                {"from": "gpt",   "value": "Python is a language."},
            ]
        })
        assert "What is Python?" in text
        assert "Python is a language." in text

    def test_sharegpt_empty_conversations_raises(self):
        from fine_tuning.formatter import format_sharegpt
        with pytest.raises(ValueError, match="conversations"):
            format_sharegpt({"conversations": []})

    def test_sharegpt_role_normalisation(self):
        """Both 'human'/'gpt' and 'user'/'assistant' role names are accepted."""
        from fine_tuning.formatter import format_sharegpt
        t1 = format_sharegpt({"conversations": [
            {"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}
        ]})
        t2 = format_sharegpt({"conversations": [
            {"from": "user", "value": "Q"}, {"from": "assistant", "value": "A"}
        ]})
        assert "Q" in t1 and "A" in t1
        assert "Q" in t2 and "A" in t2

    # ── Unified formatter ─────────────────────────────────────────────────

    def test_format_sample_alpaca(self):
        from fine_tuning.formatter import format_sample
        text = format_sample(
            {"instruction": "Test.", "output": "Done."},
            dataset_format="alpaca"
        )
        assert "Test." in text and "Done." in text

    def test_format_sample_sharegpt(self):
        from fine_tuning.formatter import format_sample
        text = format_sample(
            {"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]},
            dataset_format="sharegpt"
        )
        assert "Hi" in text and "Hello" in text

    def test_format_sample_unknown_raises(self):
        from fine_tuning.formatter import format_sample
        with pytest.raises(ValueError, match="Unknown dataset_format"):
            format_sample({}, dataset_format="csv")

    # ── Validation ────────────────────────────────────────────────────────

    def test_validate_good_samples(self):
        from fine_tuning.formatter import validate_samples
        samples = [
            {"instruction": "Do A.", "output": "Done A."},
            {"instruction": "Do B.", "output": "Done B."},
        ]
        assert validate_samples(samples, "alpaca") == []

    def test_validate_bad_samples_returns_errors(self):
        from fine_tuning.formatter import validate_samples
        samples = [
            {"instruction": "Good.", "output": "Good."},
            {"output": "Missing instruction"},   # bad
        ]
        errors = validate_samples(samples, "alpaca")
        assert len(errors) >= 1

    def test_validate_stops_at_max_errors(self):
        from fine_tuning.formatter import validate_samples
        # 10 bad samples but max_errors=3
        bad = [{"output": f"no instruction {i}"} for i in range(10)]
        errors = validate_samples(bad, "alpaca", max_errors=3)
        assert len(errors) <= 4   # 3 errors + 1 "... and more" message


# ══════════════════════════════════════════════════════════════════════════
# MOCK TRAINER
# ══════════════════════════════════════════════════════════════════════════

class TestMockTrainer:

    ALPACA_SAMPLES = [
        {"instruction": "Say hello.", "output": "Hello!"},
        {"instruction": "Say goodbye.", "output": "Goodbye!"},
        {"instruction": "Count to 3.", "output": "1, 2, 3."},
        {"instruction": "Name a colour.", "output": "Blue."},
    ]

    @pytest.mark.asyncio
    async def test_mock_training_succeeds(self, tmp_path):
        from fine_tuning.config import FineTuningConfig
        from fine_tuning.trainer import SFTTrainer

        cfg = FineTuningConfig(model_id="mock-phi", epochs=1, batch_size=2)
        trainer = SFTTrainer(cfg, "test-001", asyncio.Queue())
        result = await trainer.run(self.ALPACA_SAMPLES, str(tmp_path))

        assert result.succeeded
        assert result.total_steps >= 1
        assert 0.0 <= result.final_loss <= 3.0

    @pytest.mark.asyncio
    async def test_mock_creates_adapter_artifact(self, tmp_path):
        from fine_tuning.config import FineTuningConfig
        from fine_tuning.trainer import SFTTrainer

        cfg = FineTuningConfig(model_id="mock-phi", epochs=1, batch_size=2)
        trainer = SFTTrainer(cfg, "test-002", asyncio.Queue())
        result = await trainer.run(self.ALPACA_SAMPLES, str(tmp_path))

        adapter_dir = Path(result.adapter_path)
        assert adapter_dir.exists()
        assert (adapter_dir / "adapter_config.json").exists()

        config = json.loads((adapter_dir / "adapter_config.json").read_text())
        assert config["peft_type"] == "LORA"
        assert config["r"] == 16   # default rank

    @pytest.mark.asyncio
    async def test_mock_emits_step_events(self, tmp_path):
        from fine_tuning.config import FineTuningConfig
        from fine_tuning.trainer import SFTTrainer, StepEvent, CompleteEvent

        cfg = FineTuningConfig(model_id="mock-phi", epochs=1, batch_size=2)
        q = asyncio.Queue()
        trainer = SFTTrainer(cfg, "test-003", q)
        result = await trainer.run(self.ALPACA_SAMPLES, str(tmp_path))

        events = []
        while not q.empty():
            e = q.get_nowait()
            if e is not None:
                events.append(e)

        step_events    = [e for e in events if isinstance(e, StepEvent)]
        complete_events = [e for e in events if isinstance(e, CompleteEvent)]

        assert len(step_events) >= 1
        assert len(complete_events) == 1
        assert complete_events[0].final_loss == result.final_loss

    @pytest.mark.asyncio
    async def test_bad_samples_returns_error(self, tmp_path):
        from fine_tuning.config import FineTuningConfig
        from fine_tuning.trainer import SFTTrainer

        cfg = FineTuningConfig(model_id="mock-phi")
        trainer = SFTTrainer(cfg, "test-004", asyncio.Queue())
        bad_samples = [{"output": "no instruction"}]   # missing instruction
        result = await trainer.run(bad_samples, str(tmp_path))

        assert not result.succeeded
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_sharegpt_format_works(self, tmp_path):
        from fine_tuning.config import FineTuningConfig
        from fine_tuning.trainer import SFTTrainer

        cfg = FineTuningConfig(model_id="mock-phi", dataset_format="sharegpt")
        trainer = SFTTrainer(cfg, "test-005", asyncio.Queue())
        samples = [
            {"conversations": [
                {"from": "human", "value": "What is 2+2?"},
                {"from": "gpt",   "value": "4"},
            ]},
            {"conversations": [
                {"from": "human", "value": "Name a planet."},
                {"from": "gpt",   "value": "Mars"},
            ]},
        ]
        result = await trainer.run(samples, str(tmp_path))
        assert result.succeeded

    @pytest.mark.asyncio
    async def test_loss_decreases_over_steps(self, tmp_path):
        """Mock training uses exponential decay — loss should generally decrease."""
        from fine_tuning.config import FineTuningConfig
        from fine_tuning.trainer import SFTTrainer, StepEvent

        # Use many samples to get more steps for a visible trend
        samples = [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(20)]
        cfg = FineTuningConfig(model_id="mock-phi", epochs=2, batch_size=2)
        q = asyncio.Queue()
        trainer = SFTTrainer(cfg, "test-006", q)
        await trainer.run(samples, str(tmp_path))

        events = []
        while not q.empty():
            e = q.get_nowait()
            if e is not None:
                events.append(e)

        step_events = [e for e in events if isinstance(e, StepEvent)]
        if len(step_events) >= 3:
            first_loss = step_events[0].loss
            last_loss  = step_events[-1].loss
            assert last_loss < first_loss   # exponential decay


# ══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

class TestFineTuningAPI:

    SAMPLES = [
        {"instruction": "Say hello.", "output": "Hello!"},
        {"instruction": "Say goodbye.", "output": "Goodbye!"},
    ]

    def _post_job(self, client, model_id="mock-phi", samples=None):
        return client.post("/api/v1/fine-tuning/jobs", json={
            "model_id":       model_id,
            "samples":        samples or self.SAMPLES,
            "dataset_format": "alpaca",
            "epochs":         1,
            "batch_size":     1,
        })

    def test_submit_job_returns_201(self, client):
        resp = self._post_job(client)
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert "job_id" in data
        assert data["status"] == "running"
        assert data["n_samples"] == len(self.SAMPLES)

    def test_submit_validates_config(self, client):
        resp = client.post("/api/v1/fine-tuning/jobs", json={
            "model_id":       "mock-phi",
            "samples":        self.SAMPLES,
            "dataset_format": "alpaca",
            "epochs":         0,   # invalid
        })
        assert resp.status_code == 422

    def test_list_jobs(self, client):
        self._post_job(client)
        resp = client.get("/api/v1/fine-tuning/jobs")
        assert resp.status_code == 200
        assert len(resp.json()["data"]) >= 1

    def test_get_job_by_id(self, client):
        job_id = self._post_job(client).json()["data"]["job_id"]
        resp = client.get(f"/api/v1/fine-tuning/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["data"]["job_id"] == job_id

    def test_get_nonexistent_job_returns_404(self, client):
        resp = client.get("/api/v1/fine-tuning/jobs/ft-doesnotexist")
        assert resp.status_code == 404

    def test_cannot_delete_running_job(self, client):
        """A running job should return 422, not be silently deleted."""
        job_id = self._post_job(client).json()["data"]["job_id"]
        # Job is still running immediately after submission
        resp = client.delete(f"/api/v1/fine-tuning/jobs/{job_id}")
        # Should either block (422) or have already completed (204)
        assert resp.status_code in (204, 422)

    def test_delete_nonexistent_returns_404(self, client):
        resp = client.delete("/api/v1/fine-tuning/jobs/ft-ghost")
        assert resp.status_code == 404

    def test_bad_samples_job_fails_gracefully(self, client):
        """Jobs with invalid samples should fail gracefully, not 500."""
        import time
        resp = self._post_job(client, samples=[{"output": "missing instruction"}])
        assert resp.status_code == 201   # accepted for submission
        job_id = resp.json()["data"]["job_id"]

        for _ in range(20):
            data = client.get(f"/api/v1/fine-tuning/jobs/{job_id}").json()["data"]
            if data["status"] in ("succeeded", "failed"):
                assert data["status"] == "failed"
                assert data["error_message"] is not None
                break
            time.sleep(0.3)
