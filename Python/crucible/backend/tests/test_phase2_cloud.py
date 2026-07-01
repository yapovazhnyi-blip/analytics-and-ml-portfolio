"""
Phase 2 cloud integration tests.

Tests cover:
  - SageMaker config + mock job execution (no AWS calls)
  - SageMaker endpoint config generation (no AWS calls)
  - LLM backend resolution (resolve_backend factory)
  - AnthropicBackend message/response handling (mocked httpx)
  - OpenAICompatBackend message format translation (Anthropic <-> OpenAI)
  - BedrockBackend message format translation (mocked boto3)
  - Cloud API endpoints
  - ECS task-definition.json validity
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock


# ══════════════════════════════════════════════════════════════════════════
# SAGEMAKER CONFIG
# ══════════════════════════════════════════════════════════════════════════

class TestSageMakerConfig:

    def test_default_container_uri_set(self):
        from cloud.sagemaker import SageMakerConfig
        cfg = SageMakerConfig(role_arn="mock-role", s3_bucket="b")
        assert cfg.container_uri is not None
        assert "sagemaker-scikit-learn" in cfg.container_uri

    def test_custom_region_changes_container(self):
        from cloud.sagemaker import SageMakerConfig
        cfg = SageMakerConfig(role_arn="mock-role", s3_bucket="b", region="eu-west-1")
        assert "eu-west-1" in cfg.container_uri

    def test_unknown_region_falls_back_to_us_east_1(self):
        from cloud.sagemaker import SageMakerConfig
        cfg = SageMakerConfig(role_arn="mock-role", s3_bucket="b", region="ap-northeast-3")
        assert "us-east-1" in cfg.container_uri

    def test_default_instance_type(self):
        from cloud.sagemaker import SageMakerConfig
        cfg = SageMakerConfig(role_arn="mock-role", s3_bucket="b")
        assert cfg.instance_type == "ml.m5.xlarge"

    def test_instance_type_presets_exist(self):
        from cloud.sagemaker import INSTANCE_TYPES
        assert "cpu-medium" in INSTANCE_TYPES
        assert "gpu-v100" in INSTANCE_TYPES
        assert INSTANCE_TYPES["cpu-medium"] == "ml.m5.xlarge"


# ══════════════════════════════════════════════════════════════════════════
# SAGEMAKER MOCK TRAINING
# ══════════════════════════════════════════════════════════════════════════

class TestSageMakerMockTraining:

    @pytest.mark.asyncio
    async def test_mock_role_triggers_mock_job(self):
        from cloud.sagemaker import SageMakerTrainingRunner, SageMakerConfig
        cfg = SageMakerConfig(role_arn="mock-role", s3_bucket="test-bucket")
        runner = SageMakerTrainingRunner(cfg)
        result = await runner.run(
            local_data_path="/tmp/fake.csv",
            target_column="label",
            task_type="classification",
            experiment_name="test-exp",
        )
        assert result.succeeded
        assert result.status == "Completed"

    @pytest.mark.asyncio
    async def test_mock_job_creates_artifact(self):
        from cloud.sagemaker import SageMakerTrainingRunner, SageMakerConfig
        from pathlib import Path
        cfg = SageMakerConfig(role_arn="mock-role", s3_bucket="test-bucket")
        runner = SageMakerTrainingRunner(cfg)
        result = await runner.run(
            local_data_path="/tmp/fake.csv",
            target_column="label",
            task_type="classification",
            experiment_name="artifact-test",
        )
        assert result.local_artifact_path is not None
        assert Path(result.local_artifact_path).exists()

    @pytest.mark.asyncio
    async def test_mock_job_artifact_is_loadable(self):
        import joblib
        from cloud.sagemaker import SageMakerTrainingRunner, SageMakerConfig
        cfg = SageMakerConfig(role_arn="mock-role", s3_bucket="test-bucket")
        runner = SageMakerTrainingRunner(cfg)
        result = await runner.run(
            local_data_path="/tmp/fake.csv",
            target_column="label",
            task_type="classification",
            experiment_name="load-test",
        )
        model = joblib.load(result.local_artifact_path)
        assert hasattr(model, "predict")

    @pytest.mark.asyncio
    async def test_mock_job_has_s3_uri(self):
        from cloud.sagemaker import SageMakerTrainingRunner, SageMakerConfig
        cfg = SageMakerConfig(role_arn="mock-role", s3_bucket="test-bucket")
        runner = SageMakerTrainingRunner(cfg)
        result = await runner.run(
            local_data_path="/tmp/fake.csv",
            target_column="label",
            task_type="classification",
            experiment_name="s3-test",
        )
        assert result.model_s3_uri.startswith("s3://")

    @pytest.mark.asyncio
    async def test_well_known_test_account_id_triggers_mock(self):
        """role_arn containing the AWS docs test account ID should also mock."""
        from cloud.sagemaker import SageMakerTrainingRunner, SageMakerConfig
        cfg = SageMakerConfig(
            role_arn="arn:aws:iam::000000000000:role/test", s3_bucket="b"
        )
        runner = SageMakerTrainingRunner(cfg)
        result = await runner.run(
            local_data_path="/tmp/fake.csv", target_column="y",
            task_type="classification", experiment_name="acct-test",
        )
        assert result.succeeded


# ══════════════════════════════════════════════════════════════════════════
# SAGEMAKER ENDPOINT CONFIG
# ══════════════════════════════════════════════════════════════════════════

class TestSageMakerEndpointConfig:

    def test_describe_endpoint_config_no_aws_calls(self):
        from cloud.sagemaker import SageMakerEndpointConfig, describe_endpoint_config
        cfg = SageMakerEndpointConfig(
            role_arn="arn:aws:iam::123:role/x",
            model_s3_uri="s3://bucket/model.tar.gz",
        )
        result = describe_endpoint_config(cfg)
        assert result["ModelDataUrl"] == "s3://bucket/model.tar.gz"
        assert result["RoleArn"] == "arn:aws:iam::123:role/x"

    def test_endpoint_config_generates_name_if_not_set(self):
        from cloud.sagemaker import SageMakerEndpointConfig, describe_endpoint_config
        cfg = SageMakerEndpointConfig(role_arn="arn:x", model_s3_uri="s3://b/m.tar.gz")
        result = describe_endpoint_config(cfg)
        assert result["EndpointName"].startswith("crucible-")

    def test_endpoint_config_uses_custom_name(self):
        from cloud.sagemaker import SageMakerEndpointConfig, describe_endpoint_config
        cfg = SageMakerEndpointConfig(
            role_arn="arn:x", model_s3_uri="s3://b/m.tar.gz",
            endpoint_name="my-custom-endpoint",
        )
        result = describe_endpoint_config(cfg)
        assert result["EndpointName"] == "my-custom-endpoint"


# ══════════════════════════════════════════════════════════════════════════
# LLM BACKEND RESOLUTION
# ══════════════════════════════════════════════════════════════════════════

class TestBackendResolution:

    def test_default_resolves_to_anthropic(self):
        from llm.base import resolve_backend
        from llm.anthropic_backend import AnthropicBackend
        from config import settings
        original = settings.llm_provider
        try:
            settings.llm_provider = "anthropic"
            backend = resolve_backend(api_key="sk-ant-test")
        finally:
            settings.llm_provider = original
        assert isinstance(backend, AnthropicBackend)

    def test_bedrock_provider_resolves(self):
        from llm.base import resolve_backend
        from llm.bedrock import BedrockBackend
        from config import settings
        original = settings.llm_provider
        try:
            settings.llm_provider = "bedrock"
            backend = resolve_backend()
        finally:
            settings.llm_provider = original
        assert isinstance(backend, BedrockBackend)

    def test_ollama_provider_resolves_with_preset(self):
        from llm.base import resolve_backend
        from llm.openai_compat import OpenAICompatBackend
        from config import settings
        original = settings.llm_provider
        try:
            settings.llm_provider = "ollama"
            backend = resolve_backend()
        finally:
            settings.llm_provider = original
        assert isinstance(backend, OpenAICompatBackend)
        assert "11434" in backend._base_url

    def test_groq_provider_resolves_with_preset(self):
        from llm.base import resolve_backend
        from llm.openai_compat import OpenAICompatBackend
        from config import settings
        original = settings.llm_provider
        try:
            settings.llm_provider = "groq"
            backend = resolve_backend(api_key="gsk-test")
        finally:
            settings.llm_provider = original
        assert isinstance(backend, OpenAICompatBackend)
        assert "groq.com" in backend._base_url
        assert backend._api_key == "gsk-test"

    def test_api_key_overrides_settings_key(self):
        from llm.base import resolve_backend
        from config import settings
        original = settings.llm_provider
        try:
            settings.llm_provider = "anthropic"
            backend = resolve_backend(api_key="sk-ant-override")
        finally:
            settings.llm_provider = original
        assert backend._api_key == "sk-ant-override"


# ══════════════════════════════════════════════════════════════════════════
# ANTHROPIC BACKEND
# ══════════════════════════════════════════════════════════════════════════

class TestAnthropicBackend:

    def test_model_property(self):
        from llm.anthropic_backend import AnthropicBackend
        backend = AnthropicBackend(api_key="x", model="claude-haiku-4-5-20251001")
        assert backend.model == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_no_api_key_returns_error_response(self):
        from llm.anthropic_backend import AnthropicBackend
        backend = AnthropicBackend(api_key="")
        result = await backend.complete(messages=[{"role": "user", "content": "hi"}])
        assert result.content == ""
        assert "error" in result.raw

    @pytest.mark.asyncio
    async def test_successful_completion_parsed(self):
        from llm.anthropic_backend import AnthropicBackend
        backend = AnthropicBackend(api_key="sk-ant-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": "Hello there!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
            result = await backend.complete(messages=[{"role": "user", "content": "hi"}])

        assert result.content == "Hello there!"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    @pytest.mark.asyncio
    async def test_tool_use_blocks_extracted(self):
        from llm.anthropic_backend import AnthropicBackend
        backend = AnthropicBackend(api_key="sk-ant-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "claude-haiku-4-5-20251001",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "tu_1", "name": "list_datasets", "input": {}},
            ],
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
            result = await backend.complete(
                messages=[{"role": "user", "content": "list datasets"}],
                tools=[{"name": "list_datasets", "input_schema": {}}],
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "list_datasets"


# ══════════════════════════════════════════════════════════════════════════
# OPENAI-COMPATIBLE BACKEND
# ══════════════════════════════════════════════════════════════════════════

class TestOpenAICompatBackend:

    def test_simple_message_translation(self):
        from llm.openai_compat import OpenAICompatBackend
        backend = OpenAICompatBackend(base_url="http://x", model="llama3")
        result = backend._to_openai_messages(
            [{"role": "user", "content": "hello"}], system="You are helpful."
        )
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[1] == {"role": "user", "content": "hello"}

    def test_tool_use_block_translation(self):
        from llm.openai_compat import OpenAICompatBackend
        backend = OpenAICompatBackend(base_url="http://x", model="llama3")
        anthropic_msg = [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking..."},
                {"type": "tool_use", "id": "tu1", "name": "search", "input": {"q": "x"}},
            ],
        }]
        result = backend._to_openai_messages(anthropic_msg, system="")
        assert result[0]["tool_calls"][0]["function"]["name"] == "search"
        assert json.loads(result[0]["tool_calls"][0]["function"]["arguments"]) == {"q": "x"}

    def test_tool_result_translation(self):
        from llm.openai_compat import OpenAICompatBackend
        backend = OpenAICompatBackend(base_url="http://x", model="llama3")
        anthropic_msg = [{
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "result text"}],
        }]
        result = backend._to_openai_messages(anthropic_msg, system="")
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "tu1"
        assert result[0]["content"] == "result text"

    def test_anthropic_tools_to_openai_functions(self):
        from llm.openai_compat import OpenAICompatBackend
        backend = OpenAICompatBackend(base_url="http://x", model="llama3")
        anthropic_tools = [{
            "name": "search", "description": "Search docs",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }]
        result = backend._to_openai_tools(anthropic_tools)
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search"
        assert result[0]["function"]["parameters"]["properties"]["q"]["type"] == "string"

    def test_parse_response_extracts_content(self):
        from llm.openai_compat import OpenAICompatBackend
        backend = OpenAICompatBackend(base_url="http://x", model="llama3")
        data = {
            "model": "llama3",
            "choices": [{"message": {"content": "Hi there!", "tool_calls": None}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4},
        }
        result = backend._parse_response(data)
        assert result.content == "Hi there!"
        assert result.input_tokens == 8
        assert result.output_tokens == 4

    def test_parse_response_extracts_tool_calls(self):
        from llm.openai_compat import OpenAICompatBackend
        backend = OpenAICompatBackend(base_url="http://x", model="llama3")
        data = {
            "model": "llama3",
            "choices": [{"message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "search", "arguments": '{"q": "test"}'},
                }],
            }}],
            "usage": {},
        }
        result = backend._parse_response(data)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"
        assert result.tool_calls[0]["input"] == {"q": "test"}

    @pytest.mark.asyncio
    async def test_complete_calls_correct_endpoint(self):
        from llm.openai_compat import OpenAICompatBackend
        backend = OpenAICompatBackend(base_url="http://localhost:11434/v1", model="llama3")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "llama3",
            "choices": [{"message": {"content": "ok", "tool_calls": None}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)) as mock_post:
            await backend.complete(messages=[{"role": "user", "content": "hi"}])
            call_args = mock_post.call_args
            assert "chat/completions" in call_args[0][0]


# ══════════════════════════════════════════════════════════════════════════
# BEDROCK BACKEND
# ══════════════════════════════════════════════════════════════════════════

class TestBedrockBackend:

    def test_model_property(self):
        from llm.bedrock import BedrockBackend
        backend = BedrockBackend(model_id="anthropic.claude-haiku-4-5-20251001-v1:0")
        assert backend.model == "anthropic.claude-haiku-4-5-20251001-v1:0"

    @pytest.mark.asyncio
    async def test_complete_parses_converse_response(self):
        from llm.bedrock import BedrockBackend
        backend = BedrockBackend()

        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "Hello from Bedrock"}]}},
            "usage": {"inputTokens": 12, "outputTokens": 6},
        }

        with patch.object(backend, "_make_client", return_value=mock_client):
            result = await backend.complete(messages=[{"role": "user", "content": "hi"}])

        assert result.content == "Hello from Bedrock"
        assert result.input_tokens == 12
        assert result.output_tokens == 6

    @pytest.mark.asyncio
    async def test_complete_parses_tool_use(self):
        from llm.bedrock import BedrockBackend
        backend = BedrockBackend()

        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [
                {"toolUse": {"toolUseId": "t1", "name": "search", "input": {"q": "x"}}}
            ]}},
            "usage": {"inputTokens": 5, "outputTokens": 5},
        }

        with patch.object(backend, "_make_client", return_value=mock_client):
            result = await backend.complete(
                messages=[{"role": "user", "content": "search for x"}],
                tools=[{"name": "search", "input_schema": {}}],
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_error_handled_gracefully(self):
        from llm.bedrock import BedrockBackend
        backend = BedrockBackend()

        mock_client = MagicMock()
        mock_client.converse.side_effect = Exception("AccessDenied")

        with patch.object(backend, "_make_client", return_value=mock_client):
            result = await backend.complete(messages=[{"role": "user", "content": "hi"}])

        assert result.content == ""
        assert "error" in result.raw


# ══════════════════════════════════════════════════════════════════════════
# CLOUD API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def cloud_client():
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


class TestCloudAPI:

    def test_instance_types_endpoint(self, cloud_client):
        resp = cloud_client.get("/api/v1/cloud/sagemaker/instance-types")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "cpu-medium" in data

    def test_llm_providers_endpoint(self, cloud_client):
        resp = cloud_client.get("/api/v1/cloud/llm-providers")
        assert resp.status_code == 200
        providers = resp.json()["data"]
        names = {p["provider"] for p in providers}
        assert "anthropic" in names
        assert "bedrock" in names
        assert "ollama" in names

    def test_sagemaker_submit_missing_dataset(self, cloud_client):
        resp = cloud_client.post("/api/v1/cloud/sagemaker/submit", json={
            "dataset_id": 9999,
            "target_column": "y",
            "role_arn": "mock-role",
            "s3_bucket": "test-bucket",
        })
        assert resp.status_code == 404

    def test_sagemaker_submit_mock_succeeds(self, cloud_client, tmp_path):
        """Upload a dataset, then submit a mock SageMaker job."""
        import pandas as pd, numpy as np
        df = pd.DataFrame({"x": np.random.randn(50), "y": np.random.randint(0, 2, 50)})
        csv = df.to_csv(index=False).encode()
        ds = cloud_client.post("/api/v1/datasets/upload",
             files={"file": ("d.csv", csv, "text/csv")},
             data={"name": "sm_test"}).json()["data"]

        resp = cloud_client.post("/api/v1/cloud/sagemaker/submit", json={
            "dataset_id": ds["id"],
            "target_column": "y",
            "role_arn": "mock-role",
            "s3_bucket": "test-bucket",
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "Completed"
        assert data["local_artifact_path"] is not None


# ══════════════════════════════════════════════════════════════════════════
# ECS CONFIGURATION FILES
# ══════════════════════════════════════════════════════════════════════════

REPO_ROOT = Path(__file__).parent.parent.parent

class TestECSConfig:

    def test_task_definition_is_valid_json(self):
        path = REPO_ROOT / "infra" / "ecs" / "task-definition.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["family"] == "crucible-backend"

    def test_task_definition_uses_fargate(self):
        path = REPO_ROOT / "infra" / "ecs" / "task-definition.json"
        data = json.loads(path.read_text())
        assert "FARGATE" in data["requiresCompatibilities"]

    def test_task_definition_has_health_check(self):
        path = REPO_ROOT / "infra" / "ecs" / "task-definition.json"
        data = json.loads(path.read_text())
        container = data["containerDefinitions"][0]
        assert "healthCheck" in container

    def test_task_definition_secrets_not_plaintext(self):
        """Secrets must use valueFrom (Secrets Manager), not plaintext value."""
        path = REPO_ROOT / "infra" / "ecs" / "task-definition.json"
        data = json.loads(path.read_text())
        container = data["containerDefinitions"][0]
        for secret in container.get("secrets", []):
            assert "valueFrom" in secret
            assert secret["valueFrom"].startswith("arn:aws:secretsmanager:")

    def test_service_definition_is_valid_json(self):
        path = REPO_ROOT / "infra" / "ecs" / "service-definition.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["launchType"] == "FARGATE"

    def test_service_has_circuit_breaker_rollback(self):
        """Deployment circuit breaker must be enabled to auto-rollback bad deploys."""
        path = REPO_ROOT / "infra" / "ecs" / "service-definition.json"
        data = json.loads(path.read_text())
        cb = data["deploymentConfiguration"]["deploymentCircuitBreaker"]
        assert cb["enable"] is True
        assert cb["rollback"] is True

    def test_deploy_ecs_workflow_exists(self):
        path = REPO_ROOT / ".github" / "workflows" / "deploy-ecs.yml"
        assert path.exists()

    def test_deploy_ecs_workflow_uses_oidc(self):
        """Must use OIDC role assumption, not stored AWS access keys."""
        path = REPO_ROOT / ".github" / "workflows" / "deploy-ecs.yml"
        content = path.read_text()
        assert "id-token: write" in content
        assert "role-to-assume" in content
        assert "AWS_SECRET_ACCESS_KEY" not in content

    def test_deploy_ecs_workflow_is_manual_trigger(self):
        """Production deploys should require manual workflow_dispatch, not auto-push."""
        import yaml
        path = REPO_ROOT / ".github" / "workflows" / "deploy-ecs.yml"
        wf = yaml.safe_load(path.read_text())
        assert "workflow_dispatch" in wf.get(True, wf.get("on", {}))
