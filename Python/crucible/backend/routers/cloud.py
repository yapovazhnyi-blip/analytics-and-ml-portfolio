"""Cloud integration router — /api/v1/cloud/*"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from database import get_db
from models.dataset import Dataset
from schemas.common import DataResponse

router = APIRouter(prefix="/cloud", tags=["cloud"], dependencies=[Depends(get_current_user)])


class SageMakerSubmitRequest(BaseModel):
    dataset_id: int
    target_column: str
    task_type: str = Field(default="classification", pattern="^(classification|regression)$")
    role_arn: str = Field(
        ...,
        description="IAM role ARN with S3 and ECR permissions. Use 'mock-role' for local testing.",
    )
    s3_bucket: str
    s3_prefix: str = "crucible/training"
    region: str = "us-east-1"
    instance_type: str = "ml.m5.xlarge"
    experiment_name: str = "crucible-job"


@router.post("/sagemaker/submit")
async def submit_sagemaker_job(
    body: SageMakerSubmitRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Submits a Crucible training job to AWS SageMaker.

    Uploads the dataset to S3, submits a SageMaker training job using a
    pre-built sklearn container, and polls until the job completes.

    For local testing without AWS credentials, set role_arn='mock-role' —
    this simulates a complete job in ~2 seconds without calling AWS.

    Instance type guide:
      ml.m5.large    — 2 vCPU,  8GB — quick experiments, < 10K rows
      ml.m5.xlarge   — 4 vCPU, 16GB — recommended default, < 1M rows
      ml.m5.4xlarge  — 16 vCPU, 64GB — large datasets, > 1M rows
      ml.p3.2xlarge  — V100 GPU — deep learning workloads
    """
    from cloud.sagemaker import SageMakerTrainingRunner, SageMakerConfig

    ds = await db.get(Dataset, body.dataset_id)
    if not ds:
        raise HTTPException(404, f"Dataset {body.dataset_id} not found")
    if not ds.file_path:
        raise HTTPException(422, "Dataset has no local file path to upload.")

    config = SageMakerConfig(
        role_arn=body.role_arn,
        s3_bucket=body.s3_bucket,
        s3_prefix=body.s3_prefix,
        region=body.region,
        instance_type=body.instance_type,
    )
    runner = SageMakerTrainingRunner(config)
    result = await runner.run(
        local_data_path=ds.file_path,
        target_column=body.target_column,
        task_type=body.task_type,
        experiment_name=body.experiment_name,
    )

    if not result.succeeded:
        raise HTTPException(422, f"SageMaker job failed: {result.error}")

    return DataResponse(data={
        "job_name":         result.job_name,
        "status":           result.status,
        "model_s3_uri":     result.model_s3_uri,
        "local_artifact_path": result.local_artifact_path,
        "training_seconds": result.training_seconds,
        "instance_type":    result.instance_type,
    })


@router.get("/sagemaker/instance-types")
async def list_instance_types():
    """Returns the recommended SageMaker instance type presets."""
    from cloud.sagemaker import INSTANCE_TYPES
    return DataResponse(data=INSTANCE_TYPES)


class LLMProviderInfo(BaseModel):
    provider: str
    model: str
    requires_api_key: bool
    is_free: bool
    description: str


@router.get("/llm-providers")
async def list_llm_providers():
    """
    Returns the available LLM provider backends and their characteristics.
    Used by the frontend Settings page to let users choose a provider.
    """
    providers = [
        {"provider": "anthropic", "model": "claude-haiku-4-5-20251001",
         "requires_api_key": True, "is_free": False,
         "description": "Direct Anthropic API. Best quality, native tool-use support."},
        {"provider": "bedrock", "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
         "requires_api_key": False, "is_free": False,
         "description": "Claude via AWS Bedrock. Uses IAM credentials, billed through AWS."},
        {"provider": "ollama", "model": "llama3",
         "requires_api_key": False, "is_free": True,
         "description": "Local model via Ollama. Zero cost, runs offline, lower quality."},
        {"provider": "groq", "model": "llama-3.3-70b-versatile",
         "requires_api_key": True, "is_free": True,
         "description": "Groq cloud inference. Free tier available, very fast."},
        {"provider": "openrouter", "model": "meta-llama/llama-3-8b-instruct",
         "requires_api_key": True, "is_free": True,
         "description": "Routes to 100+ models, many with free tiers."},
    ]
    return DataResponse(data=providers)


@router.get("/tracking-providers")
async def list_tracking_providers():
    """
    Returns the available experiment tracking backends and their
    characteristics, plus which one is currently active (settings.tracking_backend).
    Used by the frontend Settings page to let users see/understand the
    active tracking provider.
    """
    from config import settings

    active = (getattr(settings, "tracking_backend", "") or "mlflow").lower()
    providers = [
        {"provider": "mlflow", "requires_api_key": False, "is_free": True,
         "self_hostable": True,
         "description": "Self-hosted, already running via docker-compose's mlflow service. "
                        "No external account required."},
        {"provider": "wandb", "requires_api_key": True, "is_free": True,
         "self_hostable": False,
         "description": "Weights & Biases cloud SaaS. Free tier for personal projects. "
                        "Richer run comparison and team collaboration UI than self-hosted MLflow."},
        {"provider": "none", "requires_api_key": False, "is_free": True,
         "self_hostable": True,
         "description": "Tracking disabled — experiments still save metrics to the DB, "
                        "just without an external tracking run."},
    ]
    for p in providers:
        p["active"] = p["provider"] == active

    return DataResponse(data=providers)
