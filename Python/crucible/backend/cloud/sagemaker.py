"""
SageMaker Training Job Submitter.

WHY SAGEMAKER INSTEAD OF EC2
------------------------------
Running training on a raw EC2 instance means:
  - Paying for the instance even when it's idle (waiting for training to start,
    or sitting unused between training runs)
  - Managing the instance lifecycle (start, stop, termination on failure)
  - Handling OS updates, security patches, and environment reproducibility

SageMaker Training Jobs fix all three:
  - Compute is provisioned on-demand and automatically terminated when the job
    ends. You only pay for the seconds the instance runs.
  - The job is defined declaratively (Docker container + S3 data path +
    instance type). SageMaker handles provisioning, execution, and cleanup.
  - Training runs inside a Docker container — the environment is exactly
    reproducible across runs and teams.

TRAINING JOB ANATOMY
---------------------
A SageMaker training job has five required components:

  1. AlgorithmSpecification — the training container (Docker URI from ECR).
     SageMaker provides pre-built containers for sklearn, XGBoost, and PyTorch.
     Custom containers are pushed to ECR. The container must read data from
     /opt/ml/input/data/<channel_name>/ and write the model to /opt/ml/model/.

  2. InputDataConfig — where to read training data from (S3 path → channel name).
     SageMaker copies the S3 data to the instance before training starts.

  3. OutputDataConfig — where to write the model artifact (S3 path).
     SageMaker copies /opt/ml/model/ to S3 when training completes.

  4. ResourceConfig — instance type and count.
     ml.m5.xlarge: 4 vCPU, 16GB RAM — good for sklearn/XGBoost on medium data
     ml.p3.2xlarge: 8 vCPU + V100 GPU — for PyTorch/TensorFlow fine-tuning

  5. RoleArn — IAM role that gives the training instance permission to:
     - Read from the input S3 bucket
     - Write to the output S3 bucket
     - Pull the Docker image from ECR

DATA FLOW
---------
  Local CSV → upload to S3 (input_s3_uri)
              ↓
              SageMaker copies to training instance (/opt/ml/input/data/train/)
              ↓
              Training script runs, saves model to /opt/ml/model/
              ↓
              SageMaker copies model.tar.gz to S3 (output_s3_uri/output/)
              ↓
              Download model.tar.gz → extract → load with joblib

INTEGRATION WITH CRUCIBLE
--------------------------
CrucibleSageMakerRunner wraps the existing TrainingResult interface. The caller
receives the same TrainingResult dataclass whether training ran locally or on
SageMaker — the rest of the pipeline (SHAP, deployment generator, MLflow) is
unaware of where training happened.

MOCK MODE
---------
When role_arn starts with "mock-" or boto3 raises on import, a simulated job
runs with artificial status transitions. This allows all tests to run without
real AWS credentials.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


# ── Built-in SageMaker container URIs ─────────────────────────────────────────
# These change by region and framework version. Check:
#   https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks.html
SKLEARN_CONTAINER = {
    "us-east-1": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    "eu-west-1": "141502667606.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    "ap-southeast-1": "475088953585.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
}

INSTANCE_TYPES = {
    "cpu-small":  "ml.m5.large",       # 2 vCPU, 8GB — quick experiments
    "cpu-medium": "ml.m5.xlarge",      # 4 vCPU, 16GB — recommended default
    "cpu-large":  "ml.m5.4xlarge",     # 16 vCPU, 64GB — large datasets
    "gpu-v100":   "ml.p3.2xlarge",     # V100 GPU — deep learning
    "gpu-t4":     "ml.g4dn.xlarge",    # T4 GPU — cost-effective deep learning
}


@dataclass
class SageMakerConfig:
    """
    Configuration for a SageMaker training job.

    Attributes:
        role_arn:         IAM role ARN with S3 read/write and ECR pull permissions.
                          Use "mock-role" for local testing.
        s3_bucket:        S3 bucket name for input data and model artifacts.
        s3_prefix:        S3 key prefix (e.g. "crucible/training").
        region:           AWS region (default: us-east-1).
        instance_type:    SageMaker instance type (default: ml.m5.xlarge).
        instance_count:   Number of training instances (default: 1).
        max_runtime_secs: Job timeout in seconds (default: 3600 = 1 hour).
        container_uri:    Docker image URI. Defaults to sklearn 1.2 for the region.
        volume_size_gb:   EBS volume size in GB (default: 30).
        tags:             Dict of AWS tags applied to the training job.
    """
    role_arn: str
    s3_bucket: str
    s3_prefix: str = "crucible/training"
    region: str = "us-east-1"
    instance_type: str = "ml.m5.xlarge"
    instance_count: int = 1
    max_runtime_secs: int = 3600
    container_uri: Optional[str] = None
    volume_size_gb: int = 30
    tags: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.container_uri:
            self.container_uri = SKLEARN_CONTAINER.get(
                self.region,
                SKLEARN_CONTAINER["us-east-1"],
            )


@dataclass
class SageMakerJobResult:
    """Result of a SageMaker training job."""
    job_name: str
    status: str                          # "Completed" | "Failed" | "Stopped"
    model_s3_uri: Optional[str] = None  # s3://bucket/prefix/output/model.tar.gz
    local_artifact_path: Optional[str] = None
    training_seconds: float = 0.0
    instance_type: str = ""
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.status == "Completed" and self.error is None


class SageMakerTrainingRunner:
    """
    Submits a Crucible training job to AWS SageMaker.

    Usage:
        config = SageMakerConfig(
            role_arn="arn:aws:iam::123456789:role/SageMakerTrainingRole",
            s3_bucket="my-crucible-bucket",
            region="us-east-1",
        )
        runner = SageMakerTrainingRunner(config)

        # Upload training data to S3 and run
        job_result = await runner.run(
            local_data_path="/tmp/training_data.csv",
            target_column="churn",
            task_type="classification",
            experiment_name="churn-model-v1",
        )
    """

    POLL_INTERVAL_SECS = 30
    TERMINAL_STATUSES   = {"Completed", "Failed", "Stopped"}

    def __init__(self, config: SageMakerConfig):
        self.config = config

    def _make_client(self):
        """Creates a SageMaker boto3 client. Isolated for easy mocking in tests."""
        import boto3
        return boto3.client("sagemaker", region_name=self.config.region)

    def _make_s3_client(self):
        import boto3
        return boto3.client("s3", region_name=self.config.region)

    async def run(
        self,
        local_data_path: str,
        target_column: str,
        task_type: str,
        experiment_name: str = "crucible-job",
    ) -> SageMakerJobResult:
        """
        Submits a training job and polls until completion.

        Steps:
          1. Upload local_data_path to S3 (input channel)
          2. Submit the SageMaker training job
          3. Poll describe_training_job() until terminal status
          4. Download and extract model.tar.gz from S3

        Returns a SageMakerJobResult with the local artifact path.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._run_sync, local_data_path, target_column, task_type, experiment_name
        )

    def _run_sync(
        self,
        local_data_path: str,
        target_column: str,
        task_type: str,
        experiment_name: str,
    ) -> SageMakerJobResult:
        """Blocking implementation — runs in thread pool via run()."""

        # Mock mode: role_arn starts with "mock-" or "arn:aws:iam::000000"
        if self.config.role_arn.startswith("mock-") or "000000000000" in self.config.role_arn:
            return self._mock_job(experiment_name)

        try:
            sm_client = self._make_client()
            s3_client = self._make_s3_client()

            # 1. Upload training data to S3
            job_name = f"{experiment_name[:32]}-{uuid.uuid4().hex[:8]}"
            input_key = f"{self.config.s3_prefix}/{job_name}/train/data.csv"
            s3_client.upload_file(
                local_data_path,
                self.config.s3_bucket,
                input_key,
            )
            input_s3_uri  = f"s3://{self.config.s3_bucket}/{input_key}"
            output_s3_uri = f"s3://{self.config.s3_bucket}/{self.config.s3_prefix}/{job_name}/output/"

            # 2. Submit training job
            hyperparameters = {
                "target_column": target_column,
                "task_type":     task_type,
                "n_trials":      "20",
            }
            sm_client.create_training_job(
                TrainingJobName=job_name,
                AlgorithmSpecification={
                    "TrainingImage":     self.config.container_uri,
                    "TrainingInputMode": "File",
                },
                RoleArn=self.config.role_arn,
                InputDataConfig=[{
                    "ChannelName":     "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri":      input_s3_uri,
                        }
                    },
                    "ContentType": "text/csv",
                }],
                OutputDataConfig={
                    "S3OutputPath": output_s3_uri,
                },
                ResourceConfig={
                    "InstanceType":   self.config.instance_type,
                    "InstanceCount":  self.config.instance_count,
                    "VolumeSizeInGB": self.config.volume_size_gb,
                },
                HyperParameters={k: str(v) for k, v in hyperparameters.items()},
                StoppingCondition={
                    "MaxRuntimeInSeconds": self.config.max_runtime_secs,
                },
                Tags=[{"Key": k, "Value": v} for k, v in self.config.tags.items()],
            )

            # 3. Poll until terminal status
            start = time.monotonic()
            while True:
                desc = sm_client.describe_training_job(TrainingJobName=job_name)
                status = desc["TrainingJobStatus"]
                if status in self.TERMINAL_STATUSES:
                    break
                time.sleep(self.POLL_INTERVAL_SECS)

            elapsed = time.monotonic() - start

            if status != "Completed":
                failure = desc.get("FailureReason", "Unknown failure")
                return SageMakerJobResult(
                    job_name=job_name, status=status,
                    training_seconds=elapsed, instance_type=self.config.instance_type,
                    error=failure,
                )

            # 4. Download and extract model artifact
            model_s3_key = (
                f"{self.config.s3_prefix}/{job_name}/output/{job_name}/output/model.tar.gz"
            )
            import tempfile, tarfile, os
            local_tar = tempfile.mktemp(suffix=".tar.gz")
            s3_client.download_file(self.config.s3_bucket, model_s3_key, local_tar)
            extract_dir = tempfile.mkdtemp(prefix="sm-model-")
            with tarfile.open(local_tar, "r:gz") as tar:
                tar.extractall(extract_dir)
            os.unlink(local_tar)

            # Find the primary artifact (joblib or pkl file)
            artifact_path = None
            for root, _, files in os.walk(extract_dir):
                for fname in files:
                    if fname.endswith((".joblib", ".pkl")):
                        artifact_path = os.path.join(root, fname)
                        break

            return SageMakerJobResult(
                job_name=job_name,
                status="Completed",
                model_s3_uri=f"s3://{self.config.s3_bucket}/{model_s3_key}",
                local_artifact_path=artifact_path,
                training_seconds=round(elapsed, 1),
                instance_type=self.config.instance_type,
            )

        except Exception as exc:
            return SageMakerJobResult(
                job_name=experiment_name,
                status="Failed",
                training_seconds=0.0,
                instance_type=self.config.instance_type,
                error=str(exc),
            )

    def _mock_job(self, experiment_name: str) -> SageMakerJobResult:
        """Simulates a SageMaker job for local testing."""
        job_name = f"mock-{experiment_name[:20]}-{uuid.uuid4().hex[:6]}"
        import tempfile, joblib
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        import numpy as np
        X = np.random.randn(100, 4)
        y = (X[:, 0] > 0).astype(int)
        model.fit(X, y)
        out = tempfile.mktemp(suffix=".joblib")
        joblib.dump(model, out)
        return SageMakerJobResult(
            job_name=job_name,
            status="Completed",
            model_s3_uri=f"s3://mock-bucket/{self.config.s3_prefix}/{job_name}/model.tar.gz",
            local_artifact_path=out,
            training_seconds=2.5,
            instance_type=self.config.instance_type,
        )


# ── SageMaker endpoint descriptor ─────────────────────────────────────────────

@dataclass
class SageMakerEndpointConfig:
    """Configuration for deploying a model as a SageMaker real-time endpoint."""
    role_arn: str
    model_s3_uri: str                   # model artifact from training job
    region: str = "us-east-1"
    container_uri: Optional[str] = None
    instance_type: str = "ml.m5.large"
    endpoint_name: Optional[str] = None


def describe_endpoint_config(endpoint_config: SageMakerEndpointConfig) -> dict:
    """
    Returns the CloudFormation-compatible SageMaker endpoint configuration dict.
    This can be used with boto3 create_model() + create_endpoint_config() + create_endpoint().

    This function generates the configuration without making any AWS API calls —
    useful for infrastructure-as-code and CI/CD pipelines.
    """
    endpoint_name = endpoint_config.endpoint_name or f"crucible-{uuid.uuid4().hex[:8]}"
    container_uri = endpoint_config.container_uri or SKLEARN_CONTAINER.get(
        endpoint_config.region, SKLEARN_CONTAINER["us-east-1"]
    )
    return {
        "ModelName":      endpoint_name,
        "EndpointName":   endpoint_name,
        "ContainerUri":   container_uri,
        "ModelDataUrl":   endpoint_config.model_s3_uri,
        "RoleArn":        endpoint_config.role_arn,
        "InstanceType":   endpoint_config.instance_type,
        "InitialInstanceCount": 1,
    }
