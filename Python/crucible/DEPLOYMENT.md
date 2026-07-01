# Deploying Crucible

Three deployment paths, in order of effort. Pick based on what you need.

---

## 1. Render — fastest path to a public URL (5 minutes)

Best for: portfolio demos, sharing a live link with interviewers.

```
1. Push this repo to GitHub (already done if you're reading this from the repo).
2. Go to https://dashboard.render.com/blueprints
3. Connect your GitHub repo — Render detects render.yaml automatically.
4. Set ANTHROPIC_API_KEY in the Render dashboard (Environment tab).
5. Click "Apply" — Render builds and deploys both services.
```

Free tier sleeps after 15 minutes of inactivity (first request after sleep takes
~30s to wake up). Upgrade to the $7/mo "starter" plan for always-on if doing a
live interview demo.

---

## 2. Google Cloud Run — scales to zero, pairs with BigQuery (15 minutes)

Best for: cost-conscious public deployment; you pay $0 when nobody's calling the API.

```bash
# One-time setup
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com

# Build and deploy
cd backend
gcloud run deploy crucible-backend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars STORAGE_BACKEND=local,DISABLE_AUTH=false \
  --set-secrets ANTHROPIC_API_KEY=crucible-anthropic-key:latest
```

New GCP accounts get $300 in free credits. Cloud Run's free tier (2M requests/month)
covers most demo and portfolio traffic indefinitely.

---

## 3. AWS ECS Fargate — production-realistic, most interview-relevant (30–60 minutes)

Best for: demonstrating real production deployment architecture in interviews.
This is what most companies actually use.

### Prerequisites
- AWS account with Fargate, ECR, Secrets Manager, and IAM permissions
- AWS CLI configured (`aws configure`)

### Steps

```bash
# 1. Create an ECR repository
aws ecr create-repository --repository-name crucible-backend

# 2. Build and push the image
aws ecr get-login-password | docker login --username AWS \
  --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker build -t crucible-backend backend/
docker tag crucible-backend:latest \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com/crucible-backend:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/crucible-backend:latest

# 3. Store secrets in AWS Secrets Manager
aws secretsmanager create-secret --name crucible/database-url \
  --secret-string "postgresql+asyncpg://..."
aws secretsmanager create-secret --name crucible/secret-key \
  --secret-string "$(python -c 'import secrets; print(secrets.token_hex(32))')"
aws secretsmanager create-secret --name crucible/anthropic-key \
  --secret-string "sk-ant-..."

# 4. Create the ECS cluster, task definition, and service
aws ecs create-cluster --cluster-name crucible-cluster
aws ecs register-task-definition --cli-input-json file://infra/ecs/task-definition.json
aws ecs create-service --cli-input-json file://infra/ecs/service-definition.json
```

Or trigger the GitHub Actions workflow instead of running these manually:
```
Actions tab → "Deploy to ECS" → Run workflow → select environment
```
This requires configuring OIDC trust between GitHub and an IAM role first —
see `.github/workflows/deploy-ecs.yml` for the role ARN format expected.

AWS Free Tier covers 750 hours/month of a single Fargate task for 12 months —
enough to run Crucible continuously at zero cost during that period.

---

## Interview talking point

> "Crucible is containerised with a multi-stage Dockerfile and deploys identically
> across Render (demo), Cloud Run (cost-optimised), or AWS ECS Fargate (production).
> The same Docker image runs in all three — only the orchestration layer changes.
> Secrets are never baked into the image; they're injected at runtime via each
> platform's secret manager (Render env vars, GCP Secret Manager, AWS Secrets Manager)."
