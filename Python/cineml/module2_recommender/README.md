# Module 2 — Personalisation Engine

Two recommender models and a FastAPI serving layer. ALS provides the control-arm baseline; the Two-Tower neural model generates treatment-arm rankings that M3 A/B-tests.

## Run

```bash
# Train ALS (fast, CPU, ~2 min)
python module2_recommender/train.py --model als

# Train Two-Tower (GPU recommended; use --sample-size for CPU)
python module2_recommender/train.py --model two-tower --epochs 20

# Serve via FastAPI (or via Docker)
uvicorn module2_recommender.api.main:app --port 8001
```

## Architecture: Two-Tower

```
User ID ──▶ Embedding(64) ──▶ Linear(128) + LayerNorm + ReLU ──┐
                                                                  ▼ cosine similarity / temperature
Item ID ──▶ Embedding(64) ──┐                                   ──┤
Genre IDs ▶ EmbeddingBag ──┘ ──▶ Linear(128) + LayerNorm + ReLU ──┘
```

Training uses BPR loss — directly optimises ranking rather than pointwise rating prediction. L2-normalised vectors + temperature scaling (τ=0.07) give stable cosine similarity scoring.

## API endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness probe |
| `GET /recommend/{user_id}?n=10` | Top-N recommendations (ALS) |
| `GET /similar/{movie_id}?n=10` | Similar items via ALS item-item CF |
| `POST /batch-recommend` | Recommendations for multiple users |
| `GET /docs` | Interactive Swagger UI |

## Evaluation metrics

| Metric | Description |
|--------|-------------|
| NDCG@10 | Ranking quality — penalises relevant items ranked lower |
| MAP@10 | Mean average precision — rewards early hits |
| Hit@10 | Did any top-10 item match? |
| Novelty | Mean self-information — higher = less popular items |
| Coverage | % of catalogue ever recommended |

## Files

| File | Purpose |
|------|---------|
| `models/two_tower.py` | UserTower, ItemTower, BPR forward, cosine scoring |
| `models/als_model.py` | ALS with confidence weighting (α=40), similar-items |
| `evaluation.py` | Full offline evaluation framework |
| `train.py` | Training entry-point — saves `two_tower_config.json` for dimension-safe loading |
| `api/main.py` | FastAPI service — lazy torch import, sys.path fix for pickle |
