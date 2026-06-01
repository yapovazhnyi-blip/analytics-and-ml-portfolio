"""
Module 2 — Recommender FastAPI service.

Endpoints:
  GET  /health
  GET  /recommend/{user_id}?n=10&model=two-tower
  GET  /similar/{movie_id}?n=10
  POST /batch-recommend

Usage:
    uvicorn module2_recommender.api.main:app --port 8001 --reload
"""
import logging
import os
import pickle
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# torch imported lazily inside load_models() — keeps container startup fast
# and allows the API to run without torch if only ALS is needed
torch = None

log = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_PATH", "data/models"))
DATA_DIR = Path("data/processed")

app = FastAPI(
    title="CineML Recommender API",
    description="Two-Tower & ALS recommendation service",
    version="1.0.0",
)

# ── State ──────────────────────────────────────────────────────────────────────
_models: dict = {}
_movies: pd.DataFrame | None = None


@app.on_event("startup")
async def load_models():
    import sys
    global _movies

    # Add module2_recommender/ to sys.path so pickle can resolve
    # 'models.als_model.ALSRecommender' when unpickling the ALS model
    m2_path = str(Path(__file__).resolve().parent.parent)
    if m2_path not in sys.path:
        sys.path.insert(0, m2_path)

    # Load movies metadata for title lookups
    movies_path = DATA_DIR / "movies.parquet"
    if movies_path.exists():
        _movies = pd.read_parquet(movies_path).set_index("movieId")

    # Load ALS
    als_path = MODEL_DIR / "als_model.pkl"
    if als_path.exists():
        with open(als_path, "rb") as f:
            _models["als"] = pickle.load(f)
        log.info("ALS model loaded")

    # Load Two-Tower — lazy import torch so container starts without it
    tt_path = MODEL_DIR / "two_tower.pt"
    if tt_path.exists():
        try:
            import torch as _torch
            from module2_recommender.models.two_tower import TwoTowerModel
            import json
            cfg_path = MODEL_DIR / "two_tower_config.json"
            if cfg_path.exists():
                cfg = json.load(open(cfg_path, encoding="utf-8"))
            else:
                ckpt = _torch.load(tt_path, map_location="cpu")
                cfg = {
                    "n_users":  ckpt["user_tower.user_embed.weight"].shape[0],
                    "n_items":  ckpt["item_tower.item_embed.weight"].shape[0],
                    "n_genres": ckpt["item_tower.genre_embed.weight"].shape[0],
                }
            model = TwoTowerModel(**cfg)
            model.load_state_dict(_torch.load(tt_path, map_location="cpu"))
            model.eval()
            _models["two-tower"] = model
            log.info("Two-Tower model loaded")
        except (ImportError, OSError) as e:
            log.warning("torch not available, Two-Tower skipped: %s", e)


# ── Schemas ────────────────────────────────────────────────────────────────────

class RecommendationItem(BaseModel):
    movie_id: int
    title: str | None
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    model: str
    recommendations: list[RecommendationItem]


class BatchRequest(BaseModel):
    user_ids: list[int]
    n: int = 10
    model: Literal["als", "two-tower"] = "two-tower"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _title(movie_id: int) -> str | None:
    if _movies is not None and movie_id in _movies.index:
        return _movies.loc[movie_id, "title_clean"]
    return None


def _recommend_als(user_id: int, n: int) -> list[RecommendationItem]:
    als = _models.get("als")
    if not als:
        raise HTTPException(503, "ALS model not loaded")
    try:
        recs = als.recommend(user_id, n)
    except KeyError:
        raise HTTPException(404, f"User {user_id} not in training set (cold start)")
    return [RecommendationItem(movie_id=mid, title=_title(mid), score=score)
            for mid, score in recs]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": list(_models.keys()),
    }


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
async def recommend(
    user_id: int,
    n: int = Query(default=10, ge=1, le=100),
    model: Literal["als", "two-tower"] = "two-tower",
):
    """Return top-N recommendations for a user."""
    if model == "als":
        recs = _recommend_als(user_id, n)
    else:
        raise HTTPException(501, "Two-Tower inference not yet wired to this endpoint")

    return RecommendResponse(user_id=user_id, model=model, recommendations=recs)


@app.get("/similar/{movie_id}", response_model=list[RecommendationItem])
async def similar_items(
    movie_id: int,
    n: int = Query(default=10, ge=1, le=50),
):
    """Return items similar to a given movie (ALS item-item CF)."""
    als = _models.get("als")
    if not als:
        raise HTTPException(503, "ALS model not loaded")
    try:
        sims = als.similar_items(movie_id, n)
    except KeyError:
        raise HTTPException(404, f"Movie {movie_id} not found")
    return [RecommendationItem(movie_id=mid, title=_title(mid), score=score)
            for mid, score in sims]


@app.post("/batch-recommend", response_model=list[RecommendResponse])
async def batch_recommend(request: BatchRequest):
    """Batch recommendations for multiple users."""
    results = []
    for uid in request.user_ids:
        try:
            recs = _recommend_als(uid, request.n)
            results.append(RecommendResponse(user_id=uid, model=request.model, recommendations=recs))
        except HTTPException:
            pass  # Skip cold-start users in batch
    return results
