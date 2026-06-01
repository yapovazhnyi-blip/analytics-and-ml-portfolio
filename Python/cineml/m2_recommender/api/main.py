"""
m2_recommender/api/main.py
FastAPI serving layer for the personalisation engine.

Endpoints:
    GET  /health
    GET  /recommend/{user_id}?model=two_tower&n=10
    POST /recommend/batch
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from m2_recommender.models.als_model import ALSRecommender
from m2_recommender.models.two_tower import TwoTowerModel

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

app = FastAPI(title="CineML Recommender API", version="1.0.0")

# ── Model registry ────────────────────────────────────────────────────────────

_als: ALSRecommender | None = None
_two_tower: TwoTowerModel | None = None


def _load_als() -> ALSRecommender:
    global _als
    if _als is None:
        _als = ALSRecommender.load(ARTIFACTS / "als")
    return _als


def _load_two_tower() -> TwoTowerModel:
    global _two_tower
    if _two_tower is None:
        # Load with dummy config — replace with saved config.json in production
        model = TwoTowerModel(n_users=200_000, n_items=70_000)
        state = torch.load(ARTIFACTS / "two_tower.pt", map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _two_tower = model
    return _two_tower


# ── Schemas ───────────────────────────────────────────────────────────────────

class RecommendationItem(BaseModel):
    movie_id: int
    score: float
    rank: int


class RecommendationResponse(BaseModel):
    user_id: int
    model: str
    recommendations: list[RecommendationItem]


class BatchRequest(BaseModel):
    user_ids: list[int]
    model: Literal["als", "two_tower"] = "two_tower"
    n: int = 10


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models": ["als", "two_tower"]}


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(
    user_id: int,
    model: Literal["als", "two_tower"] = Query(default="two_tower"),
    n: int = Query(default=10, ge=1, le=100),
) -> RecommendationResponse:
    if model == "als":
        m = _load_als()
        raw = m.recommend(user_id, n=n)
        items = [
            RecommendationItem(movie_id=r["movie_id"], score=r["score"], rank=i + 1)
            for i, r in enumerate(raw)
        ]
    elif model == "two_tower":
        # Placeholder: in production, retrieve top-N via ANN (FAISS / ScaNN)
        raise HTTPException(
            status_code=501,
            detail="Two-tower endpoint requires FAISS index. Run build_index.py first.",
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    return RecommendationResponse(user_id=user_id, model=model, recommendations=items)


@app.post("/recommend/batch")
def recommend_batch(req: BatchRequest) -> list[RecommendationResponse]:
    return [
        recommend(uid, model=req.model, n=req.n)  # type: ignore[arg-type]
        for uid in req.user_ids
    ]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
