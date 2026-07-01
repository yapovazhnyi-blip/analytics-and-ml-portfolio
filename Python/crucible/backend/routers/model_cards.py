"""Model card router — /api/v1/experiments/{id}/model-card"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from database import get_db
from model_cards.generator import generate_model_card
from model_cards.renderer import render_markdown, render_html
from models.experiment import Experiment
from models.dataset import Dataset
from schemas.common import DataResponse

router = APIRouter(tags=["model-cards"], dependencies=[Depends(get_current_user)])


@router.get("/experiments/{experiment_id}/model-card")
async def get_model_card(
    experiment_id: int,
    format: str = Query(
        default="json",
        description="Output format: 'json' | 'markdown' | 'html'",
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Generates a model card for a completed experiment.

    The model card documents:
      - Model identity (family, task type, target column)
      - Training data (dataset, rows, features)
      - Performance metrics (CV score, holdout accuracy/F1/R²)
      - Fairness assessment (if POST /experiments/{id}/fairness was run)
      - Feature importances (SHAP, top 10)
      - Data contract summary (if generated)
      - Limitations, ethical considerations, and monitoring recommendations

    Format options:
      json     → structured dict (default, for programmatic use)
      markdown → GitHub-flavoured Markdown (for READMEs, HF Hub)
      html     → self-contained HTML report (for stakeholder sharing)

    Model cards are required by the EU AI Act for high-risk AI systems
    and recommended by the US NIST AI Risk Management Framework.
    """
    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(404, f"Experiment {experiment_id} not found")
    if exp.status != "completed":
        raise HTTPException(
            422,
            f"Model card requires a completed experiment (status: {exp.status}). "
            "Run the experiment to completion first."
        )

    dataset = await db.get(Dataset, exp.dataset_id) if exp.dataset_id else None
    card = generate_model_card(exp, dataset)

    if format == "markdown":
        md = render_markdown(card)
        return PlainTextResponse(
            content=md,
            headers={"Content-Disposition": f'attachment; filename="model_card_{experiment_id}.md"'},
        )

    if format == "html":
        html = render_html(card)
        return HTMLResponse(
            content=html,
            headers={"Content-Disposition": f'attachment; filename="model_card_{experiment_id}.html"'},
        )

    # Default: JSON
    return DataResponse(data=card.to_dict())
