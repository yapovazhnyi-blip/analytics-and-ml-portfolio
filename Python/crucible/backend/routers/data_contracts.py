"""
Data contracts router — /api/v1/datasets/{id}/contracts

Endpoints:
  POST /datasets/{id}/contracts/generate  — auto-generate contract from dataset
  GET  /datasets/{id}/contracts           — retrieve the stored contract
  POST /datasets/{id}/contracts/validate  — validate a dataset against contract
  DELETE /datasets/{id}/contracts         — remove the stored contract
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from data_contracts.schema import (
    DataContract, generate_contract, validate_dataframe,
)
from database import get_db
from models.dataset import Dataset
from profiling.runner import ProfileRunner
from schemas.common import DataResponse

router = APIRouter(tags=["data-contracts"], dependencies=[Depends(get_current_user)])


# ── Schemas ───────────────────────────────────────────────────────────────────

class GenerateContractRequest(BaseModel):
    target_column: Optional[str] = Field(
        None,
        description="Target column to note in the contract (excluded from strict checks).",
    )
    tolerance: float = Field(
        default=0.10, ge=0.0, le=0.5,
        description=(
            "Numeric range buffer. 0.10 = allow ±10% beyond observed min/max. "
            "Increase for datasets with known legitimate outliers."
        ),
    )
    max_categories: int = Field(
        default=50,
        description="Max distinct values for an allowed_values check. "
                    "Higher-cardinality columns skip the categories check.",
    )


class ValidateRequest(BaseModel):
    dataset_id: int = Field(
        ...,
        description="ID of the dataset to validate against the stored contract.",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/datasets/{dataset_id}/contracts/generate")
async def generate(
    dataset_id: int,
    body: GenerateContractRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Auto-generates a data contract from a dataset's observed statistics.

    The contract captures:
      - Column presence (required columns are those present at generation time)
      - Data types (numeric vs categorical vs boolean)
      - Null rates (allowed null fraction ≤ 1.5× observed)
      - Numeric ranges (min/max with ±tolerance buffer)
      - Allowed categories (for low-cardinality columns)

    The contract is stored in the dataset record as JSON for later retrieval
    and use in validation.

    Intended workflow:
      1. Upload training data → Profile → Generate contract
      2. Upload new production data → Validate against contract
      3. Block the pipeline if validation fails
    """
    import asyncio

    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    if ds.status != "ready":
        raise HTTPException(422, f"Dataset not ready (status: {ds.status})")
    if not ds.file_path:
        raise HTTPException(422, "Dataset has no file path.")

    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(
        None, ProfileRunner.load_dataframe, ds.file_path, ds.source_type
    )

    contract = await loop.run_in_executor(
        None,
        generate_contract,
        df, dataset_id, ds.name,
        body.target_column, body.tolerance, body.max_categories,
    )

    # Persist on the Dataset row
    if hasattr(ds, "contract_json"):
        ds.contract_json = contract.to_json()

    return DataResponse(data={
        **contract.to_dict(),
        "message": (
            f"Contract generated for {len(contract.columns)} columns from "
            f"{contract.n_rows_reference} rows."
        ),
    })


@router.get("/datasets/{dataset_id}/contracts")
async def get_contract(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns the stored data contract for a dataset.
    Returns 404 if no contract has been generated yet.
    """
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    if not hasattr(ds, "contract_json") or not ds.contract_json:
        raise HTTPException(
            404,
            "No data contract for this dataset. "
            "Run POST /datasets/{id}/contracts/generate first."
        )
    return DataResponse(data=json.loads(ds.contract_json))


@router.post("/datasets/{dataset_id}/contracts/validate")
async def validate(
    dataset_id: int,
    body: ValidateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Validates a dataset against the stored contract of the reference dataset.

    The reference dataset (dataset_id in the URL) must already have a contract
    generated. The incoming dataset (body.dataset_id) is loaded and checked
    against that contract.

    Typical use: dataset_id = training dataset, body.dataset_id = new production batch.

    Returns a ValidationResult with:
      - passed: True only if no errors (warnings don't block)
      - violations: per-column check failures with failing value examples
    """
    import asyncio

    # Load reference contract
    ref_ds = await db.get(Dataset, dataset_id)
    if not ref_ds:
        raise HTTPException(404, f"Reference dataset {dataset_id} not found")
    if not hasattr(ref_ds, "contract_json") or not ref_ds.contract_json:
        raise HTTPException(
            422,
            f"Dataset {dataset_id} has no contract. Generate one first with "
            "POST /datasets/{id}/contracts/generate"
        )

    contract = DataContract.from_json(ref_ds.contract_json)

    # Load incoming dataset
    if body.dataset_id == dataset_id:
        # Validate against itself — useful for sanity-checking the contract
        incoming_ds = ref_ds
    else:
        incoming_ds = await db.get(Dataset, body.dataset_id)
        if not incoming_ds:
            raise HTTPException(404, f"Dataset {body.dataset_id} not found")
        if not incoming_ds.file_path:
            raise HTTPException(422, f"Dataset {body.dataset_id} has no file path.")

    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(
        None, ProfileRunner.load_dataframe,
        incoming_ds.file_path, incoming_ds.source_type,
    )

    result = await loop.run_in_executor(
        None, validate_dataframe, df, contract, body.dataset_id,
    )

    return DataResponse(data={
        **result.to_dict(),
        "reference_dataset_name": ref_ds.name,
        "incoming_dataset_name":  incoming_ds.name,
    })


@router.delete("/datasets/{dataset_id}/contracts", status_code=204)
async def delete_contract(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Removes the stored data contract from a dataset."""
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    if hasattr(ds, "contract_json"):
        ds.contract_json = None
