"""
Profiling router — /api/v1/datasets/{id}/profile

Endpoints:
  POST /datasets/{id}/profile  — run full profiling suite
  GET  /datasets/{id}/profile  — retrieve last profile result (stored as JSON)

The router's only responsibility is HTTP: validate inputs, call ProfileRunner,
map the result to a response schema. All profiling logic lives in the
profiling package.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from auth.dependencies import get_current_user
from fastapi import Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import Dataset
from profiling.runner import ProfileRunner, ProfileReport
from schemas.common import DataResponse

router = APIRouter(tags=["profiling"], dependencies=[Depends(get_current_user)])


# ── Request / Response schemas ─────────────────────────────────────────────

class ProfileRequest(BaseModel):
    target_column: Optional[str] = Field(None, description="Target variable column name")
    time_column: Optional[str] = Field(None, description="Timestamp column for temporal leakage check")
    test_fraction: float = Field(0.2, ge=0.05, le=0.5, description="Fraction of data held out for leakage split")


class MissingnessOut(BaseModel):
    column: str
    missing_count: int
    missing_rate: float
    severity: str
    likely_systematic: bool
    correlated_with: Optional[str] = None
    correlation_strength: Optional[float] = None


class HighCorrelationOut(BaseModel):
    col_a: str
    col_b: str
    correlation: float
    method: str


class VIFOut(BaseModel):
    column: str
    vif: float
    severe: bool


class ClassDistOut(BaseModel):
    label: str
    count: int
    proportion: float


class TargetAnalysisOut(BaseModel):
    column: str
    task_type: str
    n_unique: int
    null_count: int
    class_distribution: list[ClassDistOut] = []
    imbalance_ratio: Optional[float] = None
    is_imbalanced: bool = False
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None
    is_skewed: bool = False
    imbalance_warning: Optional[str] = None
    skewness_warning: Optional[str] = None


class ColumnStatsOut(BaseModel):
    column: str
    dtype: str
    n_unique: int
    null_count: int
    null_rate: float
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    p25: Optional[float] = None
    median: Optional[float] = None
    p75: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None


class LeakageFindingOut(BaseModel):
    leakage_type: str
    severity: str
    column: Optional[str]
    rationale: str
    metric_name: str
    metric_value: float
    threshold: float


class ProfileOut(BaseModel):
    dataset_id: int
    n_rows: int
    n_columns: int
    duration_secs: float
    warnings: list[str]
    missingness: list[MissingnessOut]
    high_correlations: list[HighCorrelationOut]
    vif_results: list[VIFOut]
    target_analysis: Optional[TargetAnalysisOut]
    column_stats: list[ColumnStatsOut]
    leakage_findings: list[LeakageFindingOut]


# ── Helpers ────────────────────────────────────────────────────────────────

def _report_to_out(report: ProfileReport) -> ProfileOut:
    """Map ProfileReport domain objects → Pydantic response schema."""
    import math

    def safe_float(v):
        if v is None:
            return None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    return ProfileOut(
        dataset_id=report.dataset_id,
        n_rows=report.n_rows,
        n_columns=report.n_columns,
        duration_secs=report.duration_secs,
        warnings=report.warnings,
        missingness=[
            MissingnessOut(
                column=m.column,
                missing_count=m.missing_count,
                missing_rate=m.missing_rate,
                severity=m.severity,
                likely_systematic=m.likely_systematic,
                correlated_with=m.correlated_with,
                correlation_strength=m.correlation_strength,
            )
            for m in report.missingness
        ],
        high_correlations=[
            HighCorrelationOut(col_a=p.col_a, col_b=p.col_b,
                               correlation=p.correlation, method=p.method)
            for p in report.correlation.high_pairs
        ],
        vif_results=[
            VIFOut(column=v.column, vif=safe_float(v.vif) or 0.0, severe=v.severe)
            for v in report.correlation.vif_results
        ],
        target_analysis=(
            TargetAnalysisOut(
                column=report.distributions.target_analysis.column,
                task_type=report.distributions.target_analysis.task_type,
                n_unique=report.distributions.target_analysis.n_unique,
                null_count=report.distributions.target_analysis.null_count,
                class_distribution=[
                    ClassDistOut(label=c.label, count=c.count, proportion=c.proportion)
                    for c in report.distributions.target_analysis.class_distribution
                ],
                imbalance_ratio=report.distributions.target_analysis.imbalance_ratio,
                is_imbalanced=report.distributions.target_analysis.is_imbalanced,
                mean=report.distributions.target_analysis.mean,
                std=report.distributions.target_analysis.std,
                min=report.distributions.target_analysis.min,
                max=report.distributions.target_analysis.max,
                skewness=report.distributions.target_analysis.skewness,
                is_skewed=report.distributions.target_analysis.is_skewed,
                imbalance_warning=report.distributions.target_analysis.imbalance_warning,
                skewness_warning=report.distributions.target_analysis.skewness_warning,
            )
            if report.distributions.target_analysis else None
        ),
        column_stats=[
            ColumnStatsOut(
                column=s.column, dtype=s.dtype, n_unique=s.n_unique,
                null_count=s.null_count, null_rate=s.null_rate,
                mean=safe_float(s.mean), std=safe_float(s.std),
                min=safe_float(s.min), p25=safe_float(s.p25),
                median=safe_float(s.median), p75=safe_float(s.p75),
                max=safe_float(s.max), skewness=safe_float(s.skewness),
            )
            for s in report.distributions.column_stats
        ],
        leakage_findings=[
            LeakageFindingOut(
                leakage_type=f.leakage_type,
                severity=f.severity.value,
                column=f.column,
                rationale=f.rationale,
                metric_name=f.metric_name,
                metric_value=round(f.metric_value, 4),
                threshold=f.threshold,
            )
            for f in (report.leakage.findings if report.leakage else [])
        ],
    )


# ── Endpoint ───────────────────────────────────────────────────────────────

@router.post("/datasets/{dataset_id}/profile", response_model=DataResponse[ProfileOut])
async def run_profile(
    dataset_id: int,
    body: ProfileRequest = ProfileRequest(),
    db: AsyncSession = Depends(get_db),
):
    """
    Run the full Crucible profiling suite on a dataset.

    Computes: missingness, correlations, VIF, leakage detection,
    target distribution, and per-column statistics.

    Results are cached for 1 hour, keyed by (dataset content hash, target_column,
    time_column, test_fraction). Re-profiling the same dataset with the same
    parameters returns instantly from cache instead of recomputing.
    """
    from caching.cache import get_profiling_cache, cache_key, PROFILING_TTL_SECS

    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    if ds.status != "ready":
        raise HTTPException(
            status_code=422,
            detail=f"Dataset is not ready for profiling (status: {ds.status})",
        )
    if not ds.file_path:
        raise HTTPException(status_code=422, detail="Dataset has no local file to profile")

    # Cache key: dataset content + the parameters that affect the result.
    content_key = ds.content_hash or f"id-{dataset_id}"
    key = cache_key(
        "profile", content_key,
        body.target_column or "_", body.time_column or "_", str(body.test_fraction),
    )
    cache = get_profiling_cache()
    cached_report = cache.get(key)
    if cached_report is not None:
        return DataResponse(data=_report_to_out(cached_report))

    try:
        runner = ProfileRunner()
        df = ProfileRunner.load_dataframe(ds.file_path, ds.source_type)
        report = await runner.run(
            df=df,
            dataset_id=dataset_id,
            target_column=body.target_column,
            time_column=body.time_column,
            test_fraction=body.test_fraction,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Profiling failed: {exc}")

    cache.set(key, report, ttl_secs=PROFILING_TTL_SECS)
    return DataResponse(data=_report_to_out(report))


@router.get("/cache/stats")
async def get_cache_stats():
    """Returns hit/miss statistics for the profiling and SHAP result caches."""
    from caching.cache import get_profiling_cache, get_shap_cache
    return DataResponse(data={
        "profiling": get_profiling_cache().stats(),
        "shap":      get_shap_cache().stats(),
    })
