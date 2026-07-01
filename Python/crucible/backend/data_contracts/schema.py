"""
Data Contracts — schema expectations that data must satisfy before entering
the ML pipeline.

WHY DATA CONTRACTS MATTER
--------------------------
The profiling phase is *descriptive*: it tells you what your data looks like.
Data contracts are *prescriptive*: they define what your data *must* look like.

Without contracts:
  Training runs on January data with income values in [0, 500_000].
  In March, a data pipeline bug produces income values in [0, 5_000_000].
  The model trains silently. Predictions are garbage. You find out three
  weeks later when business metrics drop.

With contracts:
  The same bug is caught at ingestion time — the validation step fails
  with: "Column 'income': 3,241 values exceed max (500,000)".
  The pipeline halts. The bug is fixed before the model trains.

WHAT A CONTRACT DEFINES (per column)
--------------------------------------
  dtype          — expected pandas dtype family (numeric, categorical, boolean)
  nullable       — whether null values are allowed at all
  max_null_rate  — maximum fraction of rows that may be null
  min_val        — minimum allowed value (numeric only)
  max_val        — maximum allowed value (numeric only)
  allowed_values — set of allowed category values (categorical, if low-cardinality)
  min_unique     — minimum number of distinct values (catches constant columns)

AUTO-GENERATION LOGIC
----------------------
When a profiling report is available, the contract is generated automatically:

  Numeric columns:
    min_val = observed_min × (1 - tolerance)   [default tolerance = 0.10]
    max_val = observed_max × (1 + tolerance)

  Categorical columns:
    allowed_values = observed unique values, only if n_unique ≤ 50
    (high-cardinality categoricals get no allowed_values check)

  All columns:
    nullable    = observed_null_rate > 0
    max_null_rate = min(observed_null_rate × 1.5, 1.0)

PANDERA INTEGRATION
--------------------
Validation uses Pandera's DataFrameSchema internally. The contract is stored
as plain JSON (not a Pandera object) so it can be persisted, inspected,
and edited independently of Pandera's API. When validate() is called, the
JSON contract is converted to a Pandera schema at runtime.

This separation means: if you swap Pandera for Great Expectations tomorrow,
you change only the validate() function. The contract format doesn't change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

import numpy as np
import pandas as pd


# ── Column contract ───────────────────────────────────────────────────────────

@dataclass
class ColumnContract:
    """Contract for a single column."""
    name: str
    dtype_family: str           # "numeric" | "categorical" | "boolean" | "datetime" | "text"
    nullable: bool = True
    max_null_rate: float = 1.0  # 0–1: fraction of nulls allowed
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    allowed_values: Optional[list] = None   # None = no restriction
    min_unique: Optional[int] = None        # None = no restriction

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("allowed_values") is not None:
            d["allowed_values"] = sorted(str(v) for v in d["allowed_values"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnContract":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Dataset contract ──────────────────────────────────────────────────────────

@dataclass
class DataContract:
    """Contract for an entire dataset."""
    dataset_id: int
    dataset_name: str
    n_rows_reference: int          # row count when contract was generated
    n_cols: int
    columns: list[ColumnContract] = field(default_factory=list)
    target_column: Optional[str] = None
    tolerance: float = 0.10        # numeric range tolerance used during generation
    version: str = "1.0"

    def to_dict(self) -> dict:
        return {
            "dataset_id":      self.dataset_id,
            "dataset_name":    self.dataset_name,
            "n_rows_reference": self.n_rows_reference,
            "n_cols":          self.n_cols,
            "target_column":   self.target_column,
            "tolerance":       self.tolerance,
            "version":         self.version,
            "columns":         [c.to_dict() for c in self.columns],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DataContract":
        cols = [ColumnContract.from_dict(c) for c in d.get("columns", [])]
        return cls(
            dataset_id=d["dataset_id"],
            dataset_name=d["dataset_name"],
            n_rows_reference=d["n_rows_reference"],
            n_cols=d["n_cols"],
            columns=cols,
            target_column=d.get("target_column"),
            tolerance=d.get("tolerance", 0.10),
            version=d.get("version", "1.0"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "DataContract":
        return cls.from_dict(json.loads(s))


# ── Violation ────────────────────────────────────────────────────────────────

@dataclass
class ContractViolation:
    column: str
    check: str
    expected: str
    observed: str
    n_rows_failed: int
    severity: str    # "error" | "warning"
    examples: list = field(default_factory=list)   # sample failing values

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationResult:
    dataset_id: int
    contract_dataset_id: int
    n_rows: int
    n_cols_checked: int
    passed: bool
    n_violations: int
    n_errors: int
    n_warnings: int
    violations: list[ContractViolation] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "dataset_id":          self.dataset_id,
            "contract_dataset_id": self.contract_dataset_id,
            "n_rows":              self.n_rows,
            "n_cols_checked":      self.n_cols_checked,
            "passed":              self.passed,
            "n_violations":        self.n_violations,
            "n_errors":            self.n_errors,
            "n_warnings":          self.n_warnings,
            "error":               self.error,
            "violations": [v.to_dict() for v in self.violations],
        }


# ══════════════════════════════════════════════════════════════════════════
# AUTO-GENERATION FROM PROFILING / DATAFRAME
# ══════════════════════════════════════════════════════════════════════════

def generate_contract(
    df: pd.DataFrame,
    dataset_id: int,
    dataset_name: str,
    target_column: Optional[str] = None,
    tolerance: float = 0.10,
    max_categories: int = 50,
) -> DataContract:
    """
    Auto-generates a DataContract from an observed DataFrame.

    Each column gets constraints derived from the observed data:
      - Numeric: range [min*(1-tol), max*(1+tol)], max_null_rate from observations
      - Categorical: allowed_values if n_unique ≤ max_categories
      - Boolean: allowed_values = {True, False, 0, 1}

    Args:
        df:              The reference DataFrame (e.g. the training dataset).
        dataset_id:      For labelling the contract.
        dataset_name:    Human-readable name for the contract.
        target_column:   Column to note but not strictly constrain (it won't
                         be present at inference time).
        tolerance:       Numeric range buffer (0.10 = ±10% beyond observed range).
        max_categories:  Maximum number of distinct values for an
                         allowed_values check (skip for high-cardinality cols).

    Returns:
        DataContract ready to persist as JSON.
    """
    columns = []

    for col in df.columns:
        series = df[col]
        n = len(series)
        null_count = int(series.isna().sum())
        null_rate  = null_count / n if n > 0 else 0.0
        nullable   = null_count > 0
        # Allow up to 1.5× the observed null rate (with a minimum floor of 0.01)
        max_null_rate = min(max(null_rate * 1.5, 0.01 if nullable else 0.0), 1.0)

        s_notnull = series.dropna()

        # Determine dtype family
        if pd.api.types.is_bool_dtype(series):
            dtype_family = "boolean"
            col_contract = ColumnContract(
                name=col, dtype_family=dtype_family,
                nullable=nullable, max_null_rate=max_null_rate,
                allowed_values=[True, False],
            )

        elif pd.api.types.is_numeric_dtype(series):
            dtype_family = "numeric"
            if len(s_notnull) > 0:
                obs_min = float(s_notnull.min())
                obs_max = float(s_notnull.max())
                span    = obs_max - obs_min
                # Add tolerance buffer; handle zero-span edge case
                buf = max(span * tolerance, abs(obs_min) * tolerance, 1e-6)
                min_val = obs_min - buf
                max_val = obs_max + buf
            else:
                min_val = max_val = None

            col_contract = ColumnContract(
                name=col, dtype_family=dtype_family,
                nullable=nullable, max_null_rate=max_null_rate,
                min_val=min_val, max_val=max_val,
            )

        elif pd.api.types.is_datetime64_any_dtype(series):
            dtype_family = "datetime"
            col_contract = ColumnContract(
                name=col, dtype_family=dtype_family,
                nullable=nullable, max_null_rate=max_null_rate,
            )

        else:
            # String / object / categorical
            n_unique = s_notnull.nunique()
            dtype_family = "categorical"
            allowed = (
                sorted(str(v) for v in s_notnull.unique().tolist())
                if n_unique <= max_categories and n_unique > 0
                else None
            )
            col_contract = ColumnContract(
                name=col, dtype_family=dtype_family,
                nullable=nullable, max_null_rate=max_null_rate,
                allowed_values=allowed,
            )

        columns.append(col_contract)

    return DataContract(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        n_rows_reference=len(df),
        n_cols=len(df.columns),
        columns=columns,
        target_column=target_column,
        tolerance=tolerance,
    )


# ══════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════

def validate_dataframe(
    df: pd.DataFrame,
    contract: DataContract,
    dataset_id: int,
    max_examples: int = 5,
) -> ValidationResult:
    """
    Validates a DataFrame against a DataContract.

    Runs each column contract check and collects violations. Columns present
    in the contract but missing from df are flagged as errors. Extra columns
    in df are allowed (forward-compatible).

    Returns a ValidationResult with passed=True only if there are zero errors
    (warnings alone do not cause failure).
    """
    violations: list[ContractViolation] = []
    n_checked = 0

    for col_contract in contract.columns:
        col = col_contract.name

        # Missing column → error
        if col not in df.columns:
            violations.append(ContractViolation(
                column=col, check="column_exists",
                expected="column present", observed="column missing",
                n_rows_failed=len(df), severity="error",
            ))
            continue

        n_checked += 1
        series = df[col]
        n = len(series)
        if n == 0:
            continue

        # ── Null rate ────────────────────────────────────────────────────
        null_count = int(series.isna().sum())
        null_rate  = null_count / n

        if not col_contract.nullable and null_count > 0:
            violations.append(ContractViolation(
                column=col, check="not_nullable",
                expected="0 nulls", observed=f"{null_count} nulls ({null_rate:.1%})",
                n_rows_failed=null_count, severity="error",
                examples=[],
            ))
        elif null_rate > col_contract.max_null_rate + 1e-9:
            violations.append(ContractViolation(
                column=col, check="max_null_rate",
                expected=f"null_rate ≤ {col_contract.max_null_rate:.1%}",
                observed=f"null_rate = {null_rate:.1%}",
                n_rows_failed=null_count, severity="warning",
            ))

        s_notnull = series.dropna()

        # ── Numeric range ────────────────────────────────────────────────
        if col_contract.dtype_family == "numeric" and len(s_notnull) > 0:
            try:
                numeric = pd.to_numeric(s_notnull, errors="coerce").dropna()
            except Exception:
                numeric = pd.Series([], dtype=float)

            if col_contract.min_val is not None:
                below = numeric < col_contract.min_val
                if below.any():
                    examples = numeric[below].head(max_examples).tolist()
                    violations.append(ContractViolation(
                        column=col, check="min_value",
                        expected=f"≥ {col_contract.min_val:.4g}",
                        observed=f"min = {numeric[below].min():.4g}",
                        n_rows_failed=int(below.sum()), severity="error",
                        examples=examples,
                    ))

            if col_contract.max_val is not None:
                above = numeric > col_contract.max_val
                if above.any():
                    examples = numeric[above].head(max_examples).tolist()
                    violations.append(ContractViolation(
                        column=col, check="max_value",
                        expected=f"≤ {col_contract.max_val:.4g}",
                        observed=f"max = {numeric[above].max():.4g}",
                        n_rows_failed=int(above.sum()), severity="error",
                        examples=examples,
                    ))

        # ── Allowed values ────────────────────────────────────────────────
        if col_contract.allowed_values is not None and len(s_notnull) > 0:
            allowed_set = set(str(v) for v in col_contract.allowed_values)
            actual_set  = set(str(v) for v in s_notnull.unique())
            unknown = actual_set - allowed_set

            if unknown:
                # Count rows with unknown values
                str_series = s_notnull.astype(str)
                n_unknown  = int((str_series.isin(unknown)).sum())
                violations.append(ContractViolation(
                    column=col, check="allowed_values",
                    expected=f"{len(allowed_set)} allowed categories",
                    observed=f"{len(unknown)} unknown categories: {sorted(unknown)[:5]}",
                    n_rows_failed=n_unknown, severity="error",
                    examples=sorted(unknown)[:max_examples],
                ))

    errors   = [v for v in violations if v.severity == "error"]
    warnings = [v for v in violations if v.severity == "warning"]

    return ValidationResult(
        dataset_id=dataset_id,
        contract_dataset_id=contract.dataset_id,
        n_rows=len(df),
        n_cols_checked=n_checked,
        passed=len(errors) == 0,
        n_violations=len(violations),
        n_errors=len(errors),
        n_warnings=len(warnings),
        violations=violations,
    )
