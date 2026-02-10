"""Canonical data schemas for SIT raw datasets.

Defines Pydantic models for each row type (Trial, Decision, Artifact)
and provides DataFrame validation that enforces required columns, types,
and schema versioning.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd


# Current schema version - increment on breaking changes
SCHEMA_VERSION: int = 1


# --- Column definitions ---

TRIAL_REQUIRED_COLUMNS: Dict[str, str] = {
    "run_id": "object",
    "config_hash": "object",
    "git_commit": "object",
    "seed": "int64",
    "trial_id": "int64",
    "scheduler_name": "object",
    "target_id": "object",
    "spectators": "object",
    "regime_id": "object",
    "n_samples": "int64",
    "mean_latency_us": "float64",
    "p95_latency_us": "float64",
    "p99_latency_us": "float64",
    "cvar99_latency_us": "float64",
    "slo_us": "float64",
    "violation_rate": "float64",
}

TRIAL_OPTIONAL_COLUMNS: Dict[str, str] = {
    "notes": "object",
    "schema_version": "int64",
    "created_at_utc": "object",
    "units_latency": "object",
}

DECISION_REQUIRED_COLUMNS: Dict[str, str] = {
    "run_id": "object",
    "config_hash": "object",
    "git_commit": "object",
    "seed": "int64",
    "decision_id": "int64",
    "scheduler_name": "object",
    "time_index": "int64",
    "target_id": "object",
    "candidate_sets": "object",
    "chosen_set": "object",
    "score_components_json": "object",
    "predicted_metrics_json": "object",
    "realized_metrics_json": "object",
    "safety_pass": "bool",
}

DECISION_OPTIONAL_COLUMNS: Dict[str, str] = {
    "schema_version": "int64",
    "created_at_utc": "object",
}

ARTIFACT_REQUIRED_COLUMNS: Dict[str, str] = {
    "run_id": "object",
    "path": "object",
    "sha256": "object",
    "bytes": "int64",
    "created_at_utc": "object",
    "type": "object",
}

ARTIFACT_OPTIONAL_COLUMNS: Dict[str, str] = {
    "schema_version": "int64",
}


@dataclass
class ValidationResult:
    """Result of schema validation on a DataFrame."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    coerced_columns: List[str] = field(default_factory=list)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Dict[str, str],
    schema_name: str = "unknown",
) -> pd.DataFrame:
    """Validate and normalize a DataFrame against a schema.

    Checks that all required columns are present, attempts type coercion,
    and returns a normalized DataFrame.

    Args:
        df: Input DataFrame to validate.
        required_columns: Dict mapping column name to expected dtype string.
        schema_name: Name of the schema (for error messages).

    Returns:
        Normalized DataFrame with correct dtypes.

    Raises:
        ValueError: If required columns are missing or coercion fails.
    """
    result = ValidationResult(valid=True)

    # Check for missing columns
    missing = set(required_columns.keys()) - set(df.columns)
    if missing:
        result.valid = False
        result.errors.append(
            f"[{schema_name}] Missing required columns: {sorted(missing)}"
        )
        raise ValueError("; ".join(result.errors))

    # Attempt dtype coercion
    df = df.copy()
    for col, expected_dtype in required_columns.items():
        if col not in df.columns:
            continue
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            try:
                if expected_dtype == "bool":
                    df[col] = df[col].astype(bool)
                elif expected_dtype == "int64":
                    df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")
                elif expected_dtype == "float64":
                    df[col] = pd.to_numeric(df[col], errors="raise").astype("float64")
                elif expected_dtype == "object":
                    df[col] = df[col].astype(str)
                result.coerced_columns.append(col)
            except (ValueError, TypeError) as e:
                result.valid = False
                result.errors.append(
                    f"[{schema_name}] Column '{col}' cannot be coerced to "
                    f"{expected_dtype}: {e}"
                )

    if not result.valid:
        raise ValueError("; ".join(result.errors))

    return df


def validate_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a trials DataFrame against the canonical schema."""
    return validate_dataframe(df, TRIAL_REQUIRED_COLUMNS, schema_name="trials")


def validate_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a decisions DataFrame against the canonical schema."""
    return validate_dataframe(df, DECISION_REQUIRED_COLUMNS, schema_name="decisions")


def validate_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an artifacts DataFrame against the canonical schema."""
    return validate_dataframe(df, ARTIFACT_REQUIRED_COLUMNS, schema_name="artifacts")
