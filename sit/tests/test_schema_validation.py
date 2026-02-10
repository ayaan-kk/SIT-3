"""Tests for sit.core.schema module.

Covers:
- Missing required columns raises ValueError
- Dtype coercion works for compatible types
- Schema validation passes for well-formed data
- Individual schema validators (trials, decisions, artifacts)
"""

import pandas as pd
import pytest

from sit.core.schema import (
    SCHEMA_VERSION,
    TRIAL_REQUIRED_COLUMNS,
    DECISION_REQUIRED_COLUMNS,
    ARTIFACT_REQUIRED_COLUMNS,
    validate_dataframe,
    validate_trials,
    validate_decisions,
    validate_artifacts,
)


def _make_trial_row(**overrides):
    """Create a minimal valid trial row dict."""
    row = {
        "run_id": "test-run-id",
        "config_hash": "abc123",
        "git_commit": "def456",
        "seed": 42,
        "trial_id": 0,
        "scheduler_name": "random",
        "target_id": "target_0",
        "spectators": '["spec_0"]',
        "regime_id": "regime_0",
        "n_samples": 1000,
        "mean_latency_us": 100.0,
        "p95_latency_us": 150.0,
        "p99_latency_us": 200.0,
        "cvar99_latency_us": 250.0,
        "slo_us": 500000.0,
        "violation_rate": 0.01,
    }
    row.update(overrides)
    return row


def _make_decision_row(**overrides):
    """Create a minimal valid decision row dict."""
    row = {
        "run_id": "test-run-id",
        "config_hash": "abc123",
        "git_commit": "def456",
        "seed": 42,
        "decision_id": 0,
        "scheduler_name": "random",
        "time_index": 0,
        "target_id": "target_0",
        "candidate_sets": '[["spec_0"]]',
        "chosen_set": '["spec_0"]',
        "score_components_json": '{"risk": 0.5}',
        "predicted_metrics_json": '{"p99": 200.0}',
        "realized_metrics_json": "null",
        "safety_pass": True,
    }
    row.update(overrides)
    return row


def _make_artifact_row(**overrides):
    """Create a minimal valid artifact row dict."""
    row = {
        "run_id": "test-run-id",
        "path": "/tmp/test.parquet",
        "sha256": "a" * 64,
        "bytes": 1024,
        "created_at_utc": "2025-01-01T00:00:00+00:00",
        "type": "raw",
    }
    row.update(overrides)
    return row


class TestValidateDataframe:
    def test_missing_column_raises(self):
        df = pd.DataFrame({"run_id": ["x"], "config_hash": ["y"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe(df, TRIAL_REQUIRED_COLUMNS, "trials")

    def test_valid_trial_passes(self):
        df = pd.DataFrame([_make_trial_row()])
        result = validate_dataframe(df, TRIAL_REQUIRED_COLUMNS, "trials")
        assert len(result) == 1

    def test_dtype_coercion_int_to_float(self):
        row = _make_trial_row(mean_latency_us=100)  # int, should coerce to float64
        df = pd.DataFrame([row])
        result = validate_dataframe(df, TRIAL_REQUIRED_COLUMNS, "trials")
        assert result["mean_latency_us"].dtype == "float64"

    def test_dtype_coercion_str_to_int(self):
        row = _make_trial_row(seed="42")  # str, should coerce to int64
        df = pd.DataFrame([row])
        result = validate_dataframe(df, TRIAL_REQUIRED_COLUMNS, "trials")
        assert result["seed"].dtype == "int64"

    def test_completely_empty_fails(self):
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe(df, TRIAL_REQUIRED_COLUMNS, "trials")


class TestValidateTrials:
    def test_valid(self):
        df = pd.DataFrame([_make_trial_row(), _make_trial_row(trial_id=1)])
        result = validate_trials(df)
        assert len(result) == 2

    def test_multiple_rows(self):
        rows = [_make_trial_row(trial_id=i) for i in range(5)]
        df = pd.DataFrame(rows)
        result = validate_trials(df)
        assert len(result) == 5


class TestValidateDecisions:
    def test_valid(self):
        df = pd.DataFrame([_make_decision_row()])
        result = validate_decisions(df)
        assert len(result) == 1


class TestValidateArtifacts:
    def test_valid(self):
        df = pd.DataFrame([_make_artifact_row()])
        result = validate_artifacts(df)
        assert len(result) == 1


class TestSchemaVersion:
    def test_version_is_positive_int(self):
        assert isinstance(SCHEMA_VERSION, int)
        assert SCHEMA_VERSION >= 1
