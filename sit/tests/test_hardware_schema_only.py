"""Hardware schema-only tests.

Validates hardware table schemas and data structures without
executing actual hardware workloads. Safe for CI environments.
"""

import json

import numpy as np
import pandas as pd
import pytest


class TestHardwareTrialsSchema:
    """Validate hw_trials DataFrame schema."""

    def _make_hw_trial_row(self) -> dict:
        return {
            "run_id": "test-run-id",
            "trial_id": 0,
            "target_id": "rpc_loop",
            "spectator_set": "[]",
            "batch_index": 0,
            "mean_latency_us": 150.0,
            "p95_latency_us": 200.0,
            "p99_latency_us": 250.0,
            "cvar99_latency_us": 300.0,
            "violation_rate": 0.0,
            "n_samples": 1000,
            "slo_us": 500000.0,
            "timestamp_start": "2026-01-01T00:00:00Z",
            "timestamp_end": "2026-01-01T00:00:01Z",
            "cpu_usage_snapshot": 25.0,
            "notes": "test",
        }

    def test_required_columns_present(self):
        """hw_trials must have all required columns."""
        row = self._make_hw_trial_row()
        df = pd.DataFrame([row])

        required = [
            "run_id", "trial_id", "target_id", "spectator_set",
            "batch_index", "mean_latency_us", "p95_latency_us",
            "p99_latency_us", "timestamp_start", "timestamp_end",
        ]
        for col in required:
            assert col in df.columns, f"Missing required column: {col}"

    def test_latency_positive(self):
        """Latency values must be positive."""
        row = self._make_hw_trial_row()
        df = pd.DataFrame([row])

        for col in ["mean_latency_us", "p95_latency_us", "p99_latency_us"]:
            assert df[col].iloc[0] > 0, f"{col} must be positive"

    def test_spectator_set_is_json(self):
        """spectator_set must be valid JSON."""
        row = self._make_hw_trial_row()
        row["spectator_set"] = json.dumps(["cpu_burn", "mem_burn"])
        df = pd.DataFrame([row])

        parsed = json.loads(df["spectator_set"].iloc[0])
        assert isinstance(parsed, list)


class TestMeasurementEventsSchema:
    """Validate hw_measurement_events DataFrame schema."""

    def _make_event_row(self) -> dict:
        return {
            "event_id": 0,
            "type": "control",
            "target_id": "rpc_loop",
            "spectator_id": "",
            "stat_type": "p99",
            "stat_value_us": 150.0,
            "batch_index": 0,
            "timestamp": "2026-01-01T00:00:00Z",
            "repeat": 0,
        }

    def test_required_columns_present(self):
        """hw_measurement_events must have all required columns."""
        row = self._make_event_row()
        df = pd.DataFrame([row])

        required = [
            "event_id", "type", "target_id", "spectator_id",
            "stat_type", "stat_value_us", "batch_index", "timestamp",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_event_types_valid(self):
        """Event type must be 'control' or 'treatment'."""
        for etype in ["control", "treatment"]:
            row = self._make_event_row()
            row["type"] = etype
            df = pd.DataFrame([row])
            assert df["type"].iloc[0] in ("control", "treatment")


class TestIRBSEstimatesSchema:
    """Validate hw_irbs_estimates DataFrame schema."""

    def _make_irbs_row(self) -> dict:
        return {
            "spectator_id": "cpu_burn",
            "repeat": 0,
            "naive_delta_us": 25.0,
            "irbs_delta_us": 20.0,
            "absolute_difference": 5.0,
            "drift_estimate_us": 2.5,
            "stat_type": "p99",
        }

    def test_required_columns_present(self):
        """hw_irbs_estimates must have all required columns."""
        row = self._make_irbs_row()
        df = pd.DataFrame([row])

        required = [
            "spectator_id", "naive_delta_us", "irbs_delta_us",
            "absolute_difference", "drift_estimate_us", "stat_type",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_absolute_difference_nonnegative(self):
        """absolute_difference must be >= 0."""
        row = self._make_irbs_row()
        df = pd.DataFrame([row])
        assert df["absolute_difference"].iloc[0] >= 0


class TestInterferenceRankingSchema:
    """Validate hw_interference_ranking DataFrame schema."""

    def _make_ranking_rows(self) -> list:
        return [
            {"spectator_id": "cpu_burn", "rank_irbs": 1, "rank_naive": 2,
             "irbs_delta_us": 30.0, "naive_delta_us": 25.0},
            {"spectator_id": "mem_burn", "rank_irbs": 2, "rank_naive": 1,
             "irbs_delta_us": 20.0, "naive_delta_us": 35.0},
            {"spectator_id": "cache_burn", "rank_irbs": 3, "rank_naive": 3,
             "irbs_delta_us": 10.0, "naive_delta_us": 10.0},
        ]

    def test_required_columns_present(self):
        """Ranking table must have expected columns."""
        df = pd.DataFrame(self._make_ranking_rows())

        required = [
            "spectator_id", "rank_irbs", "rank_naive",
            "irbs_delta_us", "naive_delta_us",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_ranks_are_positive_integers(self):
        """Ranks must be positive integers."""
        df = pd.DataFrame(self._make_ranking_rows())
        for col in ["rank_irbs", "rank_naive"]:
            assert all(df[col] > 0)


class TestTransferCorrelationsSchema:
    """Validate hw_counter_correlations DataFrame schema."""

    def _make_corr_rows(self) -> list:
        return [
            {"metric": "spearman_rho", "value": 0.85, "k": 3, "status": "computed"},
            {"metric": "kendall_tau", "value": 0.75, "k": 3, "status": "computed"},
            {"metric": "topk_overlap_k3", "value": 0.67, "k": 3, "status": "computed"},
        ]

    def test_required_columns_present(self):
        """Correlation table must have expected columns."""
        df = pd.DataFrame(self._make_corr_rows())

        required = ["metric", "value", "k", "status"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_skipped_status(self):
        """When hardware unavailable, status should indicate skip."""
        rows = [{"metric": "spearman_rho", "value": float("nan"),
                 "k": 0, "status": "SKIPPED_NO_HARDWARE"}]
        df = pd.DataFrame(rows)
        assert "SKIPPED" in df["status"].iloc[0]


class TestGateSchemas:
    """Test gate function return schemas."""

    def test_gate_h1_schema(self):
        """Gate H1 result must have expected keys."""
        from sit.eval.hardware import gate_h1_drift_reality

        drift_summary = {
            "drift_pct_of_mean": 2.5,
            "drift_magnitude_mean_us": 3.0,
            "drift_detected": True,
        }
        result = gate_h1_drift_reality(drift_summary)

        assert "status" in result
        assert "drift_magnitude_us" in result
        assert result["status"] in ("PASS", "FAIL")

    def test_gate_h2_schema_empty(self):
        """Gate H2 with empty data should return SKIP."""
        from sit.eval.hardware import gate_h2_irbs_effectiveness

        result = gate_h2_irbs_effectiveness(pd.DataFrame())
        assert result["status"] == "SKIP"

    def test_gate_h3_schema_no_data(self):
        """Gate H3 with no data should return SKIPPED."""
        from sit.eval.hardware import gate_h3_ranking_transfer

        result = gate_h3_ranking_transfer(pd.DataFrame())
        assert "SKIPPED" in result["status"]

    def test_gate_h2_with_data(self):
        """Gate H2 with valid data should return PASS or FAIL."""
        from sit.eval.hardware import gate_h2_irbs_effectiveness

        irbs_df = pd.DataFrame([
            {"spectator_id": "cpu_burn", "repeat": 0,
             "naive_delta_us": 25.0, "irbs_delta_us": 20.0,
             "absolute_difference": 5.0},
            {"spectator_id": "cpu_burn", "repeat": 1,
             "naive_delta_us": 30.0, "irbs_delta_us": 21.0,
             "absolute_difference": 9.0},
        ])
        result = gate_h2_irbs_effectiveness(irbs_df)
        assert result["status"] in ("PASS", "FAIL")

    def test_transfer_correlations_with_sim(self):
        """Transfer correlations should compute with matching data."""
        from sit.eval.hardware import compute_transfer_correlations

        hw_df = pd.DataFrame({
            "spectator_id": ["cpu_burn", "mem_burn", "cache_burn"],
            "rank_irbs": [1, 2, 3],
            "rank_naive": [1, 3, 2],
            "irbs_delta_us": [30.0, 20.0, 10.0],
            "naive_delta_us": [28.0, 15.0, 18.0],
        })
        sim_df = pd.DataFrame({
            "spectator_id": ["s0", "s1", "s2"],
            "sim_interference_us": [35.0, 22.0, 12.0],
            "sim_rank": [1, 2, 3],
        })

        corr_df = compute_transfer_correlations(hw_df, sim_df)
        assert len(corr_df) > 0
        assert "metric" in corr_df.columns
        assert "value" in corr_df.columns
