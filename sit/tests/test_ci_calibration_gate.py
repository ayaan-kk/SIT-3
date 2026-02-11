"""Gate R1: CI calibration test.

Runs a reduced stats calibration experiment and asserts that
empirical coverage for at least one key metric is >= 0.90
for nominal 0.95 CI.
"""

import numpy as np
import pytest

from sit.stats.coverage import (
    run_ci_calibration,
    compute_coverage_summary,
    gate_r1_ci_calibration,
)


class TestCICalibrationGate:
    """Test CI calibration achieves required coverage."""

    def test_mean_coverage_passes_gate(self):
        """Bootstrap CI for the mean should achieve >= 0.90 coverage
        with 30 repeats at nominal 0.95."""
        df = run_ci_calibration(
            n_repeats=30,
            n_bootstrap_resamples=200,
            reference_multiplier=5,
            nominal_alpha=0.05,
            metric_names=["mean"],
            base_seed=42,
            n_samples_base=500,
        )

        assert len(df) == 30
        coverage = df["covered"].mean()
        assert coverage >= 0.90, (
            f"Mean CI coverage {coverage:.2f} < 0.90 threshold"
        )

    def test_gate_r1_passes_for_mean(self):
        """Gate R1 should pass for the mean metric."""
        df = run_ci_calibration(
            n_repeats=30,
            n_bootstrap_resamples=200,
            reference_multiplier=5,
            nominal_alpha=0.05,
            metric_names=["mean", "p99", "cvar99"],
            base_seed=42,
            n_samples_base=500,
        )

        result = gate_r1_ci_calibration(
            df, min_coverage=0.90, key_metrics=["mean"],
        )
        assert result["overall"] == "PASS", (
            f"Gate R1 failed: {result}"
        )

    def test_coverage_summary_structure(self):
        """Coverage summary should have expected columns."""
        df = run_ci_calibration(
            n_repeats=10,
            n_bootstrap_resamples=100,
            reference_multiplier=3,
            nominal_alpha=0.05,
            metric_names=["mean", "p99"],
            base_seed=100,
            n_samples_base=300,
        )

        summary = compute_coverage_summary(df)
        assert "metric_name" in summary.columns
        assert "empirical_coverage" in summary.columns
        assert "n_repeats" in summary.columns
        assert "n_covered" in summary.columns
        assert len(summary) == 2  # mean and p99

    def test_calibration_df_columns(self):
        """CI calibration DataFrame should have all required columns."""
        df = run_ci_calibration(
            n_repeats=5,
            n_bootstrap_resamples=50,
            reference_multiplier=2,
            nominal_alpha=0.05,
            metric_names=["mean"],
            base_seed=0,
            n_samples_base=200,
        )

        expected_cols = [
            "metric_name", "scheduler_name", "regime_id", "load_param",
            "nominal_level", "ci_lo", "ci_hi", "truth_value",
            "covered", "world_seed", "n_samples", "n_resamples",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_multiple_metrics_coverage(self):
        """All three key metrics should have reasonable coverage."""
        df = run_ci_calibration(
            n_repeats=30,
            n_bootstrap_resamples=200,
            reference_multiplier=5,
            nominal_alpha=0.05,
            metric_names=["mean", "p99", "cvar99"],
            base_seed=42,
            n_samples_base=500,
        )

        summary = compute_coverage_summary(df)
        for _, row in summary.iterrows():
            # At minimum, coverage should not be pathologically bad
            assert row["empirical_coverage"] >= 0.60, (
                f"Coverage for {row['metric_name']} is pathologically low: "
                f"{row['empirical_coverage']:.2f}"
            )
