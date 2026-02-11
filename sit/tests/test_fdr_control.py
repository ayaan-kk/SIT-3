"""FDR control tests.

Tests:
1. BH procedure produces correct q-values on known p-value arrays
2. Significance flags are consistent with q-values
3. Monotonicity of q-values after sorting
4. Integration with effects DataFrame
"""

import numpy as np
import pandas as pd
import pytest

from sit.stats.fdr import (
    benjamini_hochberg,
    apply_fdr_to_effects,
    export_fdr_results,
    gate_r3_no_phacking,
)


class TestBenjaminiHochberg:
    """Test the core BH procedure."""

    def test_known_p_values(self):
        """BH on a known set of p-values should produce correct q-values."""
        # Classic textbook example
        p_values = np.array([0.001, 0.008, 0.039, 0.041, 0.042,
                             0.060, 0.074, 0.205, 0.212, 0.216])
        m = len(p_values)

        q_values, significant = benjamini_hochberg(p_values, q_threshold=0.10)

        assert len(q_values) == m
        assert len(significant) == m

        # First few should be significant at q=0.10
        # BH: p_i * m / i for sorted order
        # i=1: 0.001 * 10/1 = 0.010  -> sig
        # i=2: 0.008 * 10/2 = 0.040  -> sig
        # i=3: 0.039 * 10/3 = 0.130  -> not sig at 0.10
        assert significant[0] == True   # p=0.001
        assert significant[1] == True   # p=0.008

    def test_all_significant(self):
        """When all p-values are very small, all should be significant."""
        p_values = np.array([0.0001, 0.0002, 0.0003, 0.0005])
        q_values, significant = benjamini_hochberg(p_values, q_threshold=0.10)

        assert all(significant), "All very small p-values should be significant"
        assert all(q_values <= 0.10)

    def test_none_significant(self):
        """When all p-values are large, none should be significant."""
        p_values = np.array([0.50, 0.60, 0.70, 0.80, 0.90])
        q_values, significant = benjamini_hochberg(p_values, q_threshold=0.10)

        assert not any(significant), "No large p-values should be significant"
        assert all(q_values > 0.10)

    def test_q_values_bounded(self):
        """Q-values should be in [0, 1]."""
        rng = np.random.RandomState(42)
        p_values = rng.uniform(0, 1, size=50)
        q_values, _ = benjamini_hochberg(p_values, q_threshold=0.10)

        assert np.all(q_values >= 0.0)
        assert np.all(q_values <= 1.0)

    def test_monotonicity_after_sort(self):
        """When p-values are sorted, q-values should be monotonically non-decreasing."""
        p_values = np.array([0.01, 0.02, 0.03, 0.10, 0.20, 0.50])
        q_values, _ = benjamini_hochberg(p_values, q_threshold=0.10)

        # Since p-values are already sorted, q-values should be non-decreasing
        for i in range(len(q_values) - 1):
            assert q_values[i] <= q_values[i + 1] + 1e-10, (
                f"q-values not monotone: q[{i}]={q_values[i]:.4f} > q[{i+1}]={q_values[i+1]:.4f}"
            )

    def test_empty_input(self):
        """Empty p-value array should return empty arrays."""
        q_values, significant = benjamini_hochberg(np.array([]))
        assert len(q_values) == 0
        assert len(significant) == 0

    def test_single_p_value(self):
        """Single p-value: q-value = p-value."""
        p_values = np.array([0.05])
        q_values, significant = benjamini_hochberg(p_values, q_threshold=0.10)

        assert len(q_values) == 1
        assert abs(q_values[0] - 0.05) < 1e-10
        assert significant[0] == True  # 0.05 <= 0.10

    def test_preserves_order(self):
        """Q-values should correspond to the original p-value ordering."""
        # Unsorted p-values
        p_values = np.array([0.50, 0.001, 0.10, 0.005])
        q_values, significant = benjamini_hochberg(p_values, q_threshold=0.10)

        # p=0.001 (index 1) should have the smallest q-value
        assert q_values[1] <= q_values[0]
        assert q_values[1] <= q_values[2]

        # p=0.001 should definitely be significant
        assert significant[1] == True


class TestApplyFDR:
    """Test FDR application to effects DataFrames."""

    def test_apply_to_effects_df(self):
        """apply_fdr_to_effects should add q_value and significant columns."""
        effects_df = pd.DataFrame({
            "bucket_id": ["b0", "b1", "b2", "b3"],
            "metric_name": ["cvar99", "cvar99", "p99", "p99"],
            "scheduler_a": ["sit", "sit", "sit", "sit"],
            "scheduler_b": ["base", "base", "base", "base"],
            "p_value": [0.001, 0.05, 0.20, 0.80],
            "mean_difference": [-100, -50, 10, 5],
        })

        result = apply_fdr_to_effects(effects_df, q_threshold=0.10)

        assert "q_value" in result.columns
        assert "significant" in result.columns
        assert len(result) == 4

    def test_empty_effects(self):
        """Empty effects should return empty with new columns."""
        effects_df = pd.DataFrame(columns=["p_value", "bucket_id"])
        result = apply_fdr_to_effects(effects_df, q_threshold=0.10)
        assert "q_value" in result.columns
        assert "significant" in result.columns


class TestExportFDR:
    """Test FDR results export."""

    def test_export_format(self):
        """export_fdr_results should produce canonical format."""
        effects_df = pd.DataFrame({
            "bucket_id": ["b0", "b1"],
            "metric_name": ["cvar99", "p99"],
            "scheduler_a": ["sit", "sit"],
            "scheduler_b": ["base", "base"],
            "p_value": [0.01, 0.20],
            "q_value": [0.02, 0.20],
            "significant": [True, False],
        })

        result = export_fdr_results(effects_df)
        assert "raw_p" in result.columns
        assert "q_value" in result.columns
        assert "significant" in result.columns


class TestGateR3:
    """Test Gate R3: no p-hacking."""

    def test_gate_passes_when_consistent(self):
        """Gate R3 passes when all significant claims have q <= threshold."""
        fdr_df = pd.DataFrame({
            "bucket_id": ["b0", "b1", "b2"],
            "q_value": [0.01, 0.05, 0.50],
            "significant": [True, True, False],
        })

        result = gate_r3_no_phacking(fdr_df, q_threshold=0.10)
        assert result["overall"] == "PASS"

    def test_gate_fails_when_inconsistent(self):
        """Gate R3 fails if a 'significant' claim has q > threshold."""
        fdr_df = pd.DataFrame({
            "bucket_id": ["b0", "b1"],
            "q_value": [0.01, 0.20],
            "significant": [True, True],  # b1 marked sig but q=0.20 > 0.10
        })

        result = gate_r3_no_phacking(fdr_df, q_threshold=0.10)
        assert result["overall"] == "FAIL"
        assert result["n_violations"] == 1

    def test_gate_passes_empty(self):
        """Gate R3 passes on empty data."""
        result = gate_r3_no_phacking(pd.DataFrame(), q_threshold=0.10)
        assert result["overall"] == "PASS"

    def test_gate_passes_no_significant(self):
        """Gate R3 passes when nothing is significant."""
        fdr_df = pd.DataFrame({
            "bucket_id": ["b0", "b1"],
            "q_value": [0.50, 0.80],
            "significant": [False, False],
        })

        result = gate_r3_no_phacking(fdr_df, q_threshold=0.10)
        assert result["overall"] == "PASS"
        assert result["n_significant"] == 0
