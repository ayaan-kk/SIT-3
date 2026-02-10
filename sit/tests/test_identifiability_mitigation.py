"""Tests for identifiability diagnostics and mitigation.

Verifies that:
1. Bad design matrices are detected (high cond, low coverage)
2. Mitigation (adding coverage-aware probes) improves coverage_min
3. Well-conditioned matrices pass diagnostics
"""

import numpy as np
import pytest

from sit.probe.budget import ProbeBudget
from sit.probe.selection import coverage_aware_probes
from sit.tomography.design import build_design_matrix
from sit.tomography.diagnostics import (
    compute_diagnostics,
    is_well_conditioned,
    mitigate_design,
)


class TestDiagnostics:
    def test_identity_matrix_perfect(self):
        """Identity matrix should have perfect diagnostics."""
        A = np.eye(5)
        diag = compute_diagnostics(A)
        assert diag["rank"] == 5
        assert diag["cond_est"] == pytest.approx(1.0, abs=0.1)
        assert diag["coverage_min"] == 1.0
        assert diag["coverage_max"] == 1.0

    def test_zero_column_detected(self):
        """Matrix with a zero column should have coverage_min = 0."""
        A = np.array([
            [1, 0, 1],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=float)
        diag = compute_diagnostics(A)
        assert diag["coverage_min"] == 0.0
        assert not is_well_conditioned(diag, min_coverage=1.0)

    def test_high_coherence_detected(self):
        """Identical columns should have coherence = 1.0."""
        A = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ], dtype=float)
        diag = compute_diagnostics(A)
        assert diag["coherence"] >= 0.99

    def test_well_conditioned_random(self):
        """Well-designed random matrix should pass diagnostics."""
        rng = np.random.RandomState(42)
        n = 10
        sids = [f"s{i}" for i in range(n)]
        budget = ProbeBudget(m_probes=40, set_size=3, max_repeats=15)
        probes = coverage_aware_probes(sids, budget, rng)
        A = build_design_matrix(probes, sids)
        diag = compute_diagnostics(A)

        # Coverage-aware probes should ensure all spectators covered
        assert diag["coverage_min"] >= 1.0
        assert diag["rank"] >= min(A.shape)

    def test_empty_matrix(self):
        """Empty matrix should return safe default diagnostics."""
        A = np.zeros((0, 5))
        diag = compute_diagnostics(A)
        assert diag["rank"] == 0
        assert diag["cond_est"] == float("inf")


class TestMitigation:
    def test_mitigation_increases_coverage(self):
        """Mitigation should increase coverage_min for bad matrices."""
        rng = np.random.RandomState(99)
        n = 10
        sids = [f"s{i}" for i in range(n)]

        # Create a deliberately bad design: only cover first 5 spectators
        A_bad = np.zeros((10, n))
        for i in range(10):
            cols = rng.choice(5, size=2, replace=False)  # Only s0-s4
            A_bad[i, cols] = 1.0

        diag_before = compute_diagnostics(A_bad)
        assert diag_before["coverage_min"] == 0.0  # s5-s9 uncovered

        # Mitigate
        A_fixed = mitigate_design(
            A_bad, sids, rng,
            extra_probes=20,
            target_coverage_min=2.0,
        )

        diag_after = compute_diagnostics(A_fixed)
        assert diag_after["coverage_min"] > diag_before["coverage_min"], (
            f"Mitigation did not improve coverage_min: "
            f"{diag_before['coverage_min']} -> {diag_after['coverage_min']}"
        )
        assert A_fixed.shape[0] == A_bad.shape[0] + 20

    def test_mitigation_preserves_original(self):
        """Mitigation should preserve original rows at the top."""
        rng = np.random.RandomState(7)
        n = 5
        sids = [f"s{i}" for i in range(n)]

        A_orig = np.eye(n)
        A_aug = mitigate_design(A_orig, sids, rng, extra_probes=10)

        # First n rows should be unchanged
        np.testing.assert_array_equal(A_aug[:n], A_orig)
        assert A_aug.shape[0] == n + 10

    def test_well_conditioned_passes(self):
        """Well-conditioned matrix should pass without mitigation."""
        rng = np.random.RandomState(42)
        n = 10
        sids = [f"s{i}" for i in range(n)]
        budget = ProbeBudget(m_probes=50, set_size=3, max_repeats=20)
        probes = coverage_aware_probes(sids, budget, rng)
        A = build_design_matrix(probes, sids)
        diag = compute_diagnostics(A)

        assert is_well_conditioned(diag, max_cond=200.0, min_coverage=1.0)
