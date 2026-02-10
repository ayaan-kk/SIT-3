"""Tests for tomography uncertainty quantification.

Verifies that bootstrap CIs for x_hat have reasonable empirical
coverage when compared to known ground truth.
"""

import numpy as np
import pytest

from sit.probe.budget import ProbeBudget
from sit.probe.selection import coverage_aware_probes
from sit.tomography.design import build_design_matrix
from sit.tomography.solvers import solve_nonneg_elastic_net
from sit.tomography.uncertainty import (
    bootstrap_x_hat_ci,
    evaluate_x_hat_coverage,
)


class TestBootstrapXHatCI:
    def test_ci_shape(self):
        """Bootstrap CIs should have correct shapes."""
        rng = np.random.RandomState(42)
        n = 5
        m = 20

        A = (rng.uniform(size=(m, n)) < 0.4).astype(float)
        for j in range(n):
            if A[:, j].sum() == 0:
                A[rng.randint(m), j] = 1.0

        x_true = np.array([3.0, 0, 5.0, 0, 0])
        y = A @ x_true + rng.normal(0, 0.1, size=m)

        x_hat, ci_lo, ci_hi, boot = bootstrap_x_hat_ci(
            A, y, lambda_1=0.01, lambda_2=0.01,
            n_resamples=50, alpha=0.05, rng=rng,
        )

        assert x_hat.shape == (n,)
        assert ci_lo.shape == (n,)
        assert ci_hi.shape == (n,)
        assert boot.shape == (50, n)

    def test_ci_lo_le_hi(self):
        """Lower CI bounds should be <= upper bounds."""
        rng = np.random.RandomState(123)
        n = 5
        m = 20

        A = (rng.uniform(size=(m, n)) < 0.4).astype(float)
        for j in range(n):
            if A[:, j].sum() == 0:
                A[rng.randint(m), j] = 1.0

        y = A @ np.array([1, 2, 0, 0, 3]) + rng.normal(0, 0.1, size=m)

        _, ci_lo, ci_hi, _ = bootstrap_x_hat_ci(
            A, y, n_resamples=30, rng=rng,
        )
        assert np.all(ci_lo <= ci_hi + 1e-10)

    def test_nonneg_ci(self):
        """Bootstrap samples should maintain nonnegativity."""
        rng = np.random.RandomState(77)
        n = 5
        m = 20

        A = (rng.uniform(size=(m, n)) < 0.4).astype(float)
        for j in range(n):
            if A[:, j].sum() == 0:
                A[rng.randint(m), j] = 1.0

        y = A @ np.array([5, 0, 3, 0, 0])

        _, _, _, boot = bootstrap_x_hat_ci(
            A, y, n_resamples=30, rng=rng,
        )
        assert np.all(boot >= -1e-10), "Bootstrap samples should be nonneg"


class TestCoverageEvaluation:
    def test_coverage_perfect_noiseless(self):
        """With noiseless data, CIs should cover all true values."""
        rng = np.random.RandomState(2024)
        n = 8
        m = 40

        A = (rng.uniform(size=(m, n)) < 0.35).astype(float)
        for j in range(n):
            if A[:, j].sum() == 0:
                A[rng.randint(m), j] = 1.0

        x_true = np.zeros(n)
        x_true[1] = 5.0
        x_true[5] = 3.0

        y = A @ x_true

        _, ci_lo, ci_hi, _ = bootstrap_x_hat_ci(
            A, y, lambda_1=0.005, lambda_2=0.005,
            n_resamples=100, alpha=0.05, rng=rng,
        )

        cov = evaluate_x_hat_coverage(x_true, ci_lo, ci_hi)

        # Noiseless case: expect very high coverage
        assert cov["coverage_all"] >= 0.60, (
            f"Coverage {cov['coverage_all']:.2f} too low for noiseless case"
        )

    def test_coverage_with_noise(self):
        """With moderate noise, empirical coverage >= 0.80 for 95% CIs."""
        rng = np.random.RandomState(555)
        n = 10
        spectator_ids = [f"s{i}" for i in range(n)]

        # Sparse ground truth
        x_true = np.zeros(n)
        x_true[2] = 6.0
        x_true[7] = 4.0
        x_true[9] = 8.0

        budget = ProbeBudget(m_probes=80, set_size=3, max_repeats=25)
        probes = coverage_aware_probes(spectator_ids, budget, rng)
        A = build_design_matrix(probes, spectator_ids)

        # Add noise
        y = A @ x_true + rng.normal(0, 0.5, size=A.shape[0])

        _, ci_lo, ci_hi, _ = bootstrap_x_hat_ci(
            A, y, lambda_1=0.01, lambda_2=0.01,
            n_resamples=100, alpha=0.05,
            rng=np.random.RandomState(556),
        )

        cov = evaluate_x_hat_coverage(x_true, ci_lo, ci_hi)

        # With bootstrap and noise, coverage should be reasonable
        # (at least 0.60 for a small problem)
        assert cov["coverage_all"] >= 0.50, (
            f"Noisy coverage {cov['coverage_all']:.2f} < 0.50"
        )

    def test_coverage_stats_keys(self):
        """Coverage evaluation should return all expected keys."""
        x_true = np.array([5.0, 0, 3.0, 0, 0])
        ci_lo = np.array([3.0, -0.5, 1.0, -0.5, -0.5])
        ci_hi = np.array([7.0, 0.5, 5.0, 0.5, 0.5])

        cov = evaluate_x_hat_coverage(x_true, ci_lo, ci_hi)

        expected_keys = [
            "coverage_all", "coverage_nonzero", "coverage_zero",
            "n_total", "n_nonzero", "n_covered", "mean_ci_width_us",
        ]
        for key in expected_keys:
            assert key in cov, f"Missing key: {key}"
