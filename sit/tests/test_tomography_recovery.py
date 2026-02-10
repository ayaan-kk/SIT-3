"""Tests for tomography recovery quality.

Verifies that the nonneg elastic net solver correctly recovers
sparse interference vectors from design matrix measurements,
with top-k recall >= 0.95, NDCG >= 0.95, and relative L2 <= 0.10.
"""

import numpy as np
import pytest

from sit.probe.budget import ProbeBudget
from sit.probe.selection import coverage_aware_probes, compute_coverage
from sit.tomography.design import build_design_matrix
from sit.tomography.diagnostics import compute_diagnostics
from sit.tomography.metrics import (
    compute_all_recovery_metrics,
    l1_error,
    l2_error,
    ndcg_at_k,
    relative_l2_error,
    topk_recall,
)
from sit.tomography.solvers import solve_nonneg_elastic_net


class TestDesignMatrix:
    def test_shape_and_binary(self):
        """Design matrix should be binary with correct shape."""
        probes = [["s1", "s3"], ["s2", "s3"], ["s1", "s2"]]
        sids = ["s1", "s2", "s3"]
        A = build_design_matrix(probes, sids)
        assert A.shape == (3, 3)
        assert set(np.unique(A)).issubset({0.0, 1.0})

    def test_correct_entries(self):
        """Each row should mark the spectators in that probe."""
        probes = [["s1", "s3"], ["s2"]]
        sids = ["s1", "s2", "s3"]
        A = build_design_matrix(probes, sids)
        np.testing.assert_array_equal(A[0], [1, 0, 1])
        np.testing.assert_array_equal(A[1], [0, 1, 0])

    def test_unknown_spectator_ignored(self):
        """Spectators not in the ID list should be ignored."""
        probes = [["s1", "s_unknown"]]
        sids = ["s1", "s2"]
        A = build_design_matrix(probes, sids)
        np.testing.assert_array_equal(A[0], [1, 0])


class TestSolver:
    def test_perfect_recovery_simple(self):
        """Solver should perfectly recover a simple sparse vector."""
        rng = np.random.RandomState(42)
        n = 10
        m = 30

        # Ground truth: sparse with 3 nonzero entries
        x_true = np.zeros(n)
        x_true[1] = 5.0
        x_true[4] = 3.0
        x_true[7] = 8.0

        # Random binary design matrix
        A = (rng.uniform(size=(m, n)) < 0.3).astype(float)
        # Ensure each column has at least one nonzero
        for j in range(n):
            if A[:, j].sum() == 0:
                A[rng.randint(m), j] = 1.0

        y = A @ x_true

        x_hat = solve_nonneg_elastic_net(A, y, lambda_1=0.001, lambda_2=0.001)

        rel_err = relative_l2_error(x_hat, x_true)
        assert rel_err < 0.15, f"Relative L2 error {rel_err:.4f} too high"

    def test_nonneg_constraint(self):
        """All solution components should be nonneg."""
        rng = np.random.RandomState(123)
        A = rng.uniform(size=(20, 5))
        y = rng.uniform(size=20)

        x_hat = solve_nonneg_elastic_net(A, y, lambda_1=0.01, lambda_2=0.01)
        assert np.all(x_hat >= 0), "Solution has negative components"

    def test_sparsity_with_l1(self):
        """Strong L1 regularization should produce sparse solutions."""
        rng = np.random.RandomState(99)
        A = rng.uniform(size=(30, 10))
        y = A @ np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 3], dtype=float)

        x_hat = solve_nonneg_elastic_net(A, y, lambda_1=0.5, lambda_2=0.01)
        nnz = np.sum(x_hat > 1e-6)
        assert nnz <= 5, f"Expected sparse solution, got {nnz} nonzeros"


class TestRecoveryMetrics:
    def test_perfect_recovery(self):
        """Perfect recovery should have zero errors and 1.0 metrics."""
        x = np.array([5, 0, 3, 0, 8])
        metrics = compute_all_recovery_metrics(x, x)
        assert metrics["l1_error"] < 1e-10
        assert metrics["l2_error"] < 1e-10
        assert metrics["relative_l2"] < 1e-10
        assert metrics["topk_recall"] == 1.0
        assert metrics["ndcg_at_k"] == 1.0
        assert metrics["sign_consistency"] == 1.0

    def test_topk_recall_partial(self):
        """Partial top-k overlap should give < 1.0 recall."""
        x_true = np.array([10, 5, 0, 0, 0])
        x_hat = np.array([0, 5, 10, 0, 0])  # Gets #2 right, #1 wrong
        recall = topk_recall(x_hat, x_true, k=2)
        assert recall == 0.5

    def test_ndcg_perfect_ranking(self):
        """Perfect ranking should give NDCG = 1.0."""
        x_true = np.array([10, 5, 1, 0, 0])
        x_hat = np.array([100, 50, 10, 0, 0])  # Same order, different scale
        ndcg = ndcg_at_k(x_hat, x_true, k=3)
        assert ndcg == 1.0

    def test_relative_l2_zero_true(self):
        """Relative L2 should handle zero ground truth."""
        x_true = np.zeros(5)
        x_hat = np.zeros(5)
        assert relative_l2_error(x_hat, x_true) == 0.0

        x_hat2 = np.array([1, 0, 0, 0, 0])
        assert relative_l2_error(x_hat2, x_true) == float("inf")


class TestEndToEndRecovery:
    def test_coverage_aware_recovery(self):
        """Full pipeline: probes -> design -> solve -> metrics passes gates."""
        rng = np.random.RandomState(2024)
        n = 15
        spectator_ids = [f"s{i}" for i in range(n)]

        # Sparse ground truth (4 out of 15 nonzero)
        x_true = np.zeros(n)
        nonzero_idx = rng.choice(n, size=4, replace=False)
        for idx in nonzero_idx:
            x_true[idx] = rng.uniform(2.0, 10.0)

        # Generate probes
        budget = ProbeBudget(m_probes=80, set_size=3, max_repeats=25)
        probes = coverage_aware_probes(spectator_ids, budget, rng)

        # Build design matrix
        A = build_design_matrix(probes, spectator_ids)

        # Measurements (noiseless)
        y = A @ x_true

        # Solve
        x_hat = solve_nonneg_elastic_net(A, y, lambda_1=0.005, lambda_2=0.005)

        # Compute metrics
        metrics = compute_all_recovery_metrics(x_hat, x_true)

        # Gates
        assert metrics["topk_recall"] >= 0.95, (
            f"Top-k recall {metrics['topk_recall']:.3f} < 0.95"
        )
        assert metrics["ndcg_at_k"] >= 0.95, (
            f"NDCG {metrics['ndcg_at_k']:.3f} < 0.95"
        )
        assert metrics["relative_l2"] <= 0.10, (
            f"Relative L2 {metrics['relative_l2']:.4f} > 0.10"
        )

    def test_noisy_recovery(self):
        """Recovery with measurement noise should still pass relaxed gates."""
        rng = np.random.RandomState(7777)
        n = 15
        spectator_ids = [f"s{i}" for i in range(n)]

        x_true = np.zeros(n)
        nonzero_idx = rng.choice(n, size=4, replace=False)
        for idx in nonzero_idx:
            x_true[idx] = rng.uniform(3.0, 12.0)

        budget = ProbeBudget(m_probes=100, set_size=3, max_repeats=30)
        probes = coverage_aware_probes(spectator_ids, budget, rng)
        A = build_design_matrix(probes, spectator_ids)

        # Add measurement noise
        y = A @ x_true + rng.normal(0, 0.3, size=A.shape[0])

        x_hat = solve_nonneg_elastic_net(A, y, lambda_1=0.01, lambda_2=0.01)
        metrics = compute_all_recovery_metrics(x_hat, x_true)

        # Relaxed gates for noisy case
        assert metrics["topk_recall"] >= 0.75, (
            f"Noisy top-k recall {metrics['topk_recall']:.3f} < 0.75"
        )
        assert metrics["ndcg_at_k"] >= 0.80, (
            f"Noisy NDCG {metrics['ndcg_at_k']:.3f} < 0.80"
        )
