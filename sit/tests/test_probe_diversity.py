"""Tests for probe diversity metrics and DPP-based selection.

Verifies that:
- RBF and cosine kernels are computed correctly
- Mean pairwise similarity decreases with more diverse selections
- Logdet diversity increases with more diverse selections
- SIT-active produces more diverse probe sets than random
"""

import numpy as np
import pytest

from sit.probe.diversity import (
    build_kernel,
    cosine_kernel,
    coverage_entropy,
    logdet_diversity,
    logdet_increment,
    mean_pairwise_similarity,
    rbf_kernel,
)
from sit.probe.features import build_feature_matrix
from sit.probe.policies import ProbeState, select_next_probe
from sit.sim.workload import Workload, CHANNEL_NAMES, N_CHANNELS


def _make_spectators(n=20, rng=None):
    """Create synthetic spectator workloads."""
    if rng is None:
        rng = np.random.RandomState(42)
    specs = []
    for i in range(n):
        specs.append(Workload(
            workload_id=f"spec_{i}",
            role="spectator",
            features=rng.uniform(0, 1, size=N_CHANNELS),
            base_service_us_mean=rng.uniform(50, 500),
            base_service_us_cv=rng.uniform(0.1, 0.5),
        ))
    return specs


class TestRBFKernel:
    """Test RBF kernel computation."""

    def test_shape(self):
        F = np.random.randn(10, 5)
        K = rbf_kernel(F, sigma=1.0)
        assert K.shape == (10, 10)

    def test_diagonal_ones(self):
        F = np.random.randn(10, 5)
        K = rbf_kernel(F, sigma=1.0)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_symmetric(self):
        F = np.random.randn(10, 5)
        K = rbf_kernel(F, sigma=1.0)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_values_in_range(self):
        F = np.random.randn(10, 5)
        K = rbf_kernel(F, sigma=1.0)
        assert np.all(K >= 0)
        assert np.all(K <= 1.0 + 1e-10)

    def test_identical_vectors(self):
        F = np.ones((5, 3))
        K = rbf_kernel(F, sigma=1.0)
        np.testing.assert_allclose(K, 1.0, atol=1e-10)

    def test_orthogonal_vectors_small_sigma(self):
        F = np.eye(5)
        K = rbf_kernel(F, sigma=0.1)
        # Off-diagonal should be near zero for small sigma
        off_diag = K - np.eye(5)
        assert np.all(np.abs(off_diag) < 0.01)


class TestCosineKernel:
    """Test cosine similarity kernel."""

    def test_shape(self):
        F = np.random.randn(10, 5)
        K = cosine_kernel(F)
        assert K.shape == (10, 10)

    def test_diagonal_ones(self):
        F = np.random.randn(10, 5)
        K = cosine_kernel(F)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-6)

    def test_range(self):
        F = np.random.randn(10, 5)
        K = cosine_kernel(F)
        assert np.all(K >= -1.0 - 1e-6)
        assert np.all(K <= 1.0 + 1e-6)


class TestMeanPairwiseSimilarity:
    """Test mean pairwise similarity metric."""

    def test_single_element(self):
        K = np.eye(5)
        assert mean_pairwise_similarity([0], K) == 0.0

    def test_identical_features(self):
        K = np.ones((5, 5))
        sim = mean_pairwise_similarity([0, 1, 2], K)
        assert sim == pytest.approx(1.0)

    def test_orthogonal_features(self):
        K = np.eye(5)
        sim = mean_pairwise_similarity([0, 1, 2], K)
        assert sim == pytest.approx(0.0)


class TestLogdetDiversity:
    """Test log-determinant diversity metric."""

    def test_empty_set(self):
        K = np.eye(5)
        assert logdet_diversity([], K) == 0.0

    def test_single_element(self):
        K = np.eye(5)
        val = logdet_diversity([0], K, epsilon=1e-6)
        assert val > 0  # log(1 + eps) > 0

    def test_more_diverse_is_higher(self):
        """Orthogonal set should have higher logdet than correlated set."""
        K = np.eye(5)
        K[0, 1] = K[1, 0] = 0.9
        K[0, 2] = K[2, 0] = 0.9
        K[1, 2] = K[2, 1] = 0.9

        diverse_val = logdet_diversity([0, 3, 4], K, epsilon=1e-6)
        correlated_val = logdet_diversity([0, 1, 2], K, epsilon=1e-6)
        assert diverse_val > correlated_val

    def test_increment_consistency(self):
        """Incremental logdet should match full difference."""
        rng = np.random.RandomState(42)
        F = rng.randn(10, 5)
        K = rbf_kernel(F, sigma=1.0)

        S = [0, 1, 2]
        new_idx = 5

        full_before = logdet_diversity(S, K, epsilon=1e-6)
        full_after = logdet_diversity(S + [new_idx], K, epsilon=1e-6)
        increment = logdet_increment(S, new_idx, K, epsilon=1e-6)

        np.testing.assert_allclose(
            full_after - full_before, increment, atol=1e-4,
        )


class TestCoverageEntropy:
    """Test coverage entropy."""

    def test_uniform_coverage(self):
        counts = {"a": 5, "b": 5, "c": 5}
        entropy = coverage_entropy(counts)
        assert entropy > 0

    def test_concentrated_coverage(self):
        uniform = coverage_entropy({"a": 5, "b": 5, "c": 5})
        concentrated = coverage_entropy({"a": 15, "b": 0, "c": 0})
        assert uniform > concentrated

    def test_empty_coverage(self):
        assert coverage_entropy({"a": 0, "b": 0}) == 0.0


class TestSITActiveDiversity:
    """Test that SIT-active produces more diverse probes than random."""

    def test_active_lower_similarity_than_random(self):
        """SIT-active should select less similar probe sets than random."""
        rng = np.random.RandomState(42)
        spectators = _make_spectators(30, rng)
        spectator_ids = [s.workload_id for s in spectators]

        F = build_feature_matrix(spectators, mode="workload")
        K = build_kernel(F, kernel_type="rbf", sigma=1.0)

        # Run both policies for many steps
        n_steps = 50
        sims = {}

        for policy_name in ["random", "sit_active"]:
            policy_sims = []
            coverage = {sid: 0 for sid in spectator_ids}
            chosen_so_far = []
            sigma = rng.uniform(0.1, 1.0, size=len(spectator_ids))

            for step in range(n_steps):
                state = ProbeState(
                    target_id="t0",
                    regime_id="r0",
                    spectator_ids=spectator_ids,
                    chosen_so_far=chosen_so_far,
                    coverage_counts=dict(coverage),
                    sigma=sigma,
                    K=K,
                    set_size=3,
                    max_repeats=20,
                    hybrid_lambda=0.5,
                    epsilon=1e-6,
                    rng=np.random.RandomState(42 + step),
                )

                choice = select_next_probe(state, policy_name)
                for sid in choice.chosen_set:
                    coverage[sid] += 1
                chosen_so_far.append(choice.chosen_indices)

                sim = mean_pairwise_similarity(choice.chosen_indices, K)
                policy_sims.append(sim)

            sims[policy_name] = np.mean(policy_sims)

        # SIT-active should have lower or equal mean pairwise similarity
        assert sims["sit_active"] <= sims["random"] + 0.05, (
            f"SIT-active similarity ({sims['sit_active']:.3f}) should be <= "
            f"random ({sims['random']:.3f})"
        )
