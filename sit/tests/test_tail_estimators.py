"""Tests for tail statistic estimators.

Verifies:
- VaR is monotonically increasing with alpha
- CVaR >= VaR
- violation_rate matches proportion above SLO
- compute_all_tail_stats produces consistent results
"""

import numpy as np
import pytest

from sit.measure.tail import (
    estimate_var,
    estimate_cvar,
    estimate_violation_rate,
    compute_all_tail_stats,
)


class TestEstimateVaR:
    def test_var_increases_with_alpha(self):
        rng = np.random.RandomState(42)
        samples = rng.lognormal(mean=5.0, sigma=0.5, size=10000)

        v90 = estimate_var(samples, 0.90)
        v95 = estimate_var(samples, 0.95)
        v99 = estimate_var(samples, 0.99)

        assert v90 < v95 < v99, (
            f"VaR should increase: v90={v90:.2f}, v95={v95:.2f}, v99={v99:.2f}"
        )

    def test_var_within_sample_range(self):
        rng = np.random.RandomState(42)
        samples = rng.lognormal(mean=5.0, sigma=0.5, size=10000)

        var99 = estimate_var(samples, 0.99)
        assert samples.min() <= var99 <= samples.max()

    def test_var_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            estimate_var(np.array([]), 0.99)

    def test_var_single_sample(self):
        samples = np.array([100.0])
        var = estimate_var(samples, 0.99)
        assert var == pytest.approx(100.0)


class TestEstimateCVaR:
    def test_cvar_gte_var(self):
        rng = np.random.RandomState(42)
        samples = rng.lognormal(mean=5.0, sigma=0.5, size=10000)

        var99 = estimate_var(samples, 0.99)
        cvar99 = estimate_cvar(samples, 0.99)

        assert cvar99 >= var99, (
            f"CVaR ({cvar99:.2f}) should be >= VaR ({var99:.2f})"
        )

    def test_cvar_multiple_alphas(self):
        rng = np.random.RandomState(42)
        samples = rng.lognormal(mean=5.0, sigma=0.5, size=10000)

        for alpha in [0.90, 0.95, 0.99]:
            var_a = estimate_var(samples, alpha)
            cvar_a = estimate_cvar(samples, alpha)
            assert cvar_a >= var_a - 1e-6

    def test_cvar_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            estimate_cvar(np.array([]), 0.99)


class TestEstimateViolationRate:
    def test_known_violation_rate(self):
        # 30 samples above SLO, 70 below
        samples = np.concatenate([np.full(70, 100.0), np.full(30, 600.0)])
        vr = estimate_violation_rate(samples, slo_us=500.0)
        assert vr == pytest.approx(0.30)

    def test_no_violations(self):
        samples = np.full(100, 100.0)
        vr = estimate_violation_rate(samples, slo_us=500.0)
        assert vr == pytest.approx(0.0)

    def test_all_violations(self):
        samples = np.full(100, 600.0)
        vr = estimate_violation_rate(samples, slo_us=500.0)
        assert vr == pytest.approx(1.0)

    def test_approximate_match_for_lognormal(self):
        rng = np.random.RandomState(42)
        samples = rng.lognormal(mean=5.0, sigma=0.5, size=50000)
        slo = float(np.quantile(samples, 0.90))

        vr = estimate_violation_rate(samples, slo)
        # Should be approximately 0.10
        assert 0.05 < vr < 0.15, f"Expected ~0.10, got {vr:.3f}"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            estimate_violation_rate(np.array([]), 500.0)


class TestComputeAllTailStats:
    def test_ordering(self):
        rng = np.random.RandomState(42)
        samples = rng.lognormal(mean=5.0, sigma=0.5, size=10000)

        stats = compute_all_tail_stats(samples, slo_us=500000.0)
        assert stats["mean_latency_us"] <= stats["p95_latency_us"]
        assert stats["p95_latency_us"] <= stats["p99_latency_us"]
        assert stats["p99_latency_us"] <= stats["cvar99_latency_us"]

    def test_n_samples_field(self):
        samples = np.array([1.0, 2.0, 3.0])
        stats = compute_all_tail_stats(samples, slo_us=100.0)
        assert stats["n_samples"] == 3

    def test_violation_rate_range(self):
        rng = np.random.RandomState(42)
        samples = rng.lognormal(mean=5.0, sigma=0.5, size=10000)

        stats = compute_all_tail_stats(samples, slo_us=500000.0)
        assert 0.0 <= stats["violation_rate"] <= 1.0
