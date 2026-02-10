"""Smoke test for bootstrap CI coverage.

Generates 20 synthetic worlds from a known distribution, computes
bootstrap CIs for the p99, and checks that empirical coverage is
within a reasonable range of the nominal level.

For small N=20 worlds, we allow coverage between 0.6 and 1.0
for a 95% nominal CI (high variance is expected).
"""

import numpy as np
import pytest

from sit.eval.coverage import evaluate_ci_coverage
from sit.measure.bootstrap import bootstrap_ci


class TestCICoverageSanity:
    def test_lognormal_p99_coverage(self):
        """Bootstrap CI for lognormal p99 should have reasonable coverage."""
        mu_ln = 5.0
        sigma_ln = 0.5

        def gen_samples(rng, n):
            return rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=n)

        def stat_fn(x):
            return float(np.quantile(x, 0.99, method="linear"))

        # True p99 of lognormal
        from scipy.stats import norm
        true_p99 = float(np.exp(mu_ln + sigma_ln * norm.ppf(0.99)))

        def true_stat_fn(rng):
            return true_p99

        df = evaluate_ci_coverage(
            generate_samples_fn=gen_samples,
            stat_fn=stat_fn,
            true_stat_fn=true_stat_fn,
            n_worlds=20,
            n_samples=2000,
            n_resamples=200,
            nominal_alpha=0.05,
            block_size=None,
            base_seed=12345,
        )

        assert len(df) == 20
        coverage = df["covered"].mean()

        # With 20 worlds, coverage should be between 0.6 and 1.0
        # (high variance with small N, but should not be pathologically bad)
        assert 0.6 <= coverage <= 1.0, (
            f"Empirical coverage {coverage:.2f} outside expected range [0.6, 1.0]"
        )

    def test_mean_coverage_high(self):
        """Bootstrap CI for the mean should have very good coverage
        (CLT applies strongly for the mean)."""
        def gen_samples(rng, n):
            return rng.normal(loc=100.0, scale=20.0, size=n)

        def stat_fn(x):
            return float(np.mean(x))

        def true_stat_fn(rng):
            return 100.0

        df = evaluate_ci_coverage(
            generate_samples_fn=gen_samples,
            stat_fn=stat_fn,
            true_stat_fn=true_stat_fn,
            n_worlds=20,
            n_samples=1000,
            n_resamples=200,
            nominal_alpha=0.05,
            block_size=None,
            base_seed=54321,
        )

        coverage = df["covered"].mean()
        # Mean CI should be well-calibrated even with 20 worlds
        assert coverage >= 0.8, (
            f"Mean CI coverage {coverage:.2f} unexpectedly low"
        )

    def test_coverage_df_columns(self):
        """Verify the output DataFrame has expected columns."""
        def gen_samples(rng, n):
            return rng.normal(0, 1, size=n)

        df = evaluate_ci_coverage(
            generate_samples_fn=gen_samples,
            stat_fn=lambda x: float(np.mean(x)),
            true_stat_fn=lambda rng: 0.0,
            n_worlds=5,
            n_samples=100,
            n_resamples=50,
        )

        expected_cols = [
            "world_seed", "true_value_us", "point_estimate_us",
            "ci_lo_us", "ci_hi_us", "ci_width_us", "covered",
            "nominal_alpha", "nominal_coverage", "n_samples", "n_resamples",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
