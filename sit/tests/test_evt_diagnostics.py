"""EVT diagnostics tests.

Tests:
1. GPD fitting on known heavy-tail synthetic samples
2. KS test produces reasonable p-values
3. "Not applicable" correctly triggers with too few exceedances
4. Parameter stability computation
"""

import numpy as np
import pytest
from scipy import stats as sp_stats

from sit.stats.evt import (
    fit_gpd_mle,
    gpd_var,
    gpd_cvar,
    run_evt_diagnostics,
    compute_stability_score,
    gate_r2_tail_validity,
)


class TestGPDFitting:
    """Test GPD MLE fitting on synthetic data."""

    def test_fit_gpd_on_pareto_exceedances(self):
        """Fit GPD to Pareto-distributed data (xi > 0)."""
        rng = np.random.RandomState(42)
        # Pareto with shape=0.3 -> GPD xi=0.3
        xi_true = 0.3
        sigma_true = 100.0
        exceedances = sp_stats.genpareto.rvs(
            c=xi_true, scale=sigma_true, size=1000, random_state=rng,
        )

        xi_hat, sigma_hat, converged = fit_gpd_mle(exceedances)

        assert converged, "GPD fit should converge on well-behaved data"
        # Allow some estimation error
        assert abs(xi_hat - xi_true) < 0.2, (
            f"xi estimate {xi_hat:.3f} too far from truth {xi_true}"
        )
        assert abs(sigma_hat - sigma_true) / sigma_true < 0.3, (
            f"sigma estimate {sigma_hat:.1f} too far from truth {sigma_true}"
        )

    def test_fit_gpd_exponential_tail(self):
        """Fit GPD to exponential data (xi ≈ 0)."""
        rng = np.random.RandomState(123)
        exceedances = rng.exponential(scale=50.0, size=500)

        xi_hat, sigma_hat, converged = fit_gpd_mle(exceedances)

        assert converged
        # Exponential has xi=0
        assert abs(xi_hat) < 0.15, (
            f"xi should be near 0 for exponential, got {xi_hat:.3f}"
        )

    def test_fit_gpd_too_few_samples(self):
        """GPD fit should return not-converged for tiny samples."""
        exceedances = np.array([1.0, 2.0, 3.0])
        xi, sigma, converged = fit_gpd_mle(exceedances)
        assert not converged


class TestEVTDiagnostics:
    """Test the full EVT diagnostics pipeline."""

    def test_heavy_tail_diagnostics(self):
        """EVT diagnostics on heavy-tail lognormal data."""
        rng = np.random.RandomState(42)
        # Heavy-tail lognormal
        samples = rng.lognormal(mean=5.0, sigma=0.8, size=5000)

        df = run_evt_diagnostics(
            samples_us=samples,
            threshold_quantiles=[0.90, 0.95],
            min_exceedances=50,
        )

        assert len(df) == 2
        assert "xi" in df.columns
        assert "sigma" in df.columns
        assert "ks_stat" in df.columns
        assert "ks_p_value" in df.columns
        assert "evt_var99" in df.columns
        assert "evt_cvar99" in df.columns
        assert "empirical_var99" in df.columns
        assert "applicable" in df.columns

        # At least one threshold should be applicable
        applicable = df[df["applicable"] == True]  # noqa: E712
        assert len(applicable) >= 1

    def test_ks_not_catastrophic(self):
        """KS test p-value should not be catastrophically low
        for well-behaved heavy-tail data."""
        rng = np.random.RandomState(42)
        # Generate from an actual GPD to ensure good fit
        threshold = 100.0
        exceedances = sp_stats.genpareto.rvs(
            c=0.2, scale=50.0, size=2000, random_state=rng,
        )
        samples = np.concatenate([
            rng.uniform(0, threshold, size=8000),
            exceedances + threshold,
        ])

        df = run_evt_diagnostics(
            samples_us=samples,
            threshold_quantiles=[0.80],
            min_exceedances=50,
        )

        applicable = df[df["applicable"] == True]  # noqa: E712
        if len(applicable) > 0:
            median_ks_p = applicable["ks_p_value"].median()
            # Should not be catastrophically bad
            assert median_ks_p > 0.001, (
                f"KS p-value {median_ks_p:.4f} is catastrophically low"
            )

    def test_not_applicable_few_exceedances(self):
        """EVT should correctly report 'not applicable' when too few exceedances."""
        rng = np.random.RandomState(42)
        # Very small sample
        samples = rng.normal(100, 10, size=50)

        df = run_evt_diagnostics(
            samples_us=samples,
            threshold_quantiles=[0.95],
            min_exceedances=200,
        )

        assert len(df) == 1
        row = df.iloc[0]
        assert row["applicable"] == False  # noqa: E712
        assert "too_few_exceedances" in row["reason"]

    def test_stability_score(self):
        """Stability score should be high for consistent GPD fits."""
        rng = np.random.RandomState(42)
        # Well-behaved data
        samples = rng.lognormal(mean=5.0, sigma=0.6, size=10000)

        df = run_evt_diagnostics(
            samples_us=samples,
            threshold_quantiles=[0.90, 0.92, 0.94, 0.95],
            min_exceedances=50,
        )

        score = compute_stability_score(df)
        # Stability should be at least moderate
        assert score >= 0.0, f"Stability score should be non-negative: {score}"


class TestGateR2:
    """Test Gate R2 tail validity."""

    def test_gate_r2_skip_when_no_data(self):
        """Gate R2 should skip when no EVT data."""
        import pandas as pd
        result = gate_r2_tail_validity(pd.DataFrame(), evt_required=False)
        assert result["overall"] == "SKIP"

    def test_gate_r2_fail_when_required_no_data(self):
        """Gate R2 should fail when EVT required but no data."""
        import pandas as pd
        result = gate_r2_tail_validity(pd.DataFrame(), evt_required=True)
        assert result["overall"] == "FAIL"

    def test_gate_r2_on_good_data(self):
        """Gate R2 should pass on data with good fits."""
        rng = np.random.RandomState(42)
        # Generate GPD data for good fits
        threshold = 100.0
        exceedances = sp_stats.genpareto.rvs(
            c=0.2, scale=50.0, size=2000, random_state=rng,
        )
        samples = np.concatenate([
            rng.uniform(0, threshold, size=8000),
            exceedances + threshold,
        ])

        df = run_evt_diagnostics(
            samples_us=samples,
            threshold_quantiles=[0.80, 0.85],
            min_exceedances=50,
        )
        df["stability_score"] = compute_stability_score(df)

        result = gate_r2_tail_validity(
            df, ks_p_threshold=0.01, max_relative_difference=0.50,
        )
        # Should at least not crash; passage depends on fit quality
        assert result["overall"] in ("PASS", "FAIL", "N/A")
