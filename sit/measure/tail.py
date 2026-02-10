"""Tail statistic estimators for latency samples.

Provides VaR (quantile), CVaR (conditional value-at-risk / expected
shortfall), and SLO violation rate estimators. All inputs and outputs
in microseconds.
"""

import warnings

import numpy as np

from sit.core.logging import get_logger

logger = get_logger("measure.tail")

# Minimum sample size before we warn
_MIN_SAMPLES_WARN = 50


def estimate_var(samples_us: np.ndarray, alpha: float = 0.99) -> float:
    """Estimate Value-at-Risk (quantile) at level alpha.

    VaR_alpha is the alpha-quantile of the latency distribution.

    Args:
        samples_us: Latency samples in microseconds.
        alpha: Quantile level (e.g., 0.99 for p99).

    Returns:
        VaR estimate in microseconds.
    """
    n = len(samples_us)
    if n == 0:
        raise ValueError("Cannot estimate VaR from empty samples")
    if n < _MIN_SAMPLES_WARN:
        logger.warning("VaR estimation with only %d samples (recommend >= %d)", n, _MIN_SAMPLES_WARN)

    # Use linear interpolation method for consistency
    return float(np.quantile(samples_us, alpha, method="linear"))


def estimate_cvar(samples_us: np.ndarray, alpha: float = 0.99) -> float:
    """Estimate Conditional Value-at-Risk (expected shortfall) at level alpha.

    CVaR_alpha = E[L | L >= VaR_alpha]

    This is the mean of samples at or above the alpha-quantile.
    CVaR >= VaR by definition.

    Args:
        samples_us: Latency samples in microseconds.
        alpha: Quantile level.

    Returns:
        CVaR estimate in microseconds.
    """
    n = len(samples_us)
    if n == 0:
        raise ValueError("Cannot estimate CVaR from empty samples")
    if n < _MIN_SAMPLES_WARN:
        logger.warning("CVaR estimation with only %d samples (recommend >= %d)", n, _MIN_SAMPLES_WARN)

    var = estimate_var(samples_us, alpha)
    tail = samples_us[samples_us >= var]

    if len(tail) == 0:
        # Edge case: all samples below quantile (rounding), return VaR
        return var

    return float(np.mean(tail))


def estimate_violation_rate(samples_us: np.ndarray, slo_us: float) -> float:
    """Estimate the fraction of samples that violate the SLO.

    Args:
        samples_us: Latency samples in microseconds.
        slo_us: SLO threshold in microseconds.

    Returns:
        Violation rate in [0, 1].
    """
    if len(samples_us) == 0:
        raise ValueError("Cannot estimate violation rate from empty samples")

    return float(np.mean(samples_us > slo_us))


def compute_all_tail_stats(
    samples_us: np.ndarray,
    slo_us: float,
    alpha: float = 0.99,
) -> dict:
    """Compute all tail statistics for a set of latency samples.

    Returns dict with mean, p95, p99, cvar99, violation_rate.
    All values in microseconds except violation_rate which is [0,1].
    """
    return {
        "mean_latency_us": float(np.mean(samples_us)),
        "p95_latency_us": float(np.quantile(samples_us, 0.95, method="linear")),
        "p99_latency_us": estimate_var(samples_us, alpha=0.99),
        "cvar99_latency_us": estimate_cvar(samples_us, alpha=0.99),
        "violation_rate": estimate_violation_rate(samples_us, slo_us),
        "n_samples": len(samples_us),
    }
