"""Bootstrap confidence intervals for tomography reconstruction.

Provides uncertainty quantification for the recovered interference
vector x_hat by resampling measurement rows and re-solving.
"""

from typing import Optional, Tuple

import numpy as np

from sit.core.logging import get_logger
from sit.tomography.solvers import solve_nonneg_elastic_net

logger = get_logger("tomography.uncertainty")


def bootstrap_x_hat_ci(
    A: np.ndarray,
    y: np.ndarray,
    lambda_1: float = 0.01,
    lambda_2: float = 0.01,
    n_resamples: int = 200,
    alpha: float = 0.05,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap confidence intervals for each component of x_hat.

    Resamples rows of (A, y) with replacement, solves the nonneg elastic
    net on each resample, and computes percentile CIs for each x_j.

    Args:
        A: Design matrix (m, n).
        y: Measurement vector (m,).
        lambda_1: L1 regularization.
        lambda_2: L2 regularization.
        n_resamples: Number of bootstrap resamples.
        alpha: Significance level (0.05 for 95% CI).
        rng: Random state.

    Returns:
        Tuple of (x_hat, ci_lo, ci_hi, boot_samples):
            - x_hat: Point estimate (n,)
            - ci_lo: Lower CI bounds (n,)
            - ci_hi: Upper CI bounds (n,)
            - boot_samples: Bootstrap distribution (n_resamples, n)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    m, n = A.shape

    # Point estimate
    x_hat = solve_nonneg_elastic_net(A, y, lambda_1, lambda_2)

    # Bootstrap resamples
    boot_samples = np.zeros((n_resamples, n), dtype=np.float64)

    for b in range(n_resamples):
        idx = rng.randint(0, m, size=m)
        A_boot = A[idx]
        y_boot = y[idx]

        boot_samples[b] = solve_nonneg_elastic_net(
            A_boot, y_boot, lambda_1, lambda_2,
            max_iter=2000, tol=1e-6,
            warm_start=x_hat,
        )

    # Percentile CIs
    lo_pct = 100 * alpha / 2
    hi_pct = 100 * (1 - alpha / 2)
    ci_lo = np.percentile(boot_samples, lo_pct, axis=0)
    ci_hi = np.percentile(boot_samples, hi_pct, axis=0)

    # Coverage summary
    n_nonzero = int(np.sum(x_hat > 1e-10))
    mean_width = float(np.mean(ci_hi - ci_lo))
    logger.info(
        "Bootstrap CI: %d resamples, alpha=%.2f, %d nonzero components, "
        "mean CI width=%.4f us",
        n_resamples, alpha, n_nonzero, mean_width,
    )

    return x_hat, ci_lo, ci_hi, boot_samples


def evaluate_x_hat_coverage(
    x_true: np.ndarray,
    ci_lo: np.ndarray,
    ci_hi: np.ndarray,
) -> dict:
    """Evaluate coverage of x_hat bootstrap CIs against ground truth.

    Args:
        x_true: True interference vector (n,).
        ci_lo: Lower CI bounds (n,).
        ci_hi: Upper CI bounds (n,).

    Returns:
        Dict with coverage statistics.
    """
    n = len(x_true)
    covered = (ci_lo <= x_true) & (x_true <= ci_hi)
    coverage = float(np.mean(covered))

    # Coverage among nonzero components
    nonzero_mask = x_true > 1e-10
    if np.any(nonzero_mask):
        coverage_nonzero = float(np.mean(covered[nonzero_mask]))
    else:
        coverage_nonzero = 1.0

    # Coverage among zero components
    zero_mask = ~nonzero_mask
    if np.any(zero_mask):
        coverage_zero = float(np.mean(covered[zero_mask]))
    else:
        coverage_zero = 1.0

    ci_widths = ci_hi - ci_lo
    mean_width = float(np.mean(ci_widths))

    result = {
        "coverage_all": coverage,
        "coverage_nonzero": coverage_nonzero,
        "coverage_zero": coverage_zero,
        "n_total": n,
        "n_nonzero": int(np.sum(nonzero_mask)),
        "n_covered": int(np.sum(covered)),
        "mean_ci_width_us": mean_width,
    }

    logger.info(
        "x_hat CI coverage: %.3f overall (%.3f nonzero, %.3f zero), "
        "mean width=%.4f us",
        coverage, coverage_nonzero, coverage_zero, mean_width,
    )

    return result
