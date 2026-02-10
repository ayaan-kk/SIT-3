"""Bootstrap confidence interval estimation.

Supports both IID bootstrap and block bootstrap (for autocorrelated
samples, e.g., from queueing simulations). Returns point estimate,
CI bounds, and full bootstrap distribution.
"""

from typing import Callable, Optional, Tuple

import numpy as np

from sit.core.logging import get_logger

logger = get_logger("measure.bootstrap")


def iid_bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    samples: np.ndarray,
    n_resamples: int = 200,
    alpha: float = 0.05,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, float, np.ndarray]:
    """IID bootstrap confidence interval.

    Resamples with replacement from samples, computes stat_fn on each
    resample, and returns the (alpha/2, 1-alpha/2) percentile interval.

    Args:
        stat_fn: Function mapping samples array to scalar statistic.
        samples: Original data array.
        n_resamples: Number of bootstrap resamples.
        alpha: Significance level (0.05 for 95% CI).
        rng: Random state for reproducibility.

    Returns:
        Tuple of (ci_lo, ci_hi, bootstrap_distribution).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n = len(samples)
    boot_stats = np.empty(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        boot_stats[b] = stat_fn(samples[idx])

    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return lo, hi, boot_stats


def block_bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    samples: np.ndarray,
    n_resamples: int = 200,
    block_size: int = 64,
    alpha: float = 0.05,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, float, np.ndarray]:
    """Block bootstrap confidence interval for autocorrelated data.

    Resamples contiguous blocks of data to preserve autocorrelation
    structure, then computes the statistic on each pseudo-sample.

    Args:
        stat_fn: Function mapping samples array to scalar statistic.
        samples: Original data array (order matters).
        n_resamples: Number of bootstrap resamples.
        block_size: Size of contiguous blocks.
        alpha: Significance level.
        rng: Random state for reproducibility.

    Returns:
        Tuple of (ci_lo, ci_hi, bootstrap_distribution).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n = len(samples)
    if block_size > n:
        block_size = n

    # Number of blocks needed to cover n samples
    n_blocks = int(np.ceil(n / block_size))
    boot_stats = np.empty(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        # Sample block start indices
        starts = rng.randint(0, n - block_size + 1, size=n_blocks)
        # Build pseudo-sample by concatenating blocks
        blocks = [samples[s: s + block_size] for s in starts]
        pseudo = np.concatenate(blocks)[:n]  # Trim to original length
        boot_stats[b] = stat_fn(pseudo)

    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return lo, hi, boot_stats


def bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    samples: np.ndarray,
    n_resamples: int = 200,
    block_size: Optional[int] = None,
    alpha: float = 0.05,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, float, np.ndarray]:
    """Unified bootstrap CI interface.

    Uses block bootstrap if block_size is provided and > 1,
    otherwise uses IID bootstrap.

    Args:
        stat_fn: Statistic function.
        samples: Data array.
        n_resamples: Number of resamples.
        block_size: If provided and > 1, use block bootstrap.
        alpha: Significance level.
        rng: Random state.

    Returns:
        Tuple of (ci_lo, ci_hi, bootstrap_distribution).
    """
    if block_size is not None and block_size > 1:
        return block_bootstrap_ci(
            stat_fn, samples, n_resamples, block_size, alpha, rng
        )
    else:
        return iid_bootstrap_ci(
            stat_fn, samples, n_resamples, alpha, rng
        )
