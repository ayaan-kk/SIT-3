"""CI coverage evaluation across synthetic worlds.

Generates multiple synthetic worlds with different seeds, estimates
statistics with bootstrap CIs, and checks empirical coverage against
nominal levels. Outputs tail_ci_coverage.parquet for auditing.
"""

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.measure.bootstrap import bootstrap_ci
from sit.measure.tail import estimate_var, estimate_cvar

logger = get_logger("eval.coverage")


def evaluate_ci_coverage(
    generate_samples_fn: Callable[[np.random.RandomState, int], np.ndarray],
    stat_fn: Callable[[np.ndarray], float],
    true_stat_fn: Callable[[np.random.RandomState], float],
    n_worlds: int = 20,
    n_samples: int = 2000,
    n_resamples: int = 200,
    nominal_alpha: float = 0.05,
    block_size: Optional[int] = None,
    base_seed: int = 0,
) -> pd.DataFrame:
    """Evaluate empirical coverage of bootstrap CIs across synthetic worlds.

    For each world:
    1. Generate samples from the world
    2. Compute bootstrap CI at nominal level
    3. Check if the true parameter falls within CI

    Args:
        generate_samples_fn: Callable(rng, n) -> samples array.
        stat_fn: Statistic function (e.g., lambda x: np.quantile(x, 0.99)).
        true_stat_fn: Callable(rng) -> true statistic value for that world.
        n_worlds: Number of synthetic worlds to evaluate.
        n_samples: Samples per world.
        n_resamples: Bootstrap resamples per CI.
        nominal_alpha: Significance level (e.g., 0.05 for 95% CI).
        block_size: Block size for block bootstrap (None for IID).
        base_seed: Base seed for world generation.

    Returns:
        DataFrame with one row per world and coverage results.
    """
    rows = []

    for w in range(n_worlds):
        seed = base_seed + w
        rng = np.random.RandomState(seed)

        # Generate the world's "true" parameter
        true_val = true_stat_fn(rng)

        # Generate samples from this world
        rng2 = np.random.RandomState(seed + 10000)
        samples = generate_samples_fn(rng2, n_samples)

        # Compute bootstrap CI
        boot_rng = np.random.RandomState(seed + 20000)
        ci_lo, ci_hi, _ = bootstrap_ci(
            stat_fn=stat_fn,
            samples=samples,
            n_resamples=n_resamples,
            block_size=block_size,
            alpha=nominal_alpha,
            rng=boot_rng,
        )

        point_est = stat_fn(samples)
        covered = ci_lo <= true_val <= ci_hi

        rows.append({
            "world_seed": seed,
            "true_value_us": true_val,
            "point_estimate_us": point_est,
            "ci_lo_us": ci_lo,
            "ci_hi_us": ci_hi,
            "ci_width_us": ci_hi - ci_lo,
            "covered": covered,
            "nominal_alpha": nominal_alpha,
            "nominal_coverage": 1.0 - nominal_alpha,
            "n_samples": n_samples,
            "n_resamples": n_resamples,
        })

    df = pd.DataFrame(rows)
    empirical_coverage = df["covered"].mean()
    logger.info(
        "CI coverage evaluation: %d worlds, nominal=%.2f, empirical=%.3f",
        n_worlds, 1.0 - nominal_alpha, empirical_coverage,
    )

    return df
