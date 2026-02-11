"""Episode-level and block bootstrap for scheduling/load sweeps.

Wraps sit.measure.bootstrap with episode-level resampling logic
suitable for comparing schedulers across regimes and load points.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.measure.bootstrap import bootstrap_ci as _base_bootstrap_ci
from sit.measure.tail import estimate_cvar, estimate_var

logger = get_logger("stats.bootstrap")


def episode_bootstrap_ci(
    episode_values: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_resamples: int = 200,
    alpha: float = 0.05,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, float, np.ndarray]:
    """Bootstrap CI by resampling episodes with replacement.

    Each element of episode_values is the aggregate metric from one
    episode (e.g., mean p99 from one simulation episode). We resample
    episodes, recompute the aggregate, and build a percentile CI.

    Args:
        episode_values: Array of per-episode aggregate metric values.
        stat_fn: Aggregation function over episode values (e.g., np.mean).
        n_resamples: Number of bootstrap resamples.
        alpha: Significance level (0.05 for 95% CI).
        rng: Random state for reproducibility.

    Returns:
        Tuple of (ci_lo, ci_hi, bootstrap_distribution).
    """
    return _base_bootstrap_ci(
        stat_fn=stat_fn,
        samples=episode_values,
        n_resamples=n_resamples,
        block_size=None,
        alpha=alpha,
        rng=rng,
    )


def block_bootstrap_ci(
    samples: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_resamples: int = 200,
    block_size: int = 64,
    alpha: float = 0.05,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, float, np.ndarray]:
    """Block bootstrap for autocorrelated time-series samples.

    Delegates to sit.measure.bootstrap.bootstrap_ci with block_size.

    Args:
        samples: Time-series sample array (order matters).
        stat_fn: Statistic function.
        n_resamples: Number of resamples.
        block_size: Contiguous block size.
        alpha: Significance level.
        rng: Random state.

    Returns:
        Tuple of (ci_lo, ci_hi, bootstrap_distribution).
    """
    return _base_bootstrap_ci(
        stat_fn=stat_fn,
        samples=samples,
        n_resamples=n_resamples,
        block_size=block_size,
        alpha=alpha,
        rng=rng,
    )


def compute_condition_cis(
    trials_df: pd.DataFrame,
    metric_cols: List[str],
    group_cols: List[str],
    n_resamples: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute bootstrap CIs for each (scheduler, regime, load) condition.

    Groups trials by group_cols, then for each group and each metric,
    computes an episode-level bootstrap CI over the per-trial metric values.

    Args:
        trials_df: Trials DataFrame with metric columns.
        metric_cols: List of metric column names to compute CIs for.
        group_cols: Columns to group by (e.g., ["scheduler_name", "regime_id"]).
        n_resamples: Bootstrap resamples.
        alpha: Significance level.
        seed: Base seed for reproducibility.

    Returns:
        DataFrame with one row per (group, metric) with CI bounds.
    """
    rows = []
    group_idx = 0

    for group_key, group_df in trials_df.groupby(group_cols, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        for metric in metric_cols:
            vals = group_df[metric].dropna().values
            if len(vals) < 2:
                rows.append({
                    **dict(zip(group_cols, group_key)),
                    "metric_name": metric,
                    "point_estimate": float(vals[0]) if len(vals) == 1 else float("nan"),
                    "ci_lo": float("nan"),
                    "ci_hi": float("nan"),
                    "n_episodes": len(vals),
                    "nominal_alpha": alpha,
                })
                continue

            rng = np.random.RandomState(seed + group_idx)
            ci_lo, ci_hi, _ = episode_bootstrap_ci(
                episode_values=vals,
                stat_fn=np.mean,
                n_resamples=n_resamples,
                alpha=alpha,
                rng=rng,
            )
            rows.append({
                **dict(zip(group_cols, group_key)),
                "metric_name": metric,
                "point_estimate": float(np.mean(vals)),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_episodes": len(vals),
                "nominal_alpha": alpha,
            })
            group_idx += 1

    return pd.DataFrame(rows)
