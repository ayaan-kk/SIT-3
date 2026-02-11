"""Regime-wise effect sizes and paired comparisons.

Computes effect sizes (Cliff's delta, Cohen's d), paired statistical
tests (Wilcoxon signed-rank), and bootstrap CIs for effects across
scheduler pairs within regime buckets.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from sit.core.logging import get_logger
from sit.stats.bootstrap import episode_bootstrap_ci

logger = get_logger("stats.effects")


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta effect size.

    Cliff's delta = (# concordant - # discordant) / (n_x * n_y)
    Range: [-1, 1]. Values near 0 indicate no effect.

    Args:
        x: Sample from group A.
        y: Sample from group B.

    Returns:
        Cliff's delta.
    """
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0

    # Vectorized comparison
    more = 0
    less = 0
    for xi in x:
        more += np.sum(xi > y)
        less += np.sum(xi < y)

    return (more - less) / (n_x * n_y)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d (standardized mean difference).

    d = (mean_x - mean_y) / pooled_std

    Args:
        x: Sample from group A.
        y: Sample from group B.

    Returns:
        Cohen's d.
    """
    n_x, n_y = len(x), len(y)
    if n_x < 2 or n_y < 2:
        return 0.0

    mean_diff = np.mean(x) - np.mean(y)
    pooled_var = ((n_x - 1) * np.var(x, ddof=1) + (n_y - 1) * np.var(y, ddof=1)) / (n_x + n_y - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std < 1e-10:
        return 0.0

    return mean_diff / pooled_std


def paired_wilcoxon_test(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """Wilcoxon signed-rank test on paired differences.

    Tests H0: median(x - y) = 0.

    Args:
        x: Paired values for group A.
        y: Paired values for group B (same length as x).

    Returns:
        Tuple of (test_statistic, p_value).
    """
    if len(x) != len(y):
        raise ValueError(f"Paired test requires equal lengths: {len(x)} != {len(y)}")

    differences = x - y
    # Remove exact zeros (Wilcoxon cannot handle them)
    nonzero = differences[differences != 0]

    if len(nonzero) < 3:
        return 0.0, 1.0

    try:
        stat, p_val = sp_stats.wilcoxon(nonzero, alternative="two-sided")
        return float(stat), float(p_val)
    except Exception:
        return 0.0, 1.0


def compute_regime_effects(
    trials_df: pd.DataFrame,
    scheduler_a: str,
    scheduler_b: str,
    metric_cols: List[str],
    bucket_cols: List[str],
    n_bootstrap: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute effect sizes for scheduler_a vs scheduler_b per regime bucket.

    Uses matched episodes (same episode seeds) for paired comparisons.
    For each bucket:
    - Computes median difference, Cliff's delta, Cohen's d
    - Runs Wilcoxon signed-rank test
    - Computes bootstrap CI for the effect

    Args:
        trials_df: Trials DataFrame with scheduler_name column.
        scheduler_a: Treatment scheduler name.
        scheduler_b: Baseline scheduler name.
        metric_cols: Metrics to compare (e.g., ["cvar99_latency_us", "p99_latency_us"]).
        bucket_cols: Regime bucket columns (e.g., ["regime_id"]).
        n_bootstrap: Bootstrap resamples for effect CI.
        seed: Base seed.

    Returns:
        DataFrame with effect sizes, p-values, and CIs per bucket and metric.
    """
    df_a = trials_df[trials_df["scheduler_name"] == scheduler_a].copy()
    df_b = trials_df[trials_df["scheduler_name"] == scheduler_b].copy()

    rows = []
    bucket_idx = 0

    # Build bucket groups
    for bucket_key, bucket_a in df_a.groupby(bucket_cols, sort=True):
        if not isinstance(bucket_key, tuple):
            bucket_key = (bucket_key,)

        # Filter bucket_b to same bucket
        mask = pd.Series(True, index=df_b.index)
        for col, val in zip(bucket_cols, bucket_key):
            mask = mask & (df_b[col] == val)
        bucket_b = df_b[mask]

        # Match by trial ordering (paired by index position)
        n_episodes = min(len(bucket_a), len(bucket_b))
        if n_episodes < 3:
            continue

        bucket_id = "|".join(str(v) for v in bucket_key)

        for metric in metric_cols:
            vals_a = bucket_a[metric].values[:n_episodes]
            vals_b = bucket_b[metric].values[:n_episodes]

            # Effect metrics
            median_diff = float(np.median(vals_a - vals_b))
            mean_diff = float(np.mean(vals_a - vals_b))
            cd = cliffs_delta(vals_a, vals_b)
            cohen = cohens_d(vals_a, vals_b)

            # Wilcoxon test
            w_stat, p_value = paired_wilcoxon_test(vals_a, vals_b)

            # Bootstrap CI for the mean difference
            diffs = vals_a - vals_b
            rng = np.random.RandomState(seed + bucket_idx)
            ci_lo, ci_hi, _ = episode_bootstrap_ci(
                episode_values=diffs,
                stat_fn=np.mean,
                n_resamples=n_bootstrap,
                alpha=0.05,
                rng=rng,
            )

            rows.append({
                "bucket_id": bucket_id,
                **dict(zip(bucket_cols, bucket_key)),
                "metric_name": metric,
                "scheduler_a": scheduler_a,
                "scheduler_b": scheduler_b,
                "n_episodes": n_episodes,
                "mean_difference": mean_diff,
                "median_difference": median_diff,
                "cliffs_delta": cd,
                "cohens_d": cohen,
                "wilcoxon_stat": w_stat,
                "p_value": p_value,
                "effect_ci_lo": ci_lo,
                "effect_ci_hi": ci_hi,
            })
            bucket_idx += 1

    return pd.DataFrame(rows)


def compute_all_pairwise_effects(
    trials_df: pd.DataFrame,
    schedulers: List[str],
    baseline: str,
    metric_cols: List[str],
    bucket_cols: List[str],
    n_bootstrap: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute effects for all schedulers vs the baseline.

    Args:
        trials_df: Trials DataFrame.
        schedulers: All scheduler names.
        baseline: Baseline scheduler name.
        metric_cols: Metrics to compare.
        bucket_cols: Regime bucket columns.
        n_bootstrap: Bootstrap resamples.
        seed: Base seed.

    Returns:
        Combined effects DataFrame.
    """
    dfs = []
    for sched in schedulers:
        if sched == baseline:
            continue
        eff_df = compute_regime_effects(
            trials_df=trials_df,
            scheduler_a=sched,
            scheduler_b=baseline,
            metric_cols=metric_cols,
            bucket_cols=bucket_cols,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        if len(eff_df) > 0:
            dfs.append(eff_df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)
