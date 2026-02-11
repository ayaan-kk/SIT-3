"""CI calibration harness for validating bootstrap coverage.

Runs controlled synthetic repeats to check that nominal 95% CIs
achieve empirical coverage >= 0.90 (Gate R1).
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.measure.bootstrap import bootstrap_ci
from sit.measure.tail import estimate_cvar, estimate_var

logger = get_logger("stats.coverage")


# Stat functions for calibration
STAT_FUNCTIONS = {
    "mean": lambda x: float(np.mean(x)),
    "p99": lambda x: float(np.quantile(x, 0.99, method="linear")),
    "cvar99": lambda x: estimate_cvar(x, alpha=0.99),
}


def run_ci_calibration(
    n_repeats: int = 30,
    n_bootstrap_resamples: int = 200,
    reference_multiplier: int = 5,
    nominal_alpha: float = 0.05,
    metric_names: Optional[List[str]] = None,
    base_seed: int = 0,
    n_samples_base: int = 500,
) -> pd.DataFrame:
    """Run CI calibration experiment across synthetic worlds.

    For each repeat:
      1. Fix a world seed
      2. Generate a "small" sample and compute bootstrap CI
      3. Compute "reference truth" from a large sample (reference_multiplier x)
      4. Check whether truth falls in CI

    Args:
        n_repeats: Number of independent worlds/repeats.
        n_bootstrap_resamples: Bootstrap resamples per CI computation.
        reference_multiplier: Large-sample multiplier for reference truth.
        nominal_alpha: Significance level (0.05 for 95% CI).
        metric_names: Which metrics to calibrate. Defaults to all.
        base_seed: Base random seed.
        n_samples_base: Base number of samples per repeat.

    Returns:
        DataFrame with calibration results (one row per repeat x metric).
    """
    if metric_names is None:
        metric_names = list(STAT_FUNCTIONS.keys())

    # Lognormal parameters for synthetic data
    mu_ln = 5.0
    sigma_ln = 0.5

    rows = []

    for rep in range(n_repeats):
        world_seed = base_seed + rep

        # Generate "small" sample
        rng_small = np.random.RandomState(world_seed)
        samples_small = rng_small.lognormal(
            mean=mu_ln, sigma=sigma_ln, size=n_samples_base,
        )

        # Generate "reference" (large) sample for ground truth
        rng_ref = np.random.RandomState(world_seed + 1_000_000)
        samples_ref = rng_ref.lognormal(
            mean=mu_ln, sigma=sigma_ln,
            size=n_samples_base * reference_multiplier,
        )

        for metric_name in metric_names:
            stat_fn = STAT_FUNCTIONS[metric_name]
            truth_value = stat_fn(samples_ref)

            boot_rng = np.random.RandomState(world_seed + 2_000_000)
            ci_lo, ci_hi, _ = bootstrap_ci(
                stat_fn=stat_fn,
                samples=samples_small,
                n_resamples=n_bootstrap_resamples,
                alpha=nominal_alpha,
                rng=boot_rng,
            )

            covered = ci_lo <= truth_value <= ci_hi

            rows.append({
                "metric_name": metric_name,
                "scheduler_name": "synthetic",
                "regime_id": "calibration",
                "load_param": 0.5,
                "nominal_level": 1.0 - nominal_alpha,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "truth_value": truth_value,
                "covered": covered,
                "world_seed": world_seed,
                "n_samples": n_samples_base,
                "n_resamples": n_bootstrap_resamples,
            })

    df = pd.DataFrame(rows)

    # Log summary coverage by metric
    for metric_name in metric_names:
        sub = df[df["metric_name"] == metric_name]
        cov = sub["covered"].mean()
        logger.info(
            "CI calibration [%s]: coverage=%.3f (n=%d, nominal=%.2f)",
            metric_name, cov, len(sub), 1.0 - nominal_alpha,
        )

    return df


def compute_coverage_summary(calibration_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-metric coverage rates from calibration results.

    Args:
        calibration_df: Output of run_ci_calibration().

    Returns:
        Summary DataFrame with empirical_coverage per metric.
    """
    rows = []
    for metric_name, sub in calibration_df.groupby("metric_name"):
        n_total = len(sub)
        n_covered = int(sub["covered"].sum())
        rows.append({
            "metric_name": metric_name,
            "n_repeats": n_total,
            "n_covered": n_covered,
            "empirical_coverage": n_covered / n_total if n_total > 0 else 0.0,
            "nominal_level": sub["nominal_level"].iloc[0],
        })
    return pd.DataFrame(rows)


def gate_r1_ci_calibration(
    calibration_df: pd.DataFrame,
    min_coverage: float = 0.90,
    key_metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Gate R1: Verify CI calibration meets coverage threshold.

    Args:
        calibration_df: Output of run_ci_calibration().
        min_coverage: Minimum empirical coverage required.
        key_metrics: Metrics to check. Defaults to ["mean", "p99", "cvar99"].

    Returns:
        Dict with pass/fail status and details per metric.
    """
    if key_metrics is None:
        key_metrics = ["mean", "p99", "cvar99"]

    summary = compute_coverage_summary(calibration_df)
    results = {"gate": "R1_CI_CALIBRATION", "min_coverage": min_coverage}
    all_pass = True

    for metric in key_metrics:
        sub = summary[summary["metric_name"] == metric]
        if len(sub) == 0:
            results[metric] = {"status": "SKIP", "reason": "no data"}
            continue

        emp_cov = float(sub["empirical_coverage"].iloc[0])
        passed = emp_cov >= min_coverage
        if not passed:
            all_pass = False

        results[metric] = {
            "empirical_coverage": emp_cov,
            "n_repeats": int(sub["n_repeats"].iloc[0]),
            "status": "PASS" if passed else "FAIL",
        }

    results["overall"] = "PASS" if all_pass else "FAIL"
    logger.info("Gate R1 CI Calibration: %s", results["overall"])
    return results
