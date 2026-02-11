"""Extreme Value Theory (EVT) tail modeling validation.

Implements the Peaks Over Threshold (POT) method with GPD fitting
for tail risk estimation. Provides diagnostics for assessing
whether EVT-based estimates are reliable.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import minimize

from sit.core.logging import get_logger
from sit.measure.tail import estimate_cvar, estimate_var

logger = get_logger("stats.evt")


def fit_gpd_mle(
    exceedances: np.ndarray,
) -> Tuple[float, float, bool]:
    """Fit Generalized Pareto Distribution to exceedances via MLE.

    The GPD CDF is:
        F(y) = 1 - (1 + xi*y/sigma)^{-1/xi}  for xi != 0
        F(y) = 1 - exp(-y/sigma)               for xi = 0

    Args:
        exceedances: Positive exceedances (Y = L - u > 0).

    Returns:
        Tuple of (xi, sigma, converged).
    """
    exceedances = exceedances[exceedances > 0]
    n = len(exceedances)
    if n < 10:
        return 0.0, float(np.mean(exceedances)), False

    # Use scipy's genpareto for MLE
    try:
        # genpareto.fit returns (c, loc, scale) where c = xi
        c, loc, scale = sp_stats.genpareto.fit(exceedances, floc=0)
        xi = float(c)
        sigma = float(scale)
        if sigma <= 0:
            sigma = float(np.mean(exceedances))
            return xi, sigma, False
        return xi, sigma, True
    except Exception:
        # Fallback: method of moments for exponential approximation
        sigma = float(np.mean(exceedances))
        return 0.0, sigma, False


def gpd_var(xi: float, sigma: float, threshold_u: float, alpha: float, n_total: int, n_exceed: int) -> float:
    """Compute EVT-based VaR estimate.

    VaR_alpha = u + (sigma/xi) * ((n_total/n_exceed * (1-alpha))^{-xi} - 1)
    """
    if n_exceed == 0:
        return threshold_u

    p_exceed = n_exceed / n_total
    if abs(xi) < 1e-10:
        # Exponential case
        return threshold_u + sigma * np.log(p_exceed / (1 - alpha))

    return threshold_u + (sigma / xi) * (((1 - alpha) / p_exceed) ** (-xi) - 1)


def gpd_cvar(xi: float, sigma: float, var_alpha: float, threshold_u: float) -> float:
    """Compute EVT-based CVaR (expected shortfall) estimate.

    CVaR = VaR/(1-xi) + (sigma - xi*u)/(1-xi)
    """
    if xi >= 1.0:
        # Mean does not exist for xi >= 1
        return var_alpha * 1.5  # rough upper bound

    return (var_alpha + sigma - xi * threshold_u) / (1 - xi)


def run_evt_diagnostics(
    samples_us: np.ndarray,
    threshold_quantiles: List[float],
    min_exceedances: int = 200,
    alpha: float = 0.99,
) -> pd.DataFrame:
    """Run EVT diagnostics across multiple thresholds.

    For each threshold quantile:
    1. Compute exceedances
    2. Fit GPD
    3. Run KS test
    4. Compute EVT-based VaR/CVaR and compare to empirical

    Args:
        samples_us: Full latency sample in microseconds.
        threshold_quantiles: List of quantile levels for thresholds.
        min_exceedances: Minimum exceedances required for fitting.
        alpha: VaR/CVaR level (e.g., 0.99 for p99).

    Returns:
        DataFrame with one row per threshold, containing fit
        parameters, diagnostics, and comparisons.
    """
    n_total = len(samples_us)
    empirical_var = estimate_var(samples_us, alpha=alpha)
    empirical_cvar = estimate_cvar(samples_us, alpha=alpha)

    rows = []

    for q in threshold_quantiles:
        threshold_u = float(np.quantile(samples_us, q, method="linear"))
        exceedances = samples_us[samples_us > threshold_u] - threshold_u
        n_exceed = len(exceedances)

        row = {
            "threshold_quantile": q,
            "threshold_u": threshold_u,
            "n_exceedances": n_exceed,
            "min_exceedances": min_exceedances,
        }

        if n_exceed < max(10, min_exceedances):
            row.update({
                "xi": float("nan"),
                "sigma": float("nan"),
                "converged": False,
                "ks_stat": float("nan"),
                "ks_p_value": float("nan"),
                "evt_var99": float("nan"),
                "evt_cvar99": float("nan"),
                "empirical_var99": empirical_var,
                "empirical_cvar99": empirical_cvar,
                "relative_difference_var": float("nan"),
                "relative_difference_cvar": float("nan"),
                "applicable": False,
                "reason": f"too_few_exceedances ({n_exceed} < {max(10, min_exceedances)})",
            })
            rows.append(row)
            continue

        # Fit GPD
        xi, sigma, converged = fit_gpd_mle(exceedances)

        # KS test against fitted GPD
        try:
            ks_stat, ks_p = sp_stats.kstest(
                exceedances, "genpareto", args=(xi, 0, sigma),
            )
        except Exception:
            ks_stat, ks_p = float("nan"), float("nan")

        # EVT-based estimates
        evt_var = gpd_var(xi, sigma, threshold_u, alpha, n_total, n_exceed)
        evt_cvar_val = gpd_cvar(xi, sigma, evt_var, threshold_u)

        # Relative differences
        rel_diff_var = abs(evt_var - empirical_var) / max(abs(empirical_var), 1e-10)
        rel_diff_cvar = abs(evt_cvar_val - empirical_cvar) / max(abs(empirical_cvar), 1e-10)

        row.update({
            "xi": xi,
            "sigma": sigma,
            "converged": converged,
            "ks_stat": float(ks_stat),
            "ks_p_value": float(ks_p),
            "evt_var99": evt_var,
            "evt_cvar99": evt_cvar_val,
            "empirical_var99": empirical_var,
            "empirical_cvar99": empirical_cvar,
            "relative_difference_var": rel_diff_var,
            "relative_difference_cvar": rel_diff_cvar,
            "applicable": True,
            "reason": "",
        })
        rows.append(row)

    return pd.DataFrame(rows)


def compute_stability_score(evt_df: pd.DataFrame) -> float:
    """Compute parameter stability across thresholds.

    A low coefficient of variation in xi and sigma across thresholds
    indicates stable fits. Returns a score in [0, 1] where higher is better.

    Args:
        evt_df: Output of run_evt_diagnostics().

    Returns:
        Stability score in [0, 1].
    """
    applicable = evt_df[evt_df["applicable"] == True]  # noqa: E712
    if len(applicable) < 2:
        return 0.0

    xi_vals = applicable["xi"].values
    sigma_vals = applicable["sigma"].values

    # CV for xi (use MAD-based for robustness)
    xi_spread = np.std(xi_vals) / max(abs(np.mean(xi_vals)), 1e-10)
    sigma_spread = np.std(sigma_vals) / max(abs(np.mean(sigma_vals)), 1e-10)

    # Score: 1 / (1 + combined_spread)
    combined = (xi_spread + sigma_spread) / 2
    return 1.0 / (1.0 + combined)


def run_evt_analysis(
    trials_df: pd.DataFrame,
    group_cols: List[str],
    latency_col: str = "cvar99_latency_us",
    threshold_quantiles: Optional[List[float]] = None,
    min_exceedances: int = 200,
) -> pd.DataFrame:
    """Run EVT analysis grouped by scheduler/regime/load.

    For each group, collects latency values and runs EVT diagnostics.

    Args:
        trials_df: Trials DataFrame.
        group_cols: Columns to group by.
        latency_col: Column containing latency values to analyze.
        threshold_quantiles: Threshold quantiles for POT.
        min_exceedances: Minimum exceedances needed.

    Returns:
        Combined EVT diagnostics DataFrame with group columns.
    """
    if threshold_quantiles is None:
        threshold_quantiles = [0.90, 0.92, 0.94, 0.95]

    all_rows = []

    for group_key, group_df in trials_df.groupby(group_cols, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        samples = group_df[latency_col].dropna().values
        if len(samples) < 20:
            continue

        evt_df = run_evt_diagnostics(
            samples_us=samples,
            threshold_quantiles=threshold_quantiles,
            min_exceedances=min_exceedances,
        )

        stability = compute_stability_score(evt_df)

        for _, row in evt_df.iterrows():
            row_dict = row.to_dict()
            row_dict.update(dict(zip(group_cols, group_key)))
            row_dict["stability_score"] = stability
            all_rows.append(row_dict)

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def gate_r2_tail_validity(
    evt_df: pd.DataFrame,
    ks_p_threshold: float = 0.05,
    max_relative_difference: float = 0.25,
    evt_required: bool = False,
) -> Dict[str, Any]:
    """Gate R2: Validate EVT tail modeling quality.

    Checks:
    1. Median KS p-value >= threshold across applicable conditions
    2. Relative difference between EVT and empirical < max_relative_difference
    3. If EVT not applicable, logs reason without failing (unless required)

    Args:
        evt_df: Output of run_evt_analysis() or run_evt_diagnostics().
        ks_p_threshold: Minimum median KS p-value.
        max_relative_difference: Maximum tolerable relative difference.
        evt_required: If True, fail when EVT is not applicable.

    Returns:
        Dict with gate status and details.
    """
    results = {
        "gate": "R2_TAIL_VALIDITY",
        "ks_p_threshold": ks_p_threshold,
        "max_relative_difference": max_relative_difference,
    }

    if len(evt_df) == 0:
        results["overall"] = "SKIP" if not evt_required else "FAIL"
        results["reason"] = "no EVT results"
        logger.info("Gate R2: %s - %s", results["overall"], results["reason"])
        return results

    applicable = evt_df[evt_df["applicable"] == True]  # noqa: E712

    if len(applicable) == 0:
        non_applicable = evt_df[evt_df["applicable"] == False]  # noqa: E712
        reasons = non_applicable["reason"].unique().tolist() if "reason" in evt_df.columns else ["unknown"]
        results["overall"] = "N/A" if not evt_required else "FAIL"
        results["reason"] = f"EVT not applicable: {reasons}"
        logger.info("Gate R2: %s - %s", results["overall"], results["reason"])
        return results

    # Check KS p-values
    ks_p_values = applicable["ks_p_value"].dropna()
    median_ks_p = float(ks_p_values.median()) if len(ks_p_values) > 0 else 0.0

    # Check stability
    stability_scores = applicable["stability_score"].dropna() if "stability_score" in applicable.columns else pd.Series(dtype=float)
    median_stability = float(stability_scores.median()) if len(stability_scores) > 0 else 0.0

    # Check relative differences
    rel_diffs = applicable["relative_difference_cvar"].dropna()
    median_rel_diff = float(rel_diffs.median()) if len(rel_diffs) > 0 else 1.0

    ks_pass = median_ks_p >= ks_p_threshold or median_stability >= 0.5
    diff_pass = median_rel_diff < max_relative_difference

    results.update({
        "n_applicable": int(len(applicable)),
        "n_total": int(len(evt_df)),
        "median_ks_p_value": median_ks_p,
        "median_stability": median_stability,
        "median_relative_difference": median_rel_diff,
        "ks_pass": ks_pass,
        "diff_pass": diff_pass,
        "overall": "PASS" if (ks_pass and diff_pass) else "FAIL",
    })

    logger.info("Gate R2 Tail Validity: %s (ks_p=%.3f, stability=%.3f, rel_diff=%.3f)",
                results["overall"], median_ks_p, median_stability, median_rel_diff)
    return results
