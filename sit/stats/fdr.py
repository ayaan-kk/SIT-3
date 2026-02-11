"""Benjamini-Hochberg FDR control for multiple comparisons.

Provides FDR-adjusted q-values and significance flags at configurable
thresholds. Ensures no p-hacking in regime-wise claims.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger

logger = get_logger("stats.fdr")


def benjamini_hochberg(
    p_values: np.ndarray,
    q_threshold: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the Benjamini-Hochberg procedure for FDR control.

    Given m p-values, computes adjusted q-values and significance flags.

    The BH procedure:
    1. Sort p-values in ascending order
    2. For rank i (1-indexed), compute adjusted p_i = p_i * m / i
    3. Enforce monotonicity: q_i = min(q_i, q_{i+1}) from bottom up
    4. Flag significant if q_i <= q_threshold

    Args:
        p_values: Array of raw p-values.
        q_threshold: FDR threshold (e.g., 0.10).

    Returns:
        Tuple of (q_values, significant) arrays, same length as p_values.
    """
    m = len(p_values)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)

    p_arr = np.asarray(p_values, dtype=np.float64)

    # Sort indices
    sorted_idx = np.argsort(p_arr)
    sorted_p = p_arr[sorted_idx]

    # Compute BH adjusted p-values
    ranks = np.arange(1, m + 1, dtype=np.float64)
    adjusted = sorted_p * m / ranks

    # Enforce monotonicity (cumulative minimum from the right)
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Clip to [0, 1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Map back to original order
    q_values = np.empty(m, dtype=np.float64)
    q_values[sorted_idx] = adjusted

    significant = q_values <= q_threshold

    n_sig = int(np.sum(significant))
    logger.info(
        "BH FDR control: %d/%d significant at q<=%.2f",
        n_sig, m, q_threshold,
    )

    return q_values, significant


def apply_fdr_to_effects(
    effects_df: pd.DataFrame,
    p_col: str = "p_value",
    q_threshold: float = 0.10,
) -> pd.DataFrame:
    """Apply BH FDR correction to an effects DataFrame.

    Adds 'q_value' and 'significant' columns.

    Args:
        effects_df: DataFrame with a p-value column.
        p_col: Name of the p-value column.
        q_threshold: FDR threshold.

    Returns:
        DataFrame with q_value and significant columns added.
    """
    if len(effects_df) == 0:
        result = effects_df.copy()
        result["q_value"] = pd.Series(dtype=float)
        result["significant"] = pd.Series(dtype=bool)
        return result

    p_values = effects_df[p_col].values
    q_values, significant = benjamini_hochberg(p_values, q_threshold)

    result = effects_df.copy()
    result["q_value"] = q_values
    result["significant"] = significant
    return result


def export_fdr_results(
    effects_df: pd.DataFrame,
    bucket_col: str = "bucket_id",
) -> pd.DataFrame:
    """Export FDR results in the canonical format.

    Args:
        effects_df: Effects DataFrame with q_value and significant columns.
        bucket_col: Bucket identifier column.

    Returns:
        fdr_results DataFrame with bucket_id, raw_p, q_value, significant.
    """
    if len(effects_df) == 0:
        return pd.DataFrame(columns=["bucket_id", "metric_name", "scheduler_a",
                                      "scheduler_b", "raw_p", "q_value", "significant"])

    cols = [bucket_col, "metric_name", "scheduler_a", "scheduler_b",
            "p_value", "q_value", "significant"]
    available = [c for c in cols if c in effects_df.columns]
    result = effects_df[available].copy()

    if "p_value" in result.columns:
        result = result.rename(columns={"p_value": "raw_p"})

    return result


def gate_r3_no_phacking(
    fdr_df: pd.DataFrame,
    q_threshold: float = 0.10,
) -> dict:
    """Gate R3: Verify that all significant claims are FDR-controlled.

    Any row marked significant must have q_value <= q_threshold.

    Args:
        fdr_df: FDR results DataFrame.
        q_threshold: Maximum q-value for significance.

    Returns:
        Dict with gate status.
    """
    results = {"gate": "R3_NO_PHACKING", "q_threshold": q_threshold}

    if len(fdr_df) == 0:
        results["overall"] = "PASS"
        results["reason"] = "no claims to check"
        return results

    sig_col = "significant"
    q_col = "q_value"

    if sig_col not in fdr_df.columns or q_col not in fdr_df.columns:
        results["overall"] = "FAIL"
        results["reason"] = "missing q_value or significant columns"
        return results

    significant = fdr_df[fdr_df[sig_col] == True]  # noqa: E712
    if len(significant) == 0:
        results["overall"] = "PASS"
        results["n_significant"] = 0
        results["n_total"] = len(fdr_df)
        return results

    # All significant rows must have q_value <= threshold
    violations = significant[significant[q_col] > q_threshold]
    n_violations = len(violations)

    results["n_significant"] = len(significant)
    results["n_total"] = len(fdr_df)
    results["n_violations"] = n_violations
    results["overall"] = "PASS" if n_violations == 0 else "FAIL"

    logger.info("Gate R3 No P-hacking: %s (%d significant, %d violations)",
                results["overall"], len(significant), n_violations)
    return results
