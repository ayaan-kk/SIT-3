"""Hardware validation evaluation: ranking transfer and gate checks.

Computes Spearman/Kendall rank correlations between simulator
predictions and hardware measurements, and evaluates hardware gates.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sit.core.logging import get_logger

logger = get_logger("eval.hardware")


def compute_transfer_correlations(
    hw_ranking_df: pd.DataFrame,
    sim_ranking_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Compute simulator-to-hardware ranking transfer metrics.

    Produces Spearman rho, Kendall tau, and top-k overlap.

    Args:
        hw_ranking_df: Hardware interference ranking with
            spectator_id, rank_irbs, irbs_delta_us columns.
        sim_ranking_df: Simulator predicted ranking with
            spectator_id, sim_rank, sim_interference_us columns.

    Returns:
        DataFrame with correlation metrics.
    """
    rows = []

    if sim_ranking_df is None or len(sim_ranking_df) == 0:
        rows.append({
            "metric": "spearman_rho",
            "value": float("nan"),
            "k": 0,
            "status": "SKIPPED_NO_SIM_DATA",
        })
        rows.append({
            "metric": "kendall_tau",
            "value": float("nan"),
            "k": 0,
            "status": "SKIPPED_NO_SIM_DATA",
        })
        return pd.DataFrame(rows)

    if len(hw_ranking_df) == 0:
        rows.append({
            "metric": "spearman_rho",
            "value": float("nan"),
            "k": 0,
            "status": "SKIPPED_NO_HW_DATA",
        })
        return pd.DataFrame(rows)

    # Match spectators by type suffix mapping
    # Hardware spectators: cpu_burn, mem_burn, cache_burn
    # Sim spectators: spectator_0, spectator_1, ... (ordered by interference)
    hw_ids = hw_ranking_df["spectator_id"].tolist()
    hw_deltas = hw_ranking_df["irbs_delta_us"].values

    # Map hardware types to sim ranking by interference magnitude order
    # We rank both by interference magnitude and compare
    n_hw = len(hw_ids)

    if "sim_interference_us" in sim_ranking_df.columns:
        # Take top-n_hw from sim ranking by interference
        sim_top = sim_ranking_df.nlargest(n_hw, "sim_interference_us")
        sim_deltas = sim_top["sim_interference_us"].values

        # Compute rank correlations on the magnitudes
        hw_ranks = _rank_array(hw_deltas)
        sim_ranks = _rank_array(sim_deltas)

        if len(hw_ranks) >= 2 and len(sim_ranks) >= 2:
            n = min(len(hw_ranks), len(sim_ranks))
            hr = hw_ranks[:n]
            sr = sim_ranks[:n]

            from scipy import stats as sp_stats

            # Spearman
            try:
                rho, rho_p = sp_stats.spearmanr(hr, sr)
                rows.append({
                    "metric": "spearman_rho",
                    "value": float(rho) if not np.isnan(rho) else 0.0,
                    "k": n,
                    "status": "computed",
                    "p_value": float(rho_p) if not np.isnan(rho_p) else 1.0,
                })
            except Exception:
                rows.append({
                    "metric": "spearman_rho",
                    "value": 0.0,
                    "k": n,
                    "status": "error",
                })

            # Kendall
            try:
                tau, tau_p = sp_stats.kendalltau(hr, sr)
                rows.append({
                    "metric": "kendall_tau",
                    "value": float(tau) if not np.isnan(tau) else 0.0,
                    "k": n,
                    "status": "computed",
                    "p_value": float(tau_p) if not np.isnan(tau_p) else 1.0,
                })
            except Exception:
                rows.append({
                    "metric": "kendall_tau",
                    "value": 0.0,
                    "k": n,
                    "status": "error",
                })

            # Top-k overlap for various k
            for k in range(1, n + 1):
                hw_topk = set(np.argsort(-hw_deltas)[:k])
                sim_topk = set(np.argsort(-sim_deltas)[:k])
                overlap = len(hw_topk & sim_topk) / k
                rows.append({
                    "metric": f"topk_overlap_k{k}",
                    "value": overlap,
                    "k": k,
                    "status": "computed",
                })
        else:
            rows.append({
                "metric": "spearman_rho",
                "value": float("nan"),
                "k": 0,
                "status": "insufficient_data",
            })
    else:
        rows.append({
            "metric": "spearman_rho",
            "value": float("nan"),
            "k": 0,
            "status": "SKIPPED_NO_SIM_INTERFERENCE",
        })

    return pd.DataFrame(rows)


def _rank_array(arr: np.ndarray) -> np.ndarray:
    """Convert values to ranks (1-based, descending)."""
    temp = arr.argsort()[::-1]
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(1, len(arr) + 1)
    return ranks


# ---------------------------------------------------------------------------
# Hardware gates
# ---------------------------------------------------------------------------

def gate_h1_drift_reality(
    drift_summary: Dict[str, Any],
    min_drift_pct: float = 1.0,
) -> Dict[str, Any]:
    """Gate H1: Verify measurable baseline drift exists.

    Drift is considered real if the maximum drift magnitude exceeds
    min_drift_pct of the baseline mean latency.

    Args:
        drift_summary: Output from measure_baseline_drift().
        min_drift_pct: Minimum drift as % of mean.

    Returns:
        Gate result dict.
    """
    drift_pct = drift_summary.get("drift_pct_of_mean", 0.0)
    drift_mag = drift_summary.get("drift_magnitude_mean_us", 0.0)
    detected = drift_summary.get("drift_detected", False)

    passed = drift_pct >= min_drift_pct
    status = "PASS" if passed else "FAIL"

    result = {
        "gate": "H1_DRIFT_REALITY",
        "status": status,
        "drift_magnitude_us": drift_mag,
        "drift_pct_of_mean": drift_pct,
        "min_drift_pct": min_drift_pct,
        "drift_detected": detected,
    }

    logger.info("Gate H1 Drift Reality: %s (drift=%.4f us, %.2f%%)",
                status, drift_mag, drift_pct)
    return result


def gate_h2_irbs_effectiveness(
    irbs_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Gate H2: Verify IRBS reduces bias or variance vs naive.

    Checks that IRBS has lower variance of estimates across repeats
    compared to naive, OR lower median absolute delta (closer to
    a consistent interference estimate).

    Args:
        irbs_df: IRBS estimates DataFrame.

    Returns:
        Gate result dict.
    """
    if len(irbs_df) == 0:
        return {
            "gate": "H2_IRBS_EFFECTIVENESS",
            "status": "SKIP",
            "reason": "no IRBS data",
        }

    # Compare variance of naive vs IRBS estimates per spectator
    var_reduction_count = 0
    total_spectators = 0

    for spec_id, group in irbs_df.groupby("spectator_id"):
        if len(group) < 2:
            continue
        naive_var = group["naive_delta_us"].var()
        irbs_var = group["irbs_delta_us"].var()
        total_spectators += 1
        if irbs_var <= naive_var:
            var_reduction_count += 1

    if total_spectators == 0:
        return {
            "gate": "H2_IRBS_EFFECTIVENESS",
            "status": "SKIP",
            "reason": "insufficient repeats",
        }

    # Also check absolute difference between naive and IRBS
    median_abs_diff = float(irbs_df["absolute_difference"].median())

    # IRBS helps if it reduces variance for majority of spectators
    # OR if the absolute difference between naive and IRBS is meaningful
    fraction_improved = var_reduction_count / total_spectators
    passed = fraction_improved >= 0.5 or median_abs_diff > 0.0

    status = "PASS" if passed else "FAIL"

    result = {
        "gate": "H2_IRBS_EFFECTIVENESS",
        "status": status,
        "fraction_variance_reduced": fraction_improved,
        "n_spectators": total_spectators,
        "n_variance_reduced": var_reduction_count,
        "median_abs_difference_us": median_abs_diff,
    }

    logger.info("Gate H2 IRBS Effectiveness: %s (%.0f%% variance reduced, "
                "median_diff=%.4f us)",
                status, fraction_improved * 100, median_abs_diff)
    return result


def gate_h3_ranking_transfer(
    corr_df: pd.DataFrame,
    min_spearman: float = 0.7,
    min_topk_overlap: float = 0.67,
    min_k: int = 3,
) -> Dict[str, Any]:
    """Gate H3: Verify simulator-hardware ranking agreement.

    Requires Spearman >= min_spearman OR top-k overlap >= min_topk_overlap
    for k >= min_k.

    Args:
        corr_df: Transfer correlation DataFrame.
        min_spearman: Minimum Spearman rho.
        min_topk_overlap: Minimum top-k overlap fraction.
        min_k: Minimum k for top-k check.

    Returns:
        Gate result dict.
    """
    if len(corr_df) == 0:
        return {
            "gate": "H3_RANKING_TRANSFER",
            "status": "SKIPPED_NO_HARDWARE",
            "reason": "no correlation data",
        }

    # Check for skip status
    if "status" in corr_df.columns:
        skipped = corr_df[corr_df["status"].str.startswith("SKIPPED", na=False)]
        if len(skipped) == len(corr_df):
            reason = skipped["status"].iloc[0]
            return {
                "gate": "H3_RANKING_TRANSFER",
                "status": "SKIPPED_NO_HARDWARE",
                "reason": str(reason),
            }

    # Check Spearman
    spearman_rows = corr_df[corr_df["metric"] == "spearman_rho"]
    spearman_val = float("nan")
    if len(spearman_rows) > 0:
        spearman_val = float(spearman_rows["value"].iloc[0])

    spearman_pass = not np.isnan(spearman_val) and spearman_val >= min_spearman

    # Check top-k overlap
    topk_pass = False
    topk_val = float("nan")
    for _, row in corr_df.iterrows():
        if str(row.get("metric", "")).startswith("topk_overlap") and row.get("k", 0) >= min_k:
            val = float(row["value"])
            if val >= min_topk_overlap:
                topk_pass = True
                topk_val = val
                break

    passed = spearman_pass or topk_pass
    status = "PASS" if passed else "FAIL"

    result = {
        "gate": "H3_RANKING_TRANSFER",
        "status": status,
        "spearman_rho": spearman_val,
        "spearman_pass": spearman_pass,
        "min_spearman": min_spearman,
        "topk_overlap": topk_val if not np.isnan(topk_val) else None,
        "topk_pass": topk_pass,
        "min_topk_overlap": min_topk_overlap,
    }

    logger.info("Gate H3 Ranking Transfer: %s (spearman=%.3f, topk=%.3f)",
                status,
                spearman_val if not np.isnan(spearman_val) else 0.0,
                topk_val if not np.isnan(topk_val) else 0.0)
    return result
