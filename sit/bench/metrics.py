"""Comprehensive metrics computation and statistical analysis.

Computes all required metrics, bootstrap CIs, effect sizes, and
produces statistics summary files.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger

logger = get_logger("bench.metrics")


def compute_aggregate_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate metrics per policy/regime combination.

    Returns a DataFrame with summary statistics for each group.
    """
    group_cols = ["policy", "interference_regime", "load_regime"]
    metric_cols = [
        "mean_latency_us", "p95_latency_us", "p99_latency_us", "p999_latency_us",
        "cvar95_us", "cvar99_us", "cvar999_us",
        "violation_rate", "goodput", "throughput_inv_us",
        "queue_variance", "mean_backlog_us", "max_backlog_us",
        "decision_time_us",
    ]

    agg_funcs = {col: ["mean", "median", "std"] for col in metric_cols if col in results_df.columns}
    agg_funcs["catastrophe"] = ["sum", "mean"]
    agg_funcs["episode_id"] = "count"

    agg_df = results_df.groupby(group_cols).agg(agg_funcs)
    agg_df.columns = ["_".join(col).strip() for col in agg_df.columns.values]
    agg_df = agg_df.reset_index()
    agg_df = agg_df.rename(columns={"episode_id_count": "n_episodes"})

    return agg_df


def compute_pairwise_comparisons(
    results_df: pd.DataFrame,
    baseline: str = "partition",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute pairwise comparisons of each policy vs baseline.

    For each policy/regime, computes effect size, 95% CI, and p-value
    via paired bootstrap.
    """
    rng = np.random.RandomState(seed)
    group_cols = ["interference_regime", "load_regime"]
    rows = []

    for metric in ["cvar99_us", "goodput", "violation_rate", "p99_latency_us"]:
        if metric not in results_df.columns:
            continue

        for group_key, group_df in results_df.groupby(group_cols):
            ir, lr = group_key

            baseline_vals = group_df[group_df["policy"] == baseline][metric].values
            if len(baseline_vals) == 0:
                continue

            for policy in results_df["policy"].unique():
                if policy == baseline:
                    continue

                policy_vals = group_df[group_df["policy"] == policy][metric].values
                if len(policy_vals) == 0:
                    continue

                # Bootstrap paired difference
                n = min(len(baseline_vals), len(policy_vals))
                diffs = policy_vals[:n] - baseline_vals[:n]
                mean_diff = float(np.mean(diffs))

                boot_means = []
                for _ in range(n_bootstrap):
                    idx = rng.choice(n, size=n, replace=True)
                    boot_means.append(float(np.mean(diffs[idx])))

                boot_means = np.array(boot_means)
                ci_lo = float(np.percentile(boot_means, 2.5))
                ci_hi = float(np.percentile(boot_means, 97.5))

                # p-value: fraction of bootstrap means crossing zero
                if mean_diff < 0:
                    p_value = float(np.mean(boot_means >= 0))
                else:
                    p_value = float(np.mean(boot_means <= 0))
                p_value = max(p_value, 1.0 / n_bootstrap)

                # Cohen's d
                pooled_std = float(np.std(diffs))
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

                rows.append({
                    "metric": metric,
                    "policy": policy,
                    "baseline": baseline,
                    "interference_regime": ir,
                    "load_regime": lr,
                    "mean_diff": round(mean_diff, 4),
                    "ci_lo": round(ci_lo, 4),
                    "ci_hi": round(ci_hi, 4),
                    "p_value": round(p_value, 6),
                    "cohens_d": round(cohens_d, 4),
                    "n": n,
                })

    comp_df = pd.DataFrame(rows)

    # FDR correction (Benjamini-Hochberg)
    if len(comp_df) > 0 and "p_value" in comp_df.columns:
        comp_df = _apply_fdr(comp_df)

    return comp_df


def _apply_fdr(df: pd.DataFrame, q: float = 0.05) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction."""
    p_vals = df["p_value"].values.copy()
    n = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    q_values = np.ones(n)

    for rank_i, idx in enumerate(sorted_idx):
        bh_threshold = (rank_i + 1) / n * q
        q_values[idx] = min(p_vals[idx] * n / (rank_i + 1), 1.0)

    df = df.copy()
    df["q_value"] = np.round(q_values, 6)
    df["significant"] = q_values <= q
    return df


def compute_fairness_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Generate the fairness table documenting information budgets."""
    policies_info = {
        "SIT-safe": {"probes": True, "counters": False, "uncertainty": True, "ground_truth": False, "info_level": "estimated"},
        "SIT-no-IRBS": {"probes": True, "counters": False, "uncertainty": True, "ground_truth": False, "info_level": "biased"},
        "SIT-no-safety": {"probes": True, "counters": False, "uncertainty": False, "ground_truth": False, "info_level": "estimated"},
        "SIT-no-admission": {"probes": True, "counters": False, "uncertainty": True, "ground_truth": False, "info_level": "estimated"},
        "partition": {"probes": False, "counters": False, "uncertainty": False, "ground_truth": False, "info_level": "none"},
        "spread": {"probes": False, "counters": False, "uncertainty": False, "ground_truth": False, "info_level": "none"},
        "random": {"probes": False, "counters": False, "uncertainty": False, "ground_truth": False, "info_level": "none"},
        "round-robin": {"probes": False, "counters": False, "uncertainty": False, "ground_truth": False, "info_level": "none"},
        "binpack-greedy": {"probes": False, "counters": False, "uncertainty": False, "ground_truth": False, "info_level": "resource_proxy"},
        "tail-greedy": {"probes": False, "counters": False, "uncertainty": False, "ground_truth": True, "info_level": "advantaged"},
        "similarity-avoidance": {"probes": False, "counters": False, "uncertainty": False, "ground_truth": False, "info_level": "features"},
        "admission-only": {"probes": False, "counters": False, "uncertainty": False, "ground_truth": False, "info_level": "partial"},
        "oracle": {"probes": False, "counters": True, "uncertainty": False, "ground_truth": True, "info_level": "full (upper bound)"},
        "static-low": {"probes": False, "counters": False, "uncertainty": False, "ground_truth": False, "info_level": "none"},
        "static-high": {"probes": False, "counters": False, "uncertainty": False, "ground_truth": False, "info_level": "none"},
        "k8s-hpa": {"probes": False, "counters": False, "uncertainty": False, "ground_truth": False, "info_level": "utilization"},
        "k8s-default": {"probes": False, "counters": False, "uncertainty": False, "ground_truth": False, "info_level": "resource_fit"},
        "slurm-fcfs": {"probes": False, "counters": False, "uncertainty": False, "ground_truth": False, "info_level": "none"},
        "triton-proxy": {"probes": False, "counters": False, "uncertainty": False, "ground_truth": False, "info_level": "batching"},
    }

    rows = []
    for policy, info in policies_info.items():
        rows.append({
            "policy": policy,
            "uses_probes": info["probes"],
            "uses_hw_counters": info["counters"],
            "uses_uncertainty": info["uncertainty"],
            "uses_ground_truth": info["ground_truth"],
            "information_level": info["info_level"],
            "advantaged": info["ground_truth"],
        })

    return pd.DataFrame(rows)


def compute_dominance_metrics(
    results_df: pd.DataFrame,
    sit_policy: str = "SIT-safe",
) -> pd.DataFrame:
    """Compute dominance metrics for SIT vs all baselines."""
    rows = []
    group_cols = ["interference_regime", "load_regime"]

    for group_key, group_df in results_df.groupby(group_cols):
        ir, lr = group_key
        sit_data = group_df[group_df["policy"] == sit_policy]
        if len(sit_data) == 0:
            continue

        sit_cvar = float(sit_data["cvar99_us"].mean())
        sit_goodput = float(sit_data["goodput"].mean())
        sit_viol = float(sit_data["violation_rate"].mean())
        sit_cats = int(sit_data["catastrophe"].sum())

        for policy in results_df["policy"].unique():
            if policy == sit_policy:
                continue
            pol_data = group_df[group_df["policy"] == policy]
            if len(pol_data) == 0:
                continue

            pol_cvar = float(pol_data["cvar99_us"].mean())
            pol_goodput = float(pol_data["goodput"].mean())
            pol_viol = float(pol_data["violation_rate"].mean())
            pol_cats = int(pol_data["catastrophe"].sum())

            # Dominance checks
            cvar_improvement = (pol_cvar - sit_cvar) / pol_cvar if pol_cvar > 0 else 0
            goodput_improvement = sit_goodput - pol_goodput
            fewer_cats = sit_cats < pol_cats

            rows.append({
                "interference_regime": ir,
                "load_regime": lr,
                "compared_to": policy,
                "sit_cvar99": round(sit_cvar, 2),
                "baseline_cvar99": round(pol_cvar, 2),
                "cvar_improvement_pct": round(cvar_improvement * 100, 2),
                "sit_goodput": round(sit_goodput, 4),
                "baseline_goodput": round(pol_goodput, 4),
                "goodput_improvement": round(goodput_improvement, 4),
                "sit_catastrophes": sit_cats,
                "baseline_catastrophes": pol_cats,
                "fewer_catastrophes": fewer_cats,
                "sit_dominates": cvar_improvement > 0 and goodput_improvement >= 0,
            })

    return pd.DataFrame(rows)


def save_all_stats(
    results_df: pd.DataFrame,
    output_dir: str = "results/bench",
) -> Dict[str, str]:
    """Save all statistical outputs."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    # Aggregate metrics
    agg_df = compute_aggregate_metrics(results_df)
    p = os.path.join(output_dir, "stats_summary.csv")
    agg_df.to_csv(p, index=False)
    paths["stats_summary"] = p

    # Pairwise comparisons
    comp_df = compute_pairwise_comparisons(results_df)
    p = os.path.join(output_dir, "stats_bootstrap_results.csv")
    comp_df.to_csv(p, index=False)
    paths["stats_bootstrap"] = p

    # Fairness table
    fair_df = compute_fairness_table(results_df)
    p = os.path.join(output_dir, "fairness_table.csv")
    fair_df.to_csv(p, index=False)
    paths["fairness_table"] = p

    # Dominance metrics
    dom_df = compute_dominance_metrics(results_df)
    p = os.path.join(output_dir, "dominance_metrics.csv")
    dom_df.to_csv(p, index=False)
    paths["dominance_metrics"] = p

    logger.info("Saved %d stats files to %s", len(paths), output_dir)
    return paths
