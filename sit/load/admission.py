"""Admission control curves: max feasible load per scheduler.

For each scheduler, finds the maximum offered load (lambda_rps)
at which the violation rate remains below v_target.
"""

from typing import Dict

import numpy as np
import pandas as pd

from sit.core.logging import get_logger

logger = get_logger("load.admission")


def compute_admission_curve(
    sweep_df: pd.DataFrame,
    slo_us: float,
    v_target: float = 0.02,
) -> pd.DataFrame:
    """Compute SLO-satisfying admission curve per scheduler.

    For each scheduler, finds the maximum load level where the mean
    violation_rate <= v_target. Also reports the goodput at that load.

    Args:
        sweep_df: Load sweep results DataFrame.
        slo_us: SLO threshold (for reference).
        v_target: Target violation rate threshold.

    Returns:
        DataFrame with per-scheduler admission data.
    """
    if sweep_df.empty:
        return pd.DataFrame()

    schedulers = sweep_df["scheduler_name"].unique()
    load_levels = sorted(sweep_df["load_level"].unique())

    rows = []
    for sched in schedulers:
        sched_df = sweep_df[sweep_df["scheduler_name"] == sched]

        max_feasible_load = 0.0
        goodput_at_max = 0.0
        cvar_at_max = 0.0
        throughput_at_max = 0.0

        per_load_data = []

        for load_level in load_levels:
            level_df = sched_df[sched_df["load_level"] == load_level]
            if level_df.empty:
                continue

            mean_viol = float(level_df["violation_rate"].mean())
            mean_goodput = float(level_df["goodput_rps"].mean())
            mean_cvar = float(level_df["cvar99_latency_us"].mean())
            mean_throughput = float(level_df["throughput_rps"].mean())

            per_load_data.append({
                "load_level": load_level,
                "mean_violation_rate": mean_viol,
                "mean_goodput_rps": mean_goodput,
                "mean_cvar99_us": mean_cvar,
                "mean_throughput_rps": mean_throughput,
            })

            if mean_viol <= v_target:
                max_feasible_load = load_level
                goodput_at_max = mean_goodput
                cvar_at_max = mean_cvar
                throughput_at_max = mean_throughput

        rows.append({
            "scheduler_name": sched,
            "max_feasible_load_rps": max_feasible_load,
            "goodput_at_max_rps": goodput_at_max,
            "cvar99_at_max_us": cvar_at_max,
            "throughput_at_max_rps": throughput_at_max,
            "v_target": v_target,
            "slo_us": slo_us,
        })

        # Also add per-load detail rows
        for pld in per_load_data:
            rows.append({
                "scheduler_name": sched,
                "load_level": pld["load_level"],
                "mean_violation_rate": pld["mean_violation_rate"],
                "mean_goodput_rps": pld["mean_goodput_rps"],
                "mean_cvar99_us": pld["mean_cvar99_us"],
                "mean_throughput_rps": pld["mean_throughput_rps"],
                "v_target": v_target,
                "slo_us": slo_us,
            })

    result = pd.DataFrame(rows)

    # Log summary
    summary_rows = result[result["max_feasible_load_rps"].notna() &
                           (result["max_feasible_load_rps"] > 0)]
    summary_rows = summary_rows.drop_duplicates(subset=["scheduler_name"], keep="first")
    for _, row in summary_rows.iterrows():
        logger.info(
            "Admission: %s -> max_load=%.0f rps, goodput=%.1f rps",
            row["scheduler_name"], row["max_feasible_load_rps"],
            row.get("goodput_at_max_rps", 0),
        )

    return result
