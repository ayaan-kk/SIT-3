"""Queue diagnostics for model sanity.

Computes utilization, queue delay decomposition, stability checks,
and consistency validation for the serving model.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from sit.core.logging import get_logger

logger = get_logger("load.diagnostics")


def compute_queue_diagnostics(
    sweep_df: pd.DataFrame,
    slo_us: float = 500_000.0,
) -> pd.DataFrame:
    """Compute queue health diagnostics from load sweep results.

    For each (load_level, scheduler_name), computes:
    - Mean utilization (rho)
    - Mean/p95 queue delay
    - Tail decomposition (queue fraction of total latency)
    - Stability check (utilization < 1)
    - Sanity: high p99 implies high violation_rate

    Args:
        sweep_df: Load sweep results DataFrame.
        slo_us: SLO threshold.

    Returns:
        DataFrame with per (load_level, scheduler) diagnostics.
    """
    if sweep_df.empty:
        return pd.DataFrame()

    rows = []
    groups = sweep_df.groupby(["load_level", "scheduler_name"])

    for (load_level, sched_name), group_df in groups:
        utilization = float(group_df["utilization"].mean()) if "utilization" in group_df.columns else 0.0
        mean_qd = float(group_df["mean_queue_delay_us"].mean()) if "mean_queue_delay_us" in group_df.columns else 0.0
        p95_qd = float(group_df["p95_queue_delay_us"].mean()) if "p95_queue_delay_us" in group_df.columns else 0.0
        mean_svc = float(group_df["mean_service_us"].mean()) if "mean_service_us" in group_df.columns else 0.0
        busy_frac = float(group_df["busy_fraction"].mean()) if "busy_fraction" in group_df.columns else 0.0

        mean_lat = float(group_df["mean_latency_us"].mean())
        p99_lat = float(group_df["p99_latency_us"].mean())
        cvar99_lat = float(group_df["cvar99_latency_us"].mean())
        viol_rate = float(group_df["violation_rate"].mean())
        goodput = float(group_df["goodput_rps"].mean())
        throughput = float(group_df["throughput_rps"].mean())

        # Tail decomposition: what fraction of total p99 is from queueing
        queue_fraction = mean_qd / mean_lat if mean_lat > 0 else 0.0

        # Stability check
        stable = utilization < 0.99

        # Sanity checks
        # If p99 >> slo, violation rate should be high
        p99_vs_slo = p99_lat / slo_us if slo_us > 0 else 0.0
        sanity_p99_viol = True
        if p99_lat > 2.0 * slo_us and viol_rate < 0.005:
            sanity_p99_viol = False
            logger.warning(
                "Sanity fail: p99=%.0f >> slo=%.0f but violation_rate=%.4f "
                "(load=%s, sched=%s)",
                p99_lat, slo_us, viol_rate, load_level, sched_name,
            )

        # Throughput from timeline (goodput + violations = throughput)
        sanity_throughput = True
        if throughput > 0 and goodput > throughput * 1.01:
            sanity_throughput = False
            logger.warning(
                "Sanity fail: goodput=%.1f > throughput=%.1f "
                "(load=%s, sched=%s)",
                goodput, throughput, load_level, sched_name,
            )

        rows.append({
            "load_level": load_level,
            "scheduler_name": sched_name,
            "utilization": utilization,
            "mean_queue_delay_us": mean_qd,
            "p95_queue_delay_us": p95_qd,
            "mean_service_us": mean_svc,
            "busy_fraction": busy_frac,
            "queue_fraction_of_latency": queue_fraction,
            "mean_latency_us": mean_lat,
            "p99_latency_us": p99_lat,
            "cvar99_latency_us": cvar99_lat,
            "violation_rate": viol_rate,
            "throughput_rps": throughput,
            "goodput_rps": goodput,
            "p99_vs_slo_ratio": p99_vs_slo,
            "stable": stable,
            "sanity_p99_viol": sanity_p99_viol,
            "sanity_throughput": sanity_throughput,
            "n_targets": len(group_df),
        })

    result = pd.DataFrame(rows)

    n_unstable = int((~result["stable"]).sum()) if "stable" in result.columns else 0
    n_sanity_fail = 0
    if "sanity_p99_viol" in result.columns:
        n_sanity_fail += int((~result["sanity_p99_viol"]).sum())
    if "sanity_throughput" in result.columns:
        n_sanity_fail += int((~result["sanity_throughput"]).sum())

    logger.info(
        "Queue diagnostics: %d points, %d unstable, %d sanity failures",
        len(result), n_unstable, n_sanity_fail,
    )

    return result
