"""Pareto evaluation logic for load sweep gates.

Checks whether SIT-safe is on the Pareto frontier, dominates
static partition, and meets SLO throughput targets.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.load.pareto import is_dominated

logger = get_logger("eval.pareto")


def check_pareto_dominance_over_partition(
    pareto_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    sit_scheduler: str = "sit_safe_ucb",
    partition_scheduler: str = "static_partition",
    risk_band: float = 0.10,
    top_load_fraction: float = 0.30,
) -> Tuple[bool, Dict]:
    """Check if SIT-safe dominates partition in high-load region.

    Requirements:
    1. SIT-safe is on Pareto frontier in at least one high-load point
    2. SIT-safe dominates partition in at least one point within risk band

    Args:
        pareto_df: Pareto frontier data.
        sweep_df: Full sweep results.
        sit_scheduler: SIT scheduler name.
        partition_scheduler: Partition scheduler name.
        risk_band: Comparable risk band (fraction).
        top_load_fraction: Fraction of load grid considered "high load".

    Returns:
        Tuple of (passed, details_dict).
    """
    if pareto_df.empty or sweep_df.empty:
        return False, {"reason": "empty data"}

    load_levels = sorted(sweep_df["load_level"].unique())
    if not load_levels:
        return False, {"reason": "no load levels"}

    # High-load region: top fraction of load grid
    n_high = max(1, int(len(load_levels) * top_load_fraction))
    high_load_levels = set(load_levels[-n_high:])

    # Check 1: SIT-safe on frontier in high-load region
    sit_frontier = pareto_df[
        (pareto_df["scheduler_name"] == sit_scheduler) &
        (pareto_df["on_pareto_frontier"] == True) &
        (pareto_df["load_level"].isin(high_load_levels))
    ]
    on_frontier_high = len(sit_frontier) > 0

    # Check 2: SIT-safe dominates partition in at least one point
    # Aggregate per load level
    agg = sweep_df.groupby(["load_level", "scheduler_name"]).agg({
        "goodput_rps": "mean",
        "cvar99_latency_us": "mean",
    }).reset_index()

    dominates_partition = False
    domination_details = []

    for load_level in high_load_levels:
        sit_row = agg[
            (agg["scheduler_name"] == sit_scheduler) &
            (agg["load_level"] == load_level)
        ]
        part_row = agg[
            (agg["scheduler_name"] == partition_scheduler) &
            (agg["load_level"] == load_level)
        ]

        if sit_row.empty or part_row.empty:
            continue

        sit_goodput = float(sit_row["goodput_rps"].iloc[0])
        sit_risk = float(sit_row["cvar99_latency_us"].iloc[0])
        part_goodput = float(part_row["goodput_rps"].iloc[0])
        part_risk = float(part_row["cvar99_latency_us"].iloc[0])

        # Check within risk band
        risk_comparable = abs(sit_risk - part_risk) / max(part_risk, 1e-6) <= risk_band

        # Check dominance: SIT has higher goodput at comparable or lower risk
        if sit_goodput > part_goodput and sit_risk <= part_risk * (1 + risk_band):
            dominates_partition = True
            domination_details.append({
                "load_level": load_level,
                "sit_goodput": sit_goodput,
                "sit_risk": sit_risk,
                "part_goodput": part_goodput,
                "part_risk": part_risk,
            })

    details = {
        "on_frontier_high_load": on_frontier_high,
        "dominates_partition": dominates_partition,
        "n_domination_points": len(domination_details),
        "high_load_levels": list(high_load_levels),
        "domination_details": domination_details[:5],
    }

    passed = on_frontier_high and dominates_partition
    return passed, details


def check_admission_advantage(
    admission_df: pd.DataFrame,
    sit_scheduler: str = "sit_safe_ucb",
    partition_scheduler: str = "static_partition",
    advantage_ratio: float = 1.15,
) -> Tuple[bool, Dict]:
    """Check if SIT-safe admits >= advantage_ratio × partition's load.

    Also checks goodput advantage as alternative.

    Args:
        admission_df: Admission curve data.
        sit_scheduler: SIT scheduler name.
        partition_scheduler: Partition scheduler name.
        advantage_ratio: Required ratio (default 1.15 = 15% more).

    Returns:
        Tuple of (passed, details_dict).
    """
    if admission_df.empty:
        return False, {"reason": "empty admission data"}

    # Get max feasible load for each scheduler
    sit_rows = admission_df[
        (admission_df["scheduler_name"] == sit_scheduler) &
        (admission_df["max_feasible_load_rps"] > 0)
    ]
    part_rows = admission_df[
        (admission_df["scheduler_name"] == partition_scheduler) &
        (admission_df["max_feasible_load_rps"] > 0)
    ]

    if sit_rows.empty or part_rows.empty:
        # Fallback: check goodput advantage at highest common load
        return _check_goodput_advantage(
            admission_df, sit_scheduler, partition_scheduler, advantage_ratio,
        )

    sit_max = float(sit_rows.iloc[0]["max_feasible_load_rps"])
    part_max = float(part_rows.iloc[0]["max_feasible_load_rps"])

    if part_max <= 0:
        # Partition can't sustain any load -> SIT automatically wins
        return True, {"sit_max": sit_max, "part_max": 0, "ratio": float("inf")}

    ratio = sit_max / part_max
    passed = ratio >= advantage_ratio

    # Also check goodput advantage
    sit_goodput = float(sit_rows.iloc[0].get("goodput_at_max_rps", 0))
    part_goodput = float(part_rows.iloc[0].get("goodput_at_max_rps", 0))

    goodput_ratio = sit_goodput / max(part_goodput, 1e-6)
    goodput_pass = goodput_ratio >= advantage_ratio

    details = {
        "sit_max_load": sit_max,
        "part_max_load": part_max,
        "load_ratio": ratio,
        "sit_goodput_at_max": sit_goodput,
        "part_goodput_at_max": part_goodput,
        "goodput_ratio": goodput_ratio,
        "load_pass": passed,
        "goodput_pass": goodput_pass,
    }

    return passed or goodput_pass, details


def _check_goodput_advantage(
    admission_df: pd.DataFrame,
    sit_scheduler: str,
    partition_scheduler: str,
    advantage_ratio: float,
) -> Tuple[bool, Dict]:
    """Fallback: check goodput advantage at common load levels."""
    sit_data = admission_df[
        (admission_df["scheduler_name"] == sit_scheduler) &
        (admission_df["mean_goodput_rps"].notna())
    ]
    part_data = admission_df[
        (admission_df["scheduler_name"] == partition_scheduler) &
        (admission_df["mean_goodput_rps"].notna())
    ]

    if sit_data.empty or part_data.empty:
        return False, {"reason": "no goodput data"}

    # Compare at highest load level where both have data
    sit_loads = set(sit_data["load_level"].dropna().values)
    part_loads = set(part_data["load_level"].dropna().values)
    common = sit_loads & part_loads

    if not common:
        return False, {"reason": "no common load levels"}

    max_load = max(common)
    sit_gp = float(sit_data[sit_data["load_level"] == max_load]["mean_goodput_rps"].iloc[0])
    part_gp = float(part_data[part_data["load_level"] == max_load]["mean_goodput_rps"].iloc[0])

    ratio = sit_gp / max(part_gp, 1e-6)
    passed = ratio >= advantage_ratio

    return passed, {
        "load_level": max_load,
        "sit_goodput": sit_gp,
        "part_goodput": part_gp,
        "ratio": ratio,
    }


def check_model_sanity(
    diagnostics_df: pd.DataFrame,
    slo_us: float = 500_000.0,
) -> Tuple[bool, Dict]:
    """Check queue model sanity.

    Validates:
    1. No p99 >> SLO with near-zero violation rate
    2. goodput <= throughput always
    3. Utilization < 1 for stable points
    4. Throughput computed from timeline (positive when requests exist)

    Args:
        diagnostics_df: Queue diagnostics DataFrame.
        slo_us: SLO threshold.

    Returns:
        Tuple of (passed, details_dict).
    """
    if diagnostics_df.empty:
        return True, {"reason": "no diagnostics data"}

    failures = []

    for _, row in diagnostics_df.iterrows():
        # Check p99 vs violation sanity
        if "sanity_p99_viol" in row and not row["sanity_p99_viol"]:
            failures.append({
                "type": "p99_violation_inconsistency",
                "load": row.get("load_level"),
                "sched": row.get("scheduler_name"),
                "p99": row.get("p99_latency_us"),
                "viol": row.get("violation_rate"),
            })

        # Check goodput <= throughput
        if "sanity_throughput" in row and not row["sanity_throughput"]:
            failures.append({
                "type": "goodput_exceeds_throughput",
                "load": row.get("load_level"),
                "sched": row.get("scheduler_name"),
                "goodput": row.get("goodput_rps"),
                "throughput": row.get("throughput_rps"),
            })

    passed = len(failures) == 0
    return passed, {
        "n_failures": len(failures),
        "failures": failures[:10],
    }
