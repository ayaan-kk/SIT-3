"""Pareto frontier construction for goodput vs tail risk.

Identifies Pareto-optimal (scheduler, load) points where no other
point achieves both higher goodput and lower tail risk.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger

logger = get_logger("load.pareto")


def is_dominated(
    goodput_a: float,
    risk_a: float,
    goodput_b: float,
    risk_b: float,
) -> bool:
    """Check if point A is dominated by point B.

    B dominates A if: goodput_B >= goodput_A AND risk_B <= risk_A,
    with at least one strict inequality.
    """
    if goodput_b >= goodput_a and risk_b <= risk_a:
        if goodput_b > goodput_a or risk_b < risk_a:
            return True
    return False


def find_pareto_frontier(
    points: List[Tuple[float, float]],
) -> List[int]:
    """Find indices of Pareto-optimal points.

    Points are (goodput, risk). A point is Pareto-optimal if no
    other point has both higher goodput and lower risk.

    Args:
        points: List of (goodput, risk) tuples.

    Returns:
        List of indices of Pareto-optimal points.
    """
    n = len(points)
    if n == 0:
        return []

    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if is_dominated(points[i][0], points[i][1],
                            points[j][0], points[j][1]):
                dominated[i] = True
                break

    return [i for i in range(n) if not dominated[i]]


def construct_pareto_data(
    sweep_df: pd.DataFrame,
    risk_metric: str = "cvar99_latency_us",
    goodput_metric: str = "goodput_rps",
) -> pd.DataFrame:
    """Construct Pareto frontier data from load sweep results.

    Aggregates per (load_level, scheduler_name) and identifies
    Pareto-optimal points.

    Args:
        sweep_df: Load sweep results DataFrame.
        risk_metric: Column name for risk (to minimize).
        goodput_metric: Column name for goodput (to maximize).

    Returns:
        DataFrame with Pareto frontier points.
    """
    if sweep_df.empty:
        return pd.DataFrame()

    # Aggregate per (load_level, scheduler_name)
    agg = sweep_df.groupby(["load_level", "scheduler_name"]).agg({
        goodput_metric: "mean",
        risk_metric: "mean",
        "violation_rate": "mean",
        "throughput_rps": "mean",
        "n_requests": "sum",
    }).reset_index()

    agg = agg.rename(columns={
        goodput_metric: "mean_goodput_rps",
        risk_metric: "mean_risk_us",
        "violation_rate": "mean_violation_rate",
        "throughput_rps": "mean_throughput_rps",
    })

    # Find Pareto frontier per scheduler
    rows = []
    schedulers = agg["scheduler_name"].unique()

    # Build global set of points for frontier computation
    all_points = []
    all_indices = []

    for idx, row in agg.iterrows():
        all_points.append((row["mean_goodput_rps"], row["mean_risk_us"]))
        all_indices.append(idx)

    frontier_indices = find_pareto_frontier(all_points)
    frontier_set = {all_indices[i] for i in frontier_indices}

    for idx, row in agg.iterrows():
        rows.append({
            "load_level": row["load_level"],
            "scheduler_name": row["scheduler_name"],
            "mean_goodput_rps": row["mean_goodput_rps"],
            "mean_risk_us": row["mean_risk_us"],
            "mean_violation_rate": row["mean_violation_rate"],
            "mean_throughput_rps": row["mean_throughput_rps"],
            "on_pareto_frontier": idx in frontier_set,
        })

    result = pd.DataFrame(rows)
    n_frontier = int(result["on_pareto_frontier"].sum())
    logger.info(
        "Pareto construction: %d total points, %d on frontier",
        len(result), n_frontier,
    )

    return result
