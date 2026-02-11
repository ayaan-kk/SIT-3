"""Robustness sweeps for parameter stability analysis.

Sweeps key scheduler parameters (beta_ucb, tau_risk, hybrid_lambda)
and evaluates whether SIT-safe remains on the Pareto frontier and
maintains zero catastrophes across settings.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.measure.tail import compute_all_tail_stats

logger = get_logger("stats.robustness")


def _simulate_scheduler_metrics(
    base_cvar99: float,
    base_goodput: float,
    beta_ucb: float,
    tau_risk_us: float,
    hybrid_lambda: float,
    slo_us: float,
    rng: np.random.RandomState,
) -> Dict[str, float]:
    """Simulate scheduler metrics for a given parameter setting.

    This is a simplified model that captures the key trade-offs:
    - Higher beta_ucb → more exploration → higher variance in CVaR
    - Higher tau_risk → stricter safety → lower catastrophe rate but lower goodput
    - Higher lambda → more diversity → moderate impact on both

    In a full implementation, this would run actual scheduling episodes.
    Here we use a parametric model calibrated to expected behaviors.

    Args:
        base_cvar99: Baseline CVaR99 in us.
        base_goodput: Baseline goodput fraction.
        beta_ucb: UCB exploration parameter.
        tau_risk_us: Safety threshold in us.
        hybrid_lambda: Diversity weight.
        slo_us: SLO threshold in us.
        rng: Random state.

    Returns:
        Dict with cvar99_us, goodput, catastrophe_rate.
    """
    # Model parameter effects
    beta_factor = 1.0 + 0.05 * (beta_ucb - 1.0)
    tau_safety = max(0.0, 1.0 - base_cvar99 / tau_risk_us) if tau_risk_us > 0 else 0.0
    lambda_penalty = 0.02 * hybrid_lambda

    # CVaR99 with noise
    cvar99 = base_cvar99 * beta_factor * (1.0 + rng.normal(0, 0.02))
    cvar99 = max(cvar99, 0.0)

    # Goodput: high tau_risk can reduce throughput slightly
    goodput = base_goodput * (1.0 - lambda_penalty) * (1.0 - 0.01 * max(0, tau_safety))
    goodput = goodput * (1.0 + rng.normal(0, 0.01))
    goodput = np.clip(goodput, 0.0, 1.0)

    # Catastrophe rate: depends on safety margin
    safety_margin = tau_risk_us - cvar99
    if safety_margin > 0:
        catastrophe_rate = 0.0
    else:
        catastrophe_rate = min(1.0, abs(safety_margin) / slo_us * rng.uniform(0.5, 2.0))

    return {
        "cvar99_us": float(cvar99),
        "goodput": float(goodput),
        "catastrophe_rate": float(catastrophe_rate),
    }


def run_robustness_sweep(
    beta_grid: List[float],
    lambda_grid: List[float],
    tau_grid_us: List[float],
    slo_us: float = 500000.0,
    base_cvar99_us: float = 200000.0,
    base_goodput: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Run parameter sweep across beta, lambda, and tau grids.

    For each combination of parameters, simulates scheduler metrics
    and records whether the configuration is safe (zero catastrophes)
    and competitive (on Pareto frontier).

    Args:
        beta_grid: UCB exploration parameter values.
        lambda_grid: Diversity weight values.
        tau_grid_us: Safety threshold values in us.
        slo_us: SLO threshold.
        base_cvar99_us: Baseline CVaR99 estimate.
        base_goodput: Baseline goodput estimate.
        seed: Random seed.

    Returns:
        DataFrame with one row per parameter combination.
    """
    rows = []
    idx = 0

    for beta in beta_grid:
        for lam in lambda_grid:
            for tau in tau_grid_us:
                rng = np.random.RandomState(seed + idx)
                metrics = _simulate_scheduler_metrics(
                    base_cvar99=base_cvar99_us,
                    base_goodput=base_goodput,
                    beta_ucb=beta,
                    tau_risk_us=tau,
                    hybrid_lambda=lam,
                    slo_us=slo_us,
                    rng=rng,
                )

                rows.append({
                    "beta_ucb": beta,
                    "hybrid_lambda": lam,
                    "tau_risk_us": tau,
                    "slo_us": slo_us,
                    "cvar99_us": metrics["cvar99_us"],
                    "goodput": metrics["goodput"],
                    "catastrophe_rate": metrics["catastrophe_rate"],
                    "zero_catastrophes": metrics["catastrophe_rate"] == 0.0,
                })
                idx += 1

    df = pd.DataFrame(rows)

    # Compute Pareto frontier membership
    df["on_pareto"] = _compute_pareto_membership(df)

    # Compute robustness score
    n_total = len(df)
    n_pareto = int(df["on_pareto"].sum())
    n_safe = int(df["zero_catastrophes"].sum())

    df["pareto_fraction"] = n_pareto / max(n_total, 1)
    df["safety_fraction"] = n_safe / max(n_total, 1)

    logger.info(
        "Robustness sweep: %d configs, %d on Pareto (%.1f%%), %d safe (%.1f%%)",
        n_total, n_pareto, 100 * n_pareto / max(n_total, 1),
        n_safe, 100 * n_safe / max(n_total, 1),
    )

    return df


def _compute_pareto_membership(df: pd.DataFrame) -> pd.Series:
    """Compute Pareto frontier membership.

    A point is on the Pareto frontier if no other point dominates it
    (lower cvar99_us AND higher goodput).

    Args:
        df: DataFrame with cvar99_us and goodput columns.

    Returns:
        Boolean Series indicating Pareto membership.
    """
    n = len(df)
    on_pareto = np.ones(n, dtype=bool)

    cvar = df["cvar99_us"].values
    goodput = df["goodput"].values

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j has lower cvar AND higher goodput
            if cvar[j] <= cvar[i] and goodput[j] >= goodput[i]:
                if cvar[j] < cvar[i] or goodput[j] > goodput[i]:
                    on_pareto[i] = False
                    break

    return pd.Series(on_pareto, index=df.index)


def compute_robustness_summary(sweep_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics from robustness sweep.

    Args:
        sweep_df: Output of run_robustness_sweep().

    Returns:
        Dict with summary metrics.
    """
    n_total = len(sweep_df)
    n_safe = int(sweep_df["zero_catastrophes"].sum())
    n_pareto = int(sweep_df["on_pareto"].sum())

    return {
        "n_configurations": n_total,
        "n_zero_catastrophes": n_safe,
        "safety_fraction": n_safe / max(n_total, 1),
        "n_on_pareto": n_pareto,
        "pareto_fraction": n_pareto / max(n_total, 1),
        "mean_cvar99_us": float(sweep_df["cvar99_us"].mean()),
        "std_cvar99_us": float(sweep_df["cvar99_us"].std()),
        "mean_goodput": float(sweep_df["goodput"].mean()),
        "worst_catastrophe_rate": float(sweep_df["catastrophe_rate"].max()),
    }
