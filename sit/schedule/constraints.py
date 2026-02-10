"""Safety constraints for scheduling.

Implements hard safety thresholds, feasibility checks, and
catastrophe definitions for placement decisions.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sit.core.logging import get_logger

logger = get_logger("schedule.constraints")


@dataclass
class SafetyConfig:
    """Safety thresholds for scheduling."""
    tau_risk_us: float = 2_000_000.0  # Hard safety threshold
    beta_ucb: float = 2.0  # UCB parameter
    catastrophe_p99_us: float = 20_000_000.0
    catastrophe_cvar_us: float = 80_000_000.0
    catastrophe_violation_rate: float = 0.9

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SafetyConfig":
        sched_cfg = config.get("scheduling", {})
        sit_params = sched_cfg.get("sit_params", {})
        cat_cfg = sched_cfg.get("catastrophe_thresholds", {})
        return cls(
            tau_risk_us=float(sit_params.get("tau_risk_us", 2_000_000.0)),
            beta_ucb=float(sit_params.get("beta_ucb", 2.0)),
            catastrophe_p99_us=float(cat_cfg.get("catastrophe_p99_us", 20_000_000.0)),
            catastrophe_cvar_us=float(cat_cfg.get("catastrophe_cvar_us", 80_000_000.0)),
            catastrophe_violation_rate=float(cat_cfg.get("catastrophe_violation_rate", 0.9)),
        )


def check_safety(
    robust_risk_us: float,
    tau_risk_us: float,
) -> bool:
    """Check if a placement is safe (robust risk within threshold).

    Args:
        robust_risk_us: Robust predicted risk (UCB) in microseconds.
        tau_risk_us: Safety threshold in microseconds.

    Returns:
        True if placement is safe.
    """
    return robust_risk_us <= tau_risk_us


def is_catastrophe(
    metrics: Dict[str, float],
    safety_config: SafetyConfig,
) -> bool:
    """Check if realized metrics constitute a catastrophe.

    A catastrophe occurs if any:
    - p99_latency_us > catastrophe_p99_us
    - cvar99_latency_us > catastrophe_cvar_us
    - violation_rate > catastrophe_violation_rate

    Args:
        metrics: Realized per-target metrics dict.
        safety_config: Safety thresholds.

    Returns:
        True if this is a catastrophic event.
    """
    p99 = metrics.get("p99_latency_us", 0.0)
    cvar = metrics.get("cvar99_latency_us", 0.0)
    viol = metrics.get("violation_rate", 0.0)

    return (
        p99 > safety_config.catastrophe_p99_us
        or cvar > safety_config.catastrophe_cvar_us
        or viol > safety_config.catastrophe_violation_rate
    )


def count_catastrophes(
    episode_results: List[Any],
    safety_config: SafetyConfig,
) -> Tuple[int, List[Dict]]:
    """Count catastrophic events across episodes.

    Args:
        episode_results: List of EpisodeResult objects.
        safety_config: Safety thresholds.

    Returns:
        Tuple of (count, list of catastrophe details).
    """
    count = 0
    details = []

    for er in episode_results:
        for target_id, metrics in er.target_metrics.items():
            if is_catastrophe(metrics, safety_config):
                count += 1
                details.append({
                    "episode_id": er.episode_id,
                    "scheduler_name": er.scheduler_name,
                    "target_id": target_id,
                    "p99_latency_us": metrics.get("p99_latency_us", 0.0),
                    "cvar99_latency_us": metrics.get("cvar99_latency_us", 0.0),
                    "violation_rate": metrics.get("violation_rate", 0.0),
                    "fallback_used": er.fallback_used,
                })

    return count, details


def find_feasible_hosts(
    target_id: str,
    candidate_hosts: List[Any],
    regime_id: str,
    x_hat: Dict,
    sigma: Dict,
    safety_config: SafetyConfig,
    current_assignments: Dict[str, List[str]],
    base_risk: float = 0.0,
) -> List[Tuple[Any, float]]:
    """Find hosts that satisfy the safety constraint for a target.

    Args:
        target_id: Target workload ID.
        candidate_hosts: List of Host objects with capacity.
        regime_id: Current regime.
        x_hat: Interference estimates.
        sigma: Uncertainty estimates.
        safety_config: Safety config.
        current_assignments: host_id -> list of spectator workload IDs.
        base_risk: Baseline risk for isolated target.

    Returns:
        List of (host, robust_risk) pairs that are feasible.
    """
    from sit.schedule.objectives import robust_risk as compute_robust_risk

    feasible = []
    for host in candidate_hosts:
        if not host.can_accept():
            continue

        # Get spectator workload IDs on this host
        co_tenants = current_assignments.get(host.host_id, [])
        # Filter to only spectator workload IDs (not target workload IDs)
        spectator_co_tenants = co_tenants  # Already filtered by caller

        r_ucb = compute_robust_risk(
            target_id, spectator_co_tenants, regime_id,
            x_hat, sigma,
            beta=safety_config.beta_ucb,
            base_risk=base_risk,
        )

        if check_safety(r_ucb, safety_config.tau_risk_us):
            feasible.append((host, r_ucb))

    return feasible
