"""Scheduling objectives: risk minimization and goodput maximization.

Implements predicted risk computation, utility functions, and
objective evaluation for placement optimization.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sit.core.logging import get_logger

logger = get_logger("schedule.objectives")


def predicted_risk(
    target_id: str,
    co_tenant_ids: List[str],
    regime_id: str,
    x_hat: Dict[Tuple[str, str, str], float],
    base_risk: float = 0.0,
) -> float:
    """Compute predicted risk for a target given co-tenants.

    R_hat = base + sum_{s in co-tenants} x_hat[(target, s, regime)]

    Args:
        target_id: Target workload ID.
        co_tenant_ids: Spectator workload IDs on same host.
        regime_id: Current regime.
        x_hat: Learned interference map (target, spectator, regime) -> x_hat_us.
        base_risk: Baseline risk when isolated.

    Returns:
        Predicted risk in microseconds.
    """
    risk = base_risk
    for s_id in co_tenant_ids:
        key = (target_id, s_id, regime_id)
        risk += x_hat.get(key, 0.0)
    return max(0.0, risk)


def predicted_uncertainty(
    target_id: str,
    co_tenant_ids: List[str],
    regime_id: str,
    sigma: Dict[Tuple[str, str, str], float],
) -> float:
    """Compute predicted uncertainty for a target given co-tenants.

    U = sum_{s in co-tenants} sigma[(target, s, regime)]

    Args:
        target_id: Target workload ID.
        co_tenant_ids: Spectator workload IDs on same host.
        regime_id: Current regime.
        sigma: Uncertainty map (target, spectator, regime) -> sigma_us.

    Returns:
        Total uncertainty in microseconds.
    """
    unc = 0.0
    for s_id in co_tenant_ids:
        key = (target_id, s_id, regime_id)
        unc += sigma.get(key, 0.0)
    return max(0.0, unc)


def robust_risk(
    target_id: str,
    co_tenant_ids: List[str],
    regime_id: str,
    x_hat: Dict,
    sigma: Dict,
    beta: float = 2.0,
    base_risk: float = 0.0,
) -> float:
    """Compute robust upper bound on risk.

    R_UCB = R_hat + beta * U

    Args:
        target_id: Target workload ID.
        co_tenant_ids: Spectator IDs on same host.
        regime_id: Current regime.
        x_hat: Learned interference map.
        sigma: Uncertainty map.
        beta: UCB parameter.
        base_risk: Baseline risk.

    Returns:
        Robust risk upper bound in microseconds.
    """
    r_hat = predicted_risk(target_id, co_tenant_ids, regime_id, x_hat, base_risk)
    u = predicted_uncertainty(target_id, co_tenant_ids, regime_id, sigma)
    return r_hat + beta * u


def utility_function(
    risk_us: float,
    slo_us: float,
    sharpness: float = 5.0,
) -> float:
    """Compute utility as a function of predicted risk.

    Utility is high when risk << SLO and drops sharply near/above SLO.
    Uses sigmoid-based penalty.

    Args:
        risk_us: Predicted risk in microseconds.
        slo_us: SLO threshold in microseconds.
        sharpness: How sharply utility drops near SLO.

    Returns:
        Utility value in [0, 1].
    """
    if slo_us <= 0:
        return 0.0
    ratio = risk_us / slo_us
    # Sigmoid centered at ratio=1.0
    return 1.0 / (1.0 + np.exp(sharpness * (ratio - 0.8)))


def redundancy_penalty(
    host_spectator_ids: List[str],
    new_spectator_ids: List[str],
    K: Optional[np.ndarray] = None,
    spectator_id_to_idx: Optional[Dict[str, int]] = None,
) -> float:
    """Compute redundancy penalty for adding spectators to a host.

    Higher penalty when the host already has similar spectators.

    Args:
        host_spectator_ids: Spectator IDs already on the host.
        new_spectator_ids: Spectator IDs being considered for addition.
        K: Kernel matrix for spectator similarity.
        spectator_id_to_idx: Map from spectator ID to kernel matrix index.

    Returns:
        Redundancy penalty score (higher = more redundant).
    """
    if K is None or spectator_id_to_idx is None:
        return 0.0
    if not host_spectator_ids or not new_spectator_ids:
        return 0.0

    total_sim = 0.0
    count = 0
    for s_new in new_spectator_ids:
        idx_new = spectator_id_to_idx.get(s_new)
        if idx_new is None:
            continue
        for s_existing in host_spectator_ids:
            idx_ex = spectator_id_to_idx.get(s_existing)
            if idx_ex is None:
                continue
            total_sim += K[idx_new, idx_ex]
            count += 1

    return total_sim / max(count, 1)


def evaluate_placement_objective(
    placement_risks: Dict[str, float],
    slo_us: float,
    mode: str = "tail_risk",
) -> float:
    """Evaluate the overall placement objective.

    Args:
        placement_risks: Dict target_id -> robust_risk_us.
        slo_us: SLO threshold.
        mode: "tail_risk" (minimize sum of risks) or
              "goodput" (maximize sum of utilities).

    Returns:
        Objective value (lower is better for tail_risk,
        higher is better for goodput).
    """
    if mode == "tail_risk":
        return sum(placement_risks.values())
    elif mode == "goodput":
        return sum(utility_function(r, slo_us) for r in placement_risks.values())
    else:
        raise ValueError(f"Unknown objective mode: {mode}")
