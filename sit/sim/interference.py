"""Interference model: channel-based + sparse toxic pairs + optional interactions.

Computes ground truth pairwise interference X_{t,s}^{(r)} as:
    X = Delta^{channel} + Delta^{toxic}

Optional second-order interaction term when enable_interactions=True:
    Z_{t, s_i, s_j} = gamma * <beta_{s_i}, beta_{s_j}> * mean(alpha_t)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sit.sim.channels import channel_interference_us
from sit.sim.regimes import Regime
from sit.sim.workload import N_CHANNELS, Workload


@dataclass
class ToxicPairConfig:
    """Configuration for sparse toxic pair generation."""
    enabled: bool = True
    sparsity: float = 0.05
    lognormal_mu: float = 0.0
    lognormal_sigma: float = 0.7


@dataclass
class InteractionConfig:
    """Configuration for non-additive interaction terms."""
    enabled: bool = False
    gamma: float = 0.01


def generate_toxic_pairs(
    targets: List[Workload],
    spectators: List[Workload],
    config: ToxicPairConfig,
    rng: np.random.RandomState,
) -> Dict[Tuple[str, str], float]:
    """Generate sparse toxic pair interference values.

    A fraction (sparsity) of (target, spectator) pairs receive an
    extra interference bump drawn from LogNormal.

    Returns:
        Dict mapping (target_id, spectator_id) -> toxic_delta_us.
    """
    if not config.enabled:
        return {}

    toxic = {}
    for t in targets:
        for s in spectators:
            if rng.random() < config.sparsity:
                delta = float(rng.lognormal(
                    mean=config.lognormal_mu,
                    sigma=config.lognormal_sigma,
                ))
                # Scale to meaningful microsecond range
                toxic[(t.workload_id, s.workload_id)] = delta * 100.0

    return toxic


def compute_ground_truth_matrix(
    targets: List[Workload],
    spectators: List[Workload],
    regime: Regime,
    channel_weights: Dict[str, float],
    toxic_pairs: Dict[Tuple[str, str], float],
    channel_scale_us: float = 100.0,
) -> Dict[Tuple[str, str], float]:
    """Compute full ground truth interference matrix X_{t,s}^{(r)}.

    X_{t,s} = Delta^{channel}(t,s,r) + Delta^{toxic}(t,s)

    Returns:
        Dict mapping (target_id, spectator_id) -> X_us.
    """
    gt = {}
    for t in targets:
        for s in spectators:
            ch_delta = channel_interference_us(
                t, s, regime, channel_weights, scale_us=channel_scale_us,
            )
            toxic_delta = toxic_pairs.get((t.workload_id, s.workload_id), 0.0)
            gt[(t.workload_id, s.workload_id)] = ch_delta + toxic_delta
    return gt


def compute_set_interference_us(
    target: Workload,
    spectator_ids: List[str],
    ground_truth: Dict[Tuple[str, str], float],
    spectators_by_id: Optional[Dict[str, Workload]] = None,
    interaction_config: Optional[InteractionConfig] = None,
) -> float:
    """Compute total interference for a target with a set of spectators.

    Additive model: sum of pairwise X_{t,s} for s in spectator_ids.
    With interactions enabled, adds second-order terms.

    Args:
        target: Target workload.
        spectator_ids: List of spectator workload IDs.
        ground_truth: Pairwise ground truth matrix.
        spectators_by_id: Dict of spectator workloads (needed for interactions).
        interaction_config: Optional interaction configuration.

    Returns:
        Total interference in microseconds (non-negative).
    """
    total = 0.0

    # Additive pairwise
    for sid in spectator_ids:
        key = (target.workload_id, sid)
        total += ground_truth.get(key, 0.0)

    # Optional second-order interactions
    if (
        interaction_config is not None
        and interaction_config.enabled
        and spectators_by_id is not None
        and len(spectator_ids) >= 2
    ):
        gamma = interaction_config.gamma
        alpha_mean = float(np.mean(target.tail_sensitivity))

        for i in range(len(spectator_ids)):
            for j in range(i + 1, len(spectator_ids)):
                si = spectators_by_id.get(spectator_ids[i])
                sj = spectators_by_id.get(spectator_ids[j])
                if si is not None and sj is not None:
                    beta_dot = float(np.dot(si.features, sj.features))
                    total += gamma * beta_dot * alpha_mean * 100.0  # scale

    return max(0.0, total)
