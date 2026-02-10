"""Channel-based interference model.

Defines interference channels (LLC, MEM_BW, IO, TLB, SMT) and computes
per-channel pressure (beta) from spectator features and per-channel
sensitivity (alpha) from target features. Channel weights are regime-dependent.

The pairwise channel-based interference is:
    Delta_{t,s} = sum_k w_k(r) * alpha_{t,k} * beta_{s,k}
"""

from typing import Dict, List, Optional

import numpy as np

from sit.sim.workload import CHANNEL_NAMES, N_CHANNELS, Workload
from sit.sim.regimes import Regime


# Default channel weights
DEFAULT_CHANNEL_WEIGHTS: Dict[str, float] = {
    "LLC": 1.0,
    "MEM_BW": 1.2,
    "IO": 0.7,
    "TLB": 0.5,
    "SMT": 0.6,
}


def compute_pressure(spectator: Workload) -> np.ndarray:
    """Compute per-channel pressure beta_{s,k} from spectator features.

    Pressure is derived directly from the workload's feature vector,
    representing how much the spectator stresses each shared resource.

    Returns:
        Array of shape (N_CHANNELS,) with values in [0, 1].
    """
    return np.clip(spectator.features, 0.0, 1.0)


def compute_sensitivity(target: Workload) -> np.ndarray:
    """Compute per-channel sensitivity alpha_{t,k} from target features.

    Sensitivity is the target's tail_sensitivity vector, representing
    how much the target's tail latency degrades per unit of pressure
    on each channel.

    Returns:
        Array of shape (N_CHANNELS,) with values in [0, 1].
    """
    return np.clip(target.tail_sensitivity, 0.0, 1.0)


def regime_channel_weights(
    regime: Regime,
    base_weights: Dict[str, float],
) -> np.ndarray:
    """Compute regime-modulated channel weights w_k(r).

    Adjustments:
    - MEM_BW weight increases with load level (memory contention rises)
    - LLC weight increases with topology proximity (closer = more shared cache)
    - IO weight decreases slightly under high load (saturated IO is less variable)
    - TLB and SMT are relatively stable

    Returns:
        Array of shape (N_CHANNELS,) with positive weights.
    """
    weights = np.zeros(N_CHANNELS, dtype=np.float64)
    for i, name in enumerate(CHANNEL_NAMES):
        w = base_weights.get(name, 0.5)

        if name == "MEM_BW":
            # MEM_BW contention rises with load
            w *= (1.0 + 0.5 * regime.load_level)
        elif name == "LLC":
            # LLC contention stronger at closer topology
            proximity = max(0.0, 1.0 - 0.3 * regime.topology_distance)
            w *= (1.0 + 0.4 * proximity)
        elif name == "IO":
            # IO slightly less variable under saturation
            w *= (1.0 - 0.15 * regime.load_level)

        weights[i] = max(0.0, w)

    return weights


def channel_interference_us(
    target: Workload,
    spectator: Workload,
    regime: Regime,
    base_weights: Dict[str, float],
    scale_us: float = 100.0,
) -> float:
    """Compute channel-based pairwise interference Delta_{t,s}^{channel}.

    Delta = scale_us * sum_k w_k(r) * alpha_{t,k} * beta_{s,k}

    Args:
        target: Target workload.
        spectator: Spectator workload.
        regime: Current regime.
        base_weights: Base channel weights from config.
        scale_us: Global scaling factor (microseconds).

    Returns:
        Interference contribution in microseconds (non-negative).
    """
    alpha = compute_sensitivity(target)
    beta = compute_pressure(spectator)
    w = regime_channel_weights(regime, base_weights)

    delta = scale_us * np.sum(w * alpha * beta)
    return max(0.0, float(delta))


def channel_interference_decomposed(
    target: Workload,
    spectator: Workload,
    regime: Regime,
    base_weights: Dict[str, float],
    scale_us: float = 100.0,
) -> Dict[str, float]:
    """Compute per-channel interference contributions for interpretability.

    Returns a dict mapping channel name to its interference contribution in us.
    """
    alpha = compute_sensitivity(target)
    beta = compute_pressure(spectator)
    w = regime_channel_weights(regime, base_weights)

    result = {}
    for i, name in enumerate(CHANNEL_NAMES):
        result[name] = max(0.0, float(scale_us * w[i] * alpha[i] * beta[i]))
    return result
