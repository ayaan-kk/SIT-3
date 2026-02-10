"""Spectator feature vectors for probe selection.

Provides feature extraction from simulator workload attributes and
channel pressure vectors. Features are used by diversity kernels and
information-gain scoring to guide adaptive probe selection.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from sit.core.logging import get_logger
from sit.sim.workload import CHANNEL_NAMES, N_CHANNELS, Workload

logger = get_logger("probe.features")


def workload_features(spectator: Workload) -> np.ndarray:
    """Extract feature vector from a spectator workload.

    Uses the workload's resource intensity features (cpu/mem/io/tlb/smt).

    Args:
        spectator: A Workload with role="spectator".

    Returns:
        Feature vector f_s of shape (N_CHANNELS,).
    """
    return np.array(spectator.features, dtype=np.float64)


def channel_pressure_features(
    spectator: Workload,
    target: Workload,
    regime: Any,
    channel_weights: Dict[str, float],
    channel_scale_us: float = 100.0,
) -> np.ndarray:
    """Extract channel pressure vector beta_{s,k} for a spectator.

    Computes per-channel pressure weighted by regime-modulated weights
    and target sensitivity.

    Args:
        spectator: The spectator workload.
        target: The target workload.
        regime: The regime object.
        channel_weights: Base channel weights.
        channel_scale_us: Scaling factor.

    Returns:
        Pressure vector of shape (N_CHANNELS,).
    """
    from sit.sim.channels import (
        compute_pressure,
        compute_sensitivity,
        regime_channel_weights,
    )

    beta = compute_pressure(spectator)
    alpha = compute_sensitivity(target)
    w = regime_channel_weights(regime, channel_weights)
    return channel_scale_us * w * alpha * beta


def build_feature_matrix(
    spectators: List[Workload],
    mode: str = "workload",
    target: Optional[Workload] = None,
    regime: Optional[Any] = None,
    channel_weights: Optional[Dict[str, float]] = None,
    channel_scale_us: float = 100.0,
) -> np.ndarray:
    """Build feature matrix F where F[i] = f_{s_i}.

    Args:
        spectators: List of spectator workloads.
        mode: "workload" for raw features, "pressure" for channel pressure.
        target: Required if mode="pressure".
        regime: Required if mode="pressure".
        channel_weights: Required if mode="pressure".
        channel_scale_us: Scaling factor for pressure mode.

    Returns:
        Feature matrix of shape (n_spectators, d).
    """
    n = len(spectators)
    if n == 0:
        return np.zeros((0, N_CHANNELS), dtype=np.float64)

    if mode == "workload":
        F = np.array([workload_features(s) for s in spectators], dtype=np.float64)
    elif mode == "pressure":
        if target is None or regime is None or channel_weights is None:
            raise ValueError("pressure mode requires target, regime, channel_weights")
        F = np.array([
            channel_pressure_features(s, target, regime, channel_weights, channel_scale_us)
            for s in spectators
        ], dtype=np.float64)
    else:
        raise ValueError(f"Unknown feature mode: {mode}")

    logger.info(
        "Feature matrix: shape %s, mode=%s, norm range=[%.3f, %.3f]",
        F.shape, mode,
        np.linalg.norm(F, axis=1).min() if n > 0 else 0.0,
        np.linalg.norm(F, axis=1).max() if n > 0 else 0.0,
    )
    return F
