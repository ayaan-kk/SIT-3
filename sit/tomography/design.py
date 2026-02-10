"""Design matrix construction and measurement collection.

Builds the binary design matrix A from probe sets, where A[i,j] = 1
if spectator j appears in probe set i. Collects measurements y by
running simulated micro-runs for each probe set.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.core.units import CANONICAL_LATENCY_UNIT
from sit.measure.tail import estimate_var, estimate_cvar

logger = get_logger("tomography.design")


def build_design_matrix(
    probe_sets: List[List[str]],
    spectator_ids: List[str],
) -> np.ndarray:
    """Build binary design matrix A from probe sets.

    A[i, j] = 1 if spectator_ids[j] is in probe_sets[i], else 0.

    Args:
        probe_sets: List of m probe sets, each a list of spectator IDs.
        spectator_ids: Ordered list of n spectator IDs (defines column order).

    Returns:
        Binary numpy array of shape (m, n).
    """
    m = len(probe_sets)
    n = len(spectator_ids)
    sid_to_idx = {sid: j for j, sid in enumerate(spectator_ids)}

    A = np.zeros((m, n), dtype=np.float64)
    for i, probe in enumerate(probe_sets):
        for sid in probe:
            j = sid_to_idx.get(sid)
            if j is not None:
                A[i, j] = 1.0

    logger.info(
        "Design matrix: shape (%d, %d), density %.3f",
        m, n, A.sum() / max(A.size, 1),
    )
    return A


def collect_measurements(
    world: Any,
    target_id: str,
    regime_id: str,
    probe_sets: List[List[str]],
    n_samples: int,
    rng: np.random.RandomState,
    stat: str = "mean",
    use_irbs: bool = False,
) -> np.ndarray:
    """Collect measurement vector y by running micro-runs for each probe set.

    For each probe set, simulates a micro-run with the given spectators
    and computes the specified tail statistic on service times (not total
    latency) to avoid queueing nonlinearity that would break the linear
    y = Ax model.

    Optionally uses IRBS (C-T-C sandwich) to cancel drift.

    Args:
        world: Simulation World object.
        target_id: Target workload ID.
        regime_id: Regime ID.
        probe_sets: List of probe sets (each a list of spectator IDs).
        n_samples: Samples per micro-run.
        rng: Random state.
        stat: Statistic to collect: "mean", "p99", or "cvar99".
        use_irbs: If True, use IRBS C-T-C sandwich for drift correction.

    Returns:
        Measurement vector y of shape (m,) in microseconds.

    Raises:
        ValueError: If stat is not recognized.
    """
    from sit.sim.world import simulate_micro_run
    from sit.measure.irbs import irbs_ctc_estimate

    valid_stats = {"mean", "p99", "cvar99"}
    if stat not in valid_stats:
        raise ValueError(f"Unknown stat '{stat}', must be one of {valid_stats}")

    m = len(probe_sets)
    y = np.zeros(m, dtype=np.float64)

    for i, probe in enumerate(probe_sets):
        t_index = i + 1  # Compact time spacing

        if use_irbs:
            # IRBS C-T-C: C1 at t*3, T at t*3+1, C2 at t*3+2
            t_ctc = t_index * 3
            c1_result = simulate_micro_run(
                world, target_id, [], regime_id, t_ctc, n_samples, rng,
            )
            t_result = simulate_micro_run(
                world, target_id, probe, regime_id, t_ctc + 1, n_samples, rng,
            )
            c2_result = simulate_micro_run(
                world, target_id, [], regime_id, t_ctc + 2, n_samples, rng,
            )

            # Use service_us to avoid queueing nonlinearity
            c1_stat = _extract_stat(c1_result.service_us, stat)
            t_stat = _extract_stat(t_result.service_us, stat)
            c2_stat = _extract_stat(c2_result.service_us, stat)

            y[i] = irbs_ctc_estimate(c1_stat, t_stat, c2_stat)
        else:
            # Simple difference: treatment - control at same time
            treat_result = simulate_micro_run(
                world, target_id, probe, regime_id, t_index, n_samples, rng,
            )
            ctrl_result = simulate_micro_run(
                world, target_id, [], regime_id, t_index, n_samples, rng,
            )

            # Use service_us to avoid queueing nonlinearity
            treat_stat = _extract_stat(treat_result.service_us, stat)
            ctrl_stat = _extract_stat(ctrl_result.service_us, stat)
            y[i] = treat_stat - ctrl_stat

    logger.info(
        "Collected %d measurements (stat=%s, irbs=%s): "
        "min=%.2f, max=%.2f, mean=%.2f us",
        m, stat, use_irbs,
        y.min(), y.max(), y.mean(),
    )

    return y


def _extract_stat(latencies_us: np.ndarray, stat: str) -> float:
    """Extract the specified statistic from latency samples."""
    if stat == "mean":
        return float(np.mean(latencies_us))
    elif stat == "p99":
        return estimate_var(latencies_us, alpha=0.99)
    elif stat == "cvar99":
        return estimate_cvar(latencies_us, alpha=0.99)
    else:
        raise ValueError(f"Unknown stat: {stat}")


def collect_ground_truth_vector(
    world: Any,
    target_id: str,
    regime_id: str,
    spectator_ids: List[str],
) -> np.ndarray:
    """Extract the ground truth interference vector x_true for a target.

    Args:
        world: Simulation World object.
        target_id: Target workload ID.
        regime_id: Regime ID.
        spectator_ids: Ordered list of spectator IDs.

    Returns:
        Ground truth vector x_true of shape (n,) in microseconds.
    """
    gt = world.ground_truth.get(regime_id, {})
    x_true = np.array(
        [gt.get((target_id, sid), 0.0) for sid in spectator_ids],
        dtype=np.float64,
    )
    return x_true
