"""World: complete simulation environment with ground truth.

A World encapsulates all workloads, regimes, drift parameters,
interference ground truth, and simulation configuration. It exposes
the core simulate_micro_run() primitive used by measurement and
evaluation code.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.core.units import CANONICAL_LATENCY_UNIT
from sit.sim.channels import DEFAULT_CHANNEL_WEIGHTS, channel_interference_decomposed
from sit.sim.drift import DriftParams, mu_drift_us, parse_drift_config
from sit.sim.interference import (
    InteractionConfig,
    ToxicPairConfig,
    compute_ground_truth_matrix,
    compute_set_interference_us,
    generate_toxic_pairs,
)
from sit.sim.latency import BurstConfig, generate_latencies
from sit.sim.queue import QueueConfig
from sit.sim.regimes import Regime, generate_regimes
from sit.sim.workload import Workload, generate_workloads

logger = get_logger("sim.world")


@dataclass
class MicroRunResult:
    """Result of a single simulated micro-run.

    All latencies in microseconds.
    """
    latencies_us: np.ndarray
    service_us: np.ndarray
    queue_us: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class World:
    """Complete simulation environment.

    Contains all workloads, regimes, drift params, ground truth matrices,
    and simulation configuration needed to run micro-experiments.
    """
    targets: List[Workload]
    spectators: List[Workload]
    regimes: List[Regime]
    drift_params: DriftParams
    queue_config: QueueConfig
    burst_config: BurstConfig
    channel_weights: Dict[str, float]
    channel_scale_us: float
    toxic_pair_config: ToxicPairConfig
    interaction_config: InteractionConfig
    toxic_pairs: Dict[Tuple[str, str], float] = field(default_factory=dict)
    # ground_truth[(target_id, spectator_id)] per regime
    ground_truth: Dict[str, Dict[Tuple[str, str], float]] = field(default_factory=dict)

    def get_target(self, target_id: str) -> Workload:
        for t in self.targets:
            if t.workload_id == target_id:
                return t
        raise KeyError(f"Target not found: {target_id}")

    def get_spectator(self, spectator_id: str) -> Workload:
        for s in self.spectators:
            if s.workload_id == spectator_id:
                return s
        raise KeyError(f"Spectator not found: {spectator_id}")

    def get_regime(self, regime_id: str) -> Regime:
        for r in self.regimes:
            if r.regime_id == regime_id:
                return r
        raise KeyError(f"Regime not found: {regime_id}")

    @property
    def spectators_by_id(self) -> Dict[str, Workload]:
        return {s.workload_id: s for s in self.spectators}


def build_world(config: Dict[str, Any], rng: np.random.RandomState) -> World:
    """Build a complete World from configuration.

    Creates workloads, regimes, toxic pairs, and ground truth matrices
    deterministically from the provided RNG.

    Args:
        config: Full config dict with 'sim' section.
        rng: Seeded random state.

    Returns:
        Fully initialized World.
    """
    sim_cfg = config.get("sim", {})
    slo_us = config.get("slo_us", 500000.0)

    n_targets = sim_cfg.get("n_targets", 5)
    n_spectators = sim_cfg.get("n_spectators", 20)
    n_regimes = sim_cfg.get("n_regimes", 2)

    # Generate workloads
    workloads = generate_workloads(n_targets, n_spectators, rng, slo_us=slo_us)
    targets = [w for w in workloads if w.role == "target"]
    spectators = [w for w in workloads if w.role == "spectator"]

    # Generate regimes
    regimes = generate_regimes(n_regimes, rng)

    # Parse drift
    drift_cfg = sim_cfg.get("drift", {"type": "none"})
    drift_params = parse_drift_config(drift_cfg)

    # Parse channel weights
    ch_cfg = sim_cfg.get("channels", {})
    channel_weights = ch_cfg.get("weights", DEFAULT_CHANNEL_WEIGHTS)
    channel_scale_us = float(ch_cfg.get("scale_us", 100.0))

    # Parse toxic pairs
    tp_cfg = sim_cfg.get("toxic_pairs", {})
    toxic_config = ToxicPairConfig(
        enabled=tp_cfg.get("enabled", True),
        sparsity=float(tp_cfg.get("sparsity", 0.05)),
        lognormal_mu=float(tp_cfg.get("lognormal_mu", 0.0)),
        lognormal_sigma=float(tp_cfg.get("lognormal_sigma", 0.7)),
    )

    # Parse queue
    q_cfg = sim_cfg.get("queue", {})
    queue_config = QueueConfig(
        enabled=q_cfg.get("enabled", True),
        concurrency=int(q_cfg.get("concurrency", 16)),
        think_time_us=float(q_cfg.get("think_time_us", 50.0)),
    )

    # Parse burst
    b_cfg = sim_cfg.get("burst", {})
    burst_config = BurstConfig(
        enabled=b_cfg.get("enabled", True),
        p_burst=float(b_cfg.get("p_burst", 0.002)),
        pareto_alpha=float(b_cfg.get("pareto_alpha", 2.5)),
        scale_us=float(b_cfg.get("scale_us", 5000.0)),
    )

    # Parse interactions
    i_cfg = sim_cfg.get("interactions", {})
    interaction_config = InteractionConfig(
        enabled=i_cfg.get("enabled", False),
        gamma=float(i_cfg.get("gamma", 0.01)),
    )

    # Generate toxic pairs
    toxic_pairs = generate_toxic_pairs(targets, spectators, toxic_config, rng)

    # Compute ground truth for each regime
    ground_truth = {}
    for regime in regimes:
        gt = compute_ground_truth_matrix(
            targets, spectators, regime, channel_weights, toxic_pairs,
            channel_scale_us=channel_scale_us,
        )
        ground_truth[regime.regime_id] = gt

    world = World(
        targets=targets,
        spectators=spectators,
        regimes=regimes,
        drift_params=drift_params,
        queue_config=queue_config,
        burst_config=burst_config,
        channel_weights=channel_weights,
        channel_scale_us=channel_scale_us,
        toxic_pair_config=toxic_config,
        interaction_config=interaction_config,
        toxic_pairs=toxic_pairs,
        ground_truth=ground_truth,
    )

    logger.info(
        "World built: %d targets, %d spectators, %d regimes, %d toxic pairs",
        len(targets), len(spectators), len(regimes), len(toxic_pairs),
    )

    return world


def simulate_micro_run(
    world: World,
    target_id: str,
    spectator_ids: List[str],
    regime_id: str,
    t_index: int,
    n_samples: int,
    rng: np.random.RandomState,
) -> MicroRunResult:
    """Simulate a single micro-run: generate latency samples.

    This is the core simulation primitive. It:
    1. Looks up the target, spectators, and regime
    2. Computes drift at the given time index
    3. Computes total interference from ground truth
    4. Generates service times with appropriate distributions
    5. Optionally adds queueing delay

    Args:
        world: The simulation world.
        target_id: ID of the target workload.
        spectator_ids: List of co-located spectator IDs.
        regime_id: ID of the regime to use.
        t_index: Time index for drift computation.
        n_samples: Number of latency samples to generate.
        rng: Random state for reproducibility.

    Returns:
        MicroRunResult with latency arrays and metadata.
    """
    target = world.get_target(target_id)
    regime = world.get_regime(regime_id)
    gt = world.ground_truth.get(regime_id, {})

    # Compute drift
    drift_us = mu_drift_us(t_index, world.drift_params)

    # Compute total interference
    interference_us = compute_set_interference_us(
        target=target,
        spectator_ids=spectator_ids,
        ground_truth=gt,
        spectators_by_id=world.spectators_by_id,
        interaction_config=world.interaction_config,
    )

    # Generate latencies
    result = generate_latencies(
        target=target,
        interference_us=interference_us,
        drift_us=drift_us,
        n_samples=n_samples,
        rng=rng,
        drift_params=world.drift_params,
        t_index=t_index,
        burst_config=world.burst_config,
        queue_config=world.queue_config,
    )

    # Build per-channel decomposition for first spectator (metadata)
    channel_decomp = {}
    for sid in spectator_ids:
        spec = world.get_spectator(sid)
        decomp = channel_interference_decomposed(
            target, spec, regime, world.channel_weights,
            scale_us=world.channel_scale_us,
        )
        for ch, val in decomp.items():
            channel_decomp[ch] = channel_decomp.get(ch, 0.0) + val

    # Unit assertion
    assert CANONICAL_LATENCY_UNIT == "us"

    metadata = {
        **result["metadata"],
        "target_id": target_id,
        "spectator_ids": spectator_ids,
        "regime_id": regime_id,
        "t_index": t_index,
        "channel_decomposition_us": channel_decomp,
        "unit": CANONICAL_LATENCY_UNIT,
    }

    return MicroRunResult(
        latencies_us=result["latencies_us"],
        service_us=result["service_us"],
        queue_us=result["queue_us"],
        metadata=metadata,
    )


def ground_truth_to_dataframe(world: World) -> pd.DataFrame:
    """Export ground truth interference matrix as a DataFrame.

    One row per (target, spectator, regime) triple.
    """
    rows = []
    for regime in world.regimes:
        gt = world.ground_truth.get(regime.regime_id, {})
        for (tid, sid), x_us in gt.items():
            # Get toxic component
            toxic = world.toxic_pairs.get((tid, sid), 0.0)
            channel = x_us - toxic  # channel component

            rows.append({
                "target_id": tid,
                "spectator_id": sid,
                "regime_id": regime.regime_id,
                "x_total_us": x_us,
                "x_channel_us": channel,
                "x_toxic_us": toxic,
                "unit": CANONICAL_LATENCY_UNIT,
            })

    return pd.DataFrame(rows)
