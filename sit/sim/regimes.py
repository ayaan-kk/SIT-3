"""Regime definitions for SIT simulation.

A regime captures environmental context: device class, load level,
topology distance, and time bucket. Regimes modulate interference
channel weights.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Regime:
    """Environmental context for a simulation scenario.

    Attributes:
        regime_id: Unique identifier string.
        device_class: Hardware class descriptor (e.g., "cpu_server").
        load_level: Normalized system load in [0, 1].
        topology_distance: Distance metric (e.g., NUMA hops). 0 = same core.
        time_bucket: Discrete time period index.
    """
    regime_id: str
    device_class: str = "cpu_server"
    load_level: float = 0.5
    topology_distance: float = 1.0
    time_bucket: int = 0


def generate_regimes(
    n_regimes: int,
    rng: np.random.RandomState,
) -> List[Regime]:
    """Generate deterministic regimes with varied load and topology."""
    device_classes = ["cpu_server", "gpu_node", "mem_server"]
    regimes = []

    for i in range(n_regimes):
        regimes.append(Regime(
            regime_id=f"regime_{i}",
            device_class=device_classes[i % len(device_classes)],
            load_level=rng.uniform(0.2, 0.9),
            topology_distance=rng.choice([0.0, 1.0, 2.0]),
            time_bucket=i,
        ))

    return regimes
