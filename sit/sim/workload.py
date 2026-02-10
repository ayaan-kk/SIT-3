"""Workload definitions for SIT simulation.

Each workload has a role (target or spectator), a feature vector capturing
resource intensity across channels, and base service time parameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# Canonical channel names used throughout the simulator
CHANNEL_NAMES: List[str] = ["LLC", "MEM_BW", "IO", "TLB", "SMT"]
N_CHANNELS: int = len(CHANNEL_NAMES)


@dataclass
class Workload:
    """A workload (target or spectator) in the simulation.

    Attributes:
        workload_id: Unique identifier string.
        role: Either "target" or "spectator".
        features: Resource intensity vector [cpu, mem, io, tlb, branch] in [0,1].
        base_service_us_mean: Mean base service time in microseconds.
        base_service_us_cv: Coefficient of variation for base service time.
        slo_us: SLO threshold (only meaningful for targets).
        tail_sensitivity: Per-channel sensitivity vector alpha_t,k in [0,1].
    """
    workload_id: str
    role: str  # "target" or "spectator"
    features: np.ndarray = field(default_factory=lambda: np.zeros(N_CHANNELS))
    base_service_us_mean: float = 100.0
    base_service_us_cv: float = 0.3
    slo_us: float = 500000.0
    tail_sensitivity: np.ndarray = field(default_factory=lambda: np.zeros(N_CHANNELS))

    def __post_init__(self):
        self.features = np.asarray(self.features, dtype=np.float64)
        self.tail_sensitivity = np.asarray(self.tail_sensitivity, dtype=np.float64)
        assert self.role in ("target", "spectator"), f"Invalid role: {self.role}"
        assert len(self.features) == N_CHANNELS, f"Features must have {N_CHANNELS} dims"
        assert len(self.tail_sensitivity) == N_CHANNELS
        assert self.base_service_us_mean > 0
        assert self.base_service_us_cv >= 0


def generate_workloads(
    n_targets: int,
    n_spectators: int,
    rng: np.random.RandomState,
    slo_us: float = 500000.0,
) -> List[Workload]:
    """Generate a deterministic set of target and spectator workloads.

    Targets get higher tail sensitivity (alpha) and moderate features.
    Spectators get diverse feature vectors representing different
    resource pressure profiles.
    """
    workloads = []

    for i in range(n_targets):
        features = rng.uniform(0.2, 0.8, size=N_CHANNELS)
        # Targets: moderate features but high tail sensitivity
        tail_sens = rng.uniform(0.4, 1.0, size=N_CHANNELS)
        base_mean = rng.uniform(80.0, 200.0)
        base_cv = rng.uniform(0.15, 0.45)

        workloads.append(Workload(
            workload_id=f"target_{i}",
            role="target",
            features=features,
            base_service_us_mean=base_mean,
            base_service_us_cv=base_cv,
            slo_us=slo_us,
            tail_sensitivity=tail_sens,
        ))

    for i in range(n_spectators):
        features = rng.uniform(0.0, 1.0, size=N_CHANNELS)
        # Spectators don't need tail_sensitivity for scheduling,
        # but we store it for completeness
        tail_sens = rng.uniform(0.0, 0.5, size=N_CHANNELS)
        base_mean = rng.uniform(50.0, 500.0)
        base_cv = rng.uniform(0.1, 0.6)

        workloads.append(Workload(
            workload_id=f"spec_{i}",
            role="spectator",
            features=features,
            base_service_us_mean=base_mean,
            base_service_us_cv=base_cv,
            slo_us=slo_us,
            tail_sensitivity=tail_sens,
        ))

    return workloads
