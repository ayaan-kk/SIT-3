"""Probe budget management.

Defines the budget constraints for tomography probe experiments:
how many probes, spectator set size, and repeat limits.
"""

from dataclasses import dataclass


@dataclass
class ProbeBudget:
    """Budget constraints for probe generation.

    Attributes:
        m_probes: Total number of probe sets to generate.
        set_size: Number of spectators per probe set.
        max_repeats: Maximum times a single spectator appears across probes.
    """
    m_probes: int = 60
    set_size: int = 3
    max_repeats: int = 20

    def __post_init__(self):
        assert self.m_probes > 0, f"m_probes must be positive, got {self.m_probes}"
        assert self.set_size > 0, f"set_size must be positive, got {self.set_size}"
        assert self.max_repeats > 0, f"max_repeats must be positive, got {self.max_repeats}"
