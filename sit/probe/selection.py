"""Probe set selection strategies.

Implements uniform random and coverage-aware probe generation for
tomography design matrices.
"""

from typing import Dict, List, Tuple

import numpy as np

from sit.core.logging import get_logger
from sit.probe.budget import ProbeBudget

logger = get_logger("probe.selection")


def uniform_random_probes(
    spectator_ids: List[str],
    budget: ProbeBudget,
    rng: np.random.RandomState,
) -> List[List[str]]:
    """Generate uniformly random probe sets.

    Each probe is a random subset of spectators of size budget.set_size,
    sampled without replacement within each probe.

    Args:
        spectator_ids: List of all spectator IDs.
        budget: Probe budget constraints.
        rng: Random state for reproducibility.

    Returns:
        List of probe sets (each a sorted list of spectator IDs).
    """
    n = len(spectator_ids)
    probes = []
    for _ in range(budget.m_probes):
        size = min(budget.set_size, n)
        idx = rng.choice(n, size=size, replace=False)
        probe = sorted([spectator_ids[i] for i in idx])
        probes.append(probe)
    return probes


def coverage_aware_probes(
    spectator_ids: List[str],
    budget: ProbeBudget,
    rng: np.random.RandomState,
) -> List[List[str]]:
    """Generate coverage-aware probe sets.

    Greedily selects spectators with lowest coverage counts to ensure
    uniform representation across all spectators, subject to max_repeats.

    Args:
        spectator_ids: List of all spectator IDs.
        budget: Probe budget constraints.
        rng: Random state for reproducibility.

    Returns:
        List of probe sets (each a sorted list of spectator IDs).
    """
    n = len(spectator_ids)
    coverage: Dict[str, int] = {sid: 0 for sid in spectator_ids}
    probes = []

    for _ in range(budget.m_probes):
        size = min(budget.set_size, n)

        # Get eligible spectators (under max_repeats)
        eligible = [sid for sid in spectator_ids if coverage[sid] < budget.max_repeats]
        if len(eligible) < size:
            # Relax: allow any spectator
            eligible = list(spectator_ids)

        # Sort by coverage (ascending), break ties randomly
        # Add small random noise for tie-breaking
        scores = np.array([coverage[sid] + rng.uniform(0, 0.1) for sid in eligible])
        sorted_idx = np.argsort(scores)

        # Pick the `size` least-covered spectators
        selected = [eligible[sorted_idx[i]] for i in range(min(size, len(sorted_idx)))]

        for sid in selected:
            coverage[sid] += 1

        probes.append(sorted(selected))

    logger.info(
        "Coverage-aware probes: %d probes, coverage min=%d max=%d mean=%.1f",
        len(probes),
        min(coverage.values()),
        max(coverage.values()),
        np.mean(list(coverage.values())),
    )

    return probes


def compute_coverage(
    probes: List[List[str]],
    spectator_ids: List[str],
) -> Dict[str, int]:
    """Compute per-spectator coverage counts from probe sets."""
    coverage = {sid: 0 for sid in spectator_ids}
    for probe in probes:
        for sid in probe:
            if sid in coverage:
                coverage[sid] += 1
    return coverage
