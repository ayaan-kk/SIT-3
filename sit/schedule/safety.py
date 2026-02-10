"""Safety enforcement and fallback mechanisms.

Implements fallback strategies when no feasible placement exists,
and safety monitoring for scheduling decisions.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sit.core.logging import get_logger
from sit.schedule.state import Host, Job, Placement

logger = get_logger("schedule.safety")


def fallback_partition(
    target_job: Job,
    hosts: List[Host],
    current_placement: Placement,
) -> Optional[str]:
    """Fallback: place target on least-loaded host with no spectators.

    Tries to find a host that has only targets or is empty.
    If none available, picks the host with fewest spectators.

    Args:
        target_job: The target job needing placement.
        hosts: Available hosts.
        current_placement: Current placement state.

    Returns:
        host_id for fallback placement, or None if impossible.
    """
    # First pass: find host with no spectators and available capacity
    best_host = None
    min_spectators = float("inf")

    for host in hosts:
        if not host.can_accept():
            continue

        # Count spectator jobs on this host
        host_jobs = current_placement.host_jobs(host.host_id)
        n_specs = 0
        for jid in host_jobs:
            # Check if it's a spectator job by the naming convention
            if "_s_" in jid:
                n_specs += 1

        if n_specs < min_spectators:
            min_spectators = n_specs
            best_host = host

    if best_host is not None:
        return best_host.host_id
    return None


def fallback_admission_control(
    target_job: Job,
    hosts: List[Host],
) -> Optional[str]:
    """Fallback: place on host with most remaining capacity.

    Args:
        target_job: The target job.
        hosts: Available hosts.

    Returns:
        host_id or None.
    """
    best = None
    best_remaining = -1

    for host in hosts:
        if host.can_accept() and host.remaining > best_remaining:
            best_remaining = host.remaining
            best = host

    return best.host_id if best else None


def apply_fallback(
    target_job: Job,
    hosts: List[Host],
    current_placement: Placement,
) -> Tuple[Optional[str], str]:
    """Apply fallback strategy for infeasible target placement.

    Tries partition first, then admission control.

    Args:
        target_job: The target job needing placement.
        hosts: All hosts.
        current_placement: Current placement.

    Returns:
        Tuple of (host_id, fallback_type) where fallback_type
        is "partition" or "admission_control" or "none".
    """
    # Try partition: find host with fewest spectators
    host_id = fallback_partition(target_job, hosts, current_placement)
    if host_id is not None:
        logger.info(
            "Fallback partition for %s -> %s", target_job.job_id, host_id
        )
        return host_id, "partition"

    # Last resort: admission control
    host_id = fallback_admission_control(target_job, hosts)
    if host_id is not None:
        logger.info(
            "Fallback admission control for %s -> %s", target_job.job_id, host_id
        )
        return host_id, "admission_control"

    return None, "none"
