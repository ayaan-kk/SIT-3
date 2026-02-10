"""Local search improvement for scheduling placements.

Starts from a greedy placement and performs swap moves to
reduce the objective while maintaining safety constraints.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sit.core.logging import get_logger
from sit.schedule.constraints import SafetyConfig, check_safety
from sit.schedule.objectives import predicted_risk, predicted_uncertainty, robust_risk, redundancy_penalty
from sit.schedule.state import Episode, Host, Job, Placement

logger = get_logger("schedule.search")


def _compute_target_score(
    target_job: Job,
    host: Host,
    placement: Placement,
    jobs_by_id: Dict[str, Job],
    episode: Episode,
    x_hat: Dict,
    sigma: Dict,
    safety_config: SafetyConfig,
    K: Optional[np.ndarray],
    spectator_id_to_idx: Optional[Dict[str, int]],
    lambda_div: float,
) -> Tuple[float, float]:
    """Compute score and robust risk for a target on a host.

    Returns:
        (score, robust_risk_us)
    """
    co_tenant_ids = []
    for jid in placement.host_jobs(host.host_id):
        j = jobs_by_id.get(jid)
        if j and j.role == "spectator":
            co_tenant_ids.append(j.workload_id)

    r_hat = predicted_risk(
        target_job.workload_id, co_tenant_ids,
        episode.regime_id, x_hat,
    )
    u = predicted_uncertainty(
        target_job.workload_id, co_tenant_ids,
        episode.regime_id, sigma,
    )
    r_ucb = r_hat + safety_config.beta_ucb * u

    penalty = redundancy_penalty(
        co_tenant_ids, co_tenant_ids, K, spectator_id_to_idx,
    )

    score = r_ucb + lambda_div * penalty
    return score, r_ucb


def local_search_improve(
    placement: Placement,
    episode: Episode,
    x_hat: Dict,
    sigma: Dict,
    safety_config: SafetyConfig,
    K: Optional[np.ndarray] = None,
    spectator_id_to_idx: Optional[Dict[str, int]] = None,
    lambda_div: float = 0.5,
    max_iters: int = 50,
) -> Tuple[Placement, int]:
    """Improve placement via local swap search.

    Tries swapping pairs of jobs between hosts to improve the
    objective while maintaining safety constraints.

    Args:
        placement: Initial placement to improve.
        episode: Episode being scheduled.
        x_hat: Interference estimates.
        sigma: Uncertainty estimates.
        safety_config: Safety config.
        K: Kernel matrix for diversity.
        spectator_id_to_idx: Spectator ID to kernel index mapping.
        lambda_div: Diversity penalty weight.
        max_iters: Maximum swap iterations.

    Returns:
        Tuple of (improved_placement, n_swaps_performed).
    """
    jobs_by_id = {j.job_id: j for j in episode.jobs}
    target_job_ids = [j.job_id for j in episode.target_jobs]
    hosts = episode.fresh_hosts()

    # Rebuild host assignments from placement
    for host in hosts:
        for jid, hid in placement.mapping.items():
            if hid == host.host_id:
                host.assigned.append(jid)

    improved = Placement(mapping=dict(placement.mapping))
    n_swaps = 0

    def total_objective():
        """Sum of scores for all targets."""
        total = 0.0
        for tjid in target_job_ids:
            tj = jobs_by_id[tjid]
            hid = improved.mapping.get(tjid)
            if hid is None:
                continue
            host = next((h for h in hosts if h.host_id == hid), None)
            if host is None:
                continue
            score, _ = _compute_target_score(
                tj, host, improved, jobs_by_id, episode,
                x_hat, sigma, safety_config, K, spectator_id_to_idx, lambda_div,
            )
            total += score
        return total

    current_obj = total_objective()

    for _ in range(max_iters):
        improved_this_iter = False

        # Try swapping each target with a spectator on another host
        for t_jid in target_job_ids:
            t_host_id = improved.mapping.get(t_jid)
            if t_host_id is None:
                continue

            # Get spectators on other hosts
            for s_jid, s_host_id in list(improved.mapping.items()):
                s_job = jobs_by_id.get(s_jid)
                if s_job is None or s_job.role != "spectator":
                    continue
                if s_host_id == t_host_id:
                    continue

                # Try swap: move target to spectator's host, spectator to target's host
                improved.mapping[t_jid] = s_host_id
                improved.mapping[s_jid] = t_host_id

                # Check capacity (swaps preserve capacity)
                # Check safety for target at new host
                t_job = jobs_by_id[t_jid]
                new_host = next(h for h in hosts if h.host_id == s_host_id)
                _, r_ucb = _compute_target_score(
                    t_job, new_host, improved, jobs_by_id, episode,
                    x_hat, sigma, safety_config, K, spectator_id_to_idx, lambda_div,
                )

                if not check_safety(r_ucb, safety_config.tau_risk_us):
                    # Revert swap
                    improved.mapping[t_jid] = t_host_id
                    improved.mapping[s_jid] = s_host_id
                    continue

                new_obj = total_objective()
                if new_obj < current_obj - 1e-6:
                    current_obj = new_obj
                    n_swaps += 1
                    improved_this_iter = True
                    # Update hosts
                    for host in hosts:
                        host.assigned = [
                            jid for jid, hid in improved.mapping.items()
                            if hid == host.host_id
                        ]
                    break  # Restart target loop
                else:
                    # Revert swap
                    improved.mapping[t_jid] = t_host_id
                    improved.mapping[s_jid] = s_host_id

            if improved_this_iter:
                break

        if not improved_this_iter:
            break

    if n_swaps > 0:
        logger.info("Local search: %d swaps, obj %.2f -> %.2f",
                     n_swaps, current_obj + n_swaps, current_obj)

    return improved, n_swaps
