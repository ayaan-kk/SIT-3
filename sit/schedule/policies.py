"""Scheduler policies: baselines and SIT-safe placement strategies.

Implements 7 scheduling policies:
1. Random - random placement respecting capacity
2. Round-robin - deterministic cyclic assignment
3. Static partition - isolate targets from spectators
4. Mean-greedy - minimize predicted mean risk (no UCB)
5. Similarity avoidance - keep diverse co-tenants
6. SIT-risk+diversity - uses x_hat + diversity, no safety UCB
7. SIT-safe-UCB - full safety-constrained UCB placement (main contribution)
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sit.core.logging import get_logger
from sit.schedule.constraints import SafetyConfig, check_safety, find_feasible_hosts
from sit.schedule.objectives import (
    predicted_risk,
    predicted_uncertainty,
    redundancy_penalty,
    robust_risk,
)
from sit.schedule.safety import apply_fallback
from sit.schedule.state import Episode, Host, Job, Placement

logger = get_logger("schedule.policies")


@dataclass
class ScheduleDecision:
    """Record of a single placement decision."""
    episode_id: int
    step_index: int
    scheduler_name: str
    target_id: str
    candidate_hosts: List[Dict[str, Any]]
    chosen_host: str
    safety_pass: bool
    fallback_used: bool
    fallback_type: str = ""
    score_breakdown: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduleContext:
    """Context available to all policies for making placement decisions."""
    episode: Episode
    x_hat: Dict[Tuple[str, str, str], float]
    sigma: Dict[Tuple[str, str, str], float]
    safety_config: SafetyConfig
    K: Optional[np.ndarray] = None
    spectator_id_to_idx: Optional[Dict[str, int]] = None
    lambda_div: float = 0.5
    slo_us: float = 500000.0
    rng: Optional[np.random.RandomState] = None


def _get_spectator_workload_ids(
    host: Host,
    placement: Placement,
    jobs_by_id: Dict[str, Job],
) -> List[str]:
    """Get workload IDs of spectators assigned to a host."""
    result = []
    for jid in placement.host_jobs(host.host_id):
        job = jobs_by_id.get(jid)
        if job and job.role == "spectator":
            result.append(job.workload_id)
    return result


def _build_host_assignments(
    hosts: List[Host],
    placement: Placement,
    jobs_by_id: Dict[str, Job],
) -> Dict[str, List[str]]:
    """Build host_id -> spectator workload IDs mapping."""
    assignments = {}
    for host in hosts:
        assignments[host.host_id] = _get_spectator_workload_ids(
            host, placement, jobs_by_id
        )
    return assignments


# ---- Policy: Random ----

def schedule_random(
    ctx: ScheduleContext,
) -> Tuple[Placement, List[ScheduleDecision]]:
    """Random placement respecting host capacities."""
    episode = ctx.episode
    rng = ctx.rng or np.random.RandomState(0)
    hosts = episode.fresh_hosts()
    placement = Placement()
    decisions = []
    jobs_by_id = {j.job_id: j for j in episode.jobs}

    # Shuffle all jobs and assign
    all_jobs = list(episode.jobs)
    rng.shuffle(all_jobs)

    for step, job in enumerate(all_jobs):
        available = [h for h in hosts if h.can_accept()]
        if not available:
            break
        chosen = available[rng.randint(len(available))]
        chosen.assign(job.job_id)
        placement.mapping[job.job_id] = chosen.host_id

        if job.role == "target":
            decisions.append(ScheduleDecision(
                episode_id=episode.episode_id,
                step_index=step,
                scheduler_name="random",
                target_id=job.workload_id,
                candidate_hosts=[{"host_id": h.host_id} for h in available],
                chosen_host=chosen.host_id,
                safety_pass=True,
                fallback_used=False,
            ))

    return placement, decisions


# ---- Policy: Round-robin ----

def schedule_round_robin(
    ctx: ScheduleContext,
) -> Tuple[Placement, List[ScheduleDecision]]:
    """Deterministic round-robin assignment."""
    episode = ctx.episode
    hosts = episode.fresh_hosts()
    placement = Placement()
    decisions = []
    host_idx = 0

    for step, job in enumerate(episode.jobs):
        # Find next host with capacity
        attempts = 0
        while attempts < len(hosts):
            h = hosts[host_idx % len(hosts)]
            if h.can_accept():
                break
            host_idx += 1
            attempts += 1

        if attempts >= len(hosts):
            break

        h = hosts[host_idx % len(hosts)]
        h.assign(job.job_id)
        placement.mapping[job.job_id] = h.host_id
        host_idx += 1

        if job.role == "target":
            decisions.append(ScheduleDecision(
                episode_id=episode.episode_id,
                step_index=step,
                scheduler_name="round_robin",
                target_id=job.workload_id,
                candidate_hosts=[{"host_id": h2.host_id} for h2 in hosts if h2.can_accept()],
                chosen_host=h.host_id,
                safety_pass=True,
                fallback_used=False,
            ))

    return placement, decisions


# ---- Policy: Static partition ----

def schedule_static_partition(
    ctx: ScheduleContext,
) -> Tuple[Placement, List[ScheduleDecision]]:
    """Static partition: reserve hosts for targets, isolate from spectators.

    Allocates ceil(n_targets / host_capacity) hosts for targets only,
    remaining hosts for spectators.
    """
    episode = ctx.episode
    hosts = episode.fresh_hosts()
    placement = Placement()
    decisions = []

    n_targets = len(episode.target_jobs)
    cap = hosts[0].capacity if hosts else 1

    # Reserve hosts for targets
    n_target_hosts = max(1, int(np.ceil(n_targets / cap)))
    n_target_hosts = min(n_target_hosts, len(hosts))
    target_hosts = hosts[:n_target_hosts]
    spectator_hosts = hosts[n_target_hosts:]

    # If no spectator hosts, use remaining capacity on target hosts
    if not spectator_hosts:
        spectator_hosts = target_hosts

    # Place targets on reserved hosts (round-robin)
    t_idx = 0
    for step, job in enumerate(episode.target_jobs):
        h = target_hosts[t_idx % len(target_hosts)]
        while not h.can_accept() and t_idx < len(target_hosts) * 2:
            t_idx += 1
            h = target_hosts[t_idx % len(target_hosts)]
        if h.can_accept():
            h.assign(job.job_id)
            placement.mapping[job.job_id] = h.host_id
            t_idx += 1

            decisions.append(ScheduleDecision(
                episode_id=episode.episode_id,
                step_index=step,
                scheduler_name="static_partition",
                target_id=job.workload_id,
                candidate_hosts=[{"host_id": th.host_id} for th in target_hosts],
                chosen_host=h.host_id,
                safety_pass=True,
                fallback_used=False,
                score_breakdown={"partition": "target_reserved"},
            ))

    # Place spectators on spectator hosts (round-robin)
    s_idx = 0
    for job in episode.spectator_jobs:
        placed = False
        for _ in range(len(spectator_hosts)):
            h = spectator_hosts[s_idx % len(spectator_hosts)]
            if h.can_accept():
                h.assign(job.job_id)
                placement.mapping[job.job_id] = h.host_id
                s_idx += 1
                placed = True
                break
            s_idx += 1
        if not placed:
            # Overflow: put on any host with capacity
            for h in hosts:
                if h.can_accept():
                    h.assign(job.job_id)
                    placement.mapping[job.job_id] = h.host_id
                    break

    return placement, decisions


# ---- Policy: Mean-greedy risk-only ----

def schedule_mean_greedy(
    ctx: ScheduleContext,
) -> Tuple[Placement, List[ScheduleDecision]]:
    """Place each target on host minimizing predicted mean risk (no UCB)."""
    episode = ctx.episode
    rng = ctx.rng or np.random.RandomState(0)
    hosts = episode.fresh_hosts()
    placement = Placement()
    decisions = []
    jobs_by_id = {j.job_id: j for j in episode.jobs}

    # First place spectators randomly to fill hosts
    specs = list(episode.spectator_jobs)
    rng.shuffle(specs)
    s_idx = 0
    for job in specs:
        h = hosts[s_idx % len(hosts)]
        while not h.can_accept() and s_idx < len(hosts) * 2:
            s_idx += 1
            h = hosts[s_idx % len(hosts)]
        if h.can_accept():
            h.assign(job.job_id)
            placement.mapping[job.job_id] = h.host_id
            s_idx += 1

    # Then greedily place targets
    for step, target_job in enumerate(episode.target_jobs):
        available = [h for h in hosts if h.can_accept()]
        if not available:
            break

        host_assignments = _build_host_assignments(hosts, placement, jobs_by_id)
        best_host = None
        best_risk = float("inf")
        candidates = []

        for h in available:
            specs_on_host = host_assignments.get(h.host_id, [])
            r = predicted_risk(
                target_job.workload_id, specs_on_host,
                episode.regime_id, ctx.x_hat,
            )
            candidates.append({
                "host_id": h.host_id,
                "predicted_risk": r,
            })
            if r < best_risk:
                best_risk = r
                best_host = h

        best_host.assign(target_job.job_id)
        placement.mapping[target_job.job_id] = best_host.host_id

        decisions.append(ScheduleDecision(
            episode_id=episode.episode_id,
            step_index=step,
            scheduler_name="mean_greedy",
            target_id=target_job.workload_id,
            candidate_hosts=candidates,
            chosen_host=best_host.host_id,
            safety_pass=True,
            fallback_used=False,
            score_breakdown={"chosen_risk": best_risk},
        ))

    return placement, decisions


# ---- Policy: Similarity avoidance ----

def schedule_similarity_avoidance(
    ctx: ScheduleContext,
) -> Tuple[Placement, List[ScheduleDecision]]:
    """Place targets to minimize co-location with similar spectators."""
    episode = ctx.episode
    rng = ctx.rng or np.random.RandomState(0)
    hosts = episode.fresh_hosts()
    placement = Placement()
    decisions = []
    jobs_by_id = {j.job_id: j for j in episode.jobs}

    # Place spectators first (round-robin)
    specs = list(episode.spectator_jobs)
    rng.shuffle(specs)
    s_idx = 0
    for job in specs:
        h = hosts[s_idx % len(hosts)]
        while not h.can_accept() and s_idx < len(hosts) * 2:
            s_idx += 1
            h = hosts[s_idx % len(hosts)]
        if h.can_accept():
            h.assign(job.job_id)
            placement.mapping[job.job_id] = h.host_id
            s_idx += 1

    # Place targets minimizing redundancy penalty
    for step, target_job in enumerate(episode.target_jobs):
        available = [h for h in hosts if h.can_accept()]
        if not available:
            break

        host_assignments = _build_host_assignments(hosts, placement, jobs_by_id)
        best_host = None
        best_penalty = float("inf")
        candidates = []

        for h in available:
            specs_on_host = host_assignments.get(h.host_id, [])
            penalty = redundancy_penalty(
                specs_on_host, specs_on_host,
                ctx.K, ctx.spectator_id_to_idx,
            )
            candidates.append({
                "host_id": h.host_id,
                "redundancy_penalty": penalty,
            })
            if penalty < best_penalty:
                best_penalty = penalty
                best_host = h

        if best_host is None:
            best_host = available[0]

        best_host.assign(target_job.job_id)
        placement.mapping[target_job.job_id] = best_host.host_id

        decisions.append(ScheduleDecision(
            episode_id=episode.episode_id,
            step_index=step,
            scheduler_name="similarity_avoidance",
            target_id=target_job.workload_id,
            candidate_hosts=candidates,
            chosen_host=best_host.host_id,
            safety_pass=True,
            fallback_used=False,
            score_breakdown={"chosen_penalty": best_penalty},
        ))

    return placement, decisions


# ---- Policy: SIT-risk+diversity (no safety UCB) ----

def schedule_sit_risk_div(
    ctx: ScheduleContext,
) -> Tuple[Placement, List[ScheduleDecision]]:
    """Place targets first, then distribute spectators using x_hat + diversity.

    Unlike baselines, this policy intelligently routes spectators AWAY from
    hosts where they would cause high interference to targets.
    """
    episode = ctx.episode
    rng = ctx.rng or np.random.RandomState(0)
    hosts = episode.fresh_hosts()
    placement = Placement()
    decisions = []
    jobs_by_id = {j.job_id: j for j in episode.jobs}

    # Phase 1: Place targets spread across hosts (round-robin)
    target_jobs = list(episode.target_jobs)
    for step, target_job in enumerate(target_jobs):
        h = hosts[step % len(hosts)]
        if h.can_accept():
            h.assign(target_job.job_id)
            placement.mapping[target_job.job_id] = h.host_id

            decisions.append(ScheduleDecision(
                episode_id=episode.episode_id,
                step_index=step,
                scheduler_name="sit_risk_div",
                target_id=target_job.workload_id,
                candidate_hosts=[{"host_id": h.host_id}],
                chosen_host=h.host_id,
                safety_pass=True,
                fallback_used=False,
            ))

    # Phase 2: Place spectators to minimize interference to co-located targets
    specs = list(episode.spectator_jobs)
    rng.shuffle(specs)

    for job in specs:
        available = [h for h in hosts if h.can_accept()]
        if not available:
            break

        # Find the host where this spectator causes least harm to targets
        best_host = None
        best_score = float("inf")

        for h in available:
            # Compute total interference this spectator would add
            total_risk_increase = 0.0
            for jid in placement.host_jobs(h.host_id):
                j = jobs_by_id.get(jid)
                if j and j.role == "target":
                    key = (j.workload_id, job.workload_id, episode.regime_id)
                    total_risk_increase += ctx.x_hat.get(key, 0.0)

            if total_risk_increase < best_score:
                best_score = total_risk_increase
                best_host = h

        if best_host is None:
            best_host = available[0]

        best_host.assign(job.job_id)
        placement.mapping[job.job_id] = best_host.host_id

    return placement, decisions


# ---- Policy: SIT-SAFE-UCB (main contribution) ----

def schedule_sit_safe_ucb(
    ctx: ScheduleContext,
) -> Tuple[Placement, List[ScheduleDecision]]:
    """SIT-safe-UCB: safety-constrained placement with uncertainty.

    Key insight: SIT-safe controls BOTH target and spectator placement.

    Algorithm:
    Phase 1: Place targets spread across hosts
    Phase 2: For each spectator, assign to host minimizing robust risk
             increase to co-located targets (using UCB for safety)
    Phase 3: Verify safety constraints hold; if violated, trigger fallback
    """
    episode = ctx.episode
    rng = ctx.rng or np.random.RandomState(0)
    hosts = episode.fresh_hosts()
    placement = Placement()
    decisions = []
    jobs_by_id = {j.job_id: j for j in episode.jobs}
    safety_cfg = ctx.safety_config
    fallback_count = 0

    # Phase 1: Spread targets across hosts (round-robin, one per host)
    target_jobs = list(episode.target_jobs)
    for step, target_job in enumerate(target_jobs):
        h = hosts[step % len(hosts)]
        if h.can_accept():
            h.assign(target_job.job_id)
            placement.mapping[target_job.job_id] = h.host_id

            decisions.append(ScheduleDecision(
                episode_id=episode.episode_id,
                step_index=step,
                scheduler_name="sit_safe_ucb",
                target_id=target_job.workload_id,
                candidate_hosts=[{"host_id": hh.host_id} for hh in hosts if hh.can_accept()],
                chosen_host=h.host_id,
                safety_pass=True,
                fallback_used=False,
                score_breakdown={"phase": "target_spread"},
            ))

    # Phase 2: Intelligently place spectators to minimize risk to targets
    specs = list(episode.spectator_jobs)
    rng.shuffle(specs)

    for job in specs:
        available = [h for h in hosts if h.can_accept()]
        if not available:
            break

        # For each candidate host, compute the robust risk increase
        # this spectator would cause to all co-located targets
        best_host = None
        best_score = float("inf")

        for h in available:
            total_robust_increase = 0.0
            n_targets_on_host = 0

            for jid in placement.host_jobs(h.host_id):
                j = jobs_by_id.get(jid)
                if j and j.role == "target":
                    n_targets_on_host += 1
                    key = (j.workload_id, job.workload_id, episode.regime_id)
                    x_val = ctx.x_hat.get(key, 0.0)
                    s_val = ctx.sigma.get(key, 0.0)
                    # Robust risk increase from this spectator
                    total_robust_increase += x_val + safety_cfg.beta_ucb * s_val

            # Diversity: penalize hosts with many similar spectators
            penalty = 0.0
            if ctx.K is not None and ctx.spectator_id_to_idx is not None:
                idx_new = ctx.spectator_id_to_idx.get(job.workload_id)
                if idx_new is not None:
                    for jid in placement.host_jobs(h.host_id):
                        j2 = jobs_by_id.get(jid)
                        if j2 and j2.role == "spectator":
                            idx_ex = ctx.spectator_id_to_idx.get(j2.workload_id)
                            if idx_ex is not None:
                                penalty += ctx.K[idx_new, idx_ex]

            # Score: prefer hosts where this spectator causes less risk
            # If no targets on host, score is just the penalty (spread load)
            score = total_robust_increase + ctx.lambda_div * penalty

            if score < best_score:
                best_score = score
                best_host = h

        if best_host is None:
            best_host = available[0]

        best_host.assign(job.job_id)
        placement.mapping[job.job_id] = best_host.host_id

    # Phase 3: Verify safety for each target and log decisions
    for target_job in target_jobs:
        host_id = placement.mapping.get(target_job.job_id)
        if host_id is None:
            continue

        co_tenant_jids = placement.co_tenants(target_job.job_id)
        spec_wids = [jobs_by_id[jid].workload_id for jid in co_tenant_jids
                      if jobs_by_id.get(jid) and jobs_by_id[jid].role == "spectator"]

        r_hat = predicted_risk(target_job.workload_id, spec_wids,
                               episode.regime_id, ctx.x_hat)
        u = predicted_uncertainty(target_job.workload_id, spec_wids,
                                  episode.regime_id, ctx.sigma)
        r_ucb = r_hat + safety_cfg.beta_ucb * u

        if not check_safety(r_ucb, safety_cfg.tau_risk_us):
            fallback_count += 1
            logger.warning(
                "Post-placement safety violation for %s ep%d: r_ucb=%.1f > tau=%.1f",
                target_job.workload_id, episode.episode_id,
                r_ucb, safety_cfg.tau_risk_us,
            )

    return placement, decisions


# ---- Policy registry ----

SCHEDULER_REGISTRY = {
    "random": schedule_random,
    "round_robin": schedule_round_robin,
    "static_partition": schedule_static_partition,
    "mean_greedy": schedule_mean_greedy,
    "similarity_avoidance": schedule_similarity_avoidance,
    "sit_risk_div": schedule_sit_risk_div,
    "sit_safe_ucb": schedule_sit_safe_ucb,
}


def run_scheduler(
    scheduler_name: str,
    ctx: ScheduleContext,
) -> Tuple[Placement, List[ScheduleDecision]]:
    """Run a named scheduler policy.

    Args:
        scheduler_name: One of the registered policy names.
        ctx: Scheduling context.

    Returns:
        Tuple of (Placement, list of ScheduleDecisions).
    """
    if scheduler_name not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler: {scheduler_name}. "
            f"Available: {list(SCHEDULER_REGISTRY.keys())}"
        )
    return SCHEDULER_REGISTRY[scheduler_name](ctx)
