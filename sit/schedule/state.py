"""Scheduling state: episodes, placements, and host state.

Defines the core data structures for scheduling episodes including
hosts, jobs, placements, and episode generation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Host:
    """A physical host with a capacity for co-located jobs."""
    host_id: str
    capacity: int
    assigned: List[str] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return self.capacity - len(self.assigned)

    def can_accept(self) -> bool:
        return self.remaining > 0

    def assign(self, job_id: str) -> None:
        if not self.can_accept():
            raise ValueError(f"Host {self.host_id} at capacity")
        self.assigned.append(job_id)

    def reset(self) -> None:
        self.assigned = []


@dataclass
class Job:
    """A job to be placed on a host."""
    job_id: str
    workload_id: str
    role: str  # "target" or "spectator"


@dataclass
class Placement:
    """A complete placement mapping jobs to hosts."""
    mapping: Dict[str, str] = field(default_factory=dict)  # job_id -> host_id

    def co_tenants(self, job_id: str) -> List[str]:
        """Get job IDs on same host as job_id, excluding job_id."""
        host = self.mapping.get(job_id)
        if host is None:
            return []
        return [j for j, h in self.mapping.items() if h == host and j != job_id]

    def host_jobs(self, host_id: str) -> List[str]:
        """Get all job IDs assigned to a host."""
        return [j for j, h in self.mapping.items() if h == host_id]


@dataclass
class Episode:
    """A scheduling episode: a set of jobs to place on hosts."""
    episode_id: int
    jobs: List[Job]
    hosts: List[Host]
    regime_id: str
    t_index: int
    target_jobs: List[Job] = field(default_factory=list)
    spectator_jobs: List[Job] = field(default_factory=list)

    def __post_init__(self):
        if not self.target_jobs:
            self.target_jobs = [j for j in self.jobs if j.role == "target"]
        if not self.spectator_jobs:
            self.spectator_jobs = [j for j in self.jobs if j.role == "spectator"]

    def fresh_hosts(self) -> List[Host]:
        """Return fresh copies of hosts with empty assignments."""
        return [Host(h.host_id, h.capacity) for h in self.hosts]


@dataclass
class EpisodeResult:
    """Result of evaluating a placement for one episode."""
    episode_id: int
    scheduler_name: str
    placement: Placement
    target_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Per-target: {target_id: {mean, p95, p99, cvar99, violation_rate, ...}}
    episode_metrics: Dict[str, float] = field(default_factory=dict)
    # Aggregate: {mean_cvar99, max_cvar99, catastrophe, goodput, ...}
    fallback_used: bool = False
    fallback_count: int = 0


def generate_episodes(
    world: Any,
    config: Dict[str, Any],
    rng: np.random.RandomState,
) -> List[Episode]:
    """Generate scheduling episodes from world and config.

    Args:
        world: Simulation World.
        config: Scheduling config section.
        rng: Random state.

    Returns:
        List of Episode objects.
    """
    sched_cfg = config.get("scheduling", {})
    n_hosts = sched_cfg.get("n_hosts", 6)
    host_capacity = sched_cfg.get("host_capacity", 12)
    n_episodes = sched_cfg.get("episodes", 200)

    ep_gen = sched_cfg.get("episode_generator", {})
    mode = ep_gen.get("mode", "adversarial")
    toxic_rate = ep_gen.get("toxic_inclusion_rate", 0.6)
    hard_regime_rate = ep_gen.get("hard_regime_rate", 0.5)

    target_ids = [t.workload_id for t in world.targets]
    spectator_ids = [s.workload_id for s in world.spectators]

    # Identify toxic spectators for each target
    toxic_specs = {}
    for t_id in target_ids:
        toxic_specs[t_id] = []
        for s_id in spectator_ids:
            if (t_id, s_id) in world.toxic_pairs:
                toxic_specs[t_id].append(s_id)

    # Identify regimes by difficulty (use channel scale as proxy)
    regime_ids = [r.regime_id for r in world.regimes]

    episodes = []
    total_capacity = n_hosts * host_capacity

    for ep_idx in range(n_episodes):
        # Pick regime
        if mode == "adversarial" and rng.random() < hard_regime_rate:
            # Pick last regime as "hardest" (higher sensitivity factors)
            r_id = regime_ids[-1] if len(regime_ids) > 0 else regime_ids[0]
        else:
            r_id = regime_ids[rng.randint(len(regime_ids))]

        # Build jobs: all targets + subset of spectators
        jobs = []
        for t_id in target_ids:
            jobs.append(Job(
                job_id=f"ep{ep_idx}_t_{t_id}",
                workload_id=t_id,
                role="target",
            ))

        # Pick spectators to fill remaining capacity
        n_target_jobs = len(target_ids)
        n_spectator_slots = total_capacity - n_target_jobs
        n_spectator_slots = max(0, n_spectator_slots)

        # In adversarial mode, include toxic spectators at higher rate
        chosen_specs = []
        if mode == "adversarial":
            # Collect all toxic spectators
            all_toxic = set()
            for t_id in target_ids:
                all_toxic.update(toxic_specs.get(t_id, []))

            # Include toxic ones with probability toxic_rate
            for s_id in all_toxic:
                if rng.random() < toxic_rate and len(chosen_specs) < n_spectator_slots:
                    chosen_specs.append(s_id)

            # Fill remaining from all spectators
            remaining_specs = [s for s in spectator_ids if s not in chosen_specs]
            rng.shuffle(remaining_specs)
            need = n_spectator_slots - len(chosen_specs)
            if need > 0:
                chosen_specs.extend(remaining_specs[:need])
        else:
            # Random subset
            perm = rng.permutation(len(spectator_ids))
            chosen_specs = [spectator_ids[i] for i in perm[:n_spectator_slots]]

        for s_id in chosen_specs:
            jobs.append(Job(
                job_id=f"ep{ep_idx}_s_{s_id}",
                workload_id=s_id,
                role="spectator",
            ))

        hosts = [Host(f"host_{h}", host_capacity) for h in range(n_hosts)]
        t_index = ep_idx  # Simple linear time

        episodes.append(Episode(
            episode_id=ep_idx,
            jobs=jobs,
            hosts=hosts,
            regime_id=r_id,
            t_index=t_index,
        ))

    return episodes
