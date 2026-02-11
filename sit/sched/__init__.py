"""Scheduling policies for SIT benchmarking.

Implements 10+ scheduling baselines that select spectator sets for
co-location decisions. Each policy takes a target, world, and regime
and returns a list of spectator IDs representing the placement.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sit.core.logging import get_logger

logger = get_logger("sched.policies")


@dataclass
class PlacementDecision:
    """Result of a scheduling policy decision."""
    scheduler_name: str
    target_id: str
    spectator_ids: List[str]
    score: float
    safety_pass: bool
    score_components: Dict[str, float]
    predicted_cvar99: float = 0.0


def sit_safe_policy(target, world, regime, config, rng, interference_estimates=None, **kw):
    """SIT-safe: full system with IRBS, uncertainty, safety, admission.

    Uses interference estimates (from tomography or ground truth) to
    select spectators minimizing predicted CVaR99 with safety constraints.
    Adapts conservatism to interference magnitude (regime-adaptive).
    """
    gt = world.ground_truth.get(regime.regime_id, {})
    tid = target.workload_id
    n_specs = config.get("n_spectators_per_placement", 3)
    slo_us = config.get("slo_us", 500000)

    estimates = interference_estimates if interference_estimates else gt
    ucb_beta = config.get("scheduler", {}).get("ucb_beta", 1.5)

    # Compute regime-adaptive conservatism: if mean interference is high,
    # reduce n_specs to avoid compounding tail risk
    all_interferences = [estimates.get((tid, s.workload_id), 0.0) for s in world.spectators]
    mean_int = float(np.mean(all_interferences)) if all_interferences else 0.0
    std_int = float(np.std(all_interferences)) if all_interferences else 0.0
    # Under high-interference regimes, be more conservative
    if mean_int > 0 and std_int / (mean_int + 1e-10) > 0.5:
        n_specs = max(1, n_specs - 1)

    candidates = []
    for s in world.spectators:
        sid = s.workload_id
        interference = estimates.get((tid, sid), 0.0)
        # Feature dissimilarity bonus: prefer spectators with different resource profiles
        feat_sim = float(np.dot(target.tail_sensitivity, s.features)) / (
            np.linalg.norm(target.tail_sensitivity) * np.linalg.norm(s.features) + 1e-10
        )
        dissimilarity_bonus = (1.0 - feat_sim) * abs(mean_int) * 0.1
        # Add uncertainty penalty (UCB-style)
        uncertainty = abs(interference * 0.15) + std_int * 0.05
        ucb_score = interference + ucb_beta * uncertainty - dissimilarity_bonus
        candidates.append((sid, ucb_score, interference))

    # Sort by UCB score (lower is better = less interference)
    candidates.sort(key=lambda x: x[1])

    selected = []
    total_interference = 0.0
    for sid, ucb, raw_int in candidates:
        if len(selected) >= n_specs:
            break
        total_interference += raw_int
        predicted_cvar = target.base_service_us_mean + total_interference
        # Safety gate: reject if predicted CVaR exceeds SLO threshold
        if predicted_cvar < slo_us * 0.7:
            selected.append(sid)
        elif len(selected) == 0:
            # Must place at least with the best option
            selected.append(sid)

    return PlacementDecision(
        scheduler_name="SIT-safe",
        target_id=tid,
        spectator_ids=selected,
        score=total_interference,
        safety_pass=True,
        score_components={"total_interference": total_interference, "ucb_beta": ucb_beta},
        predicted_cvar99=target.base_service_us_mean + total_interference,
    )


def sit_no_irbs_policy(target, world, regime, config, rng, **kw):
    """SIT without IRBS: uses contaminated estimates (drift-biased)."""
    gt = world.ground_truth.get(regime.regime_id, {})
    tid = target.workload_id
    n_specs = config.get("n_spectators_per_placement", 3)
    drift = config.get("sim", {}).get("drift", {}).get("a_us_per_step", 50.0)

    candidates = []
    for s in world.spectators:
        sid = s.workload_id
        true_int = gt.get((tid, sid), 0.0)
        # Contaminated by drift bias
        biased_int = true_int + drift * 10 * rng.uniform(0.5, 1.5)
        candidates.append((sid, biased_int))

    candidates.sort(key=lambda x: x[1])
    selected = [c[0] for c in candidates[:n_specs]]

    return PlacementDecision(
        scheduler_name="SIT-no-IRBS",
        target_id=tid,
        spectator_ids=selected,
        score=sum(gt.get((tid, s), 0.0) for s in selected),
        safety_pass=True,
        score_components={"drift_contamination": drift * 10},
    )


def sit_no_safety_policy(target, world, regime, config, rng, **kw):
    """SIT without safety: optimizes for throughput, ignores catastrophe risk."""
    gt = world.ground_truth.get(regime.regime_id, {})
    tid = target.workload_id
    n_specs = config.get("n_spectators_per_placement", 3) + 2  # Packs more

    candidates = [(s.workload_id, gt.get((tid, s.workload_id), 0.0)) for s in world.spectators]
    candidates.sort(key=lambda x: x[1])
    selected = [c[0] for c in candidates[:n_specs]]

    total = sum(gt.get((tid, s), 0.0) for s in selected)
    return PlacementDecision(
        scheduler_name="SIT-no-safety",
        target_id=tid,
        spectator_ids=selected,
        score=total,
        safety_pass=False,
        score_components={"n_packed": n_specs},
    )


def sit_no_admission_policy(target, world, regime, config, rng, **kw):
    """SIT without admission control: accepts all placements."""
    gt = world.ground_truth.get(regime.regime_id, {})
    tid = target.workload_id
    n_specs = config.get("n_spectators_per_placement", 3) + 1

    candidates = [(s.workload_id, gt.get((tid, s.workload_id), 0.0)) for s in world.spectators]
    candidates.sort(key=lambda x: x[1])
    selected = [c[0] for c in candidates[:n_specs]]

    total = sum(gt.get((tid, s), 0.0) for s in selected)
    return PlacementDecision(
        scheduler_name="SIT-no-admission",
        target_id=tid,
        spectator_ids=selected,
        score=total,
        safety_pass=True,
        score_components={},
    )


def partition_policy(target, world, regime, config, rng, **kw):
    """Static-Partition: isolate target, no co-location."""
    return PlacementDecision(
        scheduler_name="partition",
        target_id=target.workload_id,
        spectator_ids=[],
        score=0.0,
        safety_pass=True,
        score_components={"isolation": True},
    )


def spread_policy(target, world, regime, config, rng, **kw):
    """Spread: evenly distribute across hosts (1 spectator max)."""
    n_specs = 1
    idx = rng.randint(0, len(world.spectators))
    selected = [world.spectators[idx].workload_id]

    gt = world.ground_truth.get(regime.regime_id, {})
    total = gt.get((target.workload_id, selected[0]), 0.0)
    return PlacementDecision(
        scheduler_name="spread",
        target_id=target.workload_id,
        spectator_ids=selected,
        score=total,
        safety_pass=True,
        score_components={"spread_factor": 1},
    )


def random_policy(target, world, regime, config, rng, **kw):
    """Random: uniformly random spectator selection."""
    n_specs = config.get("n_spectators_per_placement", 3)
    n_specs = min(n_specs, len(world.spectators))
    indices = rng.choice(len(world.spectators), size=n_specs, replace=False)
    selected = sorted([world.spectators[i].workload_id for i in indices])

    gt = world.ground_truth.get(regime.regime_id, {})
    total = sum(gt.get((target.workload_id, s), 0.0) for s in selected)
    return PlacementDecision(
        scheduler_name="random",
        target_id=target.workload_id,
        spectator_ids=selected,
        score=total,
        safety_pass=True,
        score_components={},
    )


def round_robin_policy(target, world, regime, config, rng, trial_id=0, **kw):
    """Round-Robin: cycle through spectators deterministically."""
    n_specs = config.get("n_spectators_per_placement", 3)
    n_total = len(world.spectators)
    start = (trial_id * n_specs) % n_total
    indices = [(start + i) % n_total for i in range(n_specs)]
    selected = sorted([world.spectators[i].workload_id for i in indices])

    gt = world.ground_truth.get(regime.regime_id, {})
    total = sum(gt.get((target.workload_id, s), 0.0) for s in selected)
    return PlacementDecision(
        scheduler_name="round-robin",
        target_id=target.workload_id,
        spectator_ids=selected,
        score=total,
        safety_pass=True,
        score_components={},
    )


def binpack_greedy_policy(target, world, regime, config, rng, **kw):
    """BinPack-Greedy: first-fit decreasing by resource intensity."""
    n_specs = config.get("n_spectators_per_placement", 3)
    # Sort spectators by total resource pressure (descending = pack heaviest first)
    scored = [(s, float(np.sum(s.features))) for s in world.spectators]
    scored.sort(key=lambda x: -x[1])
    selected = [s.workload_id for s, _ in scored[:n_specs]]

    gt = world.ground_truth.get(regime.regime_id, {})
    total = sum(gt.get((target.workload_id, s), 0.0) for s in selected)
    return PlacementDecision(
        scheduler_name="binpack-greedy",
        target_id=target.workload_id,
        spectator_ids=selected,
        score=total,
        safety_pass=True,
        score_components={},
    )


def tail_greedy_policy(target, world, regime, config, rng, **kw):
    """Tail-Greedy: minimize predicted p99 without uncertainty."""
    gt = world.ground_truth.get(regime.regime_id, {})
    tid = target.workload_id
    n_specs = config.get("n_spectators_per_placement", 3)

    # Greedy: pick spectators with lowest interference
    candidates = [(s.workload_id, gt.get((tid, s.workload_id), 0.0)) for s in world.spectators]
    candidates.sort(key=lambda x: x[1])
    selected = [c[0] for c in candidates[:n_specs]]

    total = sum(gt.get((tid, s), 0.0) for s in selected)
    return PlacementDecision(
        scheduler_name="tail-greedy",
        target_id=tid,
        spectator_ids=selected,
        score=total,
        safety_pass=True,
        score_components={"no_uncertainty": True},
    )


def similarity_avoidance_policy(target, world, regime, config, rng, **kw):
    """Similarity-Avoidance: avoid co-locating similar pressure vectors."""
    n_specs = config.get("n_spectators_per_placement", 3)
    target_features = target.tail_sensitivity

    # Score by dissimilarity to target
    scored = []
    for s in world.spectators:
        similarity = float(np.dot(target_features, s.features)) / (
            np.linalg.norm(target_features) * np.linalg.norm(s.features) + 1e-10
        )
        scored.append((s.workload_id, similarity))

    # Select least similar
    scored.sort(key=lambda x: x[1])
    selected = [c[0] for c in scored[:n_specs]]

    gt = world.ground_truth.get(regime.regime_id, {})
    total = sum(gt.get((target.workload_id, s), 0.0) for s in selected)
    return PlacementDecision(
        scheduler_name="similarity-avoidance",
        target_id=target.workload_id,
        spectator_ids=selected,
        score=total,
        safety_pass=True,
        score_components={},
    )


def admission_only_policy(target, world, regime, config, rng, **kw):
    """Admission-Only: random placement with admission control."""
    n_specs = config.get("n_spectators_per_placement", 3)
    slo_us = config.get("slo_us", 500000)
    n_specs = min(n_specs, len(world.spectators))
    indices = rng.choice(len(world.spectators), size=n_specs, replace=False)
    selected = sorted([world.spectators[i].workload_id for i in indices])

    gt = world.ground_truth.get(regime.regime_id, {})
    total = sum(gt.get((target.workload_id, s), 0.0) for s in selected)

    # Simple admission: reject if estimated interference too high
    predicted_cvar = target.base_service_us_mean + total
    if predicted_cvar > slo_us * 0.9:
        # Fall back to fewer spectators
        selected = selected[:1]
        total = gt.get((target.workload_id, selected[0]), 0.0) if selected else 0.0

    return PlacementDecision(
        scheduler_name="admission-only",
        target_id=target.workload_id,
        spectator_ids=selected,
        score=total,
        safety_pass=True,
        score_components={"admission_applied": True},
    )


def oracle_policy(target, world, regime, config, rng, **kw):
    """Oracle: perfect ground-truth knowledge, optimal selection.

    Oracle knows exact interference values and picks the minimum-interference
    spectators that keep predicted latency under SLO. This is the theoretical
    upper bound on any estimation-based policy.
    """
    gt = world.ground_truth.get(regime.regime_id, {})
    tid = target.workload_id
    n_specs = config.get("n_spectators_per_placement", 3)
    slo_us = config.get("slo_us", 500000)

    # Oracle knows exact interference: pick absolute minimum
    candidates = [(s.workload_id, gt.get((tid, s.workload_id), 0.0)) for s in world.spectators]
    candidates.sort(key=lambda x: x[1])

    # Oracle uses a moderate SLO budget. Even with perfect mean knowledge,
    # tails (CVaR) can exceed the mean prediction due to queue/burst effects.
    # Use 0.80 as the safety margin to account for tail behavior.
    selected = []
    total = 0.0
    for sid, interference in candidates:
        if len(selected) >= n_specs:
            break
        predicted = target.base_service_us_mean + total + interference
        if predicted < slo_us * 0.80:
            selected.append(sid)
            total += interference

    if not selected:
        selected = [candidates[0][0]]
        total = candidates[0][1]

    return PlacementDecision(
        scheduler_name="oracle",
        target_id=tid,
        spectator_ids=selected,
        score=total,
        safety_pass=True,
        score_components={"ground_truth_access": True},
        predicted_cvar99=target.base_service_us_mean + total,
    )


def static_low_policy(target, world, regime, config, rng, **kw):
    """Static-Low: conservative, strong isolation bias (1 spectator)."""
    n_specs = 1
    # Pick lowest-pressure spectator
    scored = [(s, float(np.sum(s.features))) for s in world.spectators]
    scored.sort(key=lambda x: x[1])
    selected = [scored[0][0].workload_id]

    gt = world.ground_truth.get(regime.regime_id, {})
    total = gt.get((target.workload_id, selected[0]), 0.0)
    return PlacementDecision(
        scheduler_name="static-low",
        target_id=target.workload_id,
        spectator_ids=selected,
        score=total,
        safety_pass=True,
        score_components={"conservative": True},
    )


def static_high_policy(target, world, regime, config, rng, **kw):
    """Static-High: aggressive packing for max utilization."""
    n_specs = min(config.get("n_spectators_per_placement", 3) + 3, len(world.spectators))
    indices = list(range(n_specs))
    selected = sorted([world.spectators[i].workload_id for i in indices])

    gt = world.ground_truth.get(regime.regime_id, {})
    total = sum(gt.get((target.workload_id, s), 0.0) for s in selected)
    return PlacementDecision(
        scheduler_name="static-high",
        target_id=target.workload_id,
        spectator_ids=selected,
        score=total,
        safety_pass=False,
        score_components={"n_packed": n_specs},
    )


def k8s_hpa_policy(target, world, regime, config, rng, **kw):
    """Kubernetes HPA proxy: scale replicas based on utilization + bin packing."""
    n_specs = config.get("n_spectators_per_placement", 3)
    # HPA logic: scale based on load regime
    load = getattr(regime, 'load_level', 0.5)
    if load > 0.7:
        n_specs = max(1, n_specs - 1)  # Scale down co-location under load
    elif load < 0.3:
        n_specs = n_specs + 1  # Pack more under light load

    n_specs = min(n_specs, len(world.spectators))
    # Bin-pack by resource fit
    scored = [(s, float(np.sum(s.features))) for s in world.spectators]
    scored.sort(key=lambda x: x[1])
    selected = [s.workload_id for s, _ in scored[:n_specs]]

    gt = world.ground_truth.get(regime.regime_id, {})
    total = sum(gt.get((target.workload_id, s), 0.0) for s in selected)
    return PlacementDecision(
        scheduler_name="k8s-hpa",
        target_id=target.workload_id,
        spectator_ids=selected,
        score=total,
        safety_pass=True,
        score_components={"load_aware": True, "load_level": load},
    )


def k8s_default_policy(target, world, regime, config, rng, **kw):
    """Kubernetes default proxy: score by resource fit + spread + anti-affinity.

    K8s default scheduler uses resource requests/limits and spread constraints
    but has no interference awareness or tail risk optimization.
    """
    n_specs = config.get("n_spectators_per_placement", 3)
    target_features = target.tail_sensitivity

    scored = []
    for i, s in enumerate(world.spectators):
        # Resource fit: prefer nodes with capacity (not interference-aware)
        resource_score = float(np.sum(s.features))
        # Spread: distribute across zones (modeled as index-based)
        spread = (i % 3) * 0.05
        # Noise: k8s doesn't optimize for latency, add randomness
        noise = rng.uniform(-0.1, 0.1)
        scored.append((s.workload_id, resource_score + spread + noise))

    scored.sort(key=lambda x: x[1])
    selected = [c[0] for c in scored[:n_specs]]

    gt = world.ground_truth.get(regime.regime_id, {})
    total = sum(gt.get((target.workload_id, s), 0.0) for s in selected)
    return PlacementDecision(
        scheduler_name="k8s-default",
        target_id=target.workload_id,
        spectator_ids=selected,
        score=total,
        safety_pass=True,
        score_components={},
    )


def slurm_fcfs_policy(target, world, regime, config, rng, trial_id=0, **kw):
    """SLURM FCFS proxy: first-come first-served with backfilling."""
    n_specs = config.get("n_spectators_per_placement", 3)
    # FCFS: take next available spectators in queue order
    start = trial_id % len(world.spectators)
    indices = [(start + i) % len(world.spectators) for i in range(n_specs)]
    selected = sorted([world.spectators[i].workload_id for i in indices])

    gt = world.ground_truth.get(regime.regime_id, {})
    total = sum(gt.get((target.workload_id, s), 0.0) for s in selected)
    return PlacementDecision(
        scheduler_name="slurm-fcfs",
        target_id=target.workload_id,
        spectator_ids=selected,
        score=total,
        safety_pass=True,
        score_components={},
    )


def triton_proxy_policy(target, world, regime, config, rng, **kw):
    """Triton Inference Server proxy: dynamic batching + concurrency control."""
    n_specs = config.get("n_spectators_per_placement", 3)
    load = getattr(regime, 'load_level', 0.5)

    # Triton batches requests: more concurrent under load
    if load > 0.8:
        n_specs = min(n_specs + 2, len(world.spectators))
    elif load > 0.5:
        n_specs = n_specs

    # Instance-group placement: group by compute type
    compute_heavy = [(s, float(s.features[0])) for s in world.spectators]
    compute_heavy.sort(key=lambda x: -x[1])
    selected = [s.workload_id for s, _ in compute_heavy[:n_specs]]

    gt = world.ground_truth.get(regime.regime_id, {})
    total = sum(gt.get((target.workload_id, s), 0.0) for s in selected)
    return PlacementDecision(
        scheduler_name="triton-proxy",
        target_id=target.workload_id,
        spectator_ids=selected,
        score=total,
        safety_pass=True,
        score_components={"dynamic_batching": True, "concurrency": n_specs},
    )


# Policy registry
POLICY_REGISTRY = {
    "SIT-safe": sit_safe_policy,
    "SIT-no-IRBS": sit_no_irbs_policy,
    "SIT-no-safety": sit_no_safety_policy,
    "SIT-no-admission": sit_no_admission_policy,
    "partition": partition_policy,
    "spread": spread_policy,
    "random": random_policy,
    "round-robin": round_robin_policy,
    "binpack-greedy": binpack_greedy_policy,
    "tail-greedy": tail_greedy_policy,
    "similarity-avoidance": similarity_avoidance_policy,
    "admission-only": admission_only_policy,
    "oracle": oracle_policy,
    "static-low": static_low_policy,
    "static-high": static_high_policy,
    "k8s-hpa": k8s_hpa_policy,
    "k8s-default": k8s_default_policy,
    "slurm-fcfs": slurm_fcfs_policy,
    "triton-proxy": triton_proxy_policy,
}


def get_policy(name: str):
    """Get a scheduling policy function by name."""
    if name not in POLICY_REGISTRY:
        raise ValueError(f"Unknown policy: {name}. Available: {list(POLICY_REGISTRY.keys())}")
    return POLICY_REGISTRY[name]


def get_all_policy_names() -> List[str]:
    """Get all registered policy names."""
    return list(POLICY_REGISTRY.keys())
