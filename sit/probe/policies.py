"""Probe selection policies for active tomography.

Implements five policies with a common interface:
  A) Random probing (baseline)
  B) Coverage-aware probing (baseline+)
  C) Uncertainty-weighted probing (core active)
  D) Diversity-first probing (DPP greedy)
  E) Hybrid active probing (SIT-active, the winning policy)

Each policy takes a ProbeState and returns a ProbeChoice.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sit.core.logging import get_logger
from sit.probe.diversity import (
    logdet_diversity,
    logdet_increment,
    mean_pairwise_similarity,
)
from sit.probe.info_gain import hybrid_score, uncertainty_scores

logger = get_logger("probe.policies")


@dataclass
class ProbeState:
    """State available to a probe selection policy at step m.

    Attributes:
        target_id: ID of the target workload.
        regime_id: ID of the current regime.
        spectator_ids: Ordered list of all spectator IDs.
        chosen_so_far: List of previously chosen probe sets (each a list of indices).
        coverage_counts: Dict mapping spectator_id to # times selected.
        x_hat: Current reconstruction estimate (n,), or None.
        sigma: Current per-component uncertainty (n,), or None.
        K: Kernel matrix (n, n) for diversity computation.
        set_size: Number of spectators per probe set.
        max_repeats: Maximum times a single spectator can appear.
        hybrid_lambda: Trade-off parameter for hybrid policy.
        epsilon: Regularization for logdet computation.
        safety_threshold: Optional safety constraint threshold.
        safety_beta: Safety multiplier for uncertainty (x_hat + beta*sigma).
        rng: Random state for reproducibility.
    """
    target_id: str
    regime_id: str
    spectator_ids: List[str]
    chosen_so_far: List[List[int]] = field(default_factory=list)
    coverage_counts: Dict[str, int] = field(default_factory=dict)
    x_hat: Optional[np.ndarray] = None
    sigma: Optional[np.ndarray] = None
    K: Optional[np.ndarray] = None
    set_size: int = 3
    max_repeats: int = 20
    hybrid_lambda: float = 0.5
    epsilon: float = 1e-6
    safety_threshold: Optional[float] = None
    safety_beta: float = 2.0
    rng: np.random.RandomState = field(
        default_factory=lambda: np.random.RandomState(0)
    )

    @property
    def n_spectators(self) -> int:
        return len(self.spectator_ids)

    def eligible_indices(self) -> List[int]:
        """Return indices of spectators that haven't exceeded max_repeats."""
        eligible = []
        for i, sid in enumerate(self.spectator_ids):
            if self.coverage_counts.get(sid, 0) < self.max_repeats:
                eligible.append(i)
        if len(eligible) < self.set_size:
            eligible = list(range(self.n_spectators))
        return eligible

    def is_safe(self, idx: int) -> bool:
        """Placeholder safety constraint check.

        Returns False if x_hat[idx] + safety_beta * sigma[idx] exceeds
        the safety_threshold, indicating the spectator may be too risky.
        """
        if self.safety_threshold is None:
            return True
        if self.x_hat is None or self.sigma is None:
            return True
        val = self.x_hat[idx] + self.safety_beta * self.sigma[idx]
        return val <= self.safety_threshold


@dataclass
class ProbeChoice:
    """Result of a probe selection decision.

    Attributes:
        chosen_indices: Selected spectator indices.
        chosen_set: Selected spectator IDs (sorted).
        score_breakdown: Per-candidate scoring details.
        policy_name: Name of the policy that made this choice.
    """
    chosen_indices: List[int]
    chosen_set: List[str]
    score_breakdown: Dict[str, Any]
    policy_name: str


def select_random(state: ProbeState) -> ProbeChoice:
    """Policy A: Uniformly random probe selection.

    Samples set_size spectators uniformly at random from eligible pool.
    """
    eligible = state.eligible_indices()
    size = min(state.set_size, len(eligible))
    chosen_idx = sorted(state.rng.choice(eligible, size=size, replace=False).tolist())
    chosen_set = sorted([state.spectator_ids[i] for i in chosen_idx])

    breakdown = {
        "policy": "random",
        "eligible_count": len(eligible),
        "candidates": {state.spectator_ids[i]: {"score": 1.0 / len(eligible)} for i in chosen_idx},
    }

    return ProbeChoice(
        chosen_indices=chosen_idx,
        chosen_set=chosen_set,
        score_breakdown=breakdown,
        policy_name="random",
    )


def select_coverage(state: ProbeState) -> ProbeChoice:
    """Policy B: Coverage-aware probe selection.

    Greedily includes spectators with lowest coverage first,
    then fills remaining slots randomly.
    """
    eligible = state.eligible_indices()
    size = min(state.set_size, len(eligible))

    scores = np.array([
        state.coverage_counts.get(state.spectator_ids[i], 0) + state.rng.uniform(0, 0.1)
        for i in eligible
    ])
    sorted_order = np.argsort(scores)
    chosen_idx = sorted([eligible[sorted_order[j]] for j in range(size)])
    chosen_set = sorted([state.spectator_ids[i] for i in chosen_idx])

    breakdown = {
        "policy": "coverage",
        "eligible_count": len(eligible),
        "candidates": {
            state.spectator_ids[eligible[j]]: {
                "coverage_count": state.coverage_counts.get(state.spectator_ids[eligible[j]], 0),
                "score": float(scores[j]),
            }
            for j in range(min(size * 3, len(eligible)))
        },
    }

    return ProbeChoice(
        chosen_indices=chosen_idx,
        chosen_set=chosen_set,
        score_breakdown=breakdown,
        policy_name="coverage",
    )


def select_uncertainty(state: ProbeState) -> ProbeChoice:
    """Policy C: Uncertainty-weighted probe selection.

    Scores each spectator by sigma_s and optionally |x_hat_s|,
    with redundancy penalty from pairwise similarity.
    """
    n = state.n_spectators
    eligible = state.eligible_indices()
    size = min(state.set_size, len(eligible))

    if state.sigma is not None:
        base_scores = uncertainty_scores(state.sigma, state.x_hat, alpha=1.0, beta=0.1)
    else:
        base_scores = np.ones(n, dtype=np.float64)

    # Greedy selection with similarity penalty
    chosen = []
    candidate_details = {}

    for _ in range(size):
        best_idx = -1
        best_score = -float("inf")

        for i in eligible:
            if i in chosen:
                continue
            if not state.is_safe(i):
                continue

            score = base_scores[i]

            # Redundancy penalty: reduce score if similar to already chosen
            if state.K is not None and len(chosen) > 0:
                sim_penalty = np.mean([state.K[i, j] for j in chosen])
                score = score * (1.0 - 0.5 * sim_penalty)

            candidate_details[state.spectator_ids[i]] = {
                "base_score": float(base_scores[i]),
                "adjusted_score": float(score),
                "sigma": float(state.sigma[i]) if state.sigma is not None else 0.0,
            }

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            chosen.append(best_idx)

    chosen_idx = sorted(chosen)
    chosen_set = sorted([state.spectator_ids[i] for i in chosen_idx])

    breakdown = {
        "policy": "uncertainty",
        "eligible_count": len(eligible),
        "candidates": candidate_details,
    }

    return ProbeChoice(
        chosen_indices=chosen_idx,
        chosen_set=chosen_set,
        score_breakdown=breakdown,
        policy_name="uncertainty",
    )


def select_diversity(state: ProbeState) -> ProbeChoice:
    """Policy D: Diversity-first probe selection (DPP greedy).

    Selects sets that maximize log-determinant diversity:
    max_{S:|S|=k} log det(K_S + epsilon * I)

    Greedy: start empty, add element that increases logdet most.
    """
    eligible = state.eligible_indices()
    size = min(state.set_size, len(eligible))

    if state.K is None:
        return select_random(state)

    chosen = []
    candidate_details = {}

    for _ in range(size):
        best_idx = -1
        best_gain = -float("inf")

        for i in eligible:
            if i in chosen:
                continue

            gain = logdet_increment(chosen, i, state.K, state.epsilon)

            candidate_details[state.spectator_ids[i]] = {
                "logdet_increment": float(gain),
            }

            if gain > best_gain:
                best_gain = gain
                best_idx = i

        if best_idx >= 0:
            chosen.append(best_idx)

    chosen_idx = sorted(chosen)
    chosen_set = sorted([state.spectator_ids[i] for i in chosen_idx])

    breakdown = {
        "policy": "diversity",
        "eligible_count": len(eligible),
        "candidates": candidate_details,
    }

    return ProbeChoice(
        chosen_indices=chosen_idx,
        chosen_set=chosen_set,
        score_breakdown=breakdown,
        policy_name="diversity",
    )


def select_sit_active(state: ProbeState) -> ProbeChoice:
    """Policy E: Hybrid active probing (SIT-active, the WINNING policy).

    Two-stage greedy selection combining efficiency and diversity:

    Stage 1 (primary selection): Pick the first spectator using
    coverage-priority + uncertainty scoring. This ensures each step
    targets the most informative/under-covered spectator.

    Stage 2 (diverse fill): Fill remaining set_size-1 slots by
    maximizing logdet diversity with respect to the primary, while
    penalizing redundancy with coverage weighting. This ensures
    within-set diversity exceeds random probing.

    The combined effect: coverage drives step-by-step probe allocation,
    diversity ensures each probe set contains dissimilar spectators,
    and uncertainty focuses effort on poorly-reconstructed components.

    Constraints:
    - Coverage cap per spectator
    - Optional safety constraint
    """
    n = state.n_spectators
    eligible = state.eligible_indices()
    size = min(state.set_size, len(eligible))

    if state.K is None:
        return select_coverage(state)

    # Determine if sigma is informative
    has_sigma = (
        state.sigma is not None
        and np.max(state.sigma) > 1e-10
        and np.std(state.sigma) > 0.05 * np.mean(state.sigma + 1e-15)
    )

    if has_sigma:
        sigma = state.sigma
        max_sigma = np.max(sigma)
        sigma_norm = sigma / max(max_sigma, 1e-15)
        unc_weight = 0.5
    else:
        sigma = np.ones(n, dtype=np.float64)
        sigma_norm = np.ones(n, dtype=np.float64)
        unc_weight = 0.0

    # Coverage priority: prefer under-covered spectators
    coverage_arr = np.array([
        state.coverage_counts.get(state.spectator_ids[i], 0)
        for i in range(n)
    ], dtype=np.float64)
    cov_priority = 1.0 / (1.0 + coverage_arr)

    chosen = []
    candidate_details = {}

    for step in range(size):
        best_idx = -1
        best_score = -float("inf")

        for i in eligible:
            if i in chosen:
                continue
            if not state.is_safe(i):
                continue

            if step == 0:
                # Stage 1: Primary selection by coverage + uncertainty
                score = cov_priority[i] + unc_weight * sigma_norm[i]
                div_incr = 0.0
            else:
                # Stage 2: Diversity-dominant fill
                div_incr = logdet_increment(chosen, i, state.K, state.epsilon)
                # Within-set: diversity is primary, coverage is tiebreaker
                score = (
                    state.hybrid_lambda * max(0.0, div_incr)
                    + 0.2 * cov_priority[i]
                    + 0.1 * unc_weight * sigma_norm[i]
                )

            candidate_details[state.spectator_ids[i]] = {
                "sigma": float(sigma[i]),
                "sigma_norm": float(sigma_norm[i]),
                "cov_priority": float(cov_priority[i]),
                "logdet_increment": float(div_incr),
                "hybrid_lambda": state.hybrid_lambda,
                "unc_weight": unc_weight,
                "total_score": float(score),
                "step": step,
            }

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            chosen.append(best_idx)

    chosen_idx = sorted(chosen)
    chosen_set = sorted([state.spectator_ids[i] for i in chosen_idx])

    breakdown = {
        "policy": "sit_active",
        "eligible_count": len(eligible),
        "hybrid_lambda": state.hybrid_lambda,
        "candidates": candidate_details,
    }

    return ProbeChoice(
        chosen_indices=chosen_idx,
        chosen_set=chosen_set,
        score_breakdown=breakdown,
        policy_name="sit_active",
    )


# Policy registry
POLICY_REGISTRY: Dict[str, Any] = {
    "random": select_random,
    "coverage": select_coverage,
    "uncertainty": select_uncertainty,
    "diversity": select_diversity,
    "sit_active": select_sit_active,
}


def select_next_probe(state: ProbeState, policy_name: str) -> ProbeChoice:
    """Dispatch to the named probe selection policy.

    Args:
        state: Current probe state.
        policy_name: Name of the policy to use.

    Returns:
        ProbeChoice from the selected policy.

    Raises:
        ValueError: If policy_name is not registered.
    """
    if policy_name not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy '{policy_name}'. "
            f"Available: {sorted(POLICY_REGISTRY.keys())}"
        )
    return POLICY_REGISTRY[policy_name](state)
