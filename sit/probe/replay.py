"""Deterministic replay of probe selection decisions.

Provides functions to reconstruct ProbeState from logged data and
reproduce probe selections exactly, ensuring 100% auditability.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.probe.policies import ProbeChoice, ProbeState, select_next_probe

logger = get_logger("probe.replay")


def replay_probe_selection(
    probe_plan_df: pd.DataFrame,
    policy_name: str,
    step: int,
    spectator_ids: List[str],
    K: np.ndarray,
    sigma_history: Optional[Dict[int, np.ndarray]] = None,
    x_hat_history: Optional[Dict[int, np.ndarray]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[ProbeChoice, bool]:
    """Replay a single probe selection step from logged data.

    Reconstructs the ProbeState that existed at the given step and
    re-runs the policy to verify deterministic reproduction.

    Args:
        probe_plan_df: Full probe plan DataFrame with all logged steps.
        policy_name: Policy to replay.
        step: Step number (1-indexed).
        spectator_ids: Ordered spectator IDs.
        K: Kernel matrix.
        sigma_history: Dict mapping step -> sigma vector (optional).
        x_hat_history: Dict mapping step -> x_hat vector (optional).
        config: Probe config dict for parameters.

    Returns:
        Tuple of (replayed_choice, matches_original).
    """
    cfg = config or {}
    set_size = cfg.get("set_size", 3)
    max_repeats = cfg.get("max_repeats_per_spectator", 20)
    hybrid_lambda = cfg.get("hybrid_lambda", 0.5)
    epsilon = cfg.get("epsilon", 1e-6)

    # Filter to this policy's steps before the current step
    policy_df = probe_plan_df[
        probe_plan_df["policy_name"] == policy_name
    ].sort_values("step_m")

    # Reconstruct coverage counts from prior steps
    coverage_counts: Dict[str, int] = {sid: 0 for sid in spectator_ids}
    sid_to_idx = {sid: i for i, sid in enumerate(spectator_ids)}
    chosen_so_far: List[List[int]] = []

    for _, row in policy_df.iterrows():
        if row["step_m"] >= step:
            break
        chosen_set = _parse_set(row["chosen_set"])
        indices = sorted([sid_to_idx[s] for s in chosen_set if s in sid_to_idx])
        chosen_so_far.append(indices)
        for sid in chosen_set:
            if sid in coverage_counts:
                coverage_counts[sid] += 1

    # Get seed from the plan
    seed_row = policy_df[policy_df["step_m"] == step]
    if len(seed_row) == 0:
        raise ValueError(f"No logged step {step} for policy {policy_name}")

    seed = int(seed_row.iloc[0]["seed"])

    # Reconstruct sigma and x_hat for this step
    sigma = None
    x_hat = None
    if sigma_history and step in sigma_history:
        sigma = sigma_history[step]
    if x_hat_history and step in x_hat_history:
        x_hat = x_hat_history[step]

    # Build ProbeState
    target_id = str(seed_row.iloc[0]["target_id"])
    regime_id = str(seed_row.iloc[0]["regime_id"])

    state = ProbeState(
        target_id=target_id,
        regime_id=regime_id,
        spectator_ids=spectator_ids,
        chosen_so_far=chosen_so_far,
        coverage_counts=coverage_counts,
        x_hat=x_hat,
        sigma=sigma,
        K=K,
        set_size=set_size,
        max_repeats=max_repeats,
        hybrid_lambda=hybrid_lambda,
        epsilon=epsilon,
        rng=np.random.RandomState(seed + step),
    )

    # Re-run the policy
    replayed = select_next_probe(state, policy_name)

    # Compare with original
    original_set = sorted(_parse_set(seed_row.iloc[0]["chosen_set"]))
    matches = replayed.chosen_set == original_set

    if not matches:
        logger.warning(
            "Replay mismatch at step %d for policy %s: "
            "original=%s, replayed=%s",
            step, policy_name, original_set, replayed.chosen_set,
        )
    else:
        logger.info(
            "Replay match at step %d for policy %s: %s",
            step, policy_name, replayed.chosen_set,
        )

    return replayed, matches


def replay_all_steps(
    probe_plan_df: pd.DataFrame,
    policy_name: str,
    spectator_ids: List[str],
    K: np.ndarray,
    sigma_history: Optional[Dict[int, np.ndarray]] = None,
    x_hat_history: Optional[Dict[int, np.ndarray]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int, List[int]]:
    """Replay all steps for a policy and report match rate.

    Returns:
        Tuple of (total_steps, matches, list_of_failed_steps).
    """
    policy_df = probe_plan_df[
        probe_plan_df["policy_name"] == policy_name
    ].sort_values("step_m")

    steps = sorted(policy_df["step_m"].unique())
    total = len(steps)
    match_count = 0
    failed_steps = []

    for step in steps:
        _, matches = replay_probe_selection(
            probe_plan_df, policy_name, step,
            spectator_ids, K, sigma_history, x_hat_history, config,
        )
        if matches:
            match_count += 1
        else:
            failed_steps.append(step)

    logger.info(
        "Replay summary for %s: %d/%d steps matched (%.1f%%)",
        policy_name, match_count, total,
        100.0 * match_count / max(total, 1),
    )

    return total, match_count, failed_steps


def _parse_set(val) -> List[str]:
    """Parse a chosen_set value from DataFrame (may be string or list)."""
    if isinstance(val, list):
        return sorted(val)
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return sorted(str(x) for x in parsed)
        except (json.JSONDecodeError, TypeError):
            pass
        # Try comma-separated
        return sorted(val.strip("[]").replace("'", "").replace('"', "").split(", "))
    return []
