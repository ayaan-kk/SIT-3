"""Decision replay for scheduling auditability.

Replays scheduling decisions from logged data and verifies
that placements are deterministically reproducible.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.schedule.constraints import SafetyConfig
from sit.schedule.policies import ScheduleContext, run_scheduler
from sit.schedule.state import Episode, Host, Job, generate_episodes

logger = get_logger("schedule.replay")


def replay_episode(
    episode: Episode,
    scheduler_name: str,
    x_hat: Dict,
    sigma: Dict,
    safety_config: SafetyConfig,
    K: Optional[np.ndarray] = None,
    spectator_id_to_idx: Optional[Dict[str, int]] = None,
    lambda_div: float = 0.5,
    slo_us: float = 500000.0,
    rng_seed: int = 0,
) -> Dict[str, str]:
    """Replay a scheduling episode and return the placement.

    Args:
        episode: Episode to replay.
        scheduler_name: Name of the scheduler.
        x_hat: Interference estimates.
        sigma: Uncertainty estimates.
        safety_config: Safety config.
        K: Kernel matrix.
        spectator_id_to_idx: Spectator ID to kernel index.
        lambda_div: Diversity weight.
        slo_us: SLO threshold.
        rng_seed: Seed for the scheduler's RNG.

    Returns:
        Placement mapping (job_id -> host_id).
    """
    ctx = ScheduleContext(
        episode=episode,
        x_hat=x_hat,
        sigma=sigma,
        safety_config=safety_config,
        K=K,
        spectator_id_to_idx=spectator_id_to_idx,
        lambda_div=lambda_div,
        slo_us=slo_us,
        rng=np.random.RandomState(rng_seed),
    )
    placement, _ = run_scheduler(scheduler_name, ctx)
    return dict(placement.mapping)


def verify_replay(
    original_decisions_df: pd.DataFrame,
    episode: Episode,
    scheduler_name: str,
    x_hat: Dict,
    sigma: Dict,
    safety_config: SafetyConfig,
    K: Optional[np.ndarray] = None,
    spectator_id_to_idx: Optional[Dict[str, int]] = None,
    lambda_div: float = 0.5,
    slo_us: float = 500000.0,
    rng_seed: int = 0,
) -> Tuple[bool, int, int, List[int]]:
    """Verify that replayed decisions match original logged decisions.

    Args:
        original_decisions_df: Logged decisions for this episode/scheduler.
        episode: Episode to replay.
        scheduler_name: Name of the scheduler.
        x_hat, sigma, safety_config, K, spectator_id_to_idx, lambda_div: Context.
        slo_us: SLO threshold.
        rng_seed: RNG seed.

    Returns:
        Tuple of (all_match, total, matches, failed_steps).
    """
    # Filter to this episode and scheduler
    mask = (
        (original_decisions_df["episode_id"] == episode.episode_id)
        & (original_decisions_df["scheduler_name"] == scheduler_name)
    )
    orig = original_decisions_df[mask].sort_values("step_index")

    if len(orig) == 0:
        return True, 0, 0, []

    # Replay
    replay_mapping = replay_episode(
        episode, scheduler_name, x_hat, sigma, safety_config,
        K, spectator_id_to_idx, lambda_div, slo_us, rng_seed,
    )

    total = 0
    matches = 0
    failed_steps = []

    for _, row in orig.iterrows():
        step = int(row["step_index"])
        target_id = row["target_id"]
        orig_host = row["chosen_host"]

        # Find the target job in episode
        target_jid = None
        for j in episode.target_jobs:
            if j.workload_id == target_id:
                target_jid = j.job_id
                break

        total += 1
        if target_jid and replay_mapping.get(target_jid) == orig_host:
            matches += 1
        else:
            failed_steps.append(step)

    all_match = (matches == total)

    if not all_match:
        logger.warning(
            "Replay mismatch for ep%d %s: %d/%d (failed: %s)",
            episode.episode_id, scheduler_name, matches, total,
            failed_steps[:5],
        )

    return all_match, total, matches, failed_steps


def replay_all_episodes(
    decisions_df: pd.DataFrame,
    episodes: List[Episode],
    scheduler_name: str,
    x_hat: Dict,
    sigma: Dict,
    safety_config: SafetyConfig,
    K: Optional[np.ndarray] = None,
    spectator_id_to_idx: Optional[Dict[str, int]] = None,
    lambda_div: float = 0.5,
    slo_us: float = 500000.0,
    base_seed: int = 0,
) -> Tuple[int, int, List[int]]:
    """Replay all episodes for a scheduler and verify decisions.

    Returns:
        Tuple of (total_decisions, total_matches, failed_episode_ids).
    """
    total = 0
    matches = 0
    failed_eps = []

    for ep in episodes:
        rng_seed = base_seed + ep.episode_id
        ok, ep_total, ep_matches, _ = verify_replay(
            decisions_df, ep, scheduler_name,
            x_hat, sigma, safety_config,
            K, spectator_id_to_idx, lambda_div, slo_us, rng_seed,
        )
        total += ep_total
        matches += ep_matches
        if not ok:
            failed_eps.append(ep.episode_id)

    return total, matches, failed_eps
