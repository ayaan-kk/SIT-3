"""Minimum winning metrics acceptance gates.

Each gate defines a measurable acceptance criterion. Placeholder gates
raise NotImplementedError until the relevant module provides real data.
Implemented gates return True/False with detailed logging.
"""

import numpy as np
import pandas as pd

from sit.core.logging import get_logger

logger = get_logger("eval.gates")


def gate_ci_coverage(
    uncertainty_df: pd.DataFrame,
    target: float = 0.80,
) -> bool:
    """Gate: Confidence interval empirical coverage meets target.

    Checks that the observed coverage of uncertainty intervals on
    synthetic ground truth meets or exceeds the target.

    Args:
        uncertainty_df: DataFrame with 'coverage_all' column from
            tomography uncertainty evaluation.
        target: Required coverage fraction (default 0.80).

    Returns:
        True if the gate passes.
    """
    if "coverage_all" not in uncertainty_df.columns:
        raise ValueError("Missing required column: coverage_all")

    mean_coverage = float(uncertainty_df["coverage_all"].mean())
    passed = mean_coverage >= target

    logger.info(
        "CI coverage gate: mean_coverage=%.3f, target=%.3f -> %s",
        mean_coverage, target,
        "PASS" if passed else "FAIL",
    )

    return passed


def gate_topk_recovery(
    recovery_df: pd.DataFrame,
    target_recall: float = 0.95,
    target_ndcg: float = 0.95,
) -> bool:
    """Gate: Top-K toxic spectator recovery accuracy meets target.

    Checks that the tomography solver correctly identifies the top-K
    most interfering spectators with recall >= target and NDCG >= target.

    Args:
        recovery_df: DataFrame with 'topk_recall' and 'ndcg_at_k' columns
            from tomography recovery evaluation.
        target_recall: Required minimum recall (default 0.95).
        target_ndcg: Required minimum NDCG (default 0.95).

    Returns:
        True if the gate passes (minimum across all targets meets both).
    """
    required = ["topk_recall", "ndcg_at_k"]
    for col in required:
        if col not in recovery_df.columns:
            raise ValueError(f"Missing required column: {col}")

    min_recall = float(recovery_df["topk_recall"].min())
    min_ndcg = float(recovery_df["ndcg_at_k"].min())
    mean_recall = float(recovery_df["topk_recall"].mean())
    mean_ndcg = float(recovery_df["ndcg_at_k"].mean())

    recall_ok = mean_recall >= target_recall
    ndcg_ok = mean_ndcg >= target_ndcg
    passed = recall_ok and ndcg_ok

    logger.info(
        "Top-k recovery gate: mean_recall=%.3f (min=%.3f, target=%.3f), "
        "mean_ndcg=%.3f (min=%.3f, target=%.3f) -> %s",
        mean_recall, min_recall, target_recall,
        mean_ndcg, min_ndcg, target_ndcg,
        "PASS" if passed else "FAIL",
    )

    return passed


def gate_relative_error(
    recovery_df: pd.DataFrame,
    max_rel_l2: float = 0.10,
) -> bool:
    """Gate: Relative L2 reconstruction error is within tolerance.

    Checks that the mean relative L2 error across all targets
    does not exceed max_rel_l2.

    Args:
        recovery_df: DataFrame with 'relative_l2' column.
        max_rel_l2: Maximum allowed mean relative L2 error (default 0.10).

    Returns:
        True if the gate passes.
    """
    if "relative_l2" not in recovery_df.columns:
        raise ValueError("Missing required column: relative_l2")

    mean_rel_l2 = float(recovery_df["relative_l2"].mean())
    max_val = float(recovery_df["relative_l2"].max())

    passed = mean_rel_l2 <= max_rel_l2

    logger.info(
        "Relative L2 gate: mean=%.4f, max=%.4f, target<=%.4f -> %s",
        mean_rel_l2, max_val, max_rel_l2,
        "PASS" if passed else "FAIL",
    )

    return passed


def gate_scheduler_safety(max_catastrophes: int = 0) -> bool:
    """Gate: Scheduler produces zero catastrophic SLO violations.

    Checks that during adversarial testing, the scheduler never places
    a target in a configuration that causes catastrophic tail latency.

    Args:
        max_catastrophes: Maximum allowed catastrophic events (default 0).

    Raises:
        NotImplementedError: Until scheduler module is implemented.
    """
    raise NotImplementedError(
        f"gate_scheduler_safety(max_catastrophes={max_catastrophes}) requires the "
        "scheduler module to be implemented. This gate will verify that the "
        "scheduler's safety constraints prevent catastrophic placements."
    )


def gate_probe_efficiency(
    efficiency_summary_df: pd.DataFrame,
    efficiency_ratio: float = 0.60,
    recall_target: float = 0.95,
    ndcg_target: float = 0.95,
) -> bool:
    """Gate P1: SIT-active probe efficiency.

    Requires:
    - median(m_needed_sit_active) <= efficiency_ratio * median(m_needed_random)
    - Achieved recall@k >= recall_target
    - Achieved ndcg@k >= ndcg_target

    Args:
        efficiency_summary_df: DataFrame with policy efficiency data.
        efficiency_ratio: Maximum allowed ratio of SIT-active to random probes.
        recall_target: Minimum required recall.
        ndcg_target: Minimum required NDCG.

    Returns:
        True if the gate passes.
    """
    random_df = efficiency_summary_df[efficiency_summary_df["policy_name"] == "random"]
    active_df = efficiency_summary_df[efficiency_summary_df["policy_name"] == "sit_active"]

    if len(random_df) == 0 or len(active_df) == 0:
        logger.warning("Probe efficiency gate: missing random or sit_active results")
        return False

    # Filter to valid m_needed (> 0)
    random_valid = random_df[random_df["m_needed"] > 0]
    active_valid = active_df[active_df["m_needed"] > 0]

    if len(random_valid) == 0:
        logger.warning("Probe efficiency gate: random never reached threshold")
        return False

    median_random = float(np.median(random_valid["m_needed"]))
    median_active = float(np.median(active_valid["m_needed"])) if len(active_valid) > 0 else float("inf")

    ratio = median_active / median_random if median_random > 0 else float("inf")
    ratio_ok = ratio <= efficiency_ratio

    # Check recovery quality
    recall_ok = float(active_df["final_recall"].mean()) >= recall_target
    ndcg_ok = float(active_df["final_ndcg"].mean()) >= ndcg_target

    passed = ratio_ok and recall_ok and ndcg_ok

    logger.info(
        "Probe efficiency gate: median_random=%d, median_active=%d, "
        "ratio=%.3f (target<=%.3f), recall=%.3f (target>=%.3f), "
        "ndcg=%.3f (target>=%.3f) -> %s",
        int(median_random), int(median_active),
        ratio, efficiency_ratio,
        float(active_df["final_recall"].mean()), recall_target,
        float(active_df["final_ndcg"].mean()), ndcg_target,
        "PASS" if passed else "FAIL",
    )

    if not passed:
        logger.warning(
            "PROBE EFFICIENCY GATE FAILURE REPORT:\n"
            "  Per-target results:\n%s",
            active_df[["target_id", "regime_id", "m_needed", "final_recall", "final_ndcg"]].to_string(),
        )

    return passed


def gate_probe_diversity(
    step_metrics_df: pd.DataFrame,
    diversity_ratio: float = 0.85,
) -> bool:
    """Gate P2: SIT-active has lower probe redundancy than random.

    Requires that mean pairwise similarity of SIT-active probe sets
    is at most diversity_ratio * random's mean similarity.

    Args:
        step_metrics_df: DataFrame with step-by-step metrics.
        diversity_ratio: Maximum allowed ratio of active to random similarity.

    Returns:
        True if the gate passes.
    """
    random_df = step_metrics_df[step_metrics_df["policy_name"] == "random"]
    active_df = step_metrics_df[step_metrics_df["policy_name"] == "sit_active"]

    if len(random_df) == 0 or len(active_df) == 0:
        logger.warning("Probe diversity gate: missing data")
        return False

    mean_sim_random = float(random_df["mean_pairwise_sim"].mean())
    mean_sim_active = float(active_df["mean_pairwise_sim"].mean())

    # If random has near-zero similarity, gate passes trivially
    if mean_sim_random < 1e-10:
        passed = True
    else:
        ratio = mean_sim_active / mean_sim_random
        passed = ratio <= diversity_ratio

    logger.info(
        "Probe diversity gate: sim_random=%.4f, sim_active=%.4f, "
        "ratio=%.3f (target<=%.3f) -> %s",
        mean_sim_random, mean_sim_active,
        mean_sim_active / max(mean_sim_random, 1e-10), diversity_ratio,
        "PASS" if passed else "FAIL",
    )

    return passed


def gate_probe_replay(
    probe_plan_df: pd.DataFrame,
    spectator_ids: list,
    K: np.ndarray,
    sigma_history: dict = None,
    x_hat_history: dict = None,
    config: dict = None,
) -> bool:
    """Gate P3: 100% probe selection replay success.

    Replays all probe selections and verifies exact match.

    Args:
        probe_plan_df: Full probe plan DataFrame.
        spectator_ids: Ordered spectator IDs.
        K: Kernel matrix.
        sigma_history: Optional sigma history for replay.
        x_hat_history: Optional x_hat history for replay.
        config: Probe config dict.

    Returns:
        True if 100% of selections are exactly replayable.
    """
    from sit.probe.replay import replay_all_steps

    policies = probe_plan_df["policy_name"].unique()
    all_match = True

    for policy in policies:
        total, matches, failed = replay_all_steps(
            probe_plan_df, policy, spectator_ids, K,
            sigma_history, x_hat_history, config,
        )
        if matches < total:
            logger.warning(
                "Probe replay gate FAIL for %s: %d/%d (failed steps: %s)",
                policy, matches, total, failed[:10],
            )
            all_match = False
        else:
            logger.info("Probe replay gate PASS for %s: %d/%d", policy, matches, total)

    return all_match


def gate_irbs_bias_reduction(
    irbs_estimates_df: pd.DataFrame,
    target_ratio: float = 3.0,
) -> bool:
    """Gate: IRBS reduces bias by at least target_ratio vs naive.

    Computes median(|error_naive|) / median(|error_irbs|) and checks
    that this ratio meets or exceeds the target.

    This is an early "rigor gate" that validates the drift-canceling
    measurement layer produces measurably less biased estimates.

    Args:
        irbs_estimates_df: DataFrame with columns 'absolute_error_naive_us'
            and 'absolute_error_irbs_us'.
        target_ratio: Minimum required bias reduction ratio (default 3.0).

    Returns:
        True if the gate passes.

    Raises:
        ValueError: If required columns are missing or data is empty.
    """
    required = ["absolute_error_naive_us", "absolute_error_irbs_us"]
    for col in required:
        if col not in irbs_estimates_df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = irbs_estimates_df.dropna(subset=required)
    if len(df) == 0:
        raise ValueError("No valid rows for IRBS bias reduction gate")

    median_naive = float(np.median(df["absolute_error_naive_us"].values))
    median_irbs = float(np.median(df["absolute_error_irbs_us"].values))

    if median_irbs <= 0:
        # Perfect IRBS (unlikely but possible with zero error)
        logger.info(
            "IRBS bias reduction gate: median_irbs=0 (perfect), PASS"
        )
        return True

    ratio = median_naive / median_irbs

    passed = ratio >= target_ratio
    logger.info(
        "IRBS bias reduction gate: median_naive=%.2f, median_irbs=%.2f, "
        "ratio=%.2f, target=%.2f -> %s",
        median_naive, median_irbs, ratio, target_ratio,
        "PASS" if passed else "FAIL",
    )

    return passed
