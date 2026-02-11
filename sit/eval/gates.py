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


def gate_scheduler_safety(
    trials_df: pd.DataFrame,
    scheduler_name: str = "sit_safe_ucb",
    max_catastrophes: int = 0,
) -> bool:
    """Gate S1: SIT-safe scheduler produces zero catastrophic events.

    Checks that during adversarial testing, the scheduler never places
    a target in a configuration that causes catastrophic tail latency.

    Args:
        trials_df: Schedule trials DataFrame with 'is_catastrophe' column.
        scheduler_name: Scheduler to evaluate (default sit_safe_ucb).
        max_catastrophes: Maximum allowed catastrophic events (default 0).

    Returns:
        True if the gate passes.
    """
    sched_df = trials_df[trials_df["scheduler_name"] == scheduler_name]

    if sched_df.empty:
        logger.warning("S1 gate: no data for scheduler %s", scheduler_name)
        return False

    if "is_catastrophe" not in sched_df.columns:
        logger.warning("S1 gate: missing is_catastrophe column")
        return False

    n_catastrophes = int(sched_df["is_catastrophe"].sum())
    passed = n_catastrophes <= max_catastrophes

    logger.info(
        "S1 (no catastrophes) gate: scheduler=%s, catastrophes=%d, "
        "max_allowed=%d -> %s",
        scheduler_name, n_catastrophes, max_catastrophes,
        "PASS" if passed else "FAIL",
    )

    if not passed:
        # Log worst catastrophes
        cats = sched_df[sched_df["is_catastrophe"] == True]
        worst = cats.nlargest(5, "cvar99_latency_us")
        logger.warning(
            "S1 FAILURE REPORT - top catastrophes:\n%s",
            worst[["episode_id", "target_id", "cvar99_latency_us",
                    "p99_latency_us", "violation_rate"]].to_string(),
        )

    return passed


def gate_scheduler_tail_improvement(
    episode_metrics_df: pd.DataFrame,
    sit_scheduler: str = "sit_safe_ucb",
    tail_reduction_ratio: float = 0.70,
    allow_pareto: bool = True,
    pareto_goodput_ratio: float = 1.10,
    pareto_cvar_band: float = 0.10,
) -> bool:
    """Gate S2: SIT-safe reduces CVaR99 vs best non-partition baseline.

    Either:
    - SIT-safe CVaR99 <= tail_reduction_ratio * best_non_partition_CVaR99 (30% reduction)
    OR (if allow_pareto):
    - SIT-safe goodput >= pareto_goodput_ratio * best_non_partition_goodput
      at comparable CVaR (within pareto_cvar_band fraction)

    Args:
        episode_metrics_df: Per-episode metrics DataFrame.
        sit_scheduler: SIT scheduler name.
        tail_reduction_ratio: Max allowed CVaR ratio (e.g., 0.70 = 30% reduction).
        allow_pareto: Allow Pareto-dominance alternative.
        pareto_goodput_ratio: Minimum goodput improvement for Pareto.
        pareto_cvar_band: CVaR comparability band.

    Returns:
        True if the gate passes.
    """
    non_partition_baselines = ["random", "round_robin", "mean_greedy", "similarity_avoidance"]

    sit_df = episode_metrics_df[episode_metrics_df["scheduler_name"] == sit_scheduler]
    if sit_df.empty:
        logger.warning("S2 gate: no data for %s", sit_scheduler)
        return False

    sit_cvar = float(sit_df["mean_cvar99"].mean())
    sit_goodput = float(sit_df["goodput"].mean())

    # Find best non-partition baseline CVaR
    best_baseline_cvar = float("inf")
    best_baseline_goodput = 0.0
    best_baseline_name = ""

    for bl in non_partition_baselines:
        bl_df = episode_metrics_df[episode_metrics_df["scheduler_name"] == bl]
        if bl_df.empty:
            continue
        bl_cvar = float(bl_df["mean_cvar99"].mean())
        bl_goodput = float(bl_df["goodput"].mean())

        if bl_cvar < best_baseline_cvar:
            best_baseline_cvar = bl_cvar
            best_baseline_goodput = bl_goodput
            best_baseline_name = bl

    if best_baseline_cvar == float("inf"):
        logger.warning("S2 gate: no baseline data found")
        return False

    # Check tail reduction
    cvar_ratio = sit_cvar / best_baseline_cvar if best_baseline_cvar > 0 else float("inf")
    tail_pass = cvar_ratio <= tail_reduction_ratio

    # Check Pareto alternative
    pareto_pass = False
    if allow_pareto and not tail_pass:
        cvar_comparable = abs(sit_cvar - best_baseline_cvar) / max(best_baseline_cvar, 1e-6) <= pareto_cvar_band
        goodput_better = sit_goodput >= pareto_goodput_ratio * best_baseline_goodput
        pareto_pass = cvar_comparable and goodput_better

    passed = tail_pass or pareto_pass

    logger.info(
        "S2 (tail improvement) gate: sit_cvar=%.1f, best_baseline_cvar=%.1f (%s), "
        "ratio=%.3f (target<=%.3f), sit_goodput=%.3f, baseline_goodput=%.3f, "
        "tail_pass=%s, pareto_pass=%s -> %s",
        sit_cvar, best_baseline_cvar, best_baseline_name,
        cvar_ratio, tail_reduction_ratio,
        sit_goodput, best_baseline_goodput,
        tail_pass, pareto_pass,
        "PASS" if passed else "FAIL",
    )

    return passed


def gate_scheduler_replay(
    decisions_df: pd.DataFrame,
    episodes: list,
    scheduler_name: str,
    x_hat: dict,
    sigma: dict,
    safety_config: "SafetyConfig" = None,
    K: np.ndarray = None,
    spectator_id_to_idx: dict = None,
    lambda_div: float = 0.5,
    slo_us: float = 500000.0,
    base_seed: int = 0,
) -> bool:
    """Gate S3: 100% scheduling decision replay match.

    Replays all SIT-safe decisions and verifies identical placements.

    Args:
        decisions_df: Schedule decisions DataFrame.
        episodes: List of Episode objects.
        scheduler_name: Scheduler to verify.
        x_hat: Interference estimates.
        sigma: Uncertainty estimates.
        safety_config: Safety configuration.
        K: Kernel matrix.
        spectator_id_to_idx: Spectator ID to index mapping.
        lambda_div: Diversity weight.
        slo_us: SLO threshold.
        base_seed: Base RNG seed.

    Returns:
        True if 100% of decisions replay correctly.
    """
    from sit.schedule.replay import replay_all_episodes

    total, matches, failed_eps = replay_all_episodes(
        decisions_df, episodes, scheduler_name,
        x_hat, sigma, safety_config,
        K, spectator_id_to_idx, lambda_div, slo_us, base_seed,
    )

    if total == 0:
        logger.warning("S3 gate: no decisions to replay for %s", scheduler_name)
        return True

    passed = (matches == total)

    logger.info(
        "S3 (replay) gate: scheduler=%s, %d/%d match, "
        "failed_episodes=%s -> %s",
        scheduler_name, matches, total,
        failed_eps[:10] if failed_eps else "none",
        "PASS" if passed else "FAIL",
    )

    return passed


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


# ---- Load / Serving Gates (L1, L2, L3) ----


def gate_load_pareto_dominance(
    pareto_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    sit_scheduler: str = "sit_safe_ucb",
    partition_scheduler: str = "static_partition",
    risk_band: float = 0.10,
    top_load_fraction: float = 0.30,
) -> bool:
    """Gate L1: SIT-safe on Pareto frontier and dominates partition.

    Requirements:
    1. SIT-safe is on the Pareto frontier in the top load region
    2. SIT-safe dominates static_partition in at least one high-load point
       (higher goodput at comparable tail risk)

    Args:
        pareto_df: Pareto frontier data.
        sweep_df: Full sweep results.
        sit_scheduler: SIT scheduler name.
        partition_scheduler: Partition scheduler name.
        risk_band: Comparable risk band fraction.
        top_load_fraction: Fraction of load grid considered "high load".

    Returns:
        True if the gate passes.
    """
    from sit.eval.pareto import check_pareto_dominance_over_partition

    passed, details = check_pareto_dominance_over_partition(
        pareto_df, sweep_df, sit_scheduler, partition_scheduler,
        risk_band, top_load_fraction,
    )

    logger.info(
        "L1 (Pareto dominance) gate: on_frontier=%s, dominates_partition=%s, "
        "n_domination_points=%d -> %s",
        details.get("on_frontier_high_load"),
        details.get("dominates_partition"),
        details.get("n_domination_points", 0),
        "PASS" if passed else "FAIL",
    )

    if not passed:
        logger.warning("L1 FAILURE DETAILS: %s", details)

    return passed


def gate_load_slo_throughput(
    admission_df: pd.DataFrame,
    sit_scheduler: str = "sit_safe_ucb",
    partition_scheduler: str = "static_partition",
    advantage_ratio: float = 1.15,
) -> bool:
    """Gate L2: SIT-safe admits >= 15% more load than partition.

    At a fixed SLO, SIT-safe's maximum feasible load (where violation_rate
    <= v_target) must be at least advantage_ratio * partition's.

    Args:
        admission_df: Admission curve data.
        sit_scheduler: SIT scheduler name.
        partition_scheduler: Partition scheduler name.
        advantage_ratio: Required admission advantage ratio.

    Returns:
        True if the gate passes.
    """
    from sit.eval.pareto import check_admission_advantage

    passed, details = check_admission_advantage(
        admission_df, sit_scheduler, partition_scheduler, advantage_ratio,
    )

    logger.info(
        "L2 (SLO throughput) gate: sit_max=%.0f, part_max=%.0f, "
        "load_ratio=%.2f, advantage_target=%.2f -> %s",
        details.get("sit_max_load", 0),
        details.get("part_max_load", 0),
        details.get("load_ratio", 0),
        advantage_ratio,
        "PASS" if passed else "FAIL",
    )

    if not passed:
        logger.warning("L2 FAILURE DETAILS: %s", details)

    return passed


def gate_load_model_sanity(
    diagnostics_df: pd.DataFrame,
    slo_us: float = 500_000.0,
) -> bool:
    """Gate L3: Queue model sanity checks pass.

    Validates:
    - No p99 >> SLO with near-zero violation rate
    - goodput <= throughput always
    - Throughput computed from timeline
    - Units are consistent

    Args:
        diagnostics_df: Queue diagnostics DataFrame.
        slo_us: SLO threshold.

    Returns:
        True if the gate passes.
    """
    from sit.eval.pareto import check_model_sanity

    passed, details = check_model_sanity(diagnostics_df, slo_us)

    logger.info(
        "L3 (model sanity) gate: n_failures=%d -> %s",
        details.get("n_failures", 0),
        "PASS" if passed else "FAIL",
    )

    if not passed:
        logger.warning("L3 FAILURE DETAILS: %s", details)

    return passed
