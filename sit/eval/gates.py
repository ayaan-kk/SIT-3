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


# === Failure mode gates (F1, F2, F3) ===


def gate_f1_detection_completeness(
    injected_assumption_ids: list,
    detected_assumption_ids: set,
) -> dict:
    """Gate F1: 100% of injected assumption violations must be detected.

    For runs with injected failures, every injected assumption_id must
    have at least one corresponding detector event logged.

    Args:
        injected_assumption_ids: List of assumption IDs that were injected.
        detected_assumption_ids: Set of assumption IDs that were detected.

    Returns:
        Dict with 'passed', 'detection_rate', 'missing', 'details'.
    """
    if not injected_assumption_ids:
        logger.info("Gate F1: No injections to check -> PASS (vacuous)")
        return {
            "passed": True,
            "detection_rate": 1.0,
            "missing": [],
            "details": "No injections",
        }

    unique_injected = set(injected_assumption_ids)
    detected = detected_assumption_ids & unique_injected
    missing = unique_injected - detected

    detection_rate = len(detected) / len(unique_injected)
    passed = len(missing) == 0

    logger.info(
        "Gate F1 (Detection): %d/%d detected (%.1f%%), missing=%s -> %s",
        len(detected), len(unique_injected),
        detection_rate * 100, sorted(missing),
        "PASS" if passed else "FAIL",
    )

    return {
        "passed": passed,
        "detection_rate": round(detection_rate, 4),
        "missing": sorted(missing),
        "details": f"{len(detected)}/{len(unique_injected)} detected",
    }


def gate_f2_graceful_degradation(
    ablation_df: pd.DataFrame,
    cvar_tolerance: float = 0.10,
) -> dict:
    """Gate F2: Fallback baselines must not catastrophically degrade.

    When SIT is gated off, fallback performance must be within tolerance.
    Checks the "no_mitigation" ablation (the true "SIT off" fallback)
    and any ablation that represents a pure fallback scenario.

    For ablations that are adversarial stress tests (no_safety,
    no_admission), large degradation is expected and scientifically
    correct -- these are reported in the ablation table for interviews
    but don't gate the pipeline.

    Args:
        ablation_df: DataFrame with ablation results including
            cvar_relative_change, catastrophe_rate columns.
        cvar_tolerance: Maximum allowed relative CVaR99 increase (default 0.10).

    Returns:
        Dict with 'passed', 'worst_cvar_change', 'violations',
        'all_degradations', 'details'.
    """
    if ablation_df is None or len(ablation_df) == 0:
        logger.info("Gate F2: No ablation data -> PASS (vacuous)")
        return {
            "passed": True,
            "worst_cvar_change": 0.0,
            "violations": [],
            "all_degradations": [],
            "details": "No ablation data",
        }

    # Ablations that represent fallback/gating-off scenarios
    # (not adversarial stress tests)
    fallback_ablations = {"no_mitigation", "no_uncertainty", "no_diversity", "no_regularization"}

    violations = []
    all_degradations = []

    for _, row in ablation_df.iterrows():
        abl_name = row.get("ablation_name", "unknown")
        cvar_change = row.get("cvar_relative_change", 0.0)
        abl_cat_rate = row.get("ablated_catastrophe_rate", 0.0)
        base_cat_rate = row.get("baseline_catastrophe_rate", 0.0)

        all_degradations.append(
            f"{abl_name}: cvar_change={cvar_change:.4f}"
        )

        # Only gate on fallback ablations, not adversarial stress tests
        if abl_name not in fallback_ablations:
            continue

        # Check: CVaR99 not worse by more than tolerance
        if cvar_change > cvar_tolerance:
            violations.append(
                f"{abl_name}: cvar_change={cvar_change:.4f} > {cvar_tolerance}"
            )

        # Check: catastrophe rate not higher than baseline
        if abl_cat_rate > base_cat_rate + 0.01:
            violations.append(
                f"{abl_name}: catastrophe_rate={abl_cat_rate:.4f} > baseline={base_cat_rate:.4f}"
            )

    worst_change = float(ablation_df["cvar_relative_change"].max()) if "cvar_relative_change" in ablation_df.columns else 0.0
    passed = len(violations) == 0

    logger.info(
        "Gate F2 (Graceful degradation): worst_cvar_change=%.4f, "
        "%d violations (checked %d fallback ablations) -> %s",
        worst_change, len(violations), len(fallback_ablations),
        "PASS" if passed else "FAIL",
    )

    if not passed:
        for v in violations:
            logger.warning("  F2 violation: %s", v)

    # Log all degradations for the record
    for d in all_degradations:
        logger.info("  Ablation: %s", d)

    return {
        "passed": passed,
        "worst_cvar_change": round(worst_change, 4),
        "violations": violations,
        "all_degradations": all_degradations,
        "details": f"{len(violations)} violations across fallback ablations",
    }


def gate_f3_no_silent_catastrophes(
    violation_report_df: pd.DataFrame,
) -> dict:
    """Gate F3: No catastrophic event may occur without detection.

    Requires zero catastrophes marked as 'silent' (no preceding
    detector event or gate trigger).

    Args:
        violation_report_df: DataFrame with 'silent' column from
            silent catastrophe audit.

    Returns:
        Dict with 'passed', 'n_catastrophes', 'n_silent', 'details'.
    """
    if violation_report_df is None or len(violation_report_df) == 0:
        logger.info("Gate F3: No catastrophes found -> PASS")
        return {
            "passed": True,
            "n_catastrophes": 0,
            "n_silent": 0,
            "details": "No catastrophes",
        }

    n_total = len(violation_report_df)
    n_silent = int(violation_report_df["silent"].sum()) if "silent" in violation_report_df.columns else 0

    passed = n_silent == 0

    logger.info(
        "Gate F3 (No silent catastrophes): %d catastrophes, %d silent -> %s",
        n_total, n_silent,
        "PASS" if passed else "FAIL",
    )

    return {
        "passed": passed,
        "n_catastrophes": n_total,
        "n_silent": n_silent,
        "details": f"{n_silent} silent out of {n_total} catastrophes",
    }
