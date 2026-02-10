"""Minimum winning metrics acceptance gates.

Each gate defines a measurable acceptance criterion. Placeholder gates
raise NotImplementedError until the relevant module provides real data.
Implemented gates return True/False with detailed logging.
"""

import numpy as np
import pandas as pd

from sit.core.logging import get_logger

logger = get_logger("eval.gates")


def gate_ci_coverage(target: float = 0.95) -> bool:
    """Gate: Confidence interval empirical coverage meets target.

    Checks that the observed coverage of uncertainty intervals on
    held-out data meets or exceeds the target (e.g., 95% nominal -> 95% actual).

    Args:
        target: Required coverage fraction (default 0.95).

    Raises:
        NotImplementedError: Until tomography/uncertainty module is implemented.
    """
    raise NotImplementedError(
        f"gate_ci_coverage(target={target}) requires the tomography/uncertainty "
        "module to be implemented. This gate will check that empirical CI coverage "
        "on synthetic ground truth meets the target."
    )


def gate_topk_recovery(target: float = 0.95) -> bool:
    """Gate: Top-K toxic spectator recovery accuracy meets target.

    Checks that the tomography solver correctly identifies the top-K
    most interfering spectators with accuracy >= target.

    Args:
        target: Required recovery accuracy fraction (default 0.95).

    Raises:
        NotImplementedError: Until tomography module is implemented.
    """
    raise NotImplementedError(
        f"gate_topk_recovery(target={target}) requires the tomography module "
        "to be implemented. This gate will check that the solver correctly "
        "recovers the top-K interfering spectators."
    )


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
