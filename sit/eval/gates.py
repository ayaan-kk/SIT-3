"""Minimum winning metrics acceptance gates.

Placeholder gates that will be implemented by later modules.
Each gate defines a measurable acceptance criterion (e.g., "95%+ CI coverage").
They raise NotImplementedError until the relevant module provides real data.
"""


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
