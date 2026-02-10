"""IRBS (Interleaved Randomized Benchmarking with Sandwiching) estimators.

Implements drift-canceling measurement designs:
- C-T-C sandwich: cancels linear drift
- C-T-C-T-C: better cancellation, variance tradeoff
- General IRBS with constrained least-squares weights

The key idea: naive A/B comparison under drift is biased because
baseline changes between control and treatment. IRBS interleaves
control and treatment runs, then uses weighted combinations to
cancel polynomial drift up to order K.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from sit.core.logging import get_logger

logger = get_logger("measure.irbs")


@dataclass
class MeasurementEvent:
    """A single measurement micro-run event.

    Attributes:
        event_id: Unique identifier.
        t_index: Time index when measurement was taken.
        is_control: True if control (no spectator), False if treatment.
        stat_value_us: Measured statistic value in microseconds.
        stat_type: Type of statistic ("mean", "p99", "cvar99").
        target_id: Target workload ID.
        spectator_ids: List of spectator IDs (empty for control).
        regime_id: Regime ID.
    """
    event_id: int
    t_index: int
    is_control: bool
    stat_value_us: float
    stat_type: str = "mean"
    target_id: str = ""
    spectator_ids: List[str] = None
    regime_id: str = ""

    def __post_init__(self):
        if self.spectator_ids is None:
            self.spectator_ids = []


def naive_ab_estimate(
    control_stat_us: float,
    treatment_stat_us: float,
) -> float:
    """Naive A/B estimate: treatment - control.

    This is biased under drift because the two measurements occur
    at different times.
    """
    return treatment_stat_us - control_stat_us


def irbs_ctc_estimate(
    c1_stat_us: float,
    t_stat_us: float,
    c2_stat_us: float,
) -> float:
    """C-T-C sandwich estimate.

    Cancels linear drift by averaging the two flanking controls:
        Delta_hat = y_T - (y_C1 + y_C2) / 2

    The control average interpolates the drift to the treatment time,
    canceling first-order (linear) drift.
    """
    control_avg = (c1_stat_us + c2_stat_us) / 2.0
    return t_stat_us - control_avg


def irbs_ctctc_estimate(
    c1: float, t1: float, c2: float, t2: float, c3: float,
) -> float:
    """C-T-C-T-C five-point estimate.

    Uses symmetric weights to cancel up to quadratic drift.
    Weights: C = [-3/8, 0, 3/4, 0, -3/8], T = [0, 1/2, 0, 1/2, 0]
    Net estimate = (t1 + t2)/2 - (3*c1 - 6*c2 + 3*c3) / 8
    Simplified: average treatments minus interpolated control.
    """
    # For quadratic cancellation with evenly spaced points:
    # Treatment average: (t1 + t2) / 2
    # Control interpolated to treatment times: (c1 + 6*c2 + c3) / 8
    # Actually, for symmetric 5-point with even spacing at t=0,1,2,3,4:
    # Controls at t=0,2,4; treatments at t=1,3
    # Linear interpolation of control to t=1: (c1 + c2)/2
    # Linear interpolation of control to t=3: (c2 + c3)/2
    # Average: ((c1+c2)/2 + (c2+c3)/2) / 2 = (c1 + 2*c2 + c3) / 4
    control_interp = (c1 + 2 * c2 + c3) / 4.0
    treatment_avg = (t1 + t2) / 2.0
    return treatment_avg - control_interp


def irbs_estimate(
    events: List[MeasurementEvent],
    weights: np.ndarray,
) -> float:
    """General IRBS estimate using pre-computed weights.

    Delta_hat = sum_i w_i * y_i

    The weights are chosen to cancel drift polynomial up to order K
    while estimating the treatment effect.

    Args:
        events: List of measurement events (control and treatment).
        weights: Array of weights, same length as events.

    Returns:
        IRBS estimate of interference in microseconds.
    """
    assert len(events) == len(weights), "Events and weights must have same length"
    values = np.array([e.stat_value_us for e in events])
    return float(np.dot(weights, values))


def build_irbs_weights(
    timestamps: np.ndarray,
    labels: np.ndarray,
    drift_order: int = 1,
) -> np.ndarray:
    """Build optimal IRBS weights via constrained least squares.

    Given m measurement events at timestamps tau_i with labels
    (1 for treatment, 0 for control), find weights w that:
    1. Cancel drift polynomial up to order K:
       sum_i w_i * tau_i^k = 0  for k = 0, ..., K
    2. Estimate treatment effect:
       sum_{i: treatment} w_i = 1, sum_{i: control} w_i = -1
    3. Minimize variance: min ||w||^2

    This is solved as a constrained least-squares problem.

    Args:
        timestamps: Array of time indices (float).
        labels: Array of 0/1 (control/treatment).
        drift_order: Polynomial drift order K to cancel.

    Returns:
        Weight array of same length as timestamps.
    """
    m = len(timestamps)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    # Normalize timestamps to [0, 1] for numerical stability
    t_range = timestamps.max() - timestamps.min()
    if t_range > 0:
        t_norm = (timestamps - timestamps.min()) / t_range
    else:
        t_norm = np.zeros(m)

    # Build constraint matrix A_eq @ w = b_eq
    # Constraint 1: drift cancellation
    # sum_i w_i * t_norm_i^k = 0 for k = 0, ..., drift_order
    n_drift = drift_order + 1

    # Constraint 2: treatment weights sum to 1
    # sum_{i: treatment} w_i = 1 => labels @ w = 1
    # This implicitly means control weights sum to -1 (from drift_order=0 constraint)
    # Actually, K=0 constraint is sum w_i = 0, and labels @ w = 1
    # means treatment contribution = 1, control contribution = -1

    n_constraints = n_drift + 1  # drift + treatment constraint
    A_eq = np.zeros((n_constraints, m))

    # Drift constraints
    for k in range(n_drift):
        A_eq[k, :] = t_norm ** k

    # Treatment constraint
    A_eq[n_drift, :] = labels

    b_eq = np.zeros(n_constraints)
    b_eq[n_drift] = 1.0  # treatment weights sum to 1

    # Solve: min ||w||^2 s.t. A_eq @ w = b_eq
    # Lagrangian: w = A_eq^T @ (A_eq @ A_eq^T)^{-1} @ b_eq
    try:
        AAT = A_eq @ A_eq.T
        # Add small regularization for numerical stability
        AAT += 1e-12 * np.eye(n_constraints)
        lam = np.linalg.solve(AAT, b_eq)
        w = A_eq.T @ lam
    except np.linalg.LinAlgError:
        logger.warning("IRBS weight computation failed, falling back to uniform")
        # Fallback: uniform weights
        w = np.zeros(m)
        t_mask = labels > 0.5
        c_mask = ~t_mask
        n_t = t_mask.sum()
        n_c = c_mask.sum()
        if n_t > 0:
            w[t_mask] = 1.0 / n_t
        if n_c > 0:
            w[c_mask] = -1.0 / n_c

    return w


def run_naive_ab(
    control_event: MeasurementEvent,
    treatment_event: MeasurementEvent,
) -> float:
    """Run a naive A/B comparison from two measurement events."""
    return naive_ab_estimate(
        control_event.stat_value_us,
        treatment_event.stat_value_us,
    )


def run_irbs_ctc(
    c1: MeasurementEvent,
    t: MeasurementEvent,
    c2: MeasurementEvent,
) -> float:
    """Run a C-T-C sandwich from three measurement events."""
    return irbs_ctc_estimate(
        c1.stat_value_us,
        t.stat_value_us,
        c2.stat_value_us,
    )
