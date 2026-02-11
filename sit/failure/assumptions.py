"""SIT core assumptions modeled as first-class objects.

Each assumption has an ID, description, expected signal, violation indicator,
severity, and a check_violation() function that detects violations from state.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from sit.core.logging import get_logger

logger = get_logger("failure.assumptions")


@dataclass
class Assumption:
    """A core SIT assumption that can be checked for violations."""

    assumption_id: str
    description: str
    expected_signal: str
    violation_indicator: str
    severity: str  # "low", "medium", "high"
    _check_fn: Callable[[Dict[str, Any]], Tuple[bool, float]] = field(
        default=None, repr=False,
    )

    def check_violation(self, state: Dict[str, Any]) -> Tuple[bool, float]:
        """Check whether this assumption is violated.

        Args:
            state: Dict of pipeline state variables.

        Returns:
            Tuple of (violated: bool, signal_strength: float in [0, 1]).
        """
        if self._check_fn is not None:
            return self._check_fn(state)
        return False, 0.0


# --- Violation check functions ---

def _check_additivity(state: Dict[str, Any]) -> Tuple[bool, float]:
    """Additivity: interference approximately sums over spectators.

    Detects if residual error grows nonlinearly with the number of
    spectators in a probe set.
    """
    residuals = state.get("residuals_by_set_size", {})
    if len(residuals) < 2:
        return False, 0.0

    sizes = sorted(residuals.keys())
    vals = [residuals[s] for s in sizes]

    # Fit linear model to residual vs set_size
    x = np.array(sizes, dtype=float)
    y = np.array(vals, dtype=float)

    if len(x) < 2 or np.std(y) < 1e-12:
        return False, 0.0

    # Check nonlinearity: compare quadratic fit improvement
    linear_coeff = np.polyfit(x, y, 1)
    linear_pred = np.polyval(linear_coeff, x)
    linear_sse = np.sum((y - linear_pred) ** 2)

    if len(x) >= 3:
        quad_coeff = np.polyfit(x, y, 2)
        quad_pred = np.polyval(quad_coeff, x)
        quad_sse = np.sum((y - quad_pred) ** 2)

        # Nonlinearity ratio: if quadratic is much better, additivity is broken
        if linear_sse > 0:
            improvement = 1.0 - quad_sse / linear_sse
        else:
            improvement = 0.0

        threshold = state.get("additivity_threshold", 0.05)
        violated = improvement > threshold
        signal = min(1.0, improvement / max(threshold, 1e-12))
    else:
        violated = False
        signal = 0.0

    return violated, signal


def _check_sparsity(state: Dict[str, Any]) -> Tuple[bool, float]:
    """Sparsity: only a small subset of spectators dominate tail risk.

    Detects if x_hat density exceeds a threshold.
    """
    x_hat = state.get("x_hat")
    if x_hat is None:
        return False, 0.0

    x = np.asarray(x_hat)
    n = len(x)
    if n == 0:
        return False, 0.0

    # Density = fraction of nonzero entries
    nnz = np.sum(np.abs(x) > 1e-10)
    density = nnz / n

    threshold = state.get("sparsity_threshold", 0.5)
    violated = density > threshold
    signal = min(1.0, density / max(threshold, 1e-12))

    return violated, signal


def _check_drift_smoothness(state: Dict[str, Any]) -> Tuple[bool, float]:
    """Drift smoothness: drift is slow enough to be canceled by IRBS window.

    Detects if IRBS residual bias increases across windows.
    """
    irbs_biases = state.get("irbs_residual_biases", [])
    if len(irbs_biases) < 2:
        return False, 0.0

    biases = np.array(irbs_biases)
    abs_biases = np.abs(biases)

    # Check if bias trend is increasing
    x = np.arange(len(abs_biases), dtype=float)
    if np.std(x) < 1e-12:
        return False, 0.0

    slope = np.polyfit(x, abs_biases, 1)[0]

    # If bias is growing, drift is too fast
    threshold = state.get("drift_bias_threshold", 0.1)
    mean_bias = np.mean(abs_biases)
    signal = mean_bias / max(threshold, 1e-12) if threshold > 0 else 0.0
    violated = mean_bias > threshold and slope > 0

    return violated, min(1.0, signal)


def _check_stationarity(state: Dict[str, Any]) -> Tuple[bool, float]:
    """Stationarity: regime does not change mid measurement batch.

    Detects if measurement variance spikes within a batch.
    """
    batch_variances = state.get("within_batch_variances", [])
    if len(batch_variances) < 2:
        return False, 0.0

    vars_arr = np.array(batch_variances)
    median_var = np.median(vars_arr)
    if median_var < 1e-12:
        return False, 0.0

    # Check for variance spikes (any sub-batch > 1.5x median)
    max_ratio = np.max(vars_arr) / median_var
    threshold = state.get("stationarity_spike_threshold", 1.5)

    violated = max_ratio > threshold
    signal = min(1.0, max_ratio / max(threshold, 1e-12))

    return violated, signal


def _check_coverage(state: Dict[str, Any]) -> Tuple[bool, float]:
    """Sufficient probe coverage: design matrix A has adequate rank and coverage.

    Detects if diagnostics.rank_eff < threshold.
    """
    diagnostics = state.get("diagnostics", {})
    rank_eff = diagnostics.get("rank_eff", float("inf"))
    n_spectators = state.get("n_spectators", 1)

    if n_spectators == 0:
        return False, 0.0

    coverage_ratio = rank_eff / n_spectators
    threshold = state.get("coverage_threshold", 0.5)

    violated = coverage_ratio < threshold
    signal = 1.0 - min(1.0, coverage_ratio / max(threshold, 1e-12))

    return violated, signal


def _check_tail_validity(state: Dict[str, Any]) -> Tuple[bool, float]:
    """Tail estimator validity: enough tail samples exist to estimate p99/CVaR.

    Detects if effective sample size is below minimum.
    """
    n_tail_samples = state.get("n_tail_samples", 0)
    min_required = state.get("min_tail_samples", 20)

    if n_tail_samples >= min_required:
        return False, 0.0

    signal = 1.0 - n_tail_samples / max(min_required, 1)
    return True, min(1.0, signal)


def _check_feasibility(state: Dict[str, Any]) -> Tuple[bool, float]:
    """Scheduler feasibility: at least one safe placement exists.

    Detects if SIT-safe rejects all placements.
    """
    n_candidates = state.get("n_candidate_placements", 0)
    n_safe = state.get("n_safe_placements", 0)

    if n_candidates == 0:
        return True, 1.0

    if n_safe > 0:
        return False, 0.0

    return True, 1.0


# --- Registry of all SIT assumptions ---

def get_all_assumptions() -> List[Assumption]:
    """Return the canonical list of SIT core assumptions."""
    return [
        Assumption(
            assumption_id="A1_additivity",
            description="Interference approximately sums over spectators",
            expected_signal="Linear residual growth with probe set size",
            violation_indicator="Nonlinear residual growth (quadratic improvement > 0.05)",
            severity="high",
            _check_fn=_check_additivity,
        ),
        Assumption(
            assumption_id="A2_sparsity",
            description="Only a small subset of spectators dominate tail risk",
            expected_signal="x_hat is sparse (density < 0.5)",
            violation_indicator="x_hat density exceeds threshold",
            severity="medium",
            _check_fn=_check_sparsity,
        ),
        Assumption(
            assumption_id="A3_drift_smoothness",
            description="Drift is slow enough to be canceled by IRBS window",
            expected_signal="IRBS residual bias stable across windows",
            violation_indicator="IRBS residual bias increases across windows",
            severity="high",
            _check_fn=_check_drift_smoothness,
        ),
        Assumption(
            assumption_id="A4_stationarity",
            description="Regime does not change mid measurement batch",
            expected_signal="Within-batch variance stable",
            violation_indicator="Measurement variance spikes within batch",
            severity="medium",
            _check_fn=_check_stationarity,
        ),
        Assumption(
            assumption_id="A5_coverage",
            description="Design matrix A has adequate rank and coverage",
            expected_signal="rank_eff / n_spectators >= 0.5",
            violation_indicator="diagnostics.rank_eff < threshold",
            severity="high",
            _check_fn=_check_coverage,
        ),
        Assumption(
            assumption_id="A6_tail_validity",
            description="Enough tail samples exist to estimate p99 or CVaR",
            expected_signal="n_tail_samples >= 20",
            violation_indicator="Effective sample size below minimum",
            severity="medium",
            _check_fn=_check_tail_validity,
        ),
        Assumption(
            assumption_id="A7_feasibility",
            description="At least one safe placement exists",
            expected_signal="n_safe_placements > 0",
            violation_indicator="SIT-safe rejects all placements",
            severity="high",
            _check_fn=_check_feasibility,
        ),
    ]


def get_assumption_by_id(assumption_id: str) -> Assumption:
    """Lookup an assumption by ID."""
    for a in get_all_assumptions():
        if a.assumption_id == assumption_id:
            return a
    raise KeyError(f"Unknown assumption: {assumption_id}")
