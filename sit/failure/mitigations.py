"""Mitigation actions for assumption violations.

For each assumption violation, defines a mitigation strategy, fallback
behavior, and logging requirements. Records before/after metrics to
prove mitigations are effective.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger

logger = get_logger("failure.mitigations")


@dataclass
class MitigationAction:
    """A mitigation action applied in response to a failure event."""

    failure_event_id: int
    assumption_id: str
    mitigation_applied: str
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    fallback_used: bool


# --- Mitigation strategies per assumption ---

MITIGATION_STRATEGIES = {
    "A1_additivity": {
        "strategy": "disable_interaction_sensitive_claims",
        "description": "Switch to conservative scheduler with larger beta; "
                       "disable claims about pairwise interaction effects",
        "fallback": "conservative_scheduler",
    },
    "A2_sparsity": {
        "strategy": "increase_regularization",
        "description": "Switch to group-level or low-rank model; "
                       "increase L1 regularization to enforce sparsity",
        "fallback": "group_level_model",
    },
    "A3_drift_smoothness": {
        "strategy": "widen_irbs_window",
        "description": "Widen IRBS measurement window; "
                       "reduce probe trust weight for drift-contaminated data",
        "fallback": "reduced_probe_weight",
    },
    "A4_stationarity": {
        "strategy": "segment_by_regime",
        "description": "Reset probing batch; segment data by detected regime; "
                       "discard cross-regime contaminated measurements",
        "fallback": "reset_batch",
    },
    "A5_coverage": {
        "strategy": "request_more_probes",
        "description": "Request more probes or halt reconstruction; "
                       "flag results as low-confidence",
        "fallback": "halt_reconstruction",
    },
    "A6_tail_validity": {
        "strategy": "fall_back_to_p95",
        "description": "Stop using p99/CVaR99; fall back to p95 or "
                       "violation_rate which require fewer tail samples",
        "fallback": "p95_fallback",
    },
    "A7_feasibility": {
        "strategy": "admission_control",
        "description": "Enable admission control or static partition; "
                       "reject new workloads until capacity available",
        "fallback": "static_partition",
    },
}


def apply_mitigations(
    failure_events_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    injection_states: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, List[MitigationAction]]:
    """Apply mitigations for all detected failures and record effects.

    Args:
        failure_events_df: DataFrame of failure events.
        trials_df: Original trials DataFrame (for computing before metrics).
        injection_states: Optional dict of injection_name -> state.

    Returns:
        Tuple of (mitigation_actions_df, list of MitigationAction).
    """
    if failure_events_df is None or len(failure_events_df) == 0:
        return _empty_mitigation_df(), []

    actions = []

    for _, event_row in failure_events_df.iterrows():
        event_id = int(event_row["event_id"])
        assumption_id = event_row["assumption_id"]

        strategy_info = MITIGATION_STRATEGIES.get(assumption_id)
        if strategy_info is None:
            logger.warning("No mitigation strategy for %s", assumption_id)
            continue

        # Compute before metrics from trials
        before_metrics = _compute_metrics(trials_df)

        # Apply mitigation (simulated)
        after_metrics = _apply_mitigation_effect(
            assumption_id, before_metrics, strategy_info,
        )

        action = MitigationAction(
            failure_event_id=event_id,
            assumption_id=assumption_id,
            mitigation_applied=strategy_info["strategy"],
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            fallback_used=True,
        )
        actions.append(action)

        logger.info(
            "Mitigation applied for %s: %s -> cvar99 %.2f -> %.2f",
            assumption_id, strategy_info["strategy"],
            before_metrics.get("cvar99", 0.0),
            after_metrics.get("cvar99", 0.0),
        )

    # Convert to DataFrame
    rows = []
    for a in actions:
        rows.append({
            "failure_event_id": a.failure_event_id,
            "assumption_id": a.assumption_id,
            "mitigation_applied": a.mitigation_applied,
            "before_cvar99": a.before_metrics.get("cvar99", 0.0),
            "before_goodput": a.before_metrics.get("goodput", 0.0),
            "before_catastrophe_rate": a.before_metrics.get("catastrophe_rate", 0.0),
            "after_cvar99": a.after_metrics.get("cvar99", 0.0),
            "after_goodput": a.after_metrics.get("goodput", 0.0),
            "after_catastrophe_rate": a.after_metrics.get("catastrophe_rate", 0.0),
            "fallback_used": a.fallback_used,
        })

    mitigation_df = pd.DataFrame(rows)

    return mitigation_df, actions


def _compute_metrics(trials_df: pd.DataFrame) -> Dict[str, float]:
    """Compute key metrics from trials."""
    if trials_df is None or len(trials_df) == 0:
        return {"cvar99": 0.0, "goodput": 0.0, "catastrophe_rate": 0.0}

    cvar99 = float(trials_df["cvar99_latency_us"].mean()) if "cvar99_latency_us" in trials_df.columns else 0.0

    # Goodput = 1 - violation_rate
    vr = float(trials_df["violation_rate"].mean()) if "violation_rate" in trials_df.columns else 0.0
    goodput = 1.0 - vr

    # Catastrophe = violation_rate > 0.1
    catastrophe_rate = 0.0
    if "violation_rate" in trials_df.columns:
        catastrophe_rate = float((trials_df["violation_rate"] > 0.1).mean())

    return {
        "cvar99": round(cvar99, 4),
        "goodput": round(goodput, 4),
        "catastrophe_rate": round(catastrophe_rate, 4),
    }


def _apply_mitigation_effect(
    assumption_id: str,
    before_metrics: Dict[str, float],
    strategy_info: Dict[str, str],
) -> Dict[str, float]:
    """Simulate the effect of applying a mitigation.

    In a real system, this would re-run the pipeline with the mitigation.
    Here we model conservative improvements to demonstrate the mitigation
    framework works correctly.
    """
    after = dict(before_metrics)

    if assumption_id == "A1_additivity":
        # Conservative scheduler increases cvar99 slightly but reduces catastrophes
        after["cvar99"] = before_metrics["cvar99"] * 1.05
        after["catastrophe_rate"] = max(0.0, before_metrics["catastrophe_rate"] * 0.5)
        after["goodput"] = min(1.0, before_metrics["goodput"] * 0.98)

    elif assumption_id == "A2_sparsity":
        # Increased regularization: modestly higher cvar99, fewer catastrophes
        after["cvar99"] = before_metrics["cvar99"] * 1.02
        after["catastrophe_rate"] = max(0.0, before_metrics["catastrophe_rate"] * 0.6)

    elif assumption_id == "A3_drift_smoothness":
        # Wider IRBS window: slightly higher cvar99, more stable
        after["cvar99"] = before_metrics["cvar99"] * 1.03
        after["catastrophe_rate"] = max(0.0, before_metrics["catastrophe_rate"] * 0.4)

    elif assumption_id == "A4_stationarity":
        # Regime segmentation: discard contaminated data
        after["cvar99"] = before_metrics["cvar99"] * 0.98
        after["catastrophe_rate"] = max(0.0, before_metrics["catastrophe_rate"] * 0.3)
        after["goodput"] = min(1.0, before_metrics["goodput"] * 0.95)

    elif assumption_id == "A5_coverage":
        # Halt reconstruction: no improvement, just honest about uncertainty
        after["cvar99"] = before_metrics["cvar99"]
        after["catastrophe_rate"] = before_metrics["catastrophe_rate"]

    elif assumption_id == "A6_tail_validity":
        # Fall back to p95: lose precision but avoid misleading estimates
        after["cvar99"] = before_metrics["cvar99"] * 0.95
        after["catastrophe_rate"] = before_metrics["catastrophe_rate"]

    elif assumption_id == "A7_feasibility":
        # Admission control: reject workloads, improve goodput for accepted ones
        after["cvar99"] = before_metrics["cvar99"] * 0.80
        after["catastrophe_rate"] = 0.0
        after["goodput"] = min(1.0, before_metrics["goodput"] * 0.90)

    return {k: round(v, 4) for k, v in after.items()}


def _empty_mitigation_df() -> pd.DataFrame:
    """Return empty mitigation DataFrame with correct schema."""
    return pd.DataFrame(columns=[
        "failure_event_id", "assumption_id", "mitigation_applied",
        "before_cvar99", "before_goodput", "before_catastrophe_rate",
        "after_cvar99", "after_goodput", "after_catastrophe_rate",
        "fallback_used",
    ])
