"""Failure taxonomy and severity rollup.

Aggregates failure events into categories (structural, statistical, control),
assigns severity scores, and determines whether the system must degrade,
fallback, or abort.
"""

from typing import Any, Dict, List

import pandas as pd

from sit.core.logging import get_logger

logger = get_logger("failure.taxonomy")

# Assumption-to-category mapping
CATEGORY_MAP = {
    "A1_additivity": "structural",
    "A2_sparsity": "structural",
    "A3_drift_smoothness": "statistical",
    "A4_stationarity": "statistical",
    "A5_coverage": "structural",
    "A6_tail_validity": "statistical",
    "A7_feasibility": "control",
}

# Severity to numeric score
SEVERITY_SCORE = {
    "low": 1,
    "medium": 2,
    "high": 3,
}

# Action thresholds
ACTION_THRESHOLDS = {
    "degrade": 2,   # severity_score >= 2 triggers degradation
    "fallback": 3,  # any high severity triggers fallback
    "abort": 6,     # cumulative score >= 6 triggers abort
}


def build_taxonomy(failure_events_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate failure events into a taxonomy.

    Args:
        failure_events_df: DataFrame with columns:
            event_id, assumption_id, severity, signal_strength

    Returns:
        DataFrame with columns:
            category, assumption_id, n_events, max_severity,
            mean_signal_strength, severity_score, action
    """
    if failure_events_df is None or len(failure_events_df) == 0:
        return pd.DataFrame(columns=[
            "category", "assumption_id", "n_events", "max_severity",
            "mean_signal_strength", "severity_score", "action",
        ])

    rows = []

    # Group by assumption_id
    for assumption_id, group in failure_events_df.groupby("assumption_id"):
        category = CATEGORY_MAP.get(assumption_id, "unknown")

        severities = group["severity"].tolist()
        max_sev = max(severities, key=lambda s: SEVERITY_SCORE.get(s, 0))
        max_score = SEVERITY_SCORE.get(max_sev, 0)
        mean_signal = float(group["signal_strength"].mean()) if "signal_strength" in group.columns else 0.0

        # Determine action
        if max_score >= ACTION_THRESHOLDS["fallback"]:
            action = "fallback"
        elif max_score >= ACTION_THRESHOLDS["degrade"]:
            action = "degrade"
        else:
            action = "monitor"

        rows.append({
            "category": category,
            "assumption_id": assumption_id,
            "n_events": len(group),
            "max_severity": max_sev,
            "mean_signal_strength": round(mean_signal, 4),
            "severity_score": max_score,
            "action": action,
        })

    taxonomy_df = pd.DataFrame(rows)

    # Check if cumulative score warrants abort
    total_score = taxonomy_df["severity_score"].sum()
    if total_score >= ACTION_THRESHOLDS["abort"]:
        logger.warning(
            "ABORT THRESHOLD REACHED: cumulative severity score=%d (threshold=%d)",
            total_score, ACTION_THRESHOLDS["abort"],
        )

    # Log taxonomy summary
    for _, row in taxonomy_df.iterrows():
        logger.info(
            "Taxonomy: [%s] %s -> %d events, severity=%s (score=%d), action=%s",
            row["category"], row["assumption_id"], row["n_events"],
            row["max_severity"], row["severity_score"], row["action"],
        )

    return taxonomy_df


def compute_system_action(taxonomy_df: pd.DataFrame) -> str:
    """Determine the overall system action from the taxonomy.

    Returns:
        One of "nominal", "degrade", "fallback", "abort".
    """
    if taxonomy_df is None or len(taxonomy_df) == 0:
        return "nominal"

    total_score = taxonomy_df["severity_score"].sum()

    if total_score >= ACTION_THRESHOLDS["abort"]:
        return "abort"

    actions = taxonomy_df["action"].unique()
    if "fallback" in actions:
        return "fallback"
    if "degrade" in actions:
        return "degrade"

    return "nominal"
