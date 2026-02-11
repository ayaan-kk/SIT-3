"""Failure detectors that watch live signals and detect assumption violations.

Each detector checks pipeline state for signs of assumption violation,
triggers failure events, and records evidence.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import json
import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.failure.assumptions import get_all_assumptions, get_assumption_by_id

logger = get_logger("failure.detectors")


@dataclass
class FailureEvent:
    """A detected failure event."""

    event_id: int
    assumption_id: str
    detector_name: str
    severity: str
    evidence_json: str
    triggered_at_stage: str  # "probe", "tomo", "sched", "load"
    signal_strength: float = 0.0


class DetectorEngine:
    """Runs all assumption detectors and collects failure events."""

    def __init__(self):
        self._events: List[FailureEvent] = []
        self._next_id = 0

    @property
    def events(self) -> List[FailureEvent]:
        return list(self._events)

    def run_all_detectors(
        self,
        state: Dict[str, Any],
        stage: str = "load",
    ) -> List[FailureEvent]:
        """Run all assumption checks and record failures.

        Args:
            state: Pipeline state dict with relevant signals.
            stage: Pipeline stage ("probe", "tomo", "sched", "load").

        Returns:
            List of new failure events detected in this call.
        """
        new_events = []

        for assumption in get_all_assumptions():
            violated, signal = assumption.check_violation(state)

            if violated:
                evidence = {
                    "signal_strength": round(signal, 4),
                    "assumption_description": assumption.description,
                    "violation_indicator": assumption.violation_indicator,
                }

                # Add specific evidence from state
                evidence.update(self._extract_evidence(assumption.assumption_id, state))

                event = FailureEvent(
                    event_id=self._next_id,
                    assumption_id=assumption.assumption_id,
                    detector_name=f"detector_{assumption.assumption_id}",
                    severity=assumption.severity,
                    evidence_json=json.dumps(evidence, default=str),
                    triggered_at_stage=stage,
                    signal_strength=signal,
                )

                self._events.append(event)
                new_events.append(event)
                self._next_id += 1

                logger.warning(
                    "FAILURE DETECTED: %s (severity=%s, signal=%.3f, stage=%s)",
                    assumption.assumption_id, assumption.severity,
                    signal, stage,
                )

        return new_events

    def _extract_evidence(
        self, assumption_id: str, state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract specific evidence fields for each assumption type."""
        evidence = {}

        if assumption_id == "A1_additivity":
            residuals = state.get("residuals_by_set_size", {})
            evidence["residuals_by_set_size"] = {
                str(k): round(v, 4) for k, v in residuals.items()
            }

        elif assumption_id == "A2_sparsity":
            x_hat = state.get("x_hat")
            if x_hat is not None:
                x = np.asarray(x_hat)
                n = len(x)
                nnz = int(np.sum(np.abs(x) > 1e-10))
                evidence["density"] = round(nnz / max(n, 1), 4)
                evidence["nnz"] = nnz
                evidence["n_spectators"] = n

        elif assumption_id == "A3_drift_smoothness":
            biases = state.get("irbs_residual_biases", [])
            evidence["irbs_biases"] = [round(b, 4) for b in biases[:10]]
            evidence["mean_abs_bias"] = round(float(np.mean(np.abs(biases))), 4) if biases else 0.0

        elif assumption_id == "A4_stationarity":
            variances = state.get("within_batch_variances", [])
            if variances:
                evidence["variance_range"] = [
                    round(min(variances), 2), round(max(variances), 2),
                ]
                evidence["variance_ratio"] = round(
                    max(variances) / max(min(variances), 1e-12), 2,
                )

        elif assumption_id == "A5_coverage":
            diag = state.get("diagnostics", {})
            evidence["rank_eff"] = round(diag.get("rank_eff", 0.0), 2)
            evidence["n_spectators"] = state.get("n_spectators", 0)
            evidence["cond_est"] = round(diag.get("cond_est", float("inf")), 2)

        elif assumption_id == "A6_tail_validity":
            evidence["n_tail_samples"] = state.get("n_tail_samples", 0)
            evidence["min_required"] = state.get("min_tail_samples", 20)

        elif assumption_id == "A7_feasibility":
            evidence["n_candidates"] = state.get("n_candidate_placements", 0)
            evidence["n_safe"] = state.get("n_safe_placements", 0)

        return evidence

    def to_dataframe(self) -> pd.DataFrame:
        """Export all failure events as a DataFrame."""
        if not self._events:
            return pd.DataFrame(columns=[
                "event_id", "assumption_id", "detector_name",
                "severity", "evidence_json", "triggered_at_stage",
                "signal_strength",
            ])

        rows = []
        for e in self._events:
            rows.append({
                "event_id": e.event_id,
                "assumption_id": e.assumption_id,
                "detector_name": e.detector_name,
                "severity": e.severity,
                "evidence_json": e.evidence_json,
                "triggered_at_stage": e.triggered_at_stage,
                "signal_strength": e.signal_strength,
            })

        return pd.DataFrame(rows)

    def detected_assumption_ids(self) -> set:
        """Return set of assumption_ids that were detected."""
        return {e.assumption_id for e in self._events}
