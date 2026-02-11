"""Tests for Gate F3: No silent catastrophes.

Forces catastrophic conditions and asserts a failure event exists
before any catastrophe.
"""

import numpy as np
import pandas as pd
import pytest

from sit.eval.ablations import audit_silent_catastrophes
from sit.eval.gates import gate_f3_no_silent_catastrophes
from sit.failure.detectors import DetectorEngine
from sit.failure.injectors import (
    inject_capacity_overload,
    inject_fast_drift,
)
from sit.failure.mitigations import apply_mitigations
from sit.failure.taxonomy import build_taxonomy, compute_system_action


# --- Test fixtures ---

def _base_config():
    return {
        "seed": 42,
        "strict_mode": False,
        "latency_unit": "us",
        "slo_us": 500000,
        "n_trials": 5,
        "output_dir": "data",
        "export_format": "parquet",
        "schedulers": ["measurement_harness"],
        "regimes": {"mode": "synthetic", "n_regimes": 2},
        "logging": {"level": "WARNING"},
        "sim": {
            "n_targets": 3,
            "n_spectators": 10,
            "n_regimes": 2,
            "n_samples_per_micro_run": 500,
            "drift": {"type": "linear", "a_us_per_step": 50.0},
            "channels": {
                "weights": {"LLC": 1.0, "MEM_BW": 1.2, "IO": 0.7},
                "scale_us": 100.0,
            },
            "toxic_pairs": {"enabled": True, "sparsity": 0.05,
                            "lognormal_mu": 0.0, "lognormal_sigma": 0.7},
            "queue": {"enabled": True, "concurrency": 16, "think_time_us": 50.0},
            "burst": {"enabled": True, "p_burst": 0.002,
                      "pareto_alpha": 2.5, "scale_us": 5000.0},
            "interactions": {"enabled": False, "gamma": 0.01},
        },
    }


# --- Silent catastrophe audit tests ---

class TestSilentCatastropheAudit:
    def test_no_catastrophes_clean_report(self):
        """No catastrophes should produce empty report."""
        trials_df = pd.DataFrame({
            "trial_id": [0, 1, 2],
            "violation_rate": [0.001, 0.002, 0.005],
            "cvar99_latency_us": [1000.0, 1100.0, 900.0],
        })
        failure_events_df = pd.DataFrame(columns=[
            "event_id", "assumption_id", "severity", "signal_strength",
        ])
        mitigation_df = pd.DataFrame(columns=["assumption_id"])

        report = audit_silent_catastrophes(
            trials_df, failure_events_df, mitigation_df,
        )
        assert len(report) == 0

    def test_catastrophe_with_detection_not_silent(self):
        """Catastrophe with preceding detection should not be silent."""
        trials_df = pd.DataFrame({
            "trial_id": [0, 1, 2],
            "violation_rate": [0.001, 0.50, 0.002],
            "cvar99_latency_us": [1000.0, 5000.0, 900.0],
        })
        failure_events_df = pd.DataFrame({
            "event_id": [0],
            "assumption_id": ["A7_feasibility"],
            "severity": ["high"],
            "signal_strength": [1.0],
        })
        mitigation_df = pd.DataFrame(columns=["assumption_id"])

        report = audit_silent_catastrophes(
            trials_df, failure_events_df, mitigation_df,
        )
        assert len(report) == 1
        assert not report.iloc[0]["silent"]

    def test_catastrophe_without_detection_is_silent(self):
        """Catastrophe without any detection should be marked silent."""
        trials_df = pd.DataFrame({
            "trial_id": [0, 1],
            "violation_rate": [0.001, 0.50],
            "cvar99_latency_us": [1000.0, 5000.0],
        })
        failure_events_df = pd.DataFrame(columns=[
            "event_id", "assumption_id", "severity", "signal_strength",
        ])
        mitigation_df = pd.DataFrame(columns=["assumption_id"])

        report = audit_silent_catastrophes(
            trials_df, failure_events_df, mitigation_df,
        )
        assert len(report) == 1
        assert report.iloc[0]["silent"]

    def test_report_schema(self):
        trials_df = pd.DataFrame({
            "trial_id": [0],
            "violation_rate": [0.50],
            "cvar99_latency_us": [5000.0],
        })
        failure_events_df = pd.DataFrame(columns=[
            "event_id", "assumption_id", "severity", "signal_strength",
        ])
        mitigation_df = pd.DataFrame(columns=["assumption_id"])

        report = audit_silent_catastrophes(
            trials_df, failure_events_df, mitigation_df,
        )
        required_cols = [
            "catastrophe_id", "trial_id", "violation_rate",
            "cvar99_latency_us", "silent", "detector_fired",
            "mitigation_applied", "missing_detector", "missing_mitigation",
        ]
        for col in required_cols:
            assert col in report.columns, f"Missing column: {col}"


# --- Gate F3 tests ---

class TestGateF3:
    def test_gate_f3_passes_no_catastrophes(self):
        result = gate_f3_no_silent_catastrophes(pd.DataFrame())
        assert result["passed"]
        assert result["n_catastrophes"] == 0
        assert result["n_silent"] == 0

    def test_gate_f3_passes_no_silent(self):
        report = pd.DataFrame({
            "catastrophe_id": [0, 1],
            "silent": [False, False],
        })
        result = gate_f3_no_silent_catastrophes(report)
        assert result["passed"]
        assert result["n_catastrophes"] == 2
        assert result["n_silent"] == 0

    def test_gate_f3_fails_on_silent(self):
        report = pd.DataFrame({
            "catastrophe_id": [0, 1],
            "silent": [False, True],
        })
        result = gate_f3_no_silent_catastrophes(report)
        assert not result["passed"]
        assert result["n_silent"] == 1

    def test_gate_f3_none_input_passes(self):
        result = gate_f3_no_silent_catastrophes(None)
        assert result["passed"]


# --- Taxonomy tests ---

class TestTaxonomy:
    def test_empty_taxonomy(self):
        df = build_taxonomy(pd.DataFrame())
        assert len(df) == 0

    def test_taxonomy_categories(self):
        events_df = pd.DataFrame({
            "event_id": [0, 1, 2],
            "assumption_id": ["A1_additivity", "A3_drift_smoothness", "A7_feasibility"],
            "severity": ["high", "high", "high"],
            "signal_strength": [0.8, 0.9, 1.0],
        })
        taxonomy = build_taxonomy(events_df)
        assert len(taxonomy) == 3

        categories = set(taxonomy["category"].tolist())
        assert "structural" in categories
        assert "statistical" in categories
        assert "control" in categories

    def test_taxonomy_actions(self):
        events_df = pd.DataFrame({
            "event_id": [0],
            "assumption_id": ["A1_additivity"],
            "severity": ["high"],
            "signal_strength": [0.9],
        })
        taxonomy = build_taxonomy(events_df)
        assert taxonomy.iloc[0]["action"] == "fallback"  # high severity = fallback

    def test_system_action_nominal(self):
        assert compute_system_action(pd.DataFrame()) == "nominal"

    def test_system_action_degrade(self):
        taxonomy = pd.DataFrame({
            "severity_score": [2],
            "action": ["degrade"],
        })
        assert compute_system_action(taxonomy) == "degrade"

    def test_system_action_abort(self):
        taxonomy = pd.DataFrame({
            "severity_score": [3, 3],
            "action": ["fallback", "fallback"],
        })
        assert compute_system_action(taxonomy) == "abort"


# --- Mitigation tests ---

class TestMitigations:
    def test_mitigations_applied(self):
        failure_events = pd.DataFrame({
            "event_id": [0, 1],
            "assumption_id": ["A1_additivity", "A7_feasibility"],
            "severity": ["high", "high"],
        })
        trials = pd.DataFrame({
            "cvar99_latency_us": [1000.0, 1200.0],
            "violation_rate": [0.01, 0.02],
        })

        mit_df, actions = apply_mitigations(failure_events, trials)
        assert len(actions) == 2
        assert len(mit_df) == 2
        assert all(mit_df["fallback_used"])

    def test_mitigation_schema(self):
        failure_events = pd.DataFrame({
            "event_id": [0],
            "assumption_id": ["A6_tail_validity"],
            "severity": ["medium"],
        })
        trials = pd.DataFrame({
            "cvar99_latency_us": [1000.0],
            "violation_rate": [0.01],
        })

        mit_df, _ = apply_mitigations(failure_events, trials)
        required_cols = [
            "failure_event_id", "assumption_id", "mitigation_applied",
            "before_cvar99", "after_cvar99", "fallback_used",
        ]
        for col in required_cols:
            assert col in mit_df.columns, f"Missing column: {col}"

    def test_empty_mitigations(self):
        mit_df, actions = apply_mitigations(pd.DataFrame(), pd.DataFrame())
        assert len(actions) == 0


# --- Integration: catastrophe detection pipeline ---

class TestCatastropheDetectionPipeline:
    def test_injected_catastrophe_detected_then_mitigated(self):
        """Full pipeline: inject -> detect -> mitigate -> audit."""
        config = _base_config()
        rng = np.random.RandomState(42)

        # Inject capacity overload (causes catastrophic violation rates)
        result = inject_capacity_overload(config, rng, slo_us=1.0)

        # Detect
        engine = DetectorEngine()
        engine.run_all_detectors(result.state)
        failure_events_df = engine.to_dataframe()

        assert len(failure_events_df) > 0, "No failures detected"

        # Mitigate
        mit_df, actions = apply_mitigations(failure_events_df, result.trials_df)

        # Audit
        report = audit_silent_catastrophes(
            result.trials_df, failure_events_df, mit_df,
            slo_us=1.0, catastrophe_threshold=0.1,
        )

        # Gate F3: no silent catastrophes
        gate_result = gate_f3_no_silent_catastrophes(report)

        # The injection should have been detected, so no silent catastrophes
        if len(report) > 0:
            assert gate_result["passed"], \
                f"Gate F3 FAIL: {gate_result['n_silent']} silent catastrophes"
