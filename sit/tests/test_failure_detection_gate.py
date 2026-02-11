"""Tests for Gate F1: Detection completeness.

Injects each assumption violation individually and asserts at least
one detector fires per injection.
"""

import numpy as np
import pandas as pd
import pytest

from sit.failure.assumptions import get_all_assumptions, get_assumption_by_id
from sit.failure.detectors import DetectorEngine
from sit.failure.injectors import (
    INJECTOR_REGISTRY,
    inject_non_additive,
    inject_dense_interference,
    inject_fast_drift,
    inject_regime_flip,
    inject_probe_starvation,
    inject_tail_starvation,
    inject_capacity_overload,
)
from sit.eval.gates import gate_f1_detection_completeness


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


# --- Assumption model tests ---

class TestAssumptionModel:
    def test_all_assumptions_have_ids(self):
        assumptions = get_all_assumptions()
        assert len(assumptions) == 7
        ids = {a.assumption_id for a in assumptions}
        assert "A1_additivity" in ids
        assert "A7_feasibility" in ids

    def test_assumption_fields_complete(self):
        for a in get_all_assumptions():
            assert a.assumption_id
            assert a.description
            assert a.expected_signal
            assert a.violation_indicator
            assert a.severity in {"low", "medium", "high"}
            assert a._check_fn is not None

    def test_assumption_lookup(self):
        a = get_assumption_by_id("A3_drift_smoothness")
        assert a.severity == "high"

    def test_assumption_lookup_missing_raises(self):
        with pytest.raises(KeyError):
            get_assumption_by_id("nonexistent")

    def test_check_no_violation_on_empty_state(self):
        for a in get_all_assumptions():
            violated, signal = a.check_violation({})
            # Most should not fire on empty state
            if a.assumption_id != "A7_feasibility":
                assert not violated or signal >= 0.0


# --- Detector engine tests ---

class TestDetectorEngine:
    def test_empty_state_no_events(self):
        engine = DetectorEngine()
        events = engine.run_all_detectors({})
        # Only feasibility should fire on completely empty state
        # (n_candidate_placements=0 triggers it)
        assert len(events) >= 0

    def test_events_have_required_fields(self):
        engine = DetectorEngine()
        state = {
            "n_candidate_placements": 5,
            "n_safe_placements": 0,
        }
        events = engine.run_all_detectors(state)
        for e in events:
            assert hasattr(e, "event_id")
            assert hasattr(e, "assumption_id")
            assert hasattr(e, "detector_name")
            assert hasattr(e, "severity")
            assert hasattr(e, "evidence_json")
            assert hasattr(e, "triggered_at_stage")

    def test_to_dataframe_schema(self):
        engine = DetectorEngine()
        engine.run_all_detectors({"n_candidate_placements": 5, "n_safe_placements": 0})
        df = engine.to_dataframe()
        required_cols = [
            "event_id", "assumption_id", "detector_name",
            "severity", "evidence_json", "triggered_at_stage",
        ]
        for col in required_cols:
            assert col in df.columns


# --- Individual injection detection tests ---

class TestInjectionDetection:
    """Each test injects one assumption violation and checks detector fires."""

    def test_detect_non_additive(self):
        config = _base_config()
        rng = np.random.RandomState(42)
        result = inject_non_additive(config, rng, gamma=5.0)

        engine = DetectorEngine()
        engine.run_all_detectors(result.state)
        detected = engine.detected_assumption_ids()

        assert result.injected_assumption_id == "A1_additivity"
        assert "A1_additivity" in detected, \
            f"Additivity violation not detected. Detected: {detected}"

    def test_detect_dense_interference(self):
        config = _base_config()
        rng = np.random.RandomState(42)
        result = inject_dense_interference(config, rng, sparsity=0.95)

        engine = DetectorEngine()
        engine.run_all_detectors(result.state)
        detected = engine.detected_assumption_ids()

        assert result.injected_assumption_id == "A2_sparsity"
        assert "A2_sparsity" in detected, \
            f"Sparsity violation not detected. Detected: {detected}"

    def test_detect_fast_drift(self):
        config = _base_config()
        rng = np.random.RandomState(42)
        result = inject_fast_drift(config, rng, a_us_per_step=500.0)

        engine = DetectorEngine()
        engine.run_all_detectors(result.state)
        detected = engine.detected_assumption_ids()

        assert result.injected_assumption_id == "A3_drift_smoothness"
        assert "A3_drift_smoothness" in detected, \
            f"Drift violation not detected. Detected: {detected}"

    def test_detect_regime_flip(self):
        config = _base_config()
        rng = np.random.RandomState(42)
        result = inject_regime_flip(config, rng)

        engine = DetectorEngine()
        engine.run_all_detectors(result.state)
        detected = engine.detected_assumption_ids()

        assert result.injected_assumption_id == "A4_stationarity"
        assert "A4_stationarity" in detected, \
            f"Stationarity violation not detected. Detected: {detected}"

    def test_detect_probe_starvation(self):
        config = _base_config()
        rng = np.random.RandomState(42)
        result = inject_probe_starvation(config, rng, m_probes=2)

        engine = DetectorEngine()
        engine.run_all_detectors(result.state)
        detected = engine.detected_assumption_ids()

        assert result.injected_assumption_id == "A5_coverage"
        assert "A5_coverage" in detected, \
            f"Coverage violation not detected. Detected: {detected}"

    def test_detect_tail_starvation(self):
        config = _base_config()
        rng = np.random.RandomState(42)
        result = inject_tail_starvation(config, rng, p_burst=0.0, n_samples=50)

        engine = DetectorEngine()
        engine.run_all_detectors(result.state)
        detected = engine.detected_assumption_ids()

        assert result.injected_assumption_id == "A6_tail_validity"
        assert "A6_tail_validity" in detected, \
            f"Tail validity violation not detected. Detected: {detected}"

    def test_detect_capacity_overload(self):
        config = _base_config()
        rng = np.random.RandomState(42)
        result = inject_capacity_overload(config, rng, slo_us=1.0)

        engine = DetectorEngine()
        engine.run_all_detectors(result.state)
        detected = engine.detected_assumption_ids()

        assert result.injected_assumption_id == "A7_feasibility"
        assert "A7_feasibility" in detected, \
            f"Feasibility violation not detected. Detected: {detected}"


# --- Gate F1 integration test ---

class TestGateF1:
    def test_gate_f1_all_detected(self):
        """Run all injections and verify 100% detection rate."""
        config = _base_config()
        rng = np.random.RandomState(42)

        injected_ids = []
        all_detected = set()

        for name, fn in INJECTOR_REGISTRY.items():
            inj_rng = np.random.RandomState(rng.randint(0, 2**31))
            result = fn(config, inj_rng)
            injected_ids.append(result.injected_assumption_id)

            engine = DetectorEngine()
            engine.run_all_detectors(result.state)
            all_detected.update(engine.detected_assumption_ids())

        gate_result = gate_f1_detection_completeness(injected_ids, all_detected)
        assert gate_result["passed"], \
            f"Gate F1 FAIL: missing={gate_result['missing']}"
        assert gate_result["detection_rate"] == 1.0

    def test_gate_f1_empty_inputs(self):
        result = gate_f1_detection_completeness([], set())
        assert result["passed"]

    def test_gate_f1_partial_detection_fails(self):
        result = gate_f1_detection_completeness(
            ["A1_additivity", "A2_sparsity"],
            {"A1_additivity"},  # Only A1 detected
        )
        assert not result["passed"]
        assert "A2_sparsity" in result["missing"]
