"""Tests for Gate F2: Graceful degradation.

Runs ablations disabling SIT components and asserts fallback performance
is not catastrophically worse than baseline.
"""

import numpy as np
import pandas as pd
import pytest

from sit.eval.ablations import (
    ABLATION_DEFS,
    run_ablation_study,
    audit_silent_catastrophes,
)
from sit.eval.gates import gate_f2_graceful_degradation


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


# --- Ablation framework tests ---

class TestAblationDefinitions:
    def test_all_ablation_names_exist(self):
        expected = [
            "no_irbs", "no_uncertainty", "no_diversity",
            "no_regularization", "no_safety", "no_admission",
            "no_mitigation",
        ]
        for name in expected:
            assert name in ABLATION_DEFS, f"Missing ablation: {name}"

    def test_ablation_defs_have_required_fields(self):
        for name, abl_def in ABLATION_DEFS.items():
            assert "description" in abl_def, f"{name} missing description"
            assert "component" in abl_def, f"{name} missing component"
            assert "config_overrides" in abl_def, f"{name} missing config_overrides"


class TestAblationStudy:
    def test_ablation_study_produces_results(self):
        config = _base_config()
        rng = np.random.RandomState(42)

        # Run a subset of ablations for speed
        df = run_ablation_study(config, rng, ablation_list=["no_irbs", "no_safety"])

        assert len(df) == 2
        assert "ablation_name" in df.columns
        assert "delta_cvar" in df.columns
        assert "delta_goodput" in df.columns
        assert "catastrophe_rate" in df.columns

    def test_ablation_study_all_columns(self):
        config = _base_config()
        rng = np.random.RandomState(42)

        df = run_ablation_study(config, rng, ablation_list=["no_irbs"])

        required_cols = [
            "ablation_name", "component_removed", "description",
            "baseline_cvar99", "ablated_cvar99", "delta_cvar",
            "cvar_relative_change",
            "baseline_goodput", "ablated_goodput", "delta_goodput",
            "baseline_catastrophe_rate", "ablated_catastrophe_rate",
            "catastrophe_rate", "probe_efficiency", "notes",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_ablation_with_baseline_trials(self):
        config = _base_config()
        rng = np.random.RandomState(42)

        # Create mock baseline trials
        baseline = pd.DataFrame({
            "cvar99_latency_us": [1000.0, 1200.0, 900.0],
            "violation_rate": [0.01, 0.02, 0.005],
        })

        df = run_ablation_study(
            config, rng,
            ablation_list=["no_mitigation"],
            baseline_trials_df=baseline,
        )
        assert len(df) == 1

    def test_ablation_unknown_name_skipped(self):
        config = _base_config()
        rng = np.random.RandomState(42)

        df = run_ablation_study(
            config, rng,
            ablation_list=["nonexistent_ablation"],
        )
        assert len(df) == 0


# --- Gate F2 tests ---

class TestGateF2:
    def test_gate_f2_passes_within_tolerance(self):
        """Ablation with small degradation should pass."""
        df = pd.DataFrame({
            "ablation_name": ["no_irbs", "no_safety"],
            "cvar_relative_change": [0.05, 0.08],
            "ablated_catastrophe_rate": [0.01, 0.02],
            "baseline_catastrophe_rate": [0.01, 0.01],
        })

        result = gate_f2_graceful_degradation(df, cvar_tolerance=0.10)
        assert result["passed"]

    def test_gate_f2_fails_on_cvar_violation(self):
        """Fallback ablation exceeding tolerance should fail."""
        # Use a fallback ablation name (not adversarial like no_safety)
        df = pd.DataFrame({
            "ablation_name": ["no_mitigation"],
            "cvar_relative_change": [0.25],
            "ablated_catastrophe_rate": [0.02],
            "baseline_catastrophe_rate": [0.01],
        })

        result = gate_f2_graceful_degradation(df, cvar_tolerance=0.10)
        assert not result["passed"]
        assert len(result["violations"]) > 0

    def test_gate_f2_fails_on_catastrophe_increase(self):
        """Fallback ablation with higher catastrophe rate should fail."""
        df = pd.DataFrame({
            "ablation_name": ["no_uncertainty"],
            "cvar_relative_change": [0.05],
            "ablated_catastrophe_rate": [0.20],
            "baseline_catastrophe_rate": [0.01],
        })

        result = gate_f2_graceful_degradation(df, cvar_tolerance=0.10)
        assert not result["passed"]

    def test_gate_f2_adversarial_ablations_ignored(self):
        """Adversarial ablations (no_safety, no_irbs) don't trigger gate."""
        df = pd.DataFrame({
            "ablation_name": ["no_safety", "no_irbs"],
            "cvar_relative_change": [2.5, 3.0],
            "ablated_catastrophe_rate": [0.20, 0.30],
            "baseline_catastrophe_rate": [0.01, 0.01],
        })

        result = gate_f2_graceful_degradation(df, cvar_tolerance=0.10)
        assert result["passed"]  # Adversarial ablations are not gated

    def test_gate_f2_empty_data_passes(self):
        result = gate_f2_graceful_degradation(pd.DataFrame())
        assert result["passed"]

    def test_gate_f2_none_data_passes(self):
        result = gate_f2_graceful_degradation(None)
        assert result["passed"]

    def test_gate_f2_integration(self):
        """Run actual ablation study and check gate F2."""
        config = _base_config()
        rng = np.random.RandomState(42)

        df = run_ablation_study(
            config, rng,
            ablation_list=["no_irbs", "no_diversity", "no_mitigation"],
        )

        # With the default SLO of 500000us, ablations should be within tolerance
        result = gate_f2_graceful_degradation(df, cvar_tolerance=0.10)
        assert isinstance(result, dict)
        assert "passed" in result
        assert "worst_cvar_change" in result
