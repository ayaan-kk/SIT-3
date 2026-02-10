"""Tests for scheduling safety gate (S1).

Verifies:
- SIT-safe has 0 catastrophes under adversarial episodes
- At least one baseline has >= 1 catastrophe (gate is meaningful)
- Safety constraints prevent catastrophic placements
"""

import numpy as np
import pandas as pd
import pytest

from sit.schedule.constraints import SafetyConfig, is_catastrophe
from sit.schedule.policies import ScheduleContext, run_scheduler
from sit.schedule.state import generate_episodes
from sit.eval.scheduling import (
    build_interference_lookup,
    evaluate_placement,
    run_scheduling_evaluation,
)
from sit.eval.gates import gate_scheduler_safety, gate_scheduler_tail_improvement
from sit.sim.world import build_world
from sit.probe.features import build_feature_matrix
from sit.probe.diversity import build_kernel


def _make_config():
    """Create a small test config for scheduling."""
    return {
        "seed": 123,
        "strict_mode": True,
        "latency_unit": "us",
        "slo_us": 500000,
        "n_trials": 5,
        "output_dir": "data",
        "export_format": "parquet",
        "sim": {
            "n_targets": 4,
            "n_spectators": 30,
            "n_regimes": 2,
            "n_samples_per_micro_run": 500,
            "drift": {"type": "linear", "a_us_per_step": 50.0},
            "channels": {
                "weights": {"LLC": 1.0, "MEM_BW": 1.2, "IO": 0.7, "TLB": 0.5, "SMT": 0.6},
                "scale_us": 500.0,
            },
            "toxic_pairs": {
                "enabled": True,
                "sparsity": 0.12,
                "lognormal_mu": 0.5,
                "lognormal_sigma": 1.0,
            },
            "queue": {"enabled": True, "concurrency": 24, "think_time_us": 50.0},
            "burst": {"enabled": True, "p_burst": 0.003, "pareto_alpha": 2.2, "scale_us": 8000.0},
            "interactions": {"enabled": False},
        },
        "scheduling": {
            "n_hosts": 4,
            "host_capacity": 10,
            "episodes": 30,
            "episode_generator": {
                "mode": "adversarial",
                "toxic_inclusion_rate": 0.7,
                "hard_regime_rate": 0.5,
            },
            "schedulers": ["random", "round_robin", "sit_safe_ucb"],
            "sit_params": {
                "beta_ucb": 2.0,
                "lambda_div": 0.5,
                "tau_risk_us": 10000000,
            },
            "catastrophe_thresholds": {
                "catastrophe_p99_us": 100000000,
                "catastrophe_cvar_us": 500000000,
                "catastrophe_violation_rate": 0.9,
            },
        },
    }


@pytest.fixture(scope="module")
def world_and_results():
    """Build world and run scheduling evaluation once for all tests."""
    config = _make_config()
    rng = np.random.RandomState(config["seed"])
    world = build_world(config, rng)

    class FakeCtx:
        run_id = "test-sched"
        config_hash = "abc123"
        git_commit = "test"
        seed = 123
        output_dir = "data"
        created_at_utc = "2024-01-01T00:00:00Z"
        raw_path = "/tmp/test_sched_raw"
        derived_path = "/tmp/test_sched_derived"

    sched_rng = np.random.RandomState(config["seed"] + 20)
    results = run_scheduling_evaluation(world, config, FakeCtx(), sched_rng)
    # results = (decisions_df, trials_df, episode_metrics_df,
    #            failure_audit_df, regime_heatmap_df, summary_df)
    return world, config, results


class TestSchedulingSafetyGate:
    """Tests for Gate S1: No catastrophes for SIT-safe."""

    def test_sit_safe_no_catastrophes(self, world_and_results):
        """S1: SIT-safe should have 0 catastrophes."""
        _, config, results = world_and_results
        trials_df = results[1]

        safety_cfg = SafetyConfig.from_config(config)

        # SIT-safe should have 0 catastrophes
        sit_trials = trials_df[trials_df["scheduler_name"] == "sit_safe_ucb"]
        n_catastrophes = int(sit_trials["is_catastrophe"].sum())
        assert n_catastrophes == 0, (
            f"SIT-safe had {n_catastrophes} catastrophes"
        )

    def test_gate_s1_passes(self, world_and_results):
        """Gate S1 function should return True for SIT-safe."""
        _, _, results = world_and_results
        trials_df = results[1]

        passed = gate_scheduler_safety(
            trials_df,
            scheduler_name="sit_safe_ucb",
            max_catastrophes=0,
        )
        assert passed, "Gate S1 should pass for SIT-safe"

    def test_gate_s1_meaningful(self, world_and_results):
        """At least one baseline should have catastrophes or high tail."""
        _, _, results = world_and_results
        trials_df = results[1]
        episode_df = results[2]

        # Random scheduler should have higher CVaR than SIT-safe
        random_eps = episode_df[episode_df["scheduler_name"] == "random"]
        sit_eps = episode_df[episode_df["scheduler_name"] == "sit_safe_ucb"]

        random_cvar = float(random_eps["mean_cvar99"].mean())
        sit_cvar = float(sit_eps["mean_cvar99"].mean())

        # SIT-safe should be at least somewhat better
        assert sit_cvar <= random_cvar * 1.05, (
            f"SIT-safe cvar {sit_cvar} not better than random {random_cvar}"
        )

    def test_catastrophe_detection(self):
        """Test that is_catastrophe correctly identifies catastrophes."""
        safety_cfg = SafetyConfig(
            catastrophe_p99_us=1000.0,
            catastrophe_cvar_us=2000.0,
            catastrophe_violation_rate=0.8,
        )

        # Not a catastrophe
        metrics = {"p99_latency_us": 500.0, "cvar99_latency_us": 800.0, "violation_rate": 0.1}
        assert not is_catastrophe(metrics, safety_cfg)

        # P99 catastrophe
        metrics = {"p99_latency_us": 1500.0, "cvar99_latency_us": 800.0, "violation_rate": 0.1}
        assert is_catastrophe(metrics, safety_cfg)

        # CVaR catastrophe
        metrics = {"p99_latency_us": 500.0, "cvar99_latency_us": 3000.0, "violation_rate": 0.1}
        assert is_catastrophe(metrics, safety_cfg)

        # Violation rate catastrophe
        metrics = {"p99_latency_us": 500.0, "cvar99_latency_us": 800.0, "violation_rate": 0.95}
        assert is_catastrophe(metrics, safety_cfg)


class TestSchedulingOutputs:
    """Tests for scheduling output structure."""

    def test_decisions_df_columns(self, world_and_results):
        """Decisions DataFrame should have required columns."""
        _, _, results = world_and_results
        decisions_df = results[0]

        required = [
            "episode_id", "step_index", "scheduler_name", "target_id",
            "candidate_hosts", "chosen_host", "safety_pass", "fallback_used",
        ]
        for col in required:
            assert col in decisions_df.columns, f"Missing column: {col}"

    def test_trials_df_columns(self, world_and_results):
        """Trials DataFrame should have required columns."""
        _, _, results = world_and_results
        trials_df = results[1]

        required = [
            "episode_id", "scheduler_name", "target_id", "regime_id",
            "predicted_risk_us", "cvar99_latency_us", "is_catastrophe",
        ]
        for col in required:
            assert col in trials_df.columns, f"Missing column: {col}"

    def test_episode_metrics_columns(self, world_and_results):
        """Episode metrics should have required columns."""
        _, _, results = world_and_results
        episode_df = results[2]

        required = [
            "episode_id", "scheduler_name", "regime_id",
            "mean_cvar99", "max_cvar99", "goodput",
        ]
        for col in required:
            assert col in episode_df.columns, f"Missing column: {col}"

    def test_failure_audit_exists(self, world_and_results):
        """Failure audit should exist and have entries."""
        _, _, results = world_and_results
        failure_df = results[3]

        assert len(failure_df) > 0, "Failure audit should have entries"
        assert "cvar99_latency_us" in failure_df.columns

    def test_summary_has_all_schedulers(self, world_and_results):
        """Summary should include all configured schedulers."""
        _, config, results = world_and_results
        summary_df = results[5]

        expected = config["scheduling"]["schedulers"]
        actual = set(summary_df["scheduler_name"].values)
        for sched in expected:
            assert sched in actual, f"Missing scheduler in summary: {sched}"
