"""Tests for scheduling tail improvement gate (S2).

Verifies:
- SIT-safe reduces CVaR99 vs best non-partition baseline
- Or achieves Pareto-dominant goodput at comparable CVaR
"""

import numpy as np
import pandas as pd
import pytest

from sit.schedule.constraints import SafetyConfig
from sit.eval.scheduling import run_scheduling_evaluation
from sit.eval.gates import gate_scheduler_tail_improvement
from sit.sim.world import build_world


def _make_config():
    """Create test config for tail improvement testing."""
    return {
        "seed": 456,
        "strict_mode": True,
        "latency_unit": "us",
        "slo_us": 500000,
        "n_trials": 5,
        "output_dir": "data",
        "export_format": "parquet",
        "sim": {
            "n_targets": 6,
            "n_spectators": 40,
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
            "n_hosts": 5,
            "host_capacity": 12,
            "episodes": 50,
            "episode_generator": {
                "mode": "adversarial",
                "toxic_inclusion_rate": 0.6,
                "hard_regime_rate": 0.5,
            },
            "schedulers": [
                "random", "round_robin", "mean_greedy",
                "similarity_avoidance", "sit_safe_ucb",
            ],
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
def tail_results():
    """Run scheduling evaluation for tail improvement tests."""
    config = _make_config()
    rng = np.random.RandomState(config["seed"])
    world = build_world(config, rng)

    class FakeCtx:
        run_id = "test-tail"
        config_hash = "def456"
        git_commit = "test"
        seed = 456
        output_dir = "data"
        created_at_utc = "2024-01-01T00:00:00Z"
        raw_path = "/tmp/test_tail_raw"
        derived_path = "/tmp/test_tail_derived"

    sched_rng = np.random.RandomState(config["seed"] + 20)
    results = run_scheduling_evaluation(world, config, FakeCtx(), sched_rng)
    return world, config, results


class TestTailImprovement:
    """Tests for Gate S2: Tail improvement."""

    def test_sit_safe_lower_cvar_than_random(self, tail_results):
        """SIT-safe should have lower mean CVaR99 than random."""
        _, _, results = tail_results
        episode_df = results[2]

        random_cvar = float(episode_df[
            episode_df["scheduler_name"] == "random"
        ]["mean_cvar99"].mean())

        sit_cvar = float(episode_df[
            episode_df["scheduler_name"] == "sit_safe_ucb"
        ]["mean_cvar99"].mean())

        assert sit_cvar < random_cvar, (
            f"SIT-safe CVaR {sit_cvar:.1f} not less than random {random_cvar:.1f}"
        )

    def test_sit_safe_lower_cvar_than_round_robin(self, tail_results):
        """SIT-safe should have lower CVaR than round-robin."""
        _, _, results = tail_results
        episode_df = results[2]

        rr_cvar = float(episode_df[
            episode_df["scheduler_name"] == "round_robin"
        ]["mean_cvar99"].mean())

        sit_cvar = float(episode_df[
            episode_df["scheduler_name"] == "sit_safe_ucb"
        ]["mean_cvar99"].mean())

        assert sit_cvar < rr_cvar, (
            f"SIT-safe CVaR {sit_cvar:.1f} not less than round_robin {rr_cvar:.1f}"
        )

    def test_gate_s2_passes(self, tail_results):
        """Gate S2 should pass (tail reduction or Pareto)."""
        _, _, results = tail_results
        episode_df = results[2]

        passed = gate_scheduler_tail_improvement(
            episode_df,
            sit_scheduler="sit_safe_ucb",
            tail_reduction_ratio=0.70,
            allow_pareto=True,
        )
        assert passed, "Gate S2 should pass"

    def test_gate_s2_unit_pass(self):
        """Test S2 gate with mock data that should pass."""
        ep_df = pd.DataFrame([
            {"scheduler_name": "random", "mean_cvar99": 1000.0, "goodput": 0.9},
            {"scheduler_name": "round_robin", "mean_cvar99": 900.0, "goodput": 0.9},
            {"scheduler_name": "mean_greedy", "mean_cvar99": 800.0, "goodput": 0.92},
            {"scheduler_name": "sit_safe_ucb", "mean_cvar99": 500.0, "goodput": 0.95},
        ])

        passed = gate_scheduler_tail_improvement(
            ep_df, tail_reduction_ratio=0.70,
        )
        assert passed, "0.625 ratio should pass at 0.70 threshold"

    def test_gate_s2_unit_fail(self):
        """Test S2 gate with mock data that should fail."""
        ep_df = pd.DataFrame([
            {"scheduler_name": "random", "mean_cvar99": 1000.0, "goodput": 0.9},
            {"scheduler_name": "sit_safe_ucb", "mean_cvar99": 950.0, "goodput": 0.91},
        ])

        passed = gate_scheduler_tail_improvement(
            ep_df,
            tail_reduction_ratio=0.70,
            allow_pareto=False,
        )
        assert not passed, "0.95 ratio should fail at 0.70 threshold"

    def test_gate_s2_pareto_pass(self):
        """Test S2 Pareto alternative."""
        ep_df = pd.DataFrame([
            {"scheduler_name": "random", "mean_cvar99": 1000.0, "goodput": 0.80},
            {"scheduler_name": "sit_safe_ucb", "mean_cvar99": 980.0, "goodput": 0.95},
        ])

        passed = gate_scheduler_tail_improvement(
            ep_df,
            tail_reduction_ratio=0.70,
            allow_pareto=True,
            pareto_goodput_ratio=1.10,
            pareto_cvar_band=0.10,
        )
        assert passed, "Pareto: better goodput at comparable CVaR should pass"

    def test_goodput_comparison(self, tail_results):
        """SIT-safe should have competitive goodput."""
        _, _, results = tail_results
        episode_df = results[2]

        random_goodput = float(episode_df[
            episode_df["scheduler_name"] == "random"
        ]["goodput"].mean())

        sit_goodput = float(episode_df[
            episode_df["scheduler_name"] == "sit_safe_ucb"
        ]["goodput"].mean())

        # SIT-safe should have at least comparable goodput
        assert sit_goodput >= random_goodput * 0.95, (
            f"SIT-safe goodput {sit_goodput:.3f} too much worse than "
            f"random {random_goodput:.3f}"
        )
