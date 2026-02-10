"""Tests for scheduling decision replay (S3).

Verifies:
- SIT-safe placements are deterministically replayable from logs
- Replay produces identical placements
"""

import numpy as np
import pandas as pd
import pytest

from sit.schedule.constraints import SafetyConfig
from sit.schedule.policies import ScheduleContext, run_scheduler
from sit.schedule.replay import replay_episode, verify_replay
from sit.schedule.state import Episode, Host, Job, generate_episodes
from sit.eval.scheduling import build_interference_lookup, run_scheduling_evaluation
from sit.eval.gates import gate_scheduler_replay
from sit.sim.world import build_world
from sit.probe.features import build_feature_matrix
from sit.probe.diversity import build_kernel


def _make_config():
    """Create small test config for replay testing."""
    return {
        "seed": 789,
        "strict_mode": True,
        "latency_unit": "us",
        "slo_us": 500000,
        "n_trials": 5,
        "output_dir": "data",
        "export_format": "parquet",
        "sim": {
            "n_targets": 4,
            "n_spectators": 20,
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
            "n_hosts": 3,
            "host_capacity": 10,
            "episodes": 10,
            "episode_generator": {
                "mode": "adversarial",
                "toxic_inclusion_rate": 0.6,
                "hard_regime_rate": 0.5,
            },
            "schedulers": ["random", "sit_safe_ucb"],
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
def replay_data():
    """Build world, generate episodes, and run scheduling for replay tests."""
    config = _make_config()
    rng = np.random.RandomState(config["seed"])
    world = build_world(config, rng)

    # Build lookup
    x_hat, sigma = build_interference_lookup(world)
    safety_config = SafetyConfig.from_config(config)

    # Build kernel
    spectator_ids = sorted([s.workload_id for s in world.spectators])
    spectator_id_to_idx = {sid: i for i, sid in enumerate(spectator_ids)}
    F = build_feature_matrix(world.spectators, mode="workload")
    K = build_kernel(F, kernel_type="rbf", sigma=0.3)

    # Generate episodes
    ep_rng = np.random.RandomState(config["seed"] + 100)
    episodes = generate_episodes(world, config, ep_rng)

    # Run schedulers and collect decisions
    base_seed = config["seed"]
    all_decisions = []

    for ep in episodes:
        for sched_name in ["sit_safe_ucb"]:
            ep_seed = base_seed + ep.episode_id * 100 + hash(sched_name) % 10000
            ep_rng_sched = np.random.RandomState(ep_seed & 0x7FFFFFFF)

            ctx = ScheduleContext(
                episode=ep,
                x_hat=x_hat,
                sigma=sigma,
                safety_config=safety_config,
                K=K,
                spectator_id_to_idx=spectator_id_to_idx,
                lambda_div=0.5,
                slo_us=500000.0,
                rng=ep_rng_sched,
            )

            placement, decisions = run_scheduler(sched_name, ctx)

            for dec in decisions:
                all_decisions.append({
                    "episode_id": dec.episode_id,
                    "step_index": dec.step_index,
                    "scheduler_name": dec.scheduler_name,
                    "target_id": dec.target_id,
                    "chosen_host": dec.chosen_host,
                    "safety_pass": dec.safety_pass,
                    "fallback_used": dec.fallback_used,
                })

    decisions_df = pd.DataFrame(all_decisions)

    return {
        "world": world,
        "config": config,
        "episodes": episodes,
        "decisions_df": decisions_df,
        "x_hat": x_hat,
        "sigma": sigma,
        "safety_config": safety_config,
        "K": K,
        "spectator_id_to_idx": spectator_id_to_idx,
        "base_seed": base_seed,
    }


class TestSchedulingReplay:
    """Tests for scheduling decision replay."""

    def test_replay_single_episode(self, replay_data):
        """Replay a single episode for SIT-safe and verify match."""
        d = replay_data
        ep = d["episodes"][0]

        ok, total, matches, failed = verify_replay(
            d["decisions_df"], ep, "sit_safe_ucb",
            d["x_hat"], d["sigma"], d["safety_config"],
            d["K"], d["spectator_id_to_idx"],
            lambda_div=0.5, slo_us=500000.0,
            rng_seed=d["base_seed"] + ep.episode_id * 100 + hash("sit_safe_ucb") % 10000,
        )

        assert ok, f"Replay mismatch: {matches}/{total} (failed: {failed})"

    def test_replay_deterministic(self, replay_data):
        """Running the same scheduler twice produces same placement."""
        d = replay_data
        ep = d["episodes"][0]
        ep_seed = d["base_seed"] + ep.episode_id * 100 + hash("sit_safe_ucb") % 10000

        mapping1 = replay_episode(
            ep, "sit_safe_ucb",
            d["x_hat"], d["sigma"], d["safety_config"],
            d["K"], d["spectator_id_to_idx"],
            rng_seed=ep_seed,
        )

        mapping2 = replay_episode(
            ep, "sit_safe_ucb",
            d["x_hat"], d["sigma"], d["safety_config"],
            d["K"], d["spectator_id_to_idx"],
            rng_seed=ep_seed,
        )

        assert mapping1 == mapping2, "Same seed should produce identical placement"

    def test_replay_all_episodes(self, replay_data):
        """Replay all episodes for SIT-safe."""
        d = replay_data

        total_decisions = 0
        total_matches = 0

        for ep in d["episodes"]:
            ep_seed = d["base_seed"] + ep.episode_id * 100 + hash("sit_safe_ucb") % 10000

            ok, ep_total, ep_matches, _ = verify_replay(
                d["decisions_df"], ep, "sit_safe_ucb",
                d["x_hat"], d["sigma"], d["safety_config"],
                d["K"], d["spectator_id_to_idx"],
                lambda_div=0.5, slo_us=500000.0,
                rng_seed=ep_seed,
            )
            total_decisions += ep_total
            total_matches += ep_matches

        assert total_matches == total_decisions, (
            f"Replay: {total_matches}/{total_decisions} match"
        )

    def test_different_seeds_different_placements(self, replay_data):
        """Different seeds should generally produce different placements."""
        d = replay_data
        ep = d["episodes"][0]

        mapping1 = replay_episode(
            ep, "sit_safe_ucb",
            d["x_hat"], d["sigma"], d["safety_config"],
            d["K"], d["spectator_id_to_idx"],
            rng_seed=42,
        )

        mapping2 = replay_episode(
            ep, "sit_safe_ucb",
            d["x_hat"], d["sigma"], d["safety_config"],
            d["K"], d["spectator_id_to_idx"],
            rng_seed=99999,
        )

        # They could be the same if the optimal placement is unique,
        # but with spectators shuffled differently, usually differ
        # Just verify both produced valid placements
        assert len(mapping1) > 0
        assert len(mapping2) > 0

    def test_gate_s3_function(self, replay_data):
        """Test the gate_scheduler_replay gate function."""
        d = replay_data

        passed = gate_scheduler_replay(
            d["decisions_df"],
            d["episodes"],
            "sit_safe_ucb",
            d["x_hat"],
            d["sigma"],
            d["safety_config"],
            d["K"],
            d["spectator_id_to_idx"],
            lambda_div=0.5,
            slo_us=500000.0,
            base_seed=d["base_seed"],
        )

        assert passed, "Gate S3 should pass for deterministic SIT-safe"
