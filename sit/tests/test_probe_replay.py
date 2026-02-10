"""Tests for probe selection replay (auditability).

Verifies that:
- Probe selections can be replayed from logged state
- Replay produces identical results under deterministic settings
- Gate P3 (100% replay) passes for all policies
"""

import json

import numpy as np
import pandas as pd
import pytest

from sit.probe.diversity import build_kernel
from sit.probe.features import build_feature_matrix
from sit.probe.policies import ProbeState, select_next_probe
from sit.probe.replay import replay_probe_selection, replay_all_steps
from sit.sim.workload import Workload, N_CHANNELS


def _make_spectators(n=15, rng=None):
    """Create synthetic spectator workloads."""
    if rng is None:
        rng = np.random.RandomState(42)
    specs = []
    for i in range(n):
        specs.append(Workload(
            workload_id=f"spec_{i}",
            role="spectator",
            features=rng.uniform(0, 1, size=N_CHANNELS),
            base_service_us_mean=rng.uniform(50, 500),
            base_service_us_cv=rng.uniform(0.1, 0.5),
        ))
    return specs


def _run_and_log_probe_selections(
    spectator_ids, K, policy_name, n_steps=20, seed=42
):
    """Run probe selections and produce a probe_plan DataFrame."""
    n = len(spectator_ids)
    sigma = np.random.RandomState(seed).uniform(0.1, 1.0, size=n)

    rows = []
    coverage = {sid: 0 for sid in spectator_ids}
    chosen_so_far = []

    for step in range(1, n_steps + 1):
        state = ProbeState(
            target_id="t0",
            regime_id="r0",
            spectator_ids=spectator_ids,
            chosen_so_far=chosen_so_far,
            coverage_counts=dict(coverage),
            sigma=sigma,
            K=K,
            set_size=3,
            max_repeats=20,
            hybrid_lambda=0.5,
            epsilon=1e-6,
            rng=np.random.RandomState(seed + step),
        )

        choice = select_next_probe(state, policy_name)
        for sid in choice.chosen_set:
            coverage[sid] = coverage.get(sid, 0) + 1
        chosen_so_far.append(choice.chosen_indices)

        rows.append({
            "run_id": "test_run",
            "config_hash": "test_hash",
            "git_commit": "test",
            "seed": seed,
            "policy_name": policy_name,
            "step_m": step,
            "target_id": "t0",
            "regime_id": "r0",
            "chosen_set": json.dumps(choice.chosen_set),
            "candidate_pool_hash": "test",
            "score_breakdown_json": json.dumps(choice.score_breakdown, default=str),
            "coverage_before": json.dumps({sid: 0 for sid in spectator_ids}),
            "coverage_after": json.dumps(coverage),
            "mean_pairwise_similarity_of_set": 0.0,
        })

    return pd.DataFrame(rows)


class TestReplayDeterminism:
    """Test that probe selections can be replayed exactly."""

    @pytest.mark.parametrize("policy_name", ["random", "coverage", "diversity", "sit_active"])
    def test_replay_matches_original(self, policy_name):
        """Each step should produce identical results on replay."""
        rng = np.random.RandomState(42)
        spectators = _make_spectators(15, rng)
        spectator_ids = [s.workload_id for s in spectators]

        F = build_feature_matrix(spectators, mode="workload")
        K = build_kernel(F, kernel_type="rbf", sigma=1.0)

        # Run and log
        plan_df = _run_and_log_probe_selections(
            spectator_ids, K, policy_name, n_steps=10, seed=42,
        )

        # Sigma history (constant for this test)
        n = len(spectator_ids)
        sigma = np.random.RandomState(42).uniform(0.1, 1.0, size=n)
        sigma_history = {step: sigma for step in range(1, 11)}

        config = {
            "set_size": 3,
            "max_repeats_per_spectator": 20,
            "hybrid_lambda": 0.5,
            "epsilon": 1e-6,
        }

        # Replay each step
        for step in range(1, 11):
            replayed, matches = replay_probe_selection(
                plan_df, policy_name, step,
                spectator_ids, K,
                sigma_history=sigma_history,
                config=config,
            )
            assert matches, (
                f"Replay mismatch at step {step} for {policy_name}: "
                f"got {replayed.chosen_set}"
            )

    @pytest.mark.parametrize("policy_name", ["random", "coverage", "diversity", "sit_active"])
    def test_replay_all_steps_100_percent(self, policy_name):
        """replay_all_steps should report 100% match rate."""
        rng = np.random.RandomState(42)
        spectators = _make_spectators(15, rng)
        spectator_ids = [s.workload_id for s in spectators]

        F = build_feature_matrix(spectators, mode="workload")
        K = build_kernel(F, kernel_type="rbf", sigma=1.0)

        plan_df = _run_and_log_probe_selections(
            spectator_ids, K, policy_name, n_steps=10, seed=42,
        )

        n = len(spectator_ids)
        sigma = np.random.RandomState(42).uniform(0.1, 1.0, size=n)
        sigma_history = {step: sigma for step in range(1, 11)}

        config = {
            "set_size": 3,
            "max_repeats_per_spectator": 20,
            "hybrid_lambda": 0.5,
            "epsilon": 1e-6,
        }

        total, matches, failed = replay_all_steps(
            plan_df, policy_name, spectator_ids, K,
            sigma_history=sigma_history,
            config=config,
        )

        assert total == 10
        assert matches == 10, f"Failed steps for {policy_name}: {failed}"
        assert len(failed) == 0


class TestReplayWithMultiplePolicies:
    """Test replay across multiple policies in same plan."""

    def test_mixed_policy_replay(self):
        """Replay should work when plan has multiple policies."""
        rng = np.random.RandomState(42)
        spectators = _make_spectators(15, rng)
        spectator_ids = [s.workload_id for s in spectators]

        F = build_feature_matrix(spectators, mode="workload")
        K = build_kernel(F, kernel_type="rbf", sigma=1.0)

        plans = []
        for policy in ["random", "sit_active"]:
            plan_df = _run_and_log_probe_selections(
                spectator_ids, K, policy, n_steps=5, seed=42,
            )
            plans.append(plan_df)

        combined = pd.concat(plans, ignore_index=True)

        n = len(spectator_ids)
        sigma = np.random.RandomState(42).uniform(0.1, 1.0, size=n)
        sigma_history = {step: sigma for step in range(1, 6)}

        config = {
            "set_size": 3,
            "max_repeats_per_spectator": 20,
            "hybrid_lambda": 0.5,
            "epsilon": 1e-6,
        }

        for policy in ["random", "sit_active"]:
            total, matches, failed = replay_all_steps(
                combined, policy, spectator_ids, K,
                sigma_history=sigma_history,
                config=config,
            )
            assert matches == total, (
                f"Replay failed for {policy}: {matches}/{total}, failed={failed}"
            )
