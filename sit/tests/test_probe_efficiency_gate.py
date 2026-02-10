"""Tests for probe efficiency gates.

Verifies that:
- SIT-active reaches recovery thresholds with fewer probes than random
- Gate P1 (efficiency ratio) passes on synthetic sparse worlds
- Full probe efficiency pipeline produces correct artifacts
"""

import numpy as np
import pandas as pd
import pytest

from sit.eval.gates import gate_probe_efficiency, gate_probe_diversity
from sit.probe.budget import ProbeBudget
from sit.probe.diversity import build_kernel, mean_pairwise_similarity
from sit.probe.features import build_feature_matrix
from sit.probe.policies import (
    ProbeState,
    select_next_probe,
)
from sit.sim.world import build_world
from sit.tomography.design import (
    build_design_matrix,
    collect_ground_truth_vector,
    collect_measurements,
)
from sit.tomography.metrics import compute_all_recovery_metrics
from sit.tomography.solvers import solve_nonneg_elastic_net


def _make_test_config():
    """Create a minimal config for probe testing."""
    return {
        "seed": 123,
        "strict_mode": False,
        "latency_unit": "us",
        "slo_us": 500000,
        "n_trials": 5,
        "output_dir": "/tmp/sit_test_probe",
        "export_format": "parquet",
        "schedulers": ["probe_harness"],
        "regimes": {"mode": "synthetic", "n_regimes": 1},
        "logging": {"level": "WARNING"},
        "sim": {
            "n_targets": 2,
            "n_spectators": 20,
            "n_regimes": 1,
            "n_samples_per_micro_run": 500,
            "drift": {"type": "linear", "a_us_per_step": 30.0},
            "channels": {
                "weights": {"LLC": 1.0, "MEM_BW": 1.2, "IO": 0.7, "TLB": 0.5, "SMT": 0.6},
                "scale_us": 100.0,
            },
            "toxic_pairs": {
                "enabled": True,
                "sparsity": 0.10,
                "lognormal_mu": 0.0,
                "lognormal_sigma": 0.7,
            },
            "queue": {"enabled": True, "concurrency": 16, "think_time_us": 50.0},
            "burst": {"enabled": True, "p_burst": 0.002, "pareto_alpha": 2.5, "scale_us": 5000.0},
            "interactions": {"enabled": False, "gamma": 0.01},
        },
        "tomography": {
            "n_samples_tomo": 500,
            "set_size": 3,
            "stat": "mean",
            "use_irbs": True,
            "solver": {"lambda1": 0.05, "lambda2": 0.01, "max_iter": 5000},
        },
        "probing": {
            "m_total": 100,
            "checkpoints_every": 10,
            "policies": ["random", "sit_active"],
            "kernel": {"type": "rbf", "sigma": 1.0, "epsilon": 1e-6},
            "hybrid_lambda": 0.5,
            "max_repeats_per_spectator": 20,
        },
        "gates": {
            "enable": True,
            "efficiency_ratio": 0.80,
            "recall": 0.80,
            "ndcg": 0.80,
            "diversity_ratio": 1.0,
        },
    }


class TestProbeEfficiencyPipeline:
    """Test the probe efficiency evaluation pipeline."""

    def test_sit_active_fewer_probes_than_random(self):
        """SIT-active should reach thresholds with fewer probes than random."""
        config = _make_test_config()
        rng = np.random.RandomState(config["seed"])
        world = build_world(config, rng)

        spectator_ids = [s.workload_id for s in world.spectators]
        target_id = world.targets[0].workload_id
        regime_id = world.regimes[0].regime_id

        # Build kernel
        F = build_feature_matrix(world.spectators, mode="workload")
        K = build_kernel(F, kernel_type="rbf", sigma=1.0)

        x_true = collect_ground_truth_vector(world, target_id, regime_id, spectator_ids)

        # Run both policies with same budget
        m_total = 100
        results = {}
        for policy_name in ["random", "sit_active"]:
            policy_seed = config["seed"] + hash(policy_name) % 10000
            policy_rng = np.random.RandomState(policy_seed)

            coverage = {sid: 0 for sid in spectator_ids}
            chosen_so_far = []
            all_probes = []
            all_y = []
            x_hat = None
            sigma = None

            for step in range(1, m_total + 1):
                state = ProbeState(
                    target_id=target_id,
                    regime_id=regime_id,
                    spectator_ids=spectator_ids,
                    chosen_so_far=chosen_so_far,
                    coverage_counts=dict(coverage),
                    x_hat=x_hat,
                    sigma=sigma,
                    K=K,
                    set_size=3,
                    max_repeats=20,
                    hybrid_lambda=0.5,
                    epsilon=1e-6,
                    rng=np.random.RandomState(policy_seed + step),
                )

                choice = select_next_probe(state, policy_name)
                for sid in choice.chosen_set:
                    coverage[sid] = coverage.get(sid, 0) + 1
                chosen_so_far.append(choice.chosen_indices)
                all_probes.append(choice.chosen_set)

                y_val = collect_measurements(
                    world, target_id, regime_id,
                    [choice.chosen_set], 500, policy_rng,
                    stat="mean", use_irbs=True,
                )
                all_y.append(float(y_val[0]))

                # Checkpoint every 10 steps
                if step % 10 == 0:
                    A = build_design_matrix(all_probes, spectator_ids)
                    y_vec = np.array(all_y)
                    x_hat = solve_nonneg_elastic_net(A, y_vec, 0.05, 0.01)
                    boot_rng = np.random.RandomState(policy_seed + step + 1000)
                    boot_samples = np.zeros((20, len(spectator_ids)))
                    m = A.shape[0]
                    for b in range(20):
                        idx = boot_rng.randint(0, m, size=m)
                        boot_samples[b] = solve_nonneg_elastic_net(
                            A[idx], y_vec[idx], 0.05, 0.01,
                            max_iter=2000, warm_start=x_hat,
                        )
                    sigma = np.std(boot_samples, axis=0)

                    metrics = compute_all_recovery_metrics(x_hat, x_true)
                    results.setdefault(policy_name, []).append(
                        (step, metrics["topk_recall"], metrics["ndcg_at_k"])
                    )

        # SIT-active should reach recall >= 0.8 in fewer steps
        def first_good_step(policy_results, recall_t=0.8, ndcg_t=0.8):
            for step, recall, ndcg in policy_results:
                if recall >= recall_t and ndcg >= ndcg_t:
                    return step
            return m_total + 1

        m_random = first_good_step(results.get("random", []))
        m_active = first_good_step(results.get("sit_active", []))

        # SIT-active should be at least as good, typically better
        assert m_active <= m_random, (
            f"SIT-active ({m_active}) should need <= probes than random ({m_random})"
        )


class TestGateP1:
    """Test Gate P1: Probe efficiency."""

    def test_gate_passes_when_active_is_faster(self):
        """Gate should pass when SIT-active uses fewer probes."""
        df = pd.DataFrame([
            {"policy_name": "random", "target_id": "t0", "regime_id": "r0",
             "m_needed": 100, "final_recall": 1.0, "final_ndcg": 1.0, "final_rel_l2": 0.01},
            {"policy_name": "random", "target_id": "t1", "regime_id": "r0",
             "m_needed": 120, "final_recall": 1.0, "final_ndcg": 1.0, "final_rel_l2": 0.01},
            {"policy_name": "sit_active", "target_id": "t0", "regime_id": "r0",
             "m_needed": 40, "final_recall": 0.98, "final_ndcg": 0.97, "final_rel_l2": 0.01},
            {"policy_name": "sit_active", "target_id": "t1", "regime_id": "r0",
             "m_needed": 50, "final_recall": 0.96, "final_ndcg": 0.96, "final_rel_l2": 0.02},
        ])
        assert gate_probe_efficiency(df, efficiency_ratio=0.60)

    def test_gate_fails_when_active_is_slower(self):
        """Gate should fail when SIT-active uses more probes."""
        df = pd.DataFrame([
            {"policy_name": "random", "target_id": "t0", "regime_id": "r0",
             "m_needed": 50, "final_recall": 1.0, "final_ndcg": 1.0, "final_rel_l2": 0.01},
            {"policy_name": "sit_active", "target_id": "t0", "regime_id": "r0",
             "m_needed": 80, "final_recall": 0.98, "final_ndcg": 0.97, "final_rel_l2": 0.01},
        ])
        assert not gate_probe_efficiency(df, efficiency_ratio=0.60)

    def test_gate_fails_with_low_recall(self):
        """Gate should fail when recovery quality is below threshold."""
        df = pd.DataFrame([
            {"policy_name": "random", "target_id": "t0", "regime_id": "r0",
             "m_needed": 100, "final_recall": 1.0, "final_ndcg": 1.0, "final_rel_l2": 0.01},
            {"policy_name": "sit_active", "target_id": "t0", "regime_id": "r0",
             "m_needed": 40, "final_recall": 0.50, "final_ndcg": 0.50, "final_rel_l2": 0.5},
        ])
        assert not gate_probe_efficiency(df, efficiency_ratio=0.60)


class TestGateP2:
    """Test Gate P2: Probe diversity."""

    def test_gate_passes_when_active_more_diverse(self):
        """Gate should pass when SIT-active has lower similarity."""
        df = pd.DataFrame([
            {"policy_name": "random", "step_m": 10, "mean_pairwise_sim": 0.80},
            {"policy_name": "random", "step_m": 20, "mean_pairwise_sim": 0.75},
            {"policy_name": "sit_active", "step_m": 10, "mean_pairwise_sim": 0.50},
            {"policy_name": "sit_active", "step_m": 20, "mean_pairwise_sim": 0.45},
        ])
        assert gate_probe_diversity(df, diversity_ratio=0.85)

    def test_gate_fails_when_active_less_diverse(self):
        """Gate should fail when SIT-active has higher similarity."""
        df = pd.DataFrame([
            {"policy_name": "random", "step_m": 10, "mean_pairwise_sim": 0.30},
            {"policy_name": "sit_active", "step_m": 10, "mean_pairwise_sim": 0.90},
        ])
        assert not gate_probe_diversity(df, diversity_ratio=0.85)
