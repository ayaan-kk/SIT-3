"""Tests for the full benchmarking system.

Verifies scheduling policies, execution matrix, metrics, figures,
and the end-to-end bench pipeline.
"""

import os
import unittest

import numpy as np
import pandas as pd


class TestPolicyRegistry(unittest.TestCase):
    """Test that all scheduling policies are registered and callable."""

    def test_registry_has_19_policies(self):
        from sit.sched import POLICY_REGISTRY
        self.assertEqual(len(POLICY_REGISTRY), 19)

    def test_all_policies_callable(self):
        from sit.sched import POLICY_REGISTRY
        for name, fn in POLICY_REGISTRY.items():
            self.assertTrue(callable(fn), f"Policy {name} is not callable")

    def test_get_policy(self):
        from sit.sched import get_policy
        fn = get_policy("SIT-safe")
        self.assertTrue(callable(fn))

    def test_get_policy_invalid(self):
        from sit.sched import get_policy
        with self.assertRaises(ValueError):
            get_policy("nonexistent-policy")

    def test_get_all_policy_names(self):
        from sit.sched import get_all_policy_names
        names = get_all_policy_names()
        self.assertEqual(len(names), 19)
        self.assertIn("SIT-safe", names)
        self.assertIn("oracle", names)
        self.assertIn("k8s-hpa", names)


class TestPolicyExecution(unittest.TestCase):
    """Test that each policy runs and returns a PlacementDecision."""

    def _build_world(self):
        from sit.sim.world import build_world
        config = {
            "seed": 42,
            "slo_us": 500000,
            "n_spectators_per_placement": 3,
            "sim": {
                "n_spectators": 10,
                "n_targets": 3,
                "n_regimes": 2,
                "feature_dim": 4,
                "n_samples_per_micro_run": 100,
                "queue": {"enabled": True, "concurrency": 8, "think_time_us": 50.0},
                "burst": {"enabled": False},
                "drift": {"type": "none"},
                "toxic_pairs": {"enabled": True, "sparsity": 0.05},
                "interactions": {"enabled": False},
            },
            "regimes": {"mode": "synthetic", "n_regimes": 2},
        }
        rng = np.random.RandomState(42)
        return build_world(config, rng), config

    def test_all_policies_return_placement(self):
        from sit.sched import POLICY_REGISTRY, PlacementDecision
        world, config = self._build_world()
        rng = np.random.RandomState(42)

        for name, fn in POLICY_REGISTRY.items():
            target = world.targets[0]
            regime = world.regimes[0]
            result = fn(target, world, regime, config, rng, trial_id=0)
            self.assertIsInstance(result, PlacementDecision, f"Policy {name} did not return PlacementDecision")
            self.assertEqual(result.scheduler_name, name, f"Policy {name} has wrong scheduler_name")
            self.assertIsInstance(result.spectator_ids, list)

    def test_partition_returns_empty_spectators(self):
        from sit.sched import get_policy
        world, config = self._build_world()
        rng = np.random.RandomState(42)
        fn = get_policy("partition")
        result = fn(world.targets[0], world, world.regimes[0], config, rng)
        self.assertEqual(len(result.spectator_ids), 0)

    def test_oracle_uses_ground_truth(self):
        from sit.sched import get_policy
        world, config = self._build_world()
        rng = np.random.RandomState(42)
        fn = get_policy("oracle")
        result = fn(world.targets[0], world, world.regimes[0], config, rng)
        self.assertTrue(result.score_components.get("ground_truth_access"))


class TestExecutionMatrix(unittest.TestCase):
    """Test the execution matrix components."""

    def test_build_regime_configs(self):
        from sit.bench.executor import build_regime_configs
        regimes = build_regime_configs()
        self.assertEqual(len(regimes), 6)
        self.assertIn("iid_noise", regimes)
        self.assertIn("adversarial", regimes)
        self.assertIn("drifting", regimes)

    def test_build_load_configs(self):
        from sit.bench.executor import build_load_configs
        loads = build_load_configs()
        self.assertEqual(len(loads), 4)
        self.assertIn("low", loads)
        self.assertIn("saturation", loads)

    def test_build_job_mix_configs(self):
        from sit.bench.executor import build_job_mix_configs
        mixes = build_job_mix_configs()
        self.assertEqual(len(mixes), 4)

    def test_run_single_evaluation(self):
        from sit.bench.executor import run_single_evaluation
        from sit.sim.world import build_world

        config = {
            "seed": 42,
            "slo_us": 500000,
            "n_spectators_per_placement": 3,
            "sim": {
                "n_spectators": 10, "n_targets": 3, "n_regimes": 2,
                "feature_dim": 4, "n_samples_per_micro_run": 100,
                "queue": {"enabled": True, "concurrency": 8, "think_time_us": 50.0},
                "burst": {"enabled": False},
                "drift": {"type": "none"},
                "toxic_pairs": {"enabled": True, "sparsity": 0.05},
                "interactions": {"enabled": False},
            },
            "regimes": {"mode": "synthetic", "n_regimes": 2},
        }
        rng = np.random.RandomState(42)
        world = build_world(config, rng)

        result_df = run_single_evaluation("random", world, config, n_episodes=5, seed=42)
        self.assertEqual(len(result_df), 5)
        self.assertIn("policy", result_df.columns)
        self.assertIn("cvar99_us", result_df.columns)
        self.assertIn("effective_goodput_rps", result_df.columns)
        self.assertTrue((result_df["policy"] == "random").all())


class TestMetrics(unittest.TestCase):
    """Test metrics computation."""

    def _make_results_df(self):
        rng = np.random.RandomState(42)
        n = 100
        rows = []
        for policy in ["SIT-safe", "partition", "random"]:
            for i in range(n):
                rows.append({
                    "policy": policy,
                    "episode_id": i,
                    "interference_regime": "iid_noise" if i % 2 == 0 else "structured",
                    "load_regime": "medium",
                    "mean_latency_us": rng.uniform(100, 300),
                    "p95_latency_us": rng.uniform(200, 500),
                    "p99_latency_us": rng.uniform(300, 800),
                    "p999_latency_us": rng.uniform(500, 1200),
                    "cvar95_us": rng.uniform(250, 600),
                    "cvar99_us": rng.uniform(400, 1000),
                    "cvar999_us": rng.uniform(600, 1500),
                    "violation_rate": rng.uniform(0, 0.05),
                    "success_rate": rng.uniform(0.90, 1.0),
                    "goodput_rps": rng.uniform(1000, 5000),
                    "effective_goodput_rps": rng.uniform(3000, 15000),
                    "throughput_rps": rng.uniform(1000, 5000),
                    "queue_variance": rng.uniform(0, 100),
                    "mean_backlog_us": rng.uniform(0, 50),
                    "max_backlog_us": rng.uniform(50, 200),
                    "catastrophe": rng.choice([True, False], p=[0.02, 0.98]),
                    "decision_time_us": rng.uniform(10, 500),
                    "n_spectators": rng.randint(1, 5),
                })
        return pd.DataFrame(rows)

    def test_compute_aggregate_metrics(self):
        from sit.bench.metrics import compute_aggregate_metrics
        df = self._make_results_df()
        agg = compute_aggregate_metrics(df)
        self.assertTrue(len(agg) > 0)
        self.assertIn("policy", agg.columns)

    def test_compute_pairwise_comparisons(self):
        from sit.bench.metrics import compute_pairwise_comparisons
        df = self._make_results_df()
        comp = compute_pairwise_comparisons(df, n_bootstrap=100)
        self.assertTrue(len(comp) > 0)
        self.assertIn("cohens_d", comp.columns)
        self.assertIn("p_value", comp.columns)

    def test_compute_fairness_table(self):
        from sit.bench.metrics import compute_fairness_table
        df = self._make_results_df()
        fair = compute_fairness_table(df)
        self.assertEqual(len(fair), 19)
        self.assertIn("information_level", fair.columns)

    def test_compute_dominance_metrics(self):
        from sit.bench.metrics import compute_dominance_metrics
        df = self._make_results_df()
        dom = compute_dominance_metrics(df)
        self.assertTrue(len(dom) > 0)
        self.assertIn("sit_dominates", dom.columns)


class TestFigures(unittest.TestCase):
    """Test figure generation."""

    def _make_results_df(self):
        rng = np.random.RandomState(42)
        rows = []
        policies = ["SIT-safe", "partition", "random", "oracle", "static-high"]
        for policy in policies:
            for i in range(50):
                rows.append({
                    "policy": policy,
                    "episode_id": i,
                    "interference_regime": ["iid_noise", "structured"][i % 2],
                    "load_regime": ["low", "medium", "high", "saturation"][i % 4],
                    "load_label": ["20-40%", "50-70%", "80-95%", "95-105%"][i % 4],
                    "mean_latency_us": rng.uniform(100, 300),
                    "p95_latency_us": rng.uniform(200, 500),
                    "p99_latency_us": rng.uniform(300, 800),
                    "p999_latency_us": rng.uniform(500, 1200),
                    "cvar95_us": rng.uniform(250, 600),
                    "cvar99_us": rng.uniform(400, 1000),
                    "cvar999_us": rng.uniform(600, 1500),
                    "violation_rate": rng.uniform(0, 0.05),
                    "success_rate": rng.uniform(0.90, 1.0),
                    "goodput_rps": rng.uniform(1000, 5000),
                    "effective_goodput_rps": rng.uniform(3000, 15000),
                    "throughput_rps": rng.uniform(1000, 5000),
                    "queue_variance": rng.uniform(0, 100),
                    "mean_backlog_us": rng.uniform(0, 50),
                    "max_backlog_us": rng.uniform(50, 200),
                    "catastrophe": rng.choice([True, False], p=[0.02, 0.98]),
                    "decision_time_us": rng.uniform(10, 500),
                    "n_spectators": rng.randint(1, 5),
                    "safety_pass": True,
                    "seed": 42,
                    "target_id": f"t_{i % 3}",
                    "regime_id": f"r_{i % 2}",
                    "slo_us": 500000,
                    "n_samples": 100,
                })
        return pd.DataFrame(rows)

    def test_generate_all_figures(self):
        import tempfile
        from sit.bench.figures import generate_all_figures

        df = self._make_results_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_all_figures(df, output_dir=tmpdir)
            self.assertTrue(len(paths) >= 30, f"Expected >= 30 figures, got {len(paths)}")
            # Check CSV files exist
            for fig_id, fpath in paths.items():
                self.assertTrue(os.path.exists(fpath), f"Missing figure: {fig_id} at {fpath}")


class TestPaper(unittest.TestCase):
    """Test LaTeX paper generation."""

    def test_generate_paper(self):
        import tempfile
        from sit.bench.paper import generate_paper

        rng = np.random.RandomState(42)
        rows = []
        for policy in ["SIT-safe", "partition", "random"]:
            for i in range(20):
                rows.append({
                    "policy": policy, "episode_id": i,
                    "cvar99_us": rng.uniform(300, 800),
                    "success_rate": rng.uniform(0.90, 1.0),
                    "goodput_rps": rng.uniform(1000, 5000),
                    "effective_goodput_rps": rng.uniform(3000, 15000),
                    "throughput_rps": rng.uniform(1000, 5000),
                    "p99_latency_us": rng.uniform(200, 600),
                    "violation_rate": rng.uniform(0, 0.05),
                    "catastrophe": False,
                    "decision_time_us": rng.uniform(10, 200),
                    "n_spectators": 3,
                    "interference_regime": "iid_noise",
                    "load_regime": "medium",
                })
        df = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_paper(df, {}, {}, output_dir=tmpdir)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                content = f.read()
            self.assertIn("\\documentclass", content)
            self.assertIn("\\begin{document}", content)
            self.assertIn("\\end{document}", content)


class TestAcceptanceCriteria(unittest.TestCase):
    """Test acceptance criteria verification."""

    def test_verify_criteria(self):
        from sit.bench.pipeline import verify_acceptance_criteria

        rng = np.random.RandomState(42)
        rows = []
        for policy in ["SIT-safe", "partition", "random", "oracle", "k8s-hpa", "k8s-default", "slurm-fcfs", "triton-proxy"]:
            for i in range(20):
                is_sit = policy == "SIT-safe"
                is_oracle = policy == "oracle"
                is_partition = policy == "partition"
                cvar = rng.uniform(300, 600) if (is_sit or is_oracle) else rng.uniform(400, 900)
                n_specs = 3 if not is_partition else 0
                # Effective goodput: SIT/oracle co-locate (high), partition is isolated (low)
                egp = rng.uniform(8000, 15000) if (is_sit or is_oracle) else (
                    rng.uniform(2000, 4000) if is_partition else rng.uniform(5000, 10000)
                )
                rows.append({
                    "policy": policy, "episode_id": i,
                    "cvar99_us": cvar,
                    "p99_latency_us": cvar * 0.9,
                    "success_rate": rng.uniform(0.90, 1.0) if is_sit else rng.uniform(0.80, 0.95),
                    "goodput_rps": rng.uniform(2000, 5000),
                    "effective_goodput_rps": egp,
                    "throughput_rps": rng.uniform(2000, 5000),
                    "violation_rate": rng.uniform(0, 0.03) if is_sit else rng.uniform(0.02, 0.08),
                    "catastrophe": False,
                    "decision_time_us": rng.uniform(10, 200),
                    "n_spectators": n_specs,
                    "interference_regime": "iid_noise",
                    "load_regime": "medium",
                })
        df = pd.DataFrame(rows)
        dom_df = pd.DataFrame([{"sit_dominates": True}])
        stats_paths = {"a": "a", "b": "b", "c": "c", "d": "d"}

        criteria = verify_acceptance_criteria(df, dom_df, stats_paths)
        self.assertIn("_summary", criteria)
        # At least basic criteria should pass
        self.assertTrue(criteria["A1_fairness_table_exists"]["passed"])
        self.assertTrue(criteria["C1_catastrophe_rate_low"]["passed"])


if __name__ == "__main__":
    unittest.main()
