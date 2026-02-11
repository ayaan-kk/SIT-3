"""Tests for decision replay integrity.

Verifies that probe, scheduling, and load decisions can be replayed
and produce matching results.
"""

import os
import unittest

import numpy as np
import pandas as pd


class TestProbeReplay(unittest.TestCase):
    """Test probe step replay."""

    def _get_test_data(self):
        """Generate test trials and decisions DataFrames."""
        run_id = "test-replay-run"
        trials_df = pd.DataFrame({
            "run_id": [run_id] * 5,
            "trial_id": list(range(5)),
            "scheduler_name": ["measurement_harness"] * 5,
            "target_id": [f"target_{i % 3}" for i in range(5)],
            "mean_latency_us": [100.0, 200.0, 150.0, 180.0, 120.0],
            "p99_latency_us": [500.0, 800.0, 600.0, 700.0, 550.0],
            "cvar99_latency_us": [600.0, 900.0, 700.0, 800.0, 650.0],
            "violation_rate": [0.01, 0.05, 0.02, 0.03, 0.01],
        })

        decisions_df = pd.DataFrame({
            "run_id": [run_id] * 5,
            "decision_id": list(range(5)),
            "scheduler_name": ["measurement_harness"] * 5,
            "target_id": [f"target_{i % 3}" for i in range(5)],
            "chosen_set": ['["s0","s1"]'] * 5,
            "safety_pass": [True] * 5,
            "score_components_json": ['{"risk": 0.1}'] * 5,
        })

        config = {
            "seed": 42,
            "sim": {"n_spectators": 20},
            "schedulers": ["measurement_harness"],
        }

        return run_id, config, trials_df, decisions_df

    def test_replay_probe_step(self):
        from sit.repro.replay import replay_probe_step

        run_id, config, trials_df, decisions_df = self._get_test_data()

        result = replay_probe_step(run_id, 0, config, decisions_df, trials_df)
        self.assertEqual(result["step"], 0)
        self.assertTrue(result["match"])
        self.assertIn("target_id", result)

    def test_replay_probe_step_out_of_range(self):
        from sit.repro.replay import replay_probe_step

        run_id, config, trials_df, decisions_df = self._get_test_data()

        result = replay_probe_step(run_id, 100, config, decisions_df, trials_df)
        self.assertFalse(result["match"])
        self.assertIn("error", result)

    def test_replay_deterministic(self):
        from sit.repro.replay import replay_probe_step

        run_id, config, trials_df, decisions_df = self._get_test_data()

        r1 = replay_probe_step(run_id, 2, config, decisions_df, trials_df)
        r2 = replay_probe_step(run_id, 2, config, decisions_df, trials_df)

        self.assertEqual(r1["replayed_candidates"], r2["replayed_candidates"])


class TestSchedulingReplay(unittest.TestCase):
    """Test scheduling episode replay."""

    def test_replay_scheduling_episode(self):
        from sit.repro.replay import replay_scheduling_episode

        decisions_df = pd.DataFrame({
            "decision_id": [0, 1, 2],
            "scheduler_name": ["measurement_harness"] * 3,
            "target_id": ["t0", "t1", "t2"],
            "chosen_set": ['["s0"]', '["s1"]', '["s2"]'],
            "safety_pass": [True, True, False],
            "score_components_json": ['{"risk": 0.1}', '{"risk": 0.2}', '{"risk": 0.9}'],
        })
        config = {"seed": 42}

        result = replay_scheduling_episode(0, "measurement_harness", config, decisions_df)
        self.assertTrue(result["match"])
        self.assertEqual(result["target_id"], "t0")
        self.assertEqual(result["chosen_set"], ["s0"])

    def test_replay_scheduling_out_of_range(self):
        from sit.repro.replay import replay_scheduling_episode

        decisions_df = pd.DataFrame({
            "decision_id": [0],
            "scheduler_name": ["measurement_harness"],
            "target_id": ["t0"],
            "chosen_set": ['["s0"]'],
            "safety_pass": [True],
            "score_components_json": ['{}'],
        })
        config = {"seed": 42}

        result = replay_scheduling_episode(10, "measurement_harness", config, decisions_df)
        self.assertFalse(result["match"])


class TestLoadReplay(unittest.TestCase):
    """Test load point replay."""

    def test_replay_load_point(self):
        from sit.repro.replay import replay_load_point

        trials_df = pd.DataFrame({
            "trial_id": [0, 1, 2],
            "scheduler_name": ["measurement_harness"] * 3,
            "mean_latency_us": [100.0, 200.0, 150.0],
            "p99_latency_us": [500.0, 800.0, 600.0],
            "cvar99_latency_us": [600.0, 900.0, 700.0],
            "violation_rate": [0.01, 0.05, 0.02],
        })
        config = {"seed": 42, "slo_us": 500000}

        result = replay_load_point(0, "measurement_harness", config, trials_df)
        self.assertTrue(result["match"])
        self.assertAlmostEqual(result["recomputed_goodput"], 0.99)
        self.assertAlmostEqual(result["recorded_p99_us"], 500.0)

    def test_replay_load_goodput_calculation(self):
        from sit.repro.replay import replay_load_point

        trials_df = pd.DataFrame({
            "trial_id": [0],
            "scheduler_name": ["measurement_harness"],
            "mean_latency_us": [200.0],
            "p99_latency_us": [1000.0],
            "cvar99_latency_us": [1200.0],
            "violation_rate": [0.10],
        })
        config = {"seed": 42, "slo_us": 500000}

        result = replay_load_point(0, "measurement_harness", config, trials_df)
        self.assertAlmostEqual(result["recomputed_goodput"], 0.90)

    def test_replay_load_not_found(self):
        from sit.repro.replay import replay_load_point

        trials_df = pd.DataFrame({
            "trial_id": [0],
            "scheduler_name": ["measurement_harness"],
            "mean_latency_us": [100.0],
            "p99_latency_us": [500.0],
            "cvar99_latency_us": [600.0],
            "violation_rate": [0.01],
        })
        config = {"seed": 42, "slo_us": 500000}

        result = replay_load_point(99, "measurement_harness", config, trials_df)
        self.assertFalse(result["match"])


class TestFullReplayVerification(unittest.TestCase):
    """Test the full replay verification pipeline."""

    def test_run_replay_verification(self):
        from sit.repro.replay import run_replay_verification

        run_id = "test-full-replay"
        trials_df = pd.DataFrame({
            "run_id": [run_id] * 5,
            "trial_id": list(range(5)),
            "scheduler_name": ["measurement_harness"] * 5,
            "target_id": [f"target_{i}" for i in range(5)],
            "mean_latency_us": [100.0 + i * 10 for i in range(5)],
            "p99_latency_us": [500.0 + i * 50 for i in range(5)],
            "cvar99_latency_us": [600.0 + i * 60 for i in range(5)],
            "violation_rate": [0.01] * 5,
        })

        decisions_df = pd.DataFrame({
            "run_id": [run_id] * 5,
            "decision_id": list(range(5)),
            "scheduler_name": ["measurement_harness"] * 5,
            "target_id": [f"target_{i}" for i in range(5)],
            "chosen_set": ['["s0","s1"]'] * 5,
            "safety_pass": [True] * 5,
            "score_components_json": ['{"risk": 0.1}'] * 5,
        })

        config = {
            "seed": 42,
            "sim": {"n_spectators": 20},
            "schedulers": ["measurement_harness"],
            "slo_us": 500000,
        }

        result = run_replay_verification(
            config, trials_df, decisions_df, run_id,
            n_probe_samples=3, n_sched_samples=3, n_load_samples=2,
        )

        self.assertTrue(result["all_match"])
        self.assertEqual(len(result["probe_replays"]), 3)
        self.assertEqual(len(result["sched_replays"]), 3)
        self.assertEqual(len(result["load_replays"]), 2)


class TestReportGeneration(unittest.TestCase):
    """Test report generation modules."""

    def test_generate_final_summary_table(self):
        from sit.report.tables import generate_final_summary_table

        stage_results = {
            "load": {
                "status": "completed",
                "n_artifacts": 5,
                "gates": {},
            },
            "stats": {
                "status": "completed",
                "n_artifacts": 8,
                "gates": {"R1": {"passed": True, "overall": "PASS"}},
            },
        }

        df = generate_final_summary_table(stage_results)
        self.assertTrue(len(df) > 0)
        self.assertIn("metric_name", df.columns)
        self.assertIn("value", df.columns)

    def test_generate_narrative_summary(self):
        from sit.report.summary import generate_narrative_summary

        stage_results = {
            "load": {"status": "completed", "n_artifacts": 5, "gates": {}},
        }
        env_snapshot = {
            "python_version": "3.10.0",
            "os": "Linux",
            "cpu": "test",
            "platform": "test",
        }

        summary = generate_narrative_summary(stage_results, env_snapshot, "test-run")
        self.assertIn("SIT Pipeline Summary", summary)
        self.assertIn("load", summary)

    def test_generate_figure_manifest(self):
        from sit.report.figures import generate_figure_manifest

        manifest_df = pd.DataFrame({
            "artifact_path": ["data/derived/x/irbs_estimates.parquet"],
            "sha256": ["abc123"],
        })

        md = generate_figure_manifest({}, manifest_df)
        self.assertIn("fig_irbs_bias", md)
        self.assertIn("Figure Manifest", md)

    def test_generate_interview_doc(self):
        from sit.report.faq import generate_interview_doc

        manifest_df = pd.DataFrame(columns=["artifact_path", "sha256"])

        md = generate_interview_doc({}, manifest_df)
        self.assertIn("Interview Preparation", md)
        self.assertIn("What problem", md)
        self.assertIn("One-minute", md)
        # Check all 10 sections exist
        for i in range(1, 11):
            self.assertIn(f"## {i}.", md)


if __name__ == "__main__":
    unittest.main()
