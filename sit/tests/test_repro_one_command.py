"""Tests for the one-command reproducibility pipeline.

Verifies that `sit repro --config configs/repro.yaml` runs end-to-end,
produces expected artifacts, and all stages complete successfully.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd


class TestReproEnv(unittest.TestCase):
    """Test environment capture and deterministic setup."""

    def test_set_deterministic_env(self):
        from sit.repro.env import set_deterministic_env

        set_deterministic_env(42)
        self.assertEqual(os.environ.get("PYTHONHASHSEED"), "42")

    def test_capture_environment(self):
        from sit.repro.env import capture_environment

        env = capture_environment()
        self.assertIn("python_version", env)
        self.assertIn("os", env)
        self.assertIn("cpu", env)
        self.assertIn("libraries", env)
        self.assertIn("numpy", env["libraries"])
        self.assertIn("pandas", env["libraries"])

    def test_environment_has_all_fields(self):
        from sit.repro.env import capture_environment

        env = capture_environment()
        required = ["python_version", "python_executable", "os", "platform", "machine", "cpu"]
        for field in required:
            self.assertIn(field, env, f"Missing field: {field}")


class TestReproRunner(unittest.TestCase):
    """Test the repro runner with minimal config."""

    def test_stage_order_defined(self):
        from sit.repro.runner import STAGE_ORDER

        self.assertIn("probe", STAGE_ORDER)
        self.assertIn("tomography", STAGE_ORDER)
        self.assertIn("stats", STAGE_ORDER)
        self.assertIn("failure", STAGE_ORDER)
        self.assertIn("hardware", STAGE_ORDER)

    def test_single_stage_run(self):
        """Run a single sim stage to verify the runner works."""
        from sit.core.config import load_config
        from sit.repro.runner import _run_stage

        result = _run_stage(
            "load",
            "configs/sim_smoke.yaml",
            {"seed": 123, "output_dir": "data"},
            "data",
        )

        self.assertEqual(result["status"], "completed")
        self.assertTrue(len(result["run_id"]) > 0)
        self.assertIsNotNone(result["trials_df"])
        self.assertIsNotNone(result["decisions_df"])

    def test_repro_pipeline_minimal(self):
        """Run repro pipeline with minimal stages."""
        from sit.repro.runner import run_repro_pipeline

        # Minimal config with just one stage
        config = {
            "seed": 123,
            "strict_mode": True,
            "latency_unit": "us",
            "slo_us": 500000,
            "n_trials": 10,
            "output_dir": "data",
            "export_format": "parquet",
            "schedulers": ["measurement_harness"],
            "regimes": {"mode": "synthetic", "n_regimes": 2},
            "logging": {"level": "WARNING"},
            "run_mode": "repro",
            "stages": {
                "load": {"config": "configs/sim_smoke.yaml"},
            },
            "repro": {
                "epsilon_float": 1e-8,
                "n_probe_samples": 2,
                "n_sched_samples": 2,
                "n_load_samples": 2,
            },
        }

        result = run_repro_pipeline(config)

        self.assertIn("master_run_id", result)
        self.assertIn("stage_results", result)
        self.assertIn("gates", result)
        self.assertIn("manifest_path", result)
        self.assertTrue(result["manifest_count"] > 0)

        # Check load stage completed
        load_result = result["stage_results"].get("load", {})
        self.assertEqual(load_result["status"], "completed")

    def test_repro_generates_docs(self):
        """Verify docs are generated after repro run."""
        from sit.repro.runner import run_repro_pipeline

        config = {
            "seed": 123,
            "strict_mode": True,
            "latency_unit": "us",
            "slo_us": 500000,
            "n_trials": 10,
            "output_dir": "data",
            "export_format": "parquet",
            "schedulers": ["measurement_harness"],
            "regimes": {"mode": "synthetic", "n_regimes": 2},
            "logging": {"level": "WARNING"},
            "run_mode": "repro",
            "stages": {
                "load": {"config": "configs/sim_smoke.yaml"},
            },
            "repro": {
                "epsilon_float": 1e-8,
                "n_probe_samples": 2,
                "n_sched_samples": 2,
                "n_load_samples": 2,
            },
        }

        run_repro_pipeline(config)

        # Check docs generated
        self.assertTrue(os.path.exists("docs/repro.md"))
        self.assertTrue(os.path.exists("docs/figures.md"))
        self.assertTrue(os.path.exists("docs/interview.md"))

        # Check reports generated
        self.assertTrue(os.path.exists("results/tables/final_summary.csv"))
        self.assertTrue(os.path.exists("results/reports/summary.md"))


class TestReproGates(unittest.TestCase):
    """Test reproducibility gate functions."""

    def test_gate_r0_pass(self):
        from sit.eval.gates import gate_r0_reproducibility

        stage_results = {
            "load": {"status": "completed"},
            "stats": {"status": "completed"},
        }
        manifest_df = pd.DataFrame({"artifact_path": ["a.parquet"], "sha256": ["abc"]})

        result = gate_r0_reproducibility(stage_results, manifest_df)
        self.assertTrue(result["passed"])

    def test_gate_r0_fail_incomplete(self):
        from sit.eval.gates import gate_r0_reproducibility

        stage_results = {
            "load": {"status": "completed"},
            "stats": {"status": "failed"},
        }
        manifest_df = pd.DataFrame({"artifact_path": ["a.parquet"], "sha256": ["abc"]})

        result = gate_r0_reproducibility(stage_results, manifest_df)
        self.assertFalse(result["passed"])

    def test_gate_r0_fail_empty_manifest(self):
        from sit.eval.gates import gate_r0_reproducibility

        stage_results = {"load": {"status": "completed"}}
        manifest_df = pd.DataFrame()

        result = gate_r0_reproducibility(stage_results, manifest_df)
        self.assertFalse(result["passed"])

    def test_gate_r1_pass(self):
        from sit.eval.gates import gate_r1_manifest_completeness

        # Create a temp file to reference
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            f.write(b"test")
            tmp_path = f.name

        try:
            manifest_df = pd.DataFrame({
                "artifact_path": [tmp_path],
                "sha256": ["abc"],
            })
            result = gate_r1_manifest_completeness(manifest_df)
            self.assertTrue(result["passed"])
        finally:
            os.unlink(tmp_path)

    def test_gate_r1_fail_missing(self):
        from sit.eval.gates import gate_r1_manifest_completeness

        manifest_df = pd.DataFrame({
            "artifact_path": ["/nonexistent/file.parquet"],
            "sha256": ["abc"],
        })

        result = gate_r1_manifest_completeness(manifest_df)
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["missing"]), 1)

    def test_gate_r2_pass(self):
        from sit.eval.gates import gate_r2_replay_integrity

        replay_results = {
            "probe_replays": [{"match": True}, {"match": True}],
            "sched_replays": [{"match": True}],
            "load_replays": [{"match": True}],
            "all_match": True,
        }

        result = gate_r2_replay_integrity(replay_results)
        self.assertTrue(result["passed"])

    def test_gate_r2_fail(self):
        from sit.eval.gates import gate_r2_replay_integrity

        replay_results = {
            "probe_replays": [{"match": True}, {"match": False}],
            "sched_replays": [{"match": True}],
            "load_replays": [{"match": True}],
            "all_match": False,
        }

        result = gate_r2_replay_integrity(replay_results)
        self.assertFalse(result["passed"])

    def test_gate_r2_vacuous_pass(self):
        from sit.eval.gates import gate_r2_replay_integrity

        result = gate_r2_replay_integrity(None)
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
