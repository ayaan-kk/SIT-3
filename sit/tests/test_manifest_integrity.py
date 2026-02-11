"""Tests for artifact manifest integrity.

Verifies that manifests correctly catalog artifacts, hashes match,
and the hash discipline tools work correctly.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd


class TestHashTree(unittest.TestCase):
    """Test deterministic directory hashing."""

    def test_hash_single_file(self):
        from sit.repro.hashes import hash_tree

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            tmp_path = f.name

        try:
            h = hash_tree(tmp_path)
            self.assertEqual(len(h), 64)  # SHA-256 hex digest

            # Same content = same hash
            h2 = hash_tree(tmp_path)
            self.assertEqual(h, h2)
        finally:
            os.unlink(tmp_path)

    def test_hash_directory(self):
        from sit.repro.hashes import hash_tree

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            with open(os.path.join(tmpdir, "a.txt"), "w") as f:
                f.write("file a")
            with open(os.path.join(tmpdir, "b.txt"), "w") as f:
                f.write("file b")

            h1 = hash_tree(tmpdir)
            h2 = hash_tree(tmpdir)
            self.assertEqual(h1, h2)

    def test_hash_ignores_pycache(self):
        from sit.repro.hashes import hash_tree

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "code.py"), "w") as f:
                f.write("pass")

            h1 = hash_tree(tmpdir)

            # Add __pycache__ - should not change hash
            cache_dir = os.path.join(tmpdir, "__pycache__")
            os.makedirs(cache_dir)
            with open(os.path.join(cache_dir, "code.pyc"), "w") as f:
                f.write("compiled")

            h2 = hash_tree(tmpdir)
            self.assertEqual(h1, h2)

    def test_hash_changes_on_content_change(self):
        from sit.repro.hashes import hash_tree

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "data.txt")
            with open(fpath, "w") as f:
                f.write("version 1")
            h1 = hash_tree(tmpdir)

            with open(fpath, "w") as f:
                f.write("version 2")
            h2 = hash_tree(tmpdir)

            self.assertNotEqual(h1, h2)


class TestFloatTolerantComparison(unittest.TestCase):
    """Test float-tolerant parquet comparison."""

    def test_identical_parquets_match(self):
        from sit.repro.hashes import compare_parquets_float_tolerant

        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": ["a", "b", "c"]})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path_a = f.name
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path_b = f.name

        try:
            df.to_parquet(path_a)
            df.to_parquet(path_b)

            result = compare_parquets_float_tolerant(path_a, path_b)
            self.assertTrue(result["match"])
            self.assertEqual(len(result["mismatches"]), 0)
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_within_epsilon_match(self):
        from sit.repro.hashes import compare_parquets_float_tolerant

        df_a = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        df_b = pd.DataFrame({"x": [1.0 + 1e-10, 2.0, 3.0 - 1e-10]})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path_a = f.name
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path_b = f.name

        try:
            df_a.to_parquet(path_a)
            df_b.to_parquet(path_b)

            result = compare_parquets_float_tolerant(path_a, path_b, epsilon=1e-8)
            self.assertTrue(result["match"])
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_beyond_epsilon_mismatch(self):
        from sit.repro.hashes import compare_parquets_float_tolerant

        df_a = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        df_b = pd.DataFrame({"x": [1.0, 2.1, 3.0]})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path_a = f.name
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path_b = f.name

        try:
            df_a.to_parquet(path_a)
            df_b.to_parquet(path_b)

            result = compare_parquets_float_tolerant(path_a, path_b, epsilon=1e-8)
            self.assertFalse(result["match"])
            self.assertTrue(len(result["mismatches"]) > 0)
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_shape_mismatch(self):
        from sit.repro.hashes import compare_parquets_float_tolerant

        df_a = pd.DataFrame({"x": [1.0, 2.0]})
        df_b = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path_a = f.name
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path_b = f.name

        try:
            df_a.to_parquet(path_a)
            df_b.to_parquet(path_b)

            result = compare_parquets_float_tolerant(path_a, path_b)
            self.assertFalse(result["match"])
        finally:
            os.unlink(path_a)
            os.unlink(path_b)


class TestManifestBuild(unittest.TestCase):
    """Test artifact manifest building."""

    def test_build_manifest_from_stage_results(self):
        from sit.repro.manifest import build_artifact_manifest

        # Need a stage that actually produced output
        from sit.repro.runner import _run_stage

        result = _run_stage(
            "load",
            "configs/sim_smoke.yaml",
            {"seed": 123, "output_dir": "data"},
            "data",
        )

        stage_results = {"load": result}
        manifest_df = build_artifact_manifest("data", "test-run-id", stage_results)

        self.assertTrue(len(manifest_df) > 0)
        self.assertIn("artifact_path", manifest_df.columns)
        self.assertIn("sha256", manifest_df.columns)
        self.assertIn("producing_stage", manifest_df.columns)

    def test_verify_manifest_all_exist(self):
        from sit.repro.manifest import build_artifact_manifest, verify_manifest
        from sit.repro.runner import _run_stage

        result = _run_stage(
            "load",
            "configs/sim_smoke.yaml",
            {"seed": 123, "output_dir": "data"},
            "data",
        )

        stage_results = {"load": result}
        manifest_df = build_artifact_manifest("data", "test-verify", stage_results)

        verification = verify_manifest(manifest_df)
        self.assertTrue(verification["valid"])
        self.assertEqual(len(verification["missing"]), 0)
        self.assertEqual(len(verification["hash_mismatches"]), 0)


class TestManifestCSVComparison(unittest.TestCase):
    """Test CSV float-tolerant comparison."""

    def test_csv_match(self):
        from sit.repro.hashes import compare_csvs_float_tolerant

        df = pd.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]})

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path_a = f.name
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path_b = f.name

        try:
            df.to_csv(path_a, index=False)
            df.to_csv(path_b, index=False)

            result = compare_csvs_float_tolerant(path_a, path_b)
            self.assertTrue(result["match"])
        finally:
            os.unlink(path_a)
            os.unlink(path_b)


if __name__ == "__main__":
    unittest.main()
