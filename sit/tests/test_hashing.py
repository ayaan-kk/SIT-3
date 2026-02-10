"""Tests for sit.core.hashing module.

Covers:
- Config hashing stability under key reordering
- Float normalization produces stable hashes
- File hashing correctness
- Nested dict handling
"""

import os
import tempfile

import pytest

from sit.core.hashing import hash_config, sha256_file


class TestHashConfig:
    def test_basic_hash(self):
        config = {"seed": 123, "name": "test"}
        h = hash_config(config)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest

    def test_stable_under_key_reorder(self):
        config_a = {"seed": 123, "name": "test", "value": 1.0}
        config_b = {"value": 1.0, "seed": 123, "name": "test"}
        assert hash_config(config_a) == hash_config(config_b)

    def test_stable_under_deep_key_reorder(self):
        config_a = {
            "outer": {"b_key": 2, "a_key": 1},
            "seed": 42,
        }
        config_b = {
            "seed": 42,
            "outer": {"a_key": 1, "b_key": 2},
        }
        assert hash_config(config_a) == hash_config(config_b)

    def test_float_normalization(self):
        # Floats that differ only in trailing precision should hash the same
        config_a = {"val": 1.0000000000001}
        config_b = {"val": 1.0000000000001}
        assert hash_config(config_a) == hash_config(config_b)

    def test_different_values_differ(self):
        config_a = {"seed": 123}
        config_b = {"seed": 456}
        assert hash_config(config_a) != hash_config(config_b)

    def test_list_handling(self):
        config_a = {"items": [1, 2, 3]}
        config_b = {"items": [1, 2, 3]}
        assert hash_config(config_a) == hash_config(config_b)

    def test_list_order_matters(self):
        config_a = {"items": [1, 2, 3]}
        config_b = {"items": [3, 2, 1]}
        assert hash_config(config_a) != hash_config(config_b)

    def test_nested_dict(self):
        config = {
            "level1": {
                "level2": {
                    "value": 42,
                    "name": "deep"
                }
            }
        }
        h = hash_config(config)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_empty_dict(self):
        h = hash_config({})
        assert isinstance(h, str)
        assert len(h) == 64

    def test_deterministic(self):
        config = {"a": 1, "b": [1, 2], "c": {"d": 3.14}}
        h1 = hash_config(config)
        h2 = hash_config(config)
        assert h1 == h2


class TestSha256File:
    def test_basic_file_hash(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            f.flush()
            path = f.name

        try:
            h = sha256_file(path)
            assert isinstance(h, str)
            assert len(h) == 64
        finally:
            os.unlink(path)

    def test_same_content_same_hash(self):
        content = "deterministic content"
        paths = []
        for _ in range(2):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(content)
                f.flush()
                paths.append(f.name)

        try:
            assert sha256_file(paths[0]) == sha256_file(paths[1])
        finally:
            for p in paths:
                os.unlink(p)

    def test_different_content_different_hash(self):
        paths = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(f"content {i}")
                f.flush()
                paths.append(f.name)

        try:
            assert sha256_file(paths[0]) != sha256_file(paths[1])
        finally:
            for p in paths:
                os.unlink(p)

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            sha256_file("/nonexistent/path/file.txt")
