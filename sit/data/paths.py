"""Path management for SIT output directories.

All paths are rooted under a configurable base directory (default: 'data/').
Each run gets a unique subdirectory tree organized by data type.
"""

import os
from pathlib import Path


def _ensure_base(base_dir: str) -> str:
    """Return the base directory path, ensuring it's a valid string."""
    return base_dir if base_dir else "data"


def raw_dir(run_id: str, base_dir: str = "data") -> str:
    """Path to raw data directory for a run.

    Layout: <base_dir>/raw/<run_id>/
    """
    return os.path.join(_ensure_base(base_dir), "raw", run_id)


def derived_dir(run_id: str, base_dir: str = "data") -> str:
    """Path to derived data directory for a run.

    Layout: <base_dir>/derived/<run_id>/
    """
    return os.path.join(_ensure_base(base_dir), "derived", run_id)


def figures_dir(run_id: str, base_dir: str = "data") -> str:
    """Path to figures directory for a run.

    Layout: <base_dir>/figures/<run_id>/
    """
    return os.path.join(_ensure_base(base_dir), "figures", run_id)


def tables_dir(run_id: str, base_dir: str = "data") -> str:
    """Path to tables directory for a run.

    Layout: <base_dir>/tables/<run_id>/
    """
    return os.path.join(_ensure_base(base_dir), "tables", run_id)


def trial_path(run_id: str, base_dir: str = "data", fmt: str = "parquet") -> str:
    """Path to the trials data file for a run."""
    return os.path.join(raw_dir(run_id, base_dir), f"trials.{fmt}")


def decision_path(run_id: str, base_dir: str = "data", fmt: str = "parquet") -> str:
    """Path to the decisions data file for a run."""
    return os.path.join(raw_dir(run_id, base_dir), f"decisions.{fmt}")


def artifact_path(run_id: str, base_dir: str = "data", fmt: str = "parquet") -> str:
    """Path to the artifacts manifest for a run."""
    return os.path.join(raw_dir(run_id, base_dir), f"artifacts.{fmt}")


def find_run_ids(base_dir: str = "data") -> list:
    """List all run IDs found in the raw data directory.

    Returns:
        Sorted list of run_id strings.
    """
    raw_base = os.path.join(_ensure_base(base_dir), "raw")
    if not os.path.isdir(raw_base):
        return []
    return sorted([
        d for d in os.listdir(raw_base)
        if os.path.isdir(os.path.join(raw_base, d))
    ])
