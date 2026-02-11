"""Hash discipline for reproducibility verification.

Provides deterministic directory hashing and float-tolerant comparison
for verifying artifact reproducibility across runs.
"""

import hashlib
import os
from typing import List, Optional, Set

import numpy as np
import pandas as pd

from sit.core.hashing import sha256_file


# Files/dirs to ignore when hashing directory trees
DEFAULT_IGNORE = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".mypy_cache",
    "*.pyc",
    "*.pyo",
}


def hash_tree(
    path: str,
    ignore_patterns: Optional[Set[str]] = None,
) -> str:
    """Hash an entire directory tree deterministically.

    Walks the directory in sorted order, hashing file paths and contents.
    Ignores volatile files like __pycache__ and .pytest_cache.

    Args:
        path: Directory path to hash.
        ignore_patterns: Set of basenames/patterns to ignore.

    Returns:
        Hex SHA-256 digest of the directory contents.
    """
    if ignore_patterns is None:
        ignore_patterns = DEFAULT_IGNORE

    h = hashlib.sha256()

    if os.path.isfile(path):
        h.update(os.path.basename(path).encode("utf-8"))
        h.update(sha256_file(path).encode("utf-8"))
        return h.hexdigest()

    for dirpath, dirnames, filenames in os.walk(path, topdown=True):
        # Filter and sort directories
        dirnames[:] = sorted(
            d for d in dirnames if not _should_ignore(d, ignore_patterns)
        )
        # Sort and process files
        for fname in sorted(filenames):
            if _should_ignore(fname, ignore_patterns):
                continue
            fpath = os.path.join(dirpath, fname)
            # Hash relative path for determinism
            rel = os.path.relpath(fpath, path)
            h.update(rel.encode("utf-8"))
            h.update(sha256_file(fpath).encode("utf-8"))

    return h.hexdigest()


def _should_ignore(name: str, patterns: Set[str]) -> bool:
    """Check if a file/dir name matches ignore patterns."""
    if name in patterns:
        return True
    for pat in patterns:
        if pat.startswith("*.") and name.endswith(pat[1:]):
            return True
    return False


def compare_parquets_float_tolerant(
    path_a: str,
    path_b: str,
    epsilon: float = 1e-8,
) -> dict:
    """Compare two parquet files with float tolerance.

    For numeric columns, checks that values match within epsilon.
    For non-numeric columns, checks exact equality.

    Args:
        path_a: Path to first parquet file.
        path_b: Path to second parquet file.
        epsilon: Maximum allowed difference for floats.

    Returns:
        Dict with 'match' (bool), 'mismatches' (list of column details).
    """
    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)

    mismatches = []

    # Check shape
    if df_a.shape != df_b.shape:
        return {
            "match": False,
            "mismatches": [
                f"Shape mismatch: {df_a.shape} vs {df_b.shape}"
            ],
        }

    # Check columns
    if list(df_a.columns) != list(df_b.columns):
        return {
            "match": False,
            "mismatches": [
                f"Column mismatch: {list(df_a.columns)} vs {list(df_b.columns)}"
            ],
        }

    for col in df_a.columns:
        if pd.api.types.is_numeric_dtype(df_a[col]):
            vals_a = df_a[col].values.astype(float)
            vals_b = df_b[col].values.astype(float)
            # Handle NaN: both NaN = match
            nan_a = np.isnan(vals_a)
            nan_b = np.isnan(vals_b)
            if not np.array_equal(nan_a, nan_b):
                mismatches.append(f"{col}: NaN pattern mismatch")
                continue
            valid = ~nan_a
            if valid.any():
                max_diff = float(np.max(np.abs(vals_a[valid] - vals_b[valid])))
                if max_diff > epsilon:
                    mismatches.append(
                        f"{col}: max_diff={max_diff:.2e} > epsilon={epsilon:.2e}"
                    )
        else:
            # Exact comparison for non-numeric
            if not df_a[col].equals(df_b[col]):
                n_diff = int((df_a[col] != df_b[col]).sum())
                mismatches.append(f"{col}: {n_diff} non-numeric mismatches")

    return {
        "match": len(mismatches) == 0,
        "mismatches": mismatches,
    }


def compare_csvs_float_tolerant(
    path_a: str,
    path_b: str,
    epsilon: float = 1e-8,
) -> dict:
    """Compare two CSV files with float tolerance.

    Args:
        path_a: Path to first CSV file.
        path_b: Path to second CSV file.
        epsilon: Maximum allowed difference for floats.

    Returns:
        Dict with 'match' (bool), 'mismatches' (list).
    """
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    mismatches = []

    if df_a.shape != df_b.shape:
        return {"match": False, "mismatches": [f"Shape: {df_a.shape} vs {df_b.shape}"]}

    if list(df_a.columns) != list(df_b.columns):
        return {"match": False, "mismatches": ["Column names differ"]}

    for col in df_a.columns:
        try:
            vals_a = pd.to_numeric(df_a[col], errors="raise").values.astype(float)
            vals_b = pd.to_numeric(df_b[col], errors="raise").values.astype(float)
            nan_a = np.isnan(vals_a)
            nan_b = np.isnan(vals_b)
            if not np.array_equal(nan_a, nan_b):
                mismatches.append(f"{col}: NaN pattern mismatch")
                continue
            valid = ~nan_a
            if valid.any():
                max_diff = float(np.max(np.abs(vals_a[valid] - vals_b[valid])))
                if max_diff > epsilon:
                    mismatches.append(f"{col}: max_diff={max_diff:.2e}")
        except (ValueError, TypeError):
            if not df_a[col].equals(df_b[col]):
                mismatches.append(f"{col}: non-numeric mismatch")

    return {"match": len(mismatches) == 0, "mismatches": mismatches}
