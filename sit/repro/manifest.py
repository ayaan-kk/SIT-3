"""Artifact manifest generation for reproducibility binder.

Creates a single manifest file cataloging every artifact produced
during a repro run, with SHA-256 hashes and dependency tracking.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from sit.core.hashing import sha256_file
from sit.core.logging import get_logger

logger = get_logger("repro.manifest")

# Recognized artifact extensions
ARTIFACT_EXTENSIONS = {".parquet", ".csv", ".png", ".md", ".log", ".yaml", ".json"}


def build_artifact_manifest(
    base_dir: str,
    master_run_id: str,
    stage_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Build a comprehensive artifact manifest for a repro run.

    Scans the output directory for all produced artifacts and records
    their metadata including SHA-256 hashes.

    Args:
        base_dir: Base output directory (e.g. 'data').
        master_run_id: The master repro run ID.
        stage_results: Dict mapping stage_name -> {run_id, config_hash, status, ...}.

    Returns:
        DataFrame with manifest rows.
    """
    rows = []

    # Scan each stage's output directories
    for stage_name, stage_info in stage_results.items():
        stage_run_id = stage_info.get("run_id", "")
        if not stage_run_id:
            continue

        # Scan raw, derived, figures, tables directories
        for subdir_type in ["raw", "derived", "figures", "tables"]:
            scan_dir = os.path.join(base_dir, subdir_type, stage_run_id)
            if not os.path.isdir(scan_dir):
                continue

            for fname in sorted(os.listdir(scan_dir)):
                fpath = os.path.join(scan_dir, fname)
                if not os.path.isfile(fpath):
                    continue

                ext = os.path.splitext(fname)[1].lower()
                if ext not in ARTIFACT_EXTENSIONS:
                    continue

                artifact_type = _classify_artifact(subdir_type, ext)
                try:
                    file_hash = sha256_file(fpath)
                    file_size = os.path.getsize(fpath)
                except Exception as e:
                    logger.warning("Failed to hash %s: %s", fpath, e)
                    continue

                rows.append({
                    "artifact_path": fpath,
                    "artifact_type": artifact_type,
                    "producing_stage": stage_name,
                    "producing_run_id": stage_run_id,
                    "sha256": file_hash,
                    "bytes": file_size,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "dependencies_hash": stage_info.get("config_hash", ""),
                })

    # Also scan for report files in results/
    for report_dir in ["results/tables", "results/reports"]:
        if os.path.isdir(report_dir):
            for fname in sorted(os.listdir(report_dir)):
                fpath = os.path.join(report_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in ARTIFACT_EXTENSIONS:
                    continue
                try:
                    file_hash = sha256_file(fpath)
                    file_size = os.path.getsize(fpath)
                except Exception:
                    continue
                rows.append({
                    "artifact_path": fpath,
                    "artifact_type": "report",
                    "producing_stage": "report",
                    "producing_run_id": master_run_id,
                    "sha256": file_hash,
                    "bytes": file_size,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "dependencies_hash": "",
                })

    # Scan docs/
    if os.path.isdir("docs"):
        for fname in sorted(os.listdir("docs")):
            fpath = os.path.join("docs", fname)
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ARTIFACT_EXTENSIONS:
                continue
            try:
                file_hash = sha256_file(fpath)
                file_size = os.path.getsize(fpath)
            except Exception:
                continue
            rows.append({
                "artifact_path": fpath,
                "artifact_type": "report",
                "producing_stage": "report",
                "producing_run_id": master_run_id,
                "sha256": file_hash,
                "bytes": file_size,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "dependencies_hash": "",
            })

    manifest_df = pd.DataFrame(rows)
    if len(manifest_df) == 0:
        manifest_df = pd.DataFrame(columns=[
            "artifact_path", "artifact_type", "producing_stage",
            "producing_run_id", "sha256", "bytes", "created_at_utc",
            "dependencies_hash",
        ])

    logger.info("Manifest built: %d artifacts", len(manifest_df))
    return manifest_df


def _classify_artifact(subdir_type: str, ext: str) -> str:
    """Classify artifact type from directory and extension."""
    if subdir_type == "raw":
        return "raw"
    elif subdir_type == "figures" or ext == ".png":
        return "figure"
    elif subdir_type == "tables" or ext == ".csv":
        return "report"
    elif ext == ".md":
        return "report"
    return "derived"


def verify_manifest(manifest_df: pd.DataFrame) -> dict:
    """Verify all artifacts in a manifest still exist and hashes match.

    Args:
        manifest_df: Manifest DataFrame with artifact_path and sha256 columns.

    Returns:
        Dict with 'valid' (bool), 'missing' (list), 'hash_mismatches' (list).
    """
    missing = []
    hash_mismatches = []

    for _, row in manifest_df.iterrows():
        path = row["artifact_path"]
        expected_hash = row["sha256"]

        if not os.path.exists(path):
            missing.append(path)
            continue

        try:
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                hash_mismatches.append({
                    "path": path,
                    "expected": expected_hash,
                    "actual": actual_hash,
                })
        except Exception as e:
            hash_mismatches.append({
                "path": path,
                "expected": expected_hash,
                "actual": f"ERROR: {e}",
            })

    return {
        "valid": len(missing) == 0 and len(hash_mismatches) == 0,
        "missing": missing,
        "hash_mismatches": hash_mismatches,
        "total": len(manifest_df),
        "verified": len(manifest_df) - len(missing) - len(hash_mismatches),
    }
