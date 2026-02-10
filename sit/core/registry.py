"""Experiment registry and run lifecycle management.

Manages run IDs, output directories, artifact registration, and
run finalization with manifest generation.
"""

import os
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from sit.core.hashing import hash_config, sha256_file
from sit.core.logging import get_logger, setup_logging
from sit.core.schema import SCHEMA_VERSION
from sit.data.paths import raw_dir, derived_dir, figures_dir, tables_dir

logger = get_logger("registry")


def create_run_id() -> str:
    """Generate a new unique run identifier (UUID4)."""
    return str(uuid.uuid4())


def get_git_commit() -> str:
    """Get the current short git commit hash.

    Returns 'UNKNOWN' if not in a git repository or git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return "UNKNOWN"


@dataclass
class ArtifactEntry:
    """A registered output artifact."""
    path: str
    artifact_type: str  # raw, derived, figure, report
    sha256: str = ""
    size_bytes: int = 0
    created_at_utc: str = ""


@dataclass
class RunContext:
    """Context for a single experiment run.

    Captures all metadata needed for reproducibility and auditability.
    """
    run_id: str
    config: Dict[str, Any]
    config_hash: str
    git_commit: str
    seed: int
    output_dir: str
    created_at_utc: str
    schema_version: int = SCHEMA_VERSION
    artifacts: List[ArtifactEntry] = field(default_factory=list)

    @property
    def raw_path(self) -> str:
        return raw_dir(self.run_id, base_dir=self.output_dir)

    @property
    def derived_path(self) -> str:
        return derived_dir(self.run_id, base_dir=self.output_dir)

    @property
    def figures_path(self) -> str:
        return figures_dir(self.run_id, base_dir=self.output_dir)

    @property
    def tables_path(self) -> str:
        return tables_dir(self.run_id, base_dir=self.output_dir)

    def ensure_dirs(self) -> None:
        """Create all output directories for this run."""
        for p in [self.raw_path, self.derived_path, self.figures_path, self.tables_path]:
            os.makedirs(p, exist_ok=True)


def start_run(config: Dict[str, Any]) -> RunContext:
    """Initialize a new experiment run.

    Creates the run context with all metadata, sets up logging,
    and creates output directories.

    Args:
        config: Loaded and validated configuration dict.

    Returns:
        RunContext for this run.
    """
    run_id = create_run_id()
    cfg_hash = hash_config(config)
    git_commit = get_git_commit()
    seed = config["seed"]
    output_dir = config["output_dir"]
    created_at = datetime.now(timezone.utc).isoformat()

    # Setup logging with run_id
    setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        run_id=run_id,
    )

    ctx = RunContext(
        run_id=run_id,
        config=config,
        config_hash=cfg_hash,
        git_commit=git_commit,
        seed=seed,
        output_dir=output_dir,
        created_at_utc=created_at,
    )
    ctx.ensure_dirs()

    logger.info("Run started: %s", run_id)
    logger.info("Config hash: %s", cfg_hash)
    logger.info("Git commit: %s", git_commit)
    logger.info("Seed: %d", seed)
    logger.info("Output dir: %s", output_dir)

    return ctx


def register_artifact(ctx: RunContext, path: str, artifact_type: str) -> ArtifactEntry:
    """Register an output artifact for the current run.

    Computes the SHA-256 hash and file size, and records the entry
    in the run context.

    Args:
        ctx: Current RunContext.
        path: Path to the artifact file.
        artifact_type: Type string (raw, derived, figure, report).

    Returns:
        The registered ArtifactEntry.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot register non-existent artifact: {path}")

    entry = ArtifactEntry(
        path=str(file_path),
        artifact_type=artifact_type,
        sha256=sha256_file(path),
        size_bytes=file_path.stat().st_size,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    ctx.artifacts.append(entry)
    logger.info("Registered artifact: %s (%s, %d bytes)", path, artifact_type, entry.size_bytes)
    return entry


def finalize_run(ctx: RunContext) -> pd.DataFrame:
    """Finalize a run by writing the artifacts manifest.

    Creates an artifacts DataFrame and writes it to the raw output
    directory.

    Args:
        ctx: Current RunContext.

    Returns:
        Artifacts DataFrame.
    """
    rows = []
    for a in ctx.artifacts:
        rows.append({
            "run_id": ctx.run_id,
            "path": a.path,
            "sha256": a.sha256,
            "bytes": a.size_bytes,
            "created_at_utc": a.created_at_utc,
            "type": a.artifact_type,
            "schema_version": ctx.schema_version,
        })

    artifacts_df = pd.DataFrame(rows)
    if len(artifacts_df) == 0:
        # Create empty DataFrame with correct columns
        artifacts_df = pd.DataFrame(columns=[
            "run_id", "path", "sha256", "bytes", "created_at_utc", "type", "schema_version"
        ])

    logger.info("Run finalized: %s (%d artifacts)", ctx.run_id, len(ctx.artifacts))
    return artifacts_df
