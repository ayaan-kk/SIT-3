"""Metric validation and smoke data generation.

Provides fail-fast validation checks for trial metrics (ordering,
finiteness, unit consistency, suspicious ratios) and a deterministic
smoke data generator for pipeline testing.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.core.units import CANONICAL_LATENCY_UNIT

logger = get_logger("validation")

# Epsilon for floating-point comparison tolerance
_EPSILON = 1e-6

# Ratio threshold: p99 > 50 * mean is suspicious
_SUSPICIOUS_RATIO = 50.0


@dataclass
class ValidationReport:
    """Report from trial metric validation."""
    total_rows: int = 0
    valid_rows: int = 0
    ordering_violations: int = 0
    finiteness_violations: int = 0
    negativity_violations: int = 0
    suspicious_ratios: int = 0
    unit_mismatches: int = 0
    details: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return (
            self.ordering_violations == 0
            and self.finiteness_violations == 0
            and self.negativity_violations == 0
            and self.unit_mismatches == 0
        )

    @property
    def suspicious_count(self) -> int:
        return self.suspicious_ratios

    def summary(self) -> str:
        lines = [
            f"Validation Report: {self.total_rows} rows",
            f"  Valid:                {self.valid_rows}",
            f"  Ordering violations:  {self.ordering_violations}",
            f"  Finiteness violations:{self.finiteness_violations}",
            f"  Negativity violations:{self.negativity_violations}",
            f"  Suspicious ratios:    {self.suspicious_ratios}",
            f"  Unit mismatches:      {self.unit_mismatches}",
            f"  Clean: {self.is_clean}",
        ]
        if self.details:
            lines.append("  Details:")
            for d in self.details[:20]:  # Cap detail output
                lines.append(f"    - {d}")
            if len(self.details) > 20:
                lines.append(f"    ... and {len(self.details) - 20} more")
        return "\n".join(lines)


def validate_trial_metrics(
    df: pd.DataFrame,
    strict_mode: bool = True,
    latency_unit: str = CANONICAL_LATENCY_UNIT,
) -> ValidationReport:
    """Validate metric consistency in a trials DataFrame.

    Checks:
    1. mean <= p95 <= p99 <= cvar99 (with epsilon tolerance)
    2. All latency values finite and non-negative
    3. SLO unit matches canonical
    4. Suspicious ratio: p99 > 50 * mean flagged

    Args:
        df: Trials DataFrame with required latency columns.
        strict_mode: If True, raise on any suspicious_count > 0.
        latency_unit: Expected latency unit string.

    Returns:
        ValidationReport with detailed findings.

    Raises:
        ValueError: In strict_mode when suspicious rows are found.
    """
    report = ValidationReport(total_rows=len(df))

    if latency_unit != CANONICAL_LATENCY_UNIT:
        report.unit_mismatches = len(df)
        report.details.append(
            f"Unit mismatch: expected '{CANONICAL_LATENCY_UNIT}', got '{latency_unit}'"
        )

    latency_cols = ["mean_latency_us", "p95_latency_us", "p99_latency_us", "cvar99_latency_us"]

    for idx, row in df.iterrows():
        row_valid = True

        # Check finiteness
        for col in latency_cols:
            val = row[col]
            if not np.isfinite(val):
                report.finiteness_violations += 1
                report.details.append(f"Row {idx}: {col}={val} is not finite")
                row_valid = False

        # Check non-negativity
        for col in latency_cols:
            val = row[col]
            if val < 0:
                report.negativity_violations += 1
                report.details.append(f"Row {idx}: {col}={val} is negative")
                row_valid = False

        # Check ordering: mean <= p95 <= p99 <= cvar99
        mean_val = row["mean_latency_us"]
        p95_val = row["p95_latency_us"]
        p99_val = row["p99_latency_us"]
        cvar99_val = row["cvar99_latency_us"]

        if mean_val > p95_val + _EPSILON:
            report.ordering_violations += 1
            report.details.append(
                f"Row {idx}: mean ({mean_val:.2f}) > p95 ({p95_val:.2f})"
            )
            row_valid = False
        if p95_val > p99_val + _EPSILON:
            report.ordering_violations += 1
            report.details.append(
                f"Row {idx}: p95 ({p95_val:.2f}) > p99 ({p99_val:.2f})"
            )
            row_valid = False
        if p99_val > cvar99_val + _EPSILON:
            report.ordering_violations += 1
            report.details.append(
                f"Row {idx}: p99 ({p99_val:.2f}) > cvar99 ({cvar99_val:.2f})"
            )
            row_valid = False

        # Check suspicious ratio
        if mean_val > 0 and p99_val > _SUSPICIOUS_RATIO * mean_val:
            report.suspicious_ratios += 1
            report.details.append(
                f"Row {idx}: p99/mean ratio = {p99_val / mean_val:.1f} "
                f"exceeds threshold {_SUSPICIOUS_RATIO}"
            )

        # Check violation_rate consistency with SLO
        # (informational only, not a hard failure)
        slo = row["slo_us"]
        vr = row["violation_rate"]
        if not (0.0 <= vr <= 1.0):
            report.details.append(
                f"Row {idx}: violation_rate={vr} outside [0, 1]"
            )
            row_valid = False

        if row_valid:
            report.valid_rows += 1

    if strict_mode and report.suspicious_count > 0:
        raise ValueError(
            f"Strict mode: {report.suspicious_count} suspicious rows detected. "
            f"Set strict_mode=false to allow.\n{report.summary()}"
        )

    if not report.is_clean:
        logger.warning("Validation issues found:\n%s", report.summary())

    return report


def generate_smoke_trials_and_decisions(
    config: Dict[str, Any],
    run_id: str,
    config_hash: str,
    git_commit: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate deterministic smoke trial and decision data.

    Produces coherent dummy data with correct units and metric ordering
    for pipeline validation. Uses numpy RandomState seeded from config
    for deterministic reproducibility.

    Args:
        config: Loaded configuration dict.
        run_id: UUID of the current run.
        config_hash: SHA-256 hash of the config.
        git_commit: Git commit short hash.

    Returns:
        Tuple of (trials_df, decisions_df).
    """
    seed = config["seed"]
    rng = np.random.RandomState(seed)
    n_trials = config["n_trials"]
    slo_us = config["slo_us"]
    schedulers = config["schedulers"]
    n_regimes = config["regimes"].get("n_regimes", 2)

    spectator_pool = [f"spec_{i}" for i in range(8)]
    target_pool = [f"target_{i}" for i in range(3)]

    trial_rows = []
    decision_rows = []

    for trial_id in range(n_trials):
        scheduler_name = schedulers[trial_id % len(schedulers)]
        target_id = target_pool[trial_id % len(target_pool)]
        regime_id = f"regime_{trial_id % n_regimes}"

        # Pick 1-4 spectators deterministically
        n_specs = rng.randint(1, 5)
        spec_indices = rng.choice(len(spectator_pool), size=n_specs, replace=False)
        specs = sorted([spectator_pool[i] for i in spec_indices])
        spectators_str = json.dumps(specs)

        n_samples = 1000

        # Generate coherent latency metrics: mean <= p95 <= p99 <= cvar99
        base_mean = rng.uniform(50.0, 200.0)  # microseconds
        mean_lat = base_mean
        p95_lat = mean_lat + rng.uniform(20.0, 100.0)
        p99_lat = p95_lat + rng.uniform(10.0, 80.0)
        cvar99_lat = p99_lat + rng.uniform(5.0, 50.0)

        # Compute violation rate: fraction of time p99 exceeds SLO
        # Higher p99 relative to SLO -> higher violation rate
        if p99_lat >= slo_us:
            violation_rate = min(1.0, 0.5 + rng.uniform(0, 0.5))
        else:
            # Roughly: higher p99/slo ratio -> higher violation rate
            ratio = p99_lat / slo_us
            violation_rate = max(0.0, ratio * rng.uniform(0.0, 0.1))

        trial_rows.append({
            "run_id": run_id,
            "config_hash": config_hash,
            "git_commit": git_commit,
            "seed": seed,
            "trial_id": trial_id,
            "scheduler_name": scheduler_name,
            "target_id": target_id,
            "spectators": spectators_str,
            "regime_id": regime_id,
            "n_samples": n_samples,
            "mean_latency_us": round(mean_lat, 6),
            "p95_latency_us": round(p95_lat, 6),
            "p99_latency_us": round(p99_lat, 6),
            "cvar99_latency_us": round(cvar99_lat, 6),
            "slo_us": float(slo_us),
            "violation_rate": round(violation_rate, 6),
            "notes": "",
            "schema_version": 1,
            "created_at_utc": "",  # Filled by caller
            "units_latency": CANONICAL_LATENCY_UNIT,
        })

        # Generate a corresponding decision
        # Build candidate sets (2-3 alternatives)
        n_candidates = rng.randint(2, 4)
        candidate_sets = []
        for _ in range(n_candidates):
            n_c = rng.randint(1, 5)
            c_indices = rng.choice(len(spectator_pool), size=n_c, replace=False)
            candidate_sets.append(sorted([spectator_pool[i] for i in c_indices]))

        # Chosen set is the one from the trial
        score_components = {
            "risk": round(rng.uniform(0.0, 1.0), 4),
            "diversity": round(rng.uniform(0.0, 1.0), 4),
            "uncertainty": round(rng.uniform(0.0, 1.0), 4),
        }
        predicted_metrics = {
            "predicted_p99_us": round(p99_lat + rng.uniform(-10, 10), 4),
            "predicted_mean_us": round(mean_lat + rng.uniform(-5, 5), 4),
        }

        decision_rows.append({
            "run_id": run_id,
            "config_hash": config_hash,
            "git_commit": git_commit,
            "seed": seed,
            "decision_id": trial_id,
            "scheduler_name": scheduler_name,
            "time_index": trial_id,
            "target_id": target_id,
            "candidate_sets": json.dumps(candidate_sets),
            "chosen_set": spectators_str,
            "score_components_json": json.dumps(score_components),
            "predicted_metrics_json": json.dumps(predicted_metrics),
            "realized_metrics_json": json.dumps(None),
            "safety_pass": True,
            "schema_version": 1,
            "created_at_utc": "",  # Filled by caller
        })

    trials_df = pd.DataFrame(trial_rows)
    decisions_df = pd.DataFrame(decision_rows)

    return trials_df, decisions_df
