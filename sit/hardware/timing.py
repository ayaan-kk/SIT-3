"""High-resolution timing and drift measurement for hardware validation.

Records per-request latency using time.perf_counter_ns() and detects
baseline drift across time-separated batches.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.measure.tail import compute_all_tail_stats

logger = get_logger("hardware.timing")


def measure_batch(
    target_fn,
    batch_index: int,
    n_requests: int = 1000,
    work_size: int = 64,
    warmup_requests: int = 100,
    slo_us: float = 500000.0,
    target_id: str = "rpc_loop",
    spectator_set: Optional[List[str]] = None,
    run_id: str = "",
    notes: str = "",
) -> Dict[str, Any]:
    """Run a single measurement batch and collect timing data.

    Args:
        target_fn: Callable that returns latency array in us.
        batch_index: Index of this batch in the experiment.
        n_requests: Number of requests per batch.
        work_size: Work size parameter for target.
        warmup_requests: Warmup requests (discarded).
        slo_us: SLO threshold in microseconds.
        target_id: Target workload identifier.
        spectator_set: List of active spectator IDs (empty for baseline).
        run_id: Run identifier.
        notes: Additional notes.

    Returns:
        Dict with timing results and metadata.
    """
    if spectator_set is None:
        spectator_set = []

    timestamp_start = datetime.now(timezone.utc).isoformat()

    # Warmup
    if warmup_requests > 0:
        _ = target_fn(n_requests=warmup_requests, work_size=work_size)

    # Measurement
    latencies_us = target_fn(n_requests=n_requests, work_size=work_size)

    timestamp_end = datetime.now(timezone.utc).isoformat()

    # Compute tail stats
    stats = compute_all_tail_stats(latencies_us, slo_us)

    # CPU usage snapshot (best effort)
    cpu_usage = _get_cpu_usage()

    return {
        "run_id": run_id,
        "trial_id": batch_index,
        "target_id": target_id,
        "spectator_set": spectator_set,
        "batch_index": batch_index,
        "mean_latency_us": stats["mean_latency_us"],
        "p95_latency_us": stats["p95_latency_us"],
        "p99_latency_us": stats["p99_latency_us"],
        "cvar99_latency_us": stats["cvar99_latency_us"],
        "violation_rate": stats["violation_rate"],
        "n_samples": stats["n_samples"],
        "slo_us": slo_us,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "cpu_usage_snapshot": cpu_usage,
        "notes": notes,
        "latencies_us": latencies_us,  # raw data, not stored in parquet
    }


def measure_baseline_drift(
    target_fn,
    n_batches: int = 5,
    n_requests_per_batch: int = 1000,
    inter_batch_sleep_s: float = 1.0,
    work_size: int = 64,
    warmup_requests: int = 100,
    slo_us: float = 500000.0,
    run_id: str = "",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Measure baseline drift by running target alone over time.

    Runs multiple batches separated by sleep intervals and computes
    drift as Delta(t) = mu(t) - mu(t_0).

    Args:
        target_fn: Callable returning latency array.
        n_batches: Number of time-separated batches.
        n_requests_per_batch: Requests per batch.
        inter_batch_sleep_s: Sleep between batches in seconds.
        work_size: Work size parameter.
        warmup_requests: Warmup requests per batch.
        slo_us: SLO threshold.
        run_id: Run identifier.

    Returns:
        Tuple of (trials_df, drift_summary).
    """
    batches = []

    for b in range(n_batches):
        if b > 0:
            time.sleep(inter_batch_sleep_s)

        result = measure_batch(
            target_fn=target_fn,
            batch_index=b,
            n_requests=n_requests_per_batch,
            work_size=work_size,
            warmup_requests=warmup_requests,
            slo_us=slo_us,
            target_id="rpc_loop",
            spectator_set=[],
            run_id=run_id,
            notes=f"baseline_batch_{b}",
        )
        batches.append(result)

    # Build DataFrame (without raw latencies)
    rows = []
    for b in batches:
        row = {k: v for k, v in b.items() if k != "latencies_us"}
        row["spectator_set"] = "[]"  # serialize for parquet
        rows.append(row)

    trials_df = pd.DataFrame(rows)

    # Compute drift
    means = [b["mean_latency_us"] for b in batches]
    p99s = [b["p99_latency_us"] for b in batches]
    base_mean = means[0]
    base_p99 = p99s[0]

    drift_mean = [m - base_mean for m in means]
    drift_p99 = [p - base_p99 for p in p99s]

    drift_magnitude_mean = max(abs(d) for d in drift_mean)
    drift_pct_mean = drift_magnitude_mean / max(abs(base_mean), 1e-10) * 100

    drift_summary = {
        "base_mean_us": base_mean,
        "base_p99_us": base_p99,
        "drift_mean_us": drift_mean,
        "drift_p99_us": drift_p99,
        "drift_magnitude_mean_us": drift_magnitude_mean,
        "drift_pct_of_mean": drift_pct_mean,
        "n_batches": n_batches,
        "drift_detected": drift_pct_mean > 1.0,  # > 1% threshold
    }

    logger.info(
        "Baseline drift: magnitude=%.2f us (%.2f%% of mean), detected=%s",
        drift_magnitude_mean, drift_pct_mean, drift_summary["drift_detected"],
    )

    return trials_df, drift_summary


def _get_cpu_usage() -> float:
    """Get current CPU usage percentage (best effort).

    Returns -1.0 if psutil is not available.
    """
    try:
        import psutil
        return psutil.cpu_percent(interval=0.1)
    except ImportError:
        return -1.0
    except Exception:
        return -1.0
