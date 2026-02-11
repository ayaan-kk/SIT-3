"""Warmup and calibration utilities for hardware measurements.

Handles timer resolution detection, warmup convergence,
and measurement stabilization.
"""

import time
from typing import Tuple

import numpy as np

from sit.core.logging import get_logger

logger = get_logger("hardware.calibration")


def measure_timer_resolution_ns(n_probes: int = 1000) -> float:
    """Estimate the resolution of time.perf_counter_ns().

    Measures minimum non-zero delta across many probes.

    Args:
        n_probes: Number of probe pairs.

    Returns:
        Estimated timer resolution in nanoseconds.
    """
    deltas = []
    for _ in range(n_probes):
        t0 = time.perf_counter_ns()
        t1 = time.perf_counter_ns()
        d = t1 - t0
        if d > 0:
            deltas.append(d)

    if not deltas:
        return 1000.0  # 1 us fallback

    resolution = float(np.min(deltas))
    logger.info("Timer resolution: %.1f ns (from %d non-zero deltas)",
                resolution, len(deltas))
    return resolution


def warmup_stabilize(
    target_fn,
    work_size: int = 64,
    max_warmup: int = 500,
    stability_window: int = 50,
    stability_threshold: float = 0.05,
) -> Tuple[int, float]:
    """Run warmup until latency stabilizes.

    Monitors the coefficient of variation of rolling mean over the
    last stability_window requests. Stops when CV < threshold.

    Args:
        target_fn: Callable(n_requests, work_size) -> latencies_us.
        work_size: Work size parameter.
        max_warmup: Maximum warmup requests.
        stability_window: Rolling window for stability check.
        stability_threshold: CV threshold for stability.

    Returns:
        Tuple of (n_warmup_used, final_mean_us).
    """
    latencies = target_fn(n_requests=max_warmup, work_size=work_size)

    for i in range(stability_window, max_warmup):
        window = latencies[i - stability_window:i]
        mean = np.mean(window)
        std = np.std(window)
        cv = std / max(mean, 1e-10)
        if cv < stability_threshold:
            logger.info("Warmup stabilized at request %d (CV=%.4f)", i, cv)
            return i, float(mean)

    final_mean = float(np.mean(latencies[-stability_window:]))
    logger.info("Warmup completed (max %d requests, mean=%.2f us)",
                max_warmup, final_mean)
    return max_warmup, final_mean
