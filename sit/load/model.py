"""Open-loop and closed-loop serving models.

Simulates per-host M/G/1 (open-loop) or closed-loop queueing.
The key model: each host is a single server. All TARGET workloads
on the host generate requests that share the server. Spectators
affect interference (service time) but do not generate requests.

This creates the "partition killer" effect:
- Static partition crams all targets on 1-2 hosts -> high aggregate load
- SIT-safe spreads targets across all hosts -> low per-host load
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from sit.core.logging import get_logger
from sit.measure.tail import compute_all_tail_stats

logger = get_logger("load.model")


@dataclass
class ServingConfig:
    """Configuration for the serving model."""
    mode: str = "open_loop"
    window_us: float = 10_000_000.0  # 10 seconds
    warmup_fraction: float = 0.2
    n_service_samples: int = 5000

    @classmethod
    def from_config(cls, config: dict) -> "ServingConfig":
        load_cfg = config.get("load", {})
        return cls(
            mode=load_cfg.get("mode", "open_loop"),
            window_us=float(load_cfg.get("window_us", 10_000_000.0)),
            warmup_fraction=float(load_cfg.get("warmup_fraction", 0.2)),
            n_service_samples=int(load_cfg.get("n_service_samples", 5000)),
        )


def simulate_open_loop_host(
    target_service_dists: Dict[str, np.ndarray],
    lambda_rps_per_target: float,
    window_us: float,
    warmup_fraction: float,
    slo_us: float,
    rng: np.random.RandomState,
) -> Dict[str, Dict[str, Any]]:
    """Simulate M/G/1 open-loop queue on a single host.

    All targets on this host share a single server. Each target generates
    Poisson arrivals at lambda_rps_per_target. Service times are drawn
    from each target's empirical distribution (which includes interference).

    This is the core model for the "partition killer" result: hosts with
    many targets have high aggregate load, causing queue blowup.

    Args:
        target_service_dists: target_id -> service time samples array.
        lambda_rps_per_target: Arrival rate (requests/second) per target.
        window_us: Total observation window in microseconds.
        warmup_fraction: Fraction of window to discard for warmup.
        slo_us: SLO threshold in microseconds.
        rng: Random state.

    Returns:
        Dict mapping target_id to metrics dict with throughput_rps,
        goodput_rps, violation_rate, tail stats, queue stats.
    """
    target_ids = list(target_service_dists.keys())
    if not target_ids or lambda_rps_per_target <= 0:
        return {tid: _empty_metrics(slo_us) for tid in target_ids}

    # Generate arrival events for all targets
    # Each target generates independent Poisson arrivals
    events = []  # list of (arrival_us, target_id)

    for tid in target_ids:
        t_us = 0.0
        # Inter-arrival time mean = 1e6 / lambda_rps microseconds
        mean_iat_us = 1e6 / lambda_rps_per_target
        while t_us < window_us:
            iat = rng.exponential(mean_iat_us)
            t_us += iat
            if t_us >= window_us:
                break
            events.append((t_us, tid))

    if not events:
        return {tid: _empty_metrics(slo_us) for tid in target_ids}

    # Sort by arrival time
    events.sort(key=lambda x: x[0])

    # Simulate M/G/1 queue (single server for entire host)
    server_free_at = 0.0
    warmup_cutoff_us = window_us * warmup_fraction

    results_by_target = {tid: [] for tid in target_ids}
    diag_queue_delays = []
    diag_service_times = []

    for arrival_us, tid in events:
        # Draw service time from target's empirical distribution
        svc_dist = target_service_dists[tid]
        svc_us = svc_dist[rng.randint(len(svc_dist))]

        # M/G/1: queue delay = max(0, server_free_at - arrival)
        queue_delay = max(0.0, server_free_at - arrival_us)
        service_start = max(server_free_at, arrival_us)
        service_end = service_start + svc_us
        server_free_at = service_end

        total_latency = queue_delay + svc_us

        # Only record post-warmup requests
        if arrival_us >= warmup_cutoff_us:
            results_by_target[tid].append(total_latency)
            diag_queue_delays.append(queue_delay)
            diag_service_times.append(svc_us)

    # Compute per-target metrics
    measurement_window_us = window_us * (1.0 - warmup_fraction)
    measurement_window_s = measurement_window_us / 1e6

    metrics = {}
    for tid in target_ids:
        latencies = results_by_target[tid]
        if not latencies:
            metrics[tid] = _empty_metrics(slo_us)
            continue

        latencies_arr = np.array(latencies, dtype=np.float64)
        n_requests = len(latencies_arr)
        n_successes = int(np.sum(latencies_arr <= slo_us))

        throughput_rps = n_requests / measurement_window_s if measurement_window_s > 0 else 0.0
        goodput_rps = n_successes / measurement_window_s if measurement_window_s > 0 else 0.0
        violation_rate = 1.0 - (n_successes / n_requests) if n_requests > 0 else 0.0

        tail_stats = compute_all_tail_stats(latencies_arr, slo_us)

        metrics[tid] = {
            "throughput_rps": throughput_rps,
            "goodput_rps": goodput_rps,
            "violation_rate": violation_rate,
            "n_requests": n_requests,
            "n_successes": n_successes,
            **tail_stats,
        }

    # Add queue diagnostics to each target's metrics
    if diag_queue_delays:
        qd_arr = np.array(diag_queue_delays)
        sd_arr = np.array(diag_service_times)
        total_service = float(np.sum(sd_arr))
        total_time = measurement_window_us

        host_diag = {
            "utilization": total_service / total_time if total_time > 0 else 0.0,
            "mean_queue_delay_us": float(np.mean(qd_arr)),
            "p95_queue_delay_us": float(np.percentile(qd_arr, 95)) if len(qd_arr) > 0 else 0.0,
            "mean_service_us": float(np.mean(sd_arr)),
            "busy_fraction": float(np.mean(qd_arr > 0)),
            "n_total_requests": len(diag_queue_delays),
        }
        for tid in target_ids:
            if tid in metrics:
                metrics[tid].update(host_diag)

    return metrics


def simulate_closed_loop_host(
    target_service_dists: Dict[str, np.ndarray],
    concurrency_per_target: int,
    think_time_us: float,
    window_us: float,
    warmup_fraction: float,
    slo_us: float,
    rng: np.random.RandomState,
) -> Dict[str, Dict[str, Any]]:
    """Simulate closed-loop queueing on a single host.

    Each target has C concurrent clients. After receiving a response,
    each client waits think_time_us before sending the next request.

    Args:
        target_service_dists: target_id -> service time samples.
        concurrency_per_target: Number of concurrent clients per target.
        think_time_us: Think time between response and next request.
        window_us: Observation window.
        warmup_fraction: Warmup fraction.
        slo_us: SLO threshold.
        rng: Random state.

    Returns:
        Dict mapping target_id to metrics.
    """
    target_ids = list(target_service_dists.keys())
    if not target_ids:
        return {}

    n_targets = len(target_ids)
    C = concurrency_per_target
    total_clients = n_targets * C

    # Initialize client state: (next_arrival_us, target_id, client_id)
    events = []
    for tid_idx, tid in enumerate(target_ids):
        for c in range(C):
            # Stagger initial arrivals
            initial_time = (tid_idx * C + c) * (think_time_us / total_clients)
            events.append((initial_time, tid, c))

    # Sort by arrival time
    import heapq
    heapq.heapify(events)

    server_free_at = 0.0
    warmup_cutoff_us = window_us * warmup_fraction
    results_by_target = {tid: [] for tid in target_ids}
    diag_queue_delays = []
    diag_service_times = []

    max_events = int(window_us / 10.0)  # safety limit

    n_processed = 0
    while events and n_processed < max_events:
        arrival_us, tid, client_id = heapq.heappop(events)
        if arrival_us >= window_us:
            break

        svc_dist = target_service_dists[tid]
        svc_us = svc_dist[rng.randint(len(svc_dist))]

        queue_delay = max(0.0, server_free_at - arrival_us)
        service_start = max(server_free_at, arrival_us)
        service_end = service_start + svc_us
        server_free_at = service_end

        total_latency = queue_delay + svc_us

        if arrival_us >= warmup_cutoff_us:
            results_by_target[tid].append(total_latency)
            diag_queue_delays.append(queue_delay)
            diag_service_times.append(svc_us)

        # Client sends next request after response + think time
        next_arrival = service_end + think_time_us
        if next_arrival < window_us:
            heapq.heappush(events, (next_arrival, tid, client_id))

        n_processed += 1

    # Compute metrics (same as open_loop)
    measurement_window_us = window_us * (1.0 - warmup_fraction)
    measurement_window_s = measurement_window_us / 1e6

    metrics = {}
    for tid in target_ids:
        latencies = results_by_target[tid]
        if not latencies:
            metrics[tid] = _empty_metrics(slo_us)
            continue

        latencies_arr = np.array(latencies, dtype=np.float64)
        n_requests = len(latencies_arr)
        n_successes = int(np.sum(latencies_arr <= slo_us))

        throughput_rps = n_requests / measurement_window_s if measurement_window_s > 0 else 0.0
        goodput_rps = n_successes / measurement_window_s if measurement_window_s > 0 else 0.0
        violation_rate = 1.0 - (n_successes / n_requests) if n_requests > 0 else 0.0

        tail_stats = compute_all_tail_stats(latencies_arr, slo_us)

        metrics[tid] = {
            "throughput_rps": throughput_rps,
            "goodput_rps": goodput_rps,
            "violation_rate": violation_rate,
            "n_requests": n_requests,
            "n_successes": n_successes,
            **tail_stats,
        }

    if diag_queue_delays:
        qd_arr = np.array(diag_queue_delays)
        sd_arr = np.array(diag_service_times)
        total_service = float(np.sum(sd_arr))
        total_time = measurement_window_us

        host_diag = {
            "utilization": total_service / total_time if total_time > 0 else 0.0,
            "mean_queue_delay_us": float(np.mean(qd_arr)),
            "p95_queue_delay_us": float(np.percentile(qd_arr, 95)) if len(qd_arr) > 0 else 0.0,
            "mean_service_us": float(np.mean(sd_arr)),
            "busy_fraction": float(np.mean(qd_arr > 0)),
            "n_total_requests": len(diag_queue_delays),
        }
        for tid in target_ids:
            if tid in metrics:
                metrics[tid].update(host_diag)

    return metrics


def _empty_metrics(slo_us: float) -> Dict[str, Any]:
    """Return empty metrics for an unserved target."""
    return {
        "throughput_rps": 0.0,
        "goodput_rps": 0.0,
        "violation_rate": 1.0,
        "n_requests": 0,
        "n_successes": 0,
        "mean_latency_us": 0.0,
        "p95_latency_us": 0.0,
        "p99_latency_us": 0.0,
        "cvar99_latency_us": 0.0,
        "n_samples": 0,
        "utilization": 0.0,
        "mean_queue_delay_us": 0.0,
        "p95_queue_delay_us": 0.0,
        "mean_service_us": 0.0,
        "busy_fraction": 0.0,
        "n_total_requests": 0,
    }
