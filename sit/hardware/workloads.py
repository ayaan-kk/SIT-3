"""Hardware workloads: target and spectator implementations.

Target workloads are latency-sensitive (RPC-like loop).
Spectator workloads are interference sources (CPU, memory, cache burners).
All workloads are deterministic, auditable, and require only Python/NumPy.
"""

import multiprocessing
import time
from typing import Any, Dict, List, Optional

import numpy as np

from sit.core.logging import get_logger

logger = get_logger("hardware.workloads")


# ---------------------------------------------------------------------------
# Target workload
# ---------------------------------------------------------------------------

def rpc_loop_target(
    n_requests: int = 1000,
    work_size: int = 64,
) -> np.ndarray:
    """Run an RPC-like target workload and record per-request latency.

    Each "request" performs a small matrix multiply (work_size x work_size)
    and records wall-clock latency in microseconds.

    Args:
        n_requests: Number of requests to execute.
        work_size: Matrix dimension for each request's computation.

    Returns:
        Array of per-request latencies in microseconds.
    """
    latencies_us = np.empty(n_requests, dtype=np.float64)
    # Pre-allocate matrices to avoid allocation noise
    a = np.random.RandomState(0).randn(work_size, work_size).astype(np.float32)
    b = np.random.RandomState(1).randn(work_size, work_size).astype(np.float32)

    for i in range(n_requests):
        t0 = time.perf_counter_ns()
        _ = a @ b  # deterministic computation
        t1 = time.perf_counter_ns()
        latencies_us[i] = (t1 - t0) / 1000.0  # ns -> us

    return latencies_us


def run_target_batch(
    target_type: str = "rpc_loop",
    n_requests: int = 1000,
    work_size: int = 64,
    warmup_requests: int = 100,
) -> np.ndarray:
    """Run a batch of target workload requests with warmup.

    Args:
        target_type: Type of target workload ("rpc_loop").
        n_requests: Number of measurement requests.
        work_size: Work size parameter.
        warmup_requests: Number of warmup requests (discarded).

    Returns:
        Array of per-request latencies in microseconds (after warmup).
    """
    if target_type != "rpc_loop":
        raise ValueError(f"Unknown target type: {target_type}")

    # Warmup
    if warmup_requests > 0:
        _ = rpc_loop_target(n_requests=warmup_requests, work_size=work_size)

    # Measurement
    return rpc_loop_target(n_requests=n_requests, work_size=work_size)


# ---------------------------------------------------------------------------
# Spectator workloads
# ---------------------------------------------------------------------------

class SpectatorWorkload:
    """Base class for spectator (interference source) workloads."""

    def __init__(self, workload_id: str, intensity: float = 0.8):
        self.workload_id = workload_id
        self.intensity = max(0.0, min(1.0, intensity))
        self._process: Optional[multiprocessing.Process] = None
        self._stop_event = multiprocessing.Event()

    def start(self):
        """Start the spectator workload in a background process."""
        self._stop_event.clear()
        self._process = multiprocessing.Process(
            target=self._run_loop,
            args=(self._stop_event, self.intensity),
            daemon=True,
        )
        self._process.start()
        logger.info("Started spectator %s (pid=%s)", self.workload_id,
                     self._process.pid)

    def stop(self):
        """Stop the spectator workload."""
        if self._process is not None and self._process.is_alive():
            self._stop_event.set()
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
            logger.info("Stopped spectator %s", self.workload_id)
        self._process = None

    def _run_loop(self, stop_event: multiprocessing.Event, intensity: float):
        """Override in subclasses to define the interference pattern."""
        raise NotImplementedError

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()


class CPUBurnSpectator(SpectatorWorkload):
    """CPU-intensive spectator: tight arithmetic loop.

    Maps to CPU/SMT contention channel.
    """

    def __init__(self, intensity: float = 0.8):
        super().__init__(workload_id="cpu_burn", intensity=intensity)

    def _run_loop(self, stop_event, intensity):
        """Tight floating-point loop consuming CPU cycles."""
        x = 1.0001
        batch = max(1, int(100000 * intensity))
        while not stop_event.is_set():
            for _ in range(batch):
                x = x * 1.0001
                if x > 1e100:
                    x = 1.0001


class MemBurnSpectator(SpectatorWorkload):
    """Memory bandwidth spectator: large array stride access.

    Maps to MEM_BW contention channel.
    """

    def __init__(self, intensity: float = 0.8):
        super().__init__(workload_id="mem_burn", intensity=intensity)

    def _run_loop(self, stop_event, intensity):
        """Sequential sweep through a large array to saturate memory bandwidth."""
        size = max(1024, int(4 * 1024 * 1024 * intensity))  # up to 4M elements
        arr = np.zeros(size, dtype=np.float64)
        while not stop_event.is_set():
            arr += 1.0
            arr *= 0.9999


class CacheBurnSpectator(SpectatorWorkload):
    """Cache stressor spectator: random access to medium array.

    Maps to LLC contention channel.
    """

    def __init__(self, intensity: float = 0.8):
        super().__init__(workload_id="cache_burn", intensity=intensity)

    def _run_loop(self, stop_event, intensity):
        """Random access pattern to thrash caches."""
        size = max(1024, int(1024 * 1024 * intensity))  # up to 1M elements
        arr = np.zeros(size, dtype=np.float64)
        rng = np.random.RandomState(42)
        batch = 10000
        while not stop_event.is_set():
            indices = rng.randint(0, size, batch)
            arr[indices] += 1.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

SPECTATOR_TYPES = {
    "cpu_burn": CPUBurnSpectator,
    "mem_burn": MemBurnSpectator,
    "cache_burn": CacheBurnSpectator,
}


def create_spectator(spec_type: str, intensity: float = 0.8) -> SpectatorWorkload:
    """Create a spectator workload by type name.

    Args:
        spec_type: One of "cpu_burn", "mem_burn", "cache_burn".
        intensity: Workload intensity in [0, 1].

    Returns:
        SpectatorWorkload instance.
    """
    cls = SPECTATOR_TYPES.get(spec_type)
    if cls is None:
        raise ValueError(f"Unknown spectator type: {spec_type}. "
                         f"Available: {list(SPECTATOR_TYPES.keys())}")
    return cls(intensity=intensity)
