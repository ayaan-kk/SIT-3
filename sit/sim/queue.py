"""Closed-loop queueing model for latency simulation.

Simulates a single-server queue with closed-loop concurrency:
C clients each issue a request, wait for completion + think time,
then issue the next. This produces realistic queueing delay that
amplifies tail latency under interference.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class QueueConfig:
    """Queue model configuration.

    Attributes:
        enabled: Whether to simulate queueing delay.
        concurrency: Number of concurrent clients (closed loop).
        think_time_us: Client think time between response and next request.
    """
    enabled: bool = True
    concurrency: int = 16
    think_time_us: float = 50.0


def simulate_closed_loop_queue(
    service_times_us: np.ndarray,
    config: QueueConfig,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate closed-loop queueing to produce queue delays.

    Models C clients sharing one server. Each client sends a request,
    waits for service, thinks for think_time_us, then sends again.
    Requests from different clients can overlap, creating queue delays.

    This is a discrete-event simulation approximation:
    - We track server_free_at (when server becomes available)
    - Clients arrive in a round-robin pattern with staggered starts
    - Queue delay = max(0, server_free_at - arrival_time)

    Args:
        service_times_us: Array of service times for each request.
        config: Queue configuration.
        rng: Random state (for minor jitter).

    Returns:
        Tuple of (queue_delays_us, total_latencies_us).
    """
    n = len(service_times_us)
    C = config.concurrency
    think = config.think_time_us

    queue_delays = np.zeros(n, dtype=np.float64)
    total_latencies = np.zeros(n, dtype=np.float64)

    # Track when each client's next request arrives
    # Stagger initial arrivals
    client_next_arrival = np.arange(C, dtype=np.float64) * (think / C)
    server_free_at = 0.0

    for i in range(n):
        client_id = i % C

        # Client's request arrives
        arrival = client_next_arrival[client_id]

        # Queue delay: wait for server if busy
        qd = max(0.0, server_free_at - arrival)
        queue_delays[i] = qd

        # Service starts when both server is free and client arrives
        service_start = max(server_free_at, arrival)
        service_end = service_start + service_times_us[i]
        server_free_at = service_end

        # Total latency = queue delay + service time
        total_latencies[i] = qd + service_times_us[i]

        # Client's next arrival: after response + think time
        client_next_arrival[client_id] = service_end + think

    return queue_delays, total_latencies
