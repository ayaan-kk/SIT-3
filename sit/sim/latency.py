"""Latency sample generation model.

Generates latency samples with explicit distributional choices:
- Base service time: LogNormal (right-skewed, realistic)
- Drift: deterministic baseline shift
- Interference: additive mean service time increase
- Multiplicative noise: small Normal perturbation
- Burst spikes: rare Pareto-distributed tail events
- Queueing: optional closed-loop delay

All values in microseconds.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from sit.sim.drift import DriftParams, mu_drift_us, drift_noise_std
from sit.sim.queue import QueueConfig, simulate_closed_loop_queue
from sit.sim.workload import Workload


@dataclass
class BurstConfig:
    """Configuration for rare tail burst events."""
    enabled: bool = True
    p_burst: float = 0.002
    pareto_alpha: float = 2.5
    scale_us: float = 5000.0


def generate_service_times(
    target: Workload,
    interference_us: float,
    drift_us: float,
    n_samples: int,
    rng: np.random.RandomState,
    drift_params: Optional[DriftParams] = None,
    t_index: int = 0,
    burst_config: Optional[BurstConfig] = None,
) -> np.ndarray:
    """Generate service time samples in microseconds.

    Model:
        S_i = base_i + drift + interference + noise_i + burst_i

    where:
        base_i ~ LogNormal(mu, sigma) parameterized by mean and CV
        drift = mu_drift_us(t_index) (deterministic)
        interference = sum of pairwise X_{t,s}
        noise_i = S_i * eta_i, eta_i ~ Normal(0, 0.02) clipped to [-0.1, 0.1]
        burst_i = Pareto(alpha, scale) with probability p_burst

    Args:
        target: Target workload with base service time params.
        interference_us: Total interference in microseconds.
        drift_us: Drift value in microseconds.
        n_samples: Number of samples to generate.
        rng: Random state for reproducibility.
        drift_params: Optional drift params for heteroscedastic noise.
        t_index: Current time index.
        burst_config: Optional burst spike configuration.

    Returns:
        Service times in microseconds, shape (n_samples,).
    """
    mean = target.base_service_us_mean
    cv = target.base_service_us_cv

    # LogNormal parameters from mean and CV
    # If X ~ LogNormal(mu_ln, sigma_ln), then
    #   E[X] = exp(mu_ln + sigma_ln^2/2)
    #   CV = sqrt(exp(sigma_ln^2) - 1)
    # So sigma_ln = sqrt(log(1 + cv^2)), mu_ln = log(mean) - sigma_ln^2/2
    sigma_ln = np.sqrt(np.log(1.0 + cv ** 2))
    mu_ln = np.log(max(mean, 1.0)) - 0.5 * sigma_ln ** 2

    base = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=n_samples)

    # Add interference and drift to mean
    service = base + interference_us + drift_us

    # Multiplicative noise: small perturbation
    noise_std = 0.02
    if drift_params is not None:
        noise_std = drift_noise_std(t_index, drift_params, 0.02)
    eta = rng.normal(0.0, noise_std, size=n_samples)
    eta = np.clip(eta, -0.1, 0.1)
    service = service * (1.0 + eta)

    # Burst spikes
    if burst_config is not None and burst_config.enabled:
        burst_mask = rng.random(n_samples) < burst_config.p_burst
        n_bursts = int(np.sum(burst_mask))
        if n_bursts > 0:
            # Pareto: X = scale * (U^{-1/alpha}) where U ~ Uniform(0,1)
            burst_values = burst_config.scale_us * (
                rng.pareto(burst_config.pareto_alpha, size=n_bursts)
            )
            service[burst_mask] += burst_values

    # Floor at zero (should be rare to need this)
    service = np.maximum(service, 0.0)

    return service


def generate_latencies(
    target: Workload,
    interference_us: float,
    drift_us: float,
    n_samples: int,
    rng: np.random.RandomState,
    drift_params: Optional[DriftParams] = None,
    t_index: int = 0,
    burst_config: Optional[BurstConfig] = None,
    queue_config: Optional[QueueConfig] = None,
) -> dict:
    """Generate full latency samples including optional queueing.

    Returns a dict with:
        latencies_us: Final end-to-end latency samples
        service_us: Service times (pre-queue)
        queue_us: Queue delays (zeros if queue disabled)
        metadata: Dict with drift_us, interference_us, n_bursts
    """
    service = generate_service_times(
        target=target,
        interference_us=interference_us,
        drift_us=drift_us,
        n_samples=n_samples,
        rng=rng,
        drift_params=drift_params,
        t_index=t_index,
        burst_config=burst_config,
    )

    queue_delays = np.zeros(n_samples, dtype=np.float64)
    latencies = service.copy()

    if queue_config is not None and queue_config.enabled:
        queue_delays, latencies = simulate_closed_loop_queue(
            service_times_us=service,
            config=queue_config,
            rng=rng,
        )

    # Simulator boundary assert: all in microseconds, finite, bounded
    assert np.all(np.isfinite(latencies)), "Non-finite latency detected"
    assert np.all(latencies >= 0), "Negative latency detected"
    assert np.all(latencies < 1e9), "Latency exceeds 1e9 us (1000s) bound"

    return {
        "latencies_us": latencies,
        "service_us": service,
        "queue_us": queue_delays,
        "metadata": {
            "drift_us": float(drift_us),
            "interference_us": float(interference_us),
            "n_samples": n_samples,
            "queue_enabled": queue_config is not None and queue_config.enabled,
        },
    }
