"""Test IRBS accuracy when measuring real interference under drift.

Adds a single spectator with known ground truth X and verifies that
IRBS recovers the interference more accurately than naive A/B
under linear drift.
"""

import numpy as np
import pytest

from sit.sim.drift import DriftParams
from sit.sim.latency import BurstConfig, generate_latencies
from sit.sim.queue import QueueConfig
from sit.sim.workload import Workload, N_CHANNELS
from sit.measure.irbs import naive_ab_estimate, irbs_ctc_estimate


def _make_target():
    return Workload(
        workload_id="target_test",
        role="target",
        features=np.full(N_CHANNELS, 0.5),
        base_service_us_mean=100.0,
        base_service_us_cv=0.2,
        slo_us=500000.0,
        tail_sensitivity=np.full(N_CHANNELS, 0.5),
    )


class TestIRBSWithInterference:
    """IRBS should recover true interference better than naive under drift."""

    def test_irbs_recovers_interference_under_drift(self):
        """With known interference X_us and linear drift, IRBS absolute
        error should be at least 3x smaller than naive absolute error."""
        target = _make_target()
        drift = DriftParams(drift_type="linear", a_us_per_step=50.0)
        no_queue = QueueConfig(enabled=False)
        no_burst = BurstConfig(enabled=False)
        n_samples = 5000

        # Known ground truth interference
        true_x_us = 80.0

        # Run multiple repetitions and average errors
        naive_errors = []
        irbs_errors = []

        for rep in range(10):
            base_seed = 1000 + rep * 100
            rng_ctrl0 = np.random.RandomState(base_seed)
            rng_treat = np.random.RandomState(base_seed + 1)
            rng_c1 = np.random.RandomState(base_seed + 2)
            rng_t = np.random.RandomState(base_seed + 3)
            rng_c2 = np.random.RandomState(base_seed + 4)

            t_base = rep * 20

            # --- Naive A/B ---
            # Control at t_base (no spectator)
            ctrl = generate_latencies(
                target, interference_us=0.0,
                drift_us=50.0 * t_base,
                n_samples=n_samples, rng=rng_ctrl0,
                drift_params=drift, t_index=t_base,
                burst_config=no_burst, queue_config=no_queue,
            )
            # Treatment at t_base + 10 (with spectator)
            treat = generate_latencies(
                target, interference_us=true_x_us,
                drift_us=50.0 * (t_base + 10),
                n_samples=n_samples, rng=rng_treat,
                drift_params=drift, t_index=t_base + 10,
                burst_config=no_burst, queue_config=no_queue,
            )
            naive_est = naive_ab_estimate(
                float(np.mean(ctrl["latencies_us"])),
                float(np.mean(treat["latencies_us"])),
            )
            naive_errors.append(abs(naive_est - true_x_us))

            # --- IRBS C-T-C ---
            # C1 at t_base+8
            c1 = generate_latencies(
                target, interference_us=0.0,
                drift_us=50.0 * (t_base + 8),
                n_samples=n_samples, rng=rng_c1,
                drift_params=drift, t_index=t_base + 8,
                burst_config=no_burst, queue_config=no_queue,
            )
            # T at t_base+10
            t_r = generate_latencies(
                target, interference_us=true_x_us,
                drift_us=50.0 * (t_base + 10),
                n_samples=n_samples, rng=rng_t,
                drift_params=drift, t_index=t_base + 10,
                burst_config=no_burst, queue_config=no_queue,
            )
            # C2 at t_base+12
            c2 = generate_latencies(
                target, interference_us=0.0,
                drift_us=50.0 * (t_base + 12),
                n_samples=n_samples, rng=rng_c2,
                drift_params=drift, t_index=t_base + 12,
                burst_config=no_burst, queue_config=no_queue,
            )
            irbs_est = irbs_ctc_estimate(
                float(np.mean(c1["latencies_us"])),
                float(np.mean(t_r["latencies_us"])),
                float(np.mean(c2["latencies_us"])),
            )
            irbs_errors.append(abs(irbs_est - true_x_us))

        median_naive = float(np.median(naive_errors))
        median_irbs = float(np.median(irbs_errors))

        # IRBS should have at least 3x less error
        ratio = median_naive / max(median_irbs, 1e-6)
        assert ratio >= 3.0, (
            f"Expected IRBS 3x better than naive. "
            f"Median naive error: {median_naive:.2f}, "
            f"median IRBS error: {median_irbs:.2f}, ratio: {ratio:.2f}"
        )

    def test_irbs_estimate_near_true_value(self):
        """IRBS estimate should be reasonably close to true interference."""
        target = _make_target()
        drift = DriftParams(drift_type="linear", a_us_per_step=50.0)
        no_queue = QueueConfig(enabled=False)
        no_burst = BurstConfig(enabled=False)
        true_x_us = 120.0
        n_samples = 8000

        rng1 = np.random.RandomState(555)
        rng2 = np.random.RandomState(556)
        rng3 = np.random.RandomState(557)

        c1 = generate_latencies(
            target, 0.0, 50.0 * 8, n_samples, rng1,
            drift_params=drift, t_index=8,
            burst_config=no_burst, queue_config=no_queue,
        )
        t_r = generate_latencies(
            target, true_x_us, 50.0 * 10, n_samples, rng2,
            drift_params=drift, t_index=10,
            burst_config=no_burst, queue_config=no_queue,
        )
        c2 = generate_latencies(
            target, 0.0, 50.0 * 12, n_samples, rng3,
            drift_params=drift, t_index=12,
            burst_config=no_burst, queue_config=no_queue,
        )

        irbs_est = irbs_ctc_estimate(
            float(np.mean(c1["latencies_us"])),
            float(np.mean(t_r["latencies_us"])),
            float(np.mean(c2["latencies_us"])),
        )

        # Should be within 30% of true value
        relative_error = abs(irbs_est - true_x_us) / true_x_us
        assert relative_error < 0.3, (
            f"IRBS estimate {irbs_est:.2f} too far from truth {true_x_us:.2f} "
            f"(relative error: {relative_error:.2%})"
        )
