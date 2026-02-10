"""Test that IRBS cancels drift bias while naive A/B does not.

Creates a World with strong linear drift and NO spectators,
then compares naive A/B vs IRBS C-T-C estimates. Under drift,
the naive estimate should be significantly biased (non-zero),
while the IRBS estimate should be close to zero.
"""

import numpy as np
import pytest

from sit.sim.drift import DriftParams
from sit.sim.latency import BurstConfig, generate_latencies
from sit.sim.queue import QueueConfig
from sit.sim.workload import Workload, N_CHANNELS
from sit.measure.irbs import naive_ab_estimate, irbs_ctc_estimate


def _make_target():
    """Create a simple target workload."""
    return Workload(
        workload_id="target_test",
        role="target",
        features=np.full(N_CHANNELS, 0.5),
        base_service_us_mean=100.0,
        base_service_us_cv=0.2,
        slo_us=500000.0,
        tail_sensitivity=np.full(N_CHANNELS, 0.5),
    )


class TestDriftBiasExists:
    """Verify that naive A/B is biased under linear drift."""

    def test_naive_ab_biased_under_linear_drift(self):
        """Naive estimate should be significantly non-zero under drift
        even with no spectators (zero true interference)."""
        target = _make_target()
        drift = DriftParams(drift_type="linear", a_us_per_step=50.0)
        no_queue = QueueConfig(enabled=False)
        no_burst = BurstConfig(enabled=False)
        n_samples = 5000

        rng = np.random.RandomState(42)

        # Control at t=0
        ctrl = generate_latencies(
            target, interference_us=0.0, drift_us=0.0,
            n_samples=n_samples, rng=rng,
            drift_params=drift, t_index=0,
            burst_config=no_burst, queue_config=no_queue,
        )
        ctrl_mean = float(np.mean(ctrl["latencies_us"]))

        # Treatment at t=10 (same conditions, no spectators, but drift)
        treat = generate_latencies(
            target, interference_us=0.0, drift_us=50.0 * 10,
            n_samples=n_samples, rng=rng,
            drift_params=drift, t_index=10,
            burst_config=no_burst, queue_config=no_queue,
        )
        treat_mean = float(np.mean(treat["latencies_us"]))

        naive = naive_ab_estimate(ctrl_mean, treat_mean)

        # Naive estimate should be large (close to drift of 500 us)
        # True interference is 0, so naive bias ~= 500 us
        assert abs(naive) > 200.0, (
            f"Expected naive estimate to be biased by drift, got {naive:.2f}"
        )


class TestIRBSCancelsDrift:
    """Verify that IRBS C-T-C cancels linear drift."""

    def test_irbs_ctc_cancels_linear_drift(self):
        """IRBS C-T-C should produce near-zero estimate with no spectators
        under linear drift, because the flanking controls interpolate
        the drift to the treatment time."""
        target = _make_target()
        drift = DriftParams(drift_type="linear", a_us_per_step=50.0)
        no_queue = QueueConfig(enabled=False)
        no_burst = BurstConfig(enabled=False)
        n_samples = 5000

        rng = np.random.RandomState(42)

        # C1 at t=8: drift = 400
        c1 = generate_latencies(
            target, interference_us=0.0, drift_us=50.0 * 8,
            n_samples=n_samples, rng=rng,
            drift_params=drift, t_index=8,
            burst_config=no_burst, queue_config=no_queue,
        )
        c1_mean = float(np.mean(c1["latencies_us"]))

        # T at t=10: drift = 500, no spectators
        t_result = generate_latencies(
            target, interference_us=0.0, drift_us=50.0 * 10,
            n_samples=n_samples, rng=rng,
            drift_params=drift, t_index=10,
            burst_config=no_burst, queue_config=no_queue,
        )
        t_mean = float(np.mean(t_result["latencies_us"]))

        # C2 at t=12: drift = 600
        c2 = generate_latencies(
            target, interference_us=0.0, drift_us=50.0 * 12,
            n_samples=n_samples, rng=rng,
            drift_params=drift, t_index=12,
            burst_config=no_burst, queue_config=no_queue,
        )
        c2_mean = float(np.mean(c2["latencies_us"]))

        irbs = irbs_ctc_estimate(c1_mean, t_mean, c2_mean)

        # IRBS should be close to 0 (true interference is 0)
        assert abs(irbs) < 50.0, (
            f"Expected IRBS to cancel drift, got estimate = {irbs:.2f}"
        )

    def test_naive_much_worse_than_irbs(self):
        """Directly compare naive vs IRBS on the same scenario:
        |naive| should be much larger than |irbs|."""
        target = _make_target()
        drift = DriftParams(drift_type="linear", a_us_per_step=50.0)
        no_queue = QueueConfig(enabled=False)
        no_burst = BurstConfig(enabled=False)
        n_samples = 5000

        rng1 = np.random.RandomState(99)
        rng2 = np.random.RandomState(100)
        rng3 = np.random.RandomState(101)
        rng4 = np.random.RandomState(102)

        # Naive: control at t=0, treatment at t=10
        ctrl = generate_latencies(
            target, 0.0, 0.0, n_samples, rng1,
            drift_params=drift, t_index=0,
            burst_config=no_burst, queue_config=no_queue,
        )
        treat = generate_latencies(
            target, 0.0, 50.0 * 10, n_samples, rng2,
            drift_params=drift, t_index=10,
            burst_config=no_burst, queue_config=no_queue,
        )
        naive = abs(naive_ab_estimate(
            float(np.mean(ctrl["latencies_us"])),
            float(np.mean(treat["latencies_us"])),
        ))

        # IRBS: C at t=8, T at t=10, C at t=12
        c1 = generate_latencies(
            target, 0.0, 50.0 * 8, n_samples, rng2,
            drift_params=drift, t_index=8,
            burst_config=no_burst, queue_config=no_queue,
        )
        t_r = generate_latencies(
            target, 0.0, 50.0 * 10, n_samples, rng3,
            drift_params=drift, t_index=10,
            burst_config=no_burst, queue_config=no_queue,
        )
        c2 = generate_latencies(
            target, 0.0, 50.0 * 12, n_samples, rng4,
            drift_params=drift, t_index=12,
            burst_config=no_burst, queue_config=no_queue,
        )
        irbs = abs(irbs_ctc_estimate(
            float(np.mean(c1["latencies_us"])),
            float(np.mean(t_r["latencies_us"])),
            float(np.mean(c2["latencies_us"])),
        ))

        # IRBS error should be at least 3x smaller
        assert naive > 3 * irbs, (
            f"Expected naive ({naive:.2f}) >> irbs ({irbs:.2f})"
        )
