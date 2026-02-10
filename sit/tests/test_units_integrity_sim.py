"""Unit integrity tests for the simulation layer.

Verifies that:
- All simulator output samples are in microseconds
- Derived statistics columns end with "_us" where expected
- SLO comparison uses the same unit
- No negative latencies
- All values finite and within bounds
"""

import numpy as np
import pytest

from sit.core.units import CANONICAL_LATENCY_UNIT
from sit.measure.tail import compute_all_tail_stats
from sit.sim.drift import DriftParams
from sit.sim.latency import BurstConfig, generate_latencies
from sit.sim.queue import QueueConfig
from sit.sim.workload import Workload, N_CHANNELS
from sit.sim.world import build_world, simulate_micro_run


def _make_target():
    return Workload(
        workload_id="target_0",
        role="target",
        features=np.full(N_CHANNELS, 0.5),
        base_service_us_mean=100.0,
        base_service_us_cv=0.2,
        slo_us=500000.0,
        tail_sensitivity=np.full(N_CHANNELS, 0.5),
    )


class TestSimulatorOutputUnits:
    def test_latencies_are_microseconds(self):
        """All generated latencies should be in a reasonable us range."""
        target = _make_target()
        rng = np.random.RandomState(42)

        result = generate_latencies(
            target=target,
            interference_us=50.0,
            drift_us=10.0,
            n_samples=5000,
            rng=rng,
            burst_config=BurstConfig(enabled=False),
            queue_config=QueueConfig(enabled=False),
        )

        latencies = result["latencies_us"]
        # Typical range for ~100us base: should be between 0 and ~10000 us
        assert np.all(latencies >= 0), "Negative latencies detected"
        assert np.all(np.isfinite(latencies)), "Non-finite latencies detected"
        assert np.all(latencies < 1e9), "Latencies exceed 1e9 us bound"

        # Sanity: mean should be roughly base + interference + drift
        expected_mean = 100.0 + 50.0 + 10.0
        actual_mean = float(np.mean(latencies))
        # Allow 50% tolerance for stochastic variation
        assert expected_mean * 0.5 < actual_mean < expected_mean * 2.0, (
            f"Mean {actual_mean:.1f} us not in expected range near {expected_mean:.1f} us"
        )

    def test_service_times_nonnegative(self):
        """Service times must be non-negative."""
        target = _make_target()
        rng = np.random.RandomState(42)

        result = generate_latencies(
            target=target,
            interference_us=0.0,
            drift_us=0.0,
            n_samples=5000,
            rng=rng,
            burst_config=BurstConfig(enabled=True, p_burst=0.01,
                                     pareto_alpha=2.5, scale_us=1000.0),
            queue_config=QueueConfig(enabled=False),
        )

        assert np.all(result["service_us"] >= 0)
        assert np.all(result["latencies_us"] >= 0)

    def test_queue_delays_nonnegative(self):
        """Queue delays must be non-negative when queueing is enabled."""
        target = _make_target()
        rng = np.random.RandomState(42)

        result = generate_latencies(
            target=target,
            interference_us=50.0,
            drift_us=0.0,
            n_samples=2000,
            rng=rng,
            burst_config=BurstConfig(enabled=False),
            queue_config=QueueConfig(enabled=True, concurrency=8, think_time_us=50.0),
        )

        assert np.all(result["queue_us"] >= 0)
        assert np.all(result["latencies_us"] >= result["service_us"] - 1e-6)


class TestDerivedStatsUnits:
    def test_tail_stats_columns_have_us_suffix(self):
        """All latency columns from compute_all_tail_stats should end in _us."""
        rng = np.random.RandomState(42)
        samples = rng.lognormal(mean=5.0, sigma=0.5, size=5000)

        stats = compute_all_tail_stats(samples, slo_us=500000.0)

        us_keys = [
            "mean_latency_us", "p95_latency_us",
            "p99_latency_us", "cvar99_latency_us",
        ]
        for key in us_keys:
            assert key in stats, f"Missing key: {key}"
            assert isinstance(stats[key], float), f"{key} not float"
            assert stats[key] >= 0, f"{key} is negative"

    def test_slo_comparison_consistent(self):
        """Violation rate should use the same unit as latency samples."""
        # Create samples where ~10% exceed the SLO
        rng = np.random.RandomState(42)
        samples = rng.lognormal(mean=5.0, sigma=0.5, size=10000)

        slo_us = float(np.quantile(samples, 0.90))

        stats = compute_all_tail_stats(samples, slo_us=slo_us)
        # Violation rate should be approximately 0.10
        assert 0.05 < stats["violation_rate"] < 0.15


class TestCanonicalUnitConstant:
    def test_unit_is_us(self):
        assert CANONICAL_LATENCY_UNIT == "us"


class TestWorldSimulationUnits:
    def test_micro_run_metadata_has_unit(self):
        """MicroRunResult metadata should declare the unit."""
        config = {
            "seed": 42,
            "slo_us": 500000.0,
            "sim": {
                "n_targets": 2,
                "n_spectators": 5,
                "n_regimes": 1,
                "drift": {"type": "none"},
                "channels": {"scale_us": 100.0},
                "toxic_pairs": {"enabled": False},
                "queue": {"enabled": False},
                "burst": {"enabled": False},
            },
        }
        rng = np.random.RandomState(42)
        world = build_world(config, rng)

        target = world.targets[0]
        spec = world.spectators[0]
        regime = world.regimes[0]

        sim_rng = np.random.RandomState(99)
        result = simulate_micro_run(
            world, target.workload_id, [spec.workload_id],
            regime.regime_id, t_index=0, n_samples=500, rng=sim_rng,
        )

        assert result.metadata["unit"] == "us"
        assert np.all(result.latencies_us >= 0)
        assert np.all(np.isfinite(result.latencies_us))
        assert np.all(result.latencies_us < 1e9)
