"""Tests for goodput units and model sanity (L3).

Verifies:
- violation_rate near 1 implies goodput near 0
- Throughput computed from timeline (positive when requests exist)
- goodput <= throughput always
- Open-loop model produces sensible results
"""

import numpy as np
import pandas as pd
import pytest

from sit.load.model import (
    ServingConfig,
    simulate_open_loop_host,
    simulate_closed_loop_host,
)
from sit.load.diagnostics import compute_queue_diagnostics
from sit.eval.pareto import check_model_sanity


class TestGoodputUnits:
    """Tests for goodput computation correctness."""

    def test_goodput_leq_throughput(self):
        """Goodput should never exceed throughput."""
        rng = np.random.RandomState(42)
        # Service times around 200us, SLO 500us -> most requests pass
        svc = rng.lognormal(mean=5.0, sigma=0.5, size=5000)
        service_dists = {"target_0": svc}

        metrics = simulate_open_loop_host(
            service_dists,
            lambda_rps_per_target=200.0,
            window_us=5_000_000.0,
            warmup_fraction=0.2,
            slo_us=500_000.0,
            rng=rng,
        )

        for tid, m in metrics.items():
            assert m["goodput_rps"] <= m["throughput_rps"] + 1e-6, (
                f"goodput {m['goodput_rps']} > throughput {m['throughput_rps']}"
            )

    def test_high_violation_low_goodput(self):
        """When violation_rate is near 1, goodput should be near 0."""
        rng = np.random.RandomState(42)
        # Very high service times -> everything violates SLO
        svc = np.full(5000, 1_000_000.0)  # 1 second = way above 500ms SLO
        service_dists = {"target_0": svc}

        metrics = simulate_open_loop_host(
            service_dists,
            lambda_rps_per_target=10.0,  # Low rate to avoid instability
            window_us=5_000_000.0,
            warmup_fraction=0.2,
            slo_us=500_000.0,
            rng=rng,
        )

        m = metrics["target_0"]
        # With service time = 1M us >> SLO = 500K us
        assert m["violation_rate"] > 0.95, f"Expected high violation, got {m['violation_rate']}"
        assert m["goodput_rps"] < m["throughput_rps"] * 0.1, (
            f"goodput {m['goodput_rps']} should be near 0 with high violation"
        )

    def test_low_violation_high_goodput(self):
        """When service times are well below SLO, goodput ~ throughput."""
        rng = np.random.RandomState(42)
        # Fast service -> all pass SLO
        svc = rng.lognormal(mean=4.0, sigma=0.3, size=5000)  # ~55us mean
        service_dists = {"target_0": svc}

        metrics = simulate_open_loop_host(
            service_dists,
            lambda_rps_per_target=50.0,
            window_us=5_000_000.0,
            warmup_fraction=0.2,
            slo_us=500_000.0,
            rng=rng,
        )

        m = metrics["target_0"]
        assert m["violation_rate"] < 0.05, f"Expected low violation, got {m['violation_rate']}"
        ratio = m["goodput_rps"] / max(m["throughput_rps"], 1e-6)
        assert ratio > 0.90, f"goodput/throughput ratio {ratio} should be near 1"

    def test_throughput_from_timeline(self):
        """Throughput should be computed from actual timeline."""
        rng = np.random.RandomState(42)
        svc = np.full(5000, 100.0)  # Constant 100us service
        service_dists = {"target_0": svc}

        metrics = simulate_open_loop_host(
            service_dists,
            lambda_rps_per_target=100.0,
            window_us=5_000_000.0,
            warmup_fraction=0.2,
            slo_us=500_000.0,
            rng=rng,
        )

        m = metrics["target_0"]
        # With Poisson arrivals at 100 rps, window 4s (after warmup)
        # Expected ~400 requests
        assert m["n_requests"] > 300, f"Expected ~400 requests, got {m['n_requests']}"
        assert m["n_requests"] < 600, f"Expected ~400 requests, got {m['n_requests']}"
        # Throughput should be approximately 100 rps
        assert abs(m["throughput_rps"] - 100.0) < 30, (
            f"Expected ~100 rps, got {m['throughput_rps']}"
        )

    def test_empty_target_returns_empty_metrics(self):
        """Empty service distributions should return empty metrics."""
        rng = np.random.RandomState(42)
        metrics = simulate_open_loop_host(
            {},
            lambda_rps_per_target=100.0,
            window_us=5_000_000.0,
            warmup_fraction=0.2,
            slo_us=500_000.0,
            rng=rng,
        )
        assert len(metrics) == 0

    def test_zero_lambda_returns_empty(self):
        """Zero arrival rate should produce empty metrics."""
        rng = np.random.RandomState(42)
        svc = np.full(100, 100.0)
        metrics = simulate_open_loop_host(
            {"target_0": svc},
            lambda_rps_per_target=0.0,
            window_us=5_000_000.0,
            warmup_fraction=0.2,
            slo_us=500_000.0,
            rng=rng,
        )
        m = metrics["target_0"]
        assert m["n_requests"] == 0
        assert m["goodput_rps"] == 0.0


class TestMultiTargetHost:
    """Tests for multi-target host queueing (partition killer mechanism)."""

    def test_more_targets_higher_latency(self):
        """More targets on host -> higher latency due to shared server."""
        rng = np.random.RandomState(42)
        svc = np.full(5000, 200.0)  # 200us per request

        # 1 target: low load
        metrics_1 = simulate_open_loop_host(
            {"t0": svc},
            lambda_rps_per_target=300.0,
            window_us=5_000_000.0,
            warmup_fraction=0.2,
            slo_us=500_000.0,
            rng=np.random.RandomState(42),
        )

        # 4 targets: 4x aggregate load on same server
        metrics_4 = simulate_open_loop_host(
            {"t0": svc, "t1": svc, "t2": svc, "t3": svc},
            lambda_rps_per_target=300.0,
            window_us=5_000_000.0,
            warmup_fraction=0.2,
            slo_us=500_000.0,
            rng=np.random.RandomState(42),
        )

        # Mean latency should be higher with 4 targets
        lat_1 = metrics_1["t0"]["mean_latency_us"]
        lat_4 = metrics_4["t0"]["mean_latency_us"]
        assert lat_4 > lat_1, (
            f"4 targets latency {lat_4} should be > 1 target {lat_1}"
        )

    def test_partition_vs_spread_mechanism(self):
        """Simulating partition (8 on 1) vs spread (1 per host) at high load.

        This demonstrates the partition killer mechanism: cramming targets
        on one host causes queue blowup at high load.
        """
        rng = np.random.RandomState(42)
        svc = np.full(5000, 200.0)  # 200us service time

        lambda_rps = 400.0

        # "Partition": 8 targets on 1 host (all share 1 server)
        partition_dists = {f"t{i}": svc for i in range(8)}
        m_partition = simulate_open_loop_host(
            partition_dists,
            lambda_rps_per_target=lambda_rps,
            window_us=5_000_000.0,
            warmup_fraction=0.2,
            slo_us=500_000.0,
            rng=np.random.RandomState(42),
        )

        # "Spread": 1 target per host (but with slightly higher service time
        # to simulate interference)
        svc_interference = np.full(5000, 400.0)  # 2x service time
        m_spread = simulate_open_loop_host(
            {"t0": svc_interference},
            lambda_rps_per_target=lambda_rps,
            window_us=5_000_000.0,
            warmup_fraction=0.2,
            slo_us=500_000.0,
            rng=np.random.RandomState(42),
        )

        # Partition host: 8 * 400 * 200/1e6 = 0.64 utilization
        # Spread host: 1 * 400 * 400/1e6 = 0.16 utilization
        # Spread should have lower p99 despite higher service time
        p99_partition = np.mean([m_partition[f"t{i}"]["p99_latency_us"] for i in range(8)])
        p99_spread = m_spread["t0"]["p99_latency_us"]

        # At high load, partition's queueing delays should dominate
        # The spread host (even with 2x service time) has much lower utilization
        assert p99_spread < p99_partition, (
            f"Spread p99 {p99_spread} should be < partition p99 {p99_partition}"
        )


class TestClosedLoopModel:
    """Tests for closed-loop serving model."""

    def test_closed_loop_produces_metrics(self):
        """Closed-loop model should produce valid metrics."""
        rng = np.random.RandomState(42)
        svc = rng.lognormal(mean=5.0, sigma=0.3, size=5000)
        service_dists = {"target_0": svc}

        metrics = simulate_closed_loop_host(
            service_dists,
            concurrency_per_target=4,
            think_time_us=100.0,
            window_us=5_000_000.0,
            warmup_fraction=0.2,
            slo_us=500_000.0,
            rng=rng,
        )

        m = metrics["target_0"]
        assert m["n_requests"] > 0
        assert m["throughput_rps"] > 0
        assert m["goodput_rps"] <= m["throughput_rps"] + 1e-6


class TestQueueDiagnostics:
    """Tests for queue diagnostic computation."""

    def test_diagnostics_computation(self):
        """Queue diagnostics should compute from sweep data."""
        data = [
            {
                "load_level": 100, "scheduler_name": "random",
                "target_id": "t0", "utilization": 0.3,
                "mean_queue_delay_us": 50, "p95_queue_delay_us": 200,
                "mean_service_us": 200, "busy_fraction": 0.2,
                "mean_latency_us": 250, "p99_latency_us": 1000,
                "cvar99_latency_us": 1200, "violation_rate": 0.01,
                "throughput_rps": 100, "goodput_rps": 99,
            },
        ]
        sweep_df = pd.DataFrame(data)
        diag = compute_queue_diagnostics(sweep_df, slo_us=500_000.0)
        assert len(diag) > 0
        assert "utilization" in diag.columns
        assert "stable" in diag.columns

    def test_sanity_check_passes_good_data(self):
        """Model sanity should pass for consistent data."""
        data = [
            {
                "load_level": 100, "scheduler_name": "random",
                "p99_latency_us": 1000, "violation_rate": 0.01,
                "goodput_rps": 99, "throughput_rps": 100,
                "sanity_p99_viol": True, "sanity_throughput": True,
                "stable": True,
            },
        ]
        diag_df = pd.DataFrame(data)
        passed, details = check_model_sanity(diag_df)
        assert passed, f"Sanity should pass for good data: {details}"

    def test_sanity_check_fails_bad_data(self):
        """Model sanity should fail for inconsistent data."""
        data = [
            {
                "load_level": 100, "scheduler_name": "bad",
                "p99_latency_us": 9_000_000,
                "violation_rate": 0.001,  # p99 >> SLO but low violation!
                "goodput_rps": 100, "throughput_rps": 100,
                "sanity_p99_viol": False,
                "sanity_throughput": True,
                "stable": True,
            },
        ]
        diag_df = pd.DataFrame(data)
        passed, details = check_model_sanity(diag_df)
        assert not passed, "Sanity should fail for inconsistent data"
