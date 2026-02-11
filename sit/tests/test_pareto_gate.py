"""Tests for Pareto dominance gate (L1).

Verifies:
- SIT-safe on Pareto frontier in high-load region
- SIT-safe dominates static_partition in at least one point
- Pareto frontier computation is correct
"""

import numpy as np
import pandas as pd
import pytest

from sit.load.pareto import (
    construct_pareto_data,
    find_pareto_frontier,
    is_dominated,
)
from sit.eval.pareto import (
    check_pareto_dominance_over_partition,
    check_admission_advantage,
)
from sit.eval.gates import gate_load_pareto_dominance


class TestParetoFrontier:
    """Tests for Pareto frontier computation."""

    def test_is_dominated(self):
        """Test Pareto dominance check."""
        # B dominates A: higher goodput, lower risk
        assert is_dominated(10, 100, 20, 50)
        # Not dominated: A has higher goodput
        assert not is_dominated(20, 100, 10, 50)
        # Not dominated: same point
        assert not is_dominated(10, 100, 10, 100)
        # B dominates A: same goodput but lower risk
        assert is_dominated(10, 100, 10, 50)

    def test_find_pareto_frontier_simple(self):
        """Test frontier with simple 2D points."""
        points = [
            (10, 100),  # dominated by (20, 50)
            (20, 50),   # on frontier
            (15, 40),   # on frontier (lower risk)
            (5, 200),   # dominated
            (25, 80),   # on frontier (highest goodput)
        ]
        frontier = find_pareto_frontier(points)
        # Points 1 (20,50), 2 (15,40), 4 (25,80) should be on frontier
        assert 1 in frontier
        assert 2 in frontier
        assert 4 in frontier
        assert 0 not in frontier
        assert 3 not in frontier

    def test_find_pareto_frontier_all_optimal(self):
        """When no point dominates another, all are on frontier."""
        points = [
            (10, 100),  # low goodput, high risk
            (20, 200),  # high goodput, higher risk
            (5, 50),    # lowest goodput, lowest risk
        ]
        frontier = find_pareto_frontier(points)
        assert len(frontier) == 3

    def test_construct_pareto_data(self):
        """Test Pareto data construction from sweep results."""
        data = []
        for load in [100, 200, 300]:
            for sched in ["random", "sit_safe_ucb"]:
                goodput = 90 if sched == "sit_safe_ucb" else 70
                risk = 500 if sched == "sit_safe_ucb" else 800
                data.append({
                    "load_level": load,
                    "scheduler_name": sched,
                    "goodput_rps": goodput,
                    "cvar99_latency_us": risk,
                    "violation_rate": 0.01,
                    "throughput_rps": 100,
                    "n_requests": 500,
                })

        sweep_df = pd.DataFrame(data)
        pareto_df = construct_pareto_data(sweep_df)

        assert len(pareto_df) > 0
        assert "on_pareto_frontier" in pareto_df.columns

        # SIT-safe should be on frontier (higher goodput, lower risk)
        sit_frontier = pareto_df[
            (pareto_df["scheduler_name"] == "sit_safe_ucb") &
            (pareto_df["on_pareto_frontier"] == True)
        ]
        assert len(sit_frontier) > 0


class TestParetoGate:
    """Tests for L1 gate (Pareto dominance over partition)."""

    def _make_sweep_df(self, partition_penalty_factor=4.0):
        """Create sweep data where SIT-safe beats partition at high load."""
        data = []
        for load in [100, 200, 300, 400, 500, 600]:
            for sched in ["random", "static_partition", "sit_safe_ucb"]:
                if sched == "static_partition":
                    # Partition: high utilization (8 targets on 1 host)
                    # At high load, queue blows up
                    utilization = load * 8 * 200 / 1e6
                    if utilization > 0.95:
                        goodput = 5.0  # Near zero
                        risk = 5_000_000.0  # Very high
                        viol = 0.99
                    else:
                        goodput = load * 0.98
                        risk = 200 / (1 - utilization) * 10
                        viol = 0.02
                elif sched == "sit_safe_ucb":
                    # SIT-safe: low utilization (targets spread)
                    utilization = load * 1.3 * 400 / 1e6
                    goodput = load * 0.95
                    risk = 400 + 400 * utilization / max(1 - utilization, 0.01) * 5
                    viol = max(0.01, utilization * 0.02)
                else:
                    # Random: moderate
                    utilization = load * 2 * 500 / 1e6
                    goodput = load * 0.8
                    risk = 500 + 500 * utilization / max(1 - utilization, 0.01) * 5
                    viol = max(0.02, utilization * 0.05)

                data.append({
                    "load_level": load,
                    "episode_id": 0,
                    "scheduler_name": sched,
                    "target_id": "t0",
                    "goodput_rps": goodput,
                    "cvar99_latency_us": risk,
                    "violation_rate": viol,
                    "throughput_rps": load,
                    "n_requests": 500,
                })

        return pd.DataFrame(data)

    def test_gate_l1_passes(self):
        """L1 gate should pass when SIT-safe dominates partition."""
        sweep_df = self._make_sweep_df()
        pareto_df = construct_pareto_data(sweep_df)

        passed = gate_load_pareto_dominance(
            pareto_df, sweep_df,
            risk_band=0.10,
            top_load_fraction=0.30,
        )
        assert passed, "L1 should pass: SIT-safe should dominate partition at high load"

    def test_gate_l1_fails_when_partition_wins(self):
        """L1 gate should fail when partition dominates."""
        data = []
        for load in [100, 200, 300]:
            for sched in ["static_partition", "sit_safe_ucb"]:
                # Make partition BETTER than SIT-safe
                goodput = 100 if sched == "static_partition" else 50
                risk = 500 if sched == "static_partition" else 1000
                data.append({
                    "load_level": load,
                    "episode_id": 0,
                    "scheduler_name": sched,
                    "target_id": "t0",
                    "goodput_rps": goodput,
                    "cvar99_latency_us": risk,
                    "violation_rate": 0.01,
                    "throughput_rps": 100,
                    "n_requests": 500,
                })

        sweep_df = pd.DataFrame(data)
        pareto_df = construct_pareto_data(sweep_df)

        passed = gate_load_pareto_dominance(
            pareto_df, sweep_df,
            risk_band=0.10,
            top_load_fraction=0.30,
        )
        assert not passed, "L1 should fail when partition dominates"

    def test_pareto_dominance_check(self):
        """Test the dominance check helper directly."""
        sweep_df = self._make_sweep_df()
        pareto_df = construct_pareto_data(sweep_df)

        passed, details = check_pareto_dominance_over_partition(
            pareto_df, sweep_df,
            risk_band=0.10,
            top_load_fraction=0.30,
        )

        assert "on_frontier_high_load" in details
        assert "dominates_partition" in details


class TestAdmissionAdvantage:
    """Tests for admission control advantage check."""

    def test_admission_advantage_passes(self):
        """SIT-safe with higher max load should pass."""
        data = [
            {"scheduler_name": "sit_safe_ucb", "max_feasible_load_rps": 500,
             "goodput_at_max_rps": 475, "load_level": None,
             "mean_violation_rate": None, "mean_goodput_rps": None,
             "mean_cvar99_us": None, "mean_throughput_rps": None,
             "v_target": 0.02, "slo_us": 500000, "cvar99_at_max_us": 1000,
             "throughput_at_max_rps": 500},
            {"scheduler_name": "static_partition", "max_feasible_load_rps": 300,
             "goodput_at_max_rps": 294, "load_level": None,
             "mean_violation_rate": None, "mean_goodput_rps": None,
             "mean_cvar99_us": None, "mean_throughput_rps": None,
             "v_target": 0.02, "slo_us": 500000, "cvar99_at_max_us": 1000,
             "throughput_at_max_rps": 300},
        ]
        admission_df = pd.DataFrame(data)

        passed, details = check_admission_advantage(admission_df)
        assert passed, f"Advantage 500/300 = 1.67 >= 1.15: {details}"

    def test_admission_advantage_fails(self):
        """Similar max load should fail the advantage check."""
        data = [
            {"scheduler_name": "sit_safe_ucb", "max_feasible_load_rps": 310,
             "goodput_at_max_rps": 300, "load_level": None,
             "mean_violation_rate": None, "mean_goodput_rps": None,
             "mean_cvar99_us": None, "mean_throughput_rps": None,
             "v_target": 0.02, "slo_us": 500000, "cvar99_at_max_us": 1000,
             "throughput_at_max_rps": 310},
            {"scheduler_name": "static_partition", "max_feasible_load_rps": 300,
             "goodput_at_max_rps": 294, "load_level": None,
             "mean_violation_rate": None, "mean_goodput_rps": None,
             "mean_cvar99_us": None, "mean_throughput_rps": None,
             "v_target": 0.02, "slo_us": 500000, "cvar99_at_max_us": 1000,
             "throughput_at_max_rps": 300},
        ]
        admission_df = pd.DataFrame(data)

        passed, details = check_admission_advantage(admission_df)
        assert not passed, f"Advantage 310/300 = 1.03 < 1.15: {details}"
