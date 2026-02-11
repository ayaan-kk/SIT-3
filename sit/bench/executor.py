"""Full execution matrix: runs all policies across all regimes.

Evaluates every scheduling policy across load, interference, and job mix
configurations, collecting comprehensive metrics for each run.
"""

import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.measure.tail import compute_all_tail_stats
from sit.sched import POLICY_REGISTRY, PlacementDecision, get_policy
from sit.sim.world import World, build_world, simulate_micro_run

logger = get_logger("bench.executor")


def build_regime_configs():
    """Generate all interference regime configurations."""
    return {
        "iid_noise": {
            "description": "IID noise (baseline)",
            "drift": {"type": "none"},
            "interactions": {"enabled": False, "gamma": 0.0},
            "toxic_pairs": {"enabled": True, "sparsity": 0.05},
            "burst": {"enabled": False},
        },
        "structured": {
            "description": "Structured interference (channel-dominated)",
            "drift": {"type": "none"},
            "interactions": {"enabled": False},
            "toxic_pairs": {"enabled": True, "sparsity": 0.10},
            "burst": {"enabled": False},
        },
        "heavy_tail_burst": {
            "description": "Heavy-tailed burst interference",
            "drift": {"type": "none"},
            "interactions": {"enabled": False},
            "toxic_pairs": {"enabled": True, "sparsity": 0.05},
            "burst": {"enabled": True, "p_burst": 0.01, "pareto_alpha": 1.5, "scale_us": 10000.0},
        },
        "adversarial": {
            "description": "Adversarial co-location",
            "drift": {"type": "none"},
            "interactions": {"enabled": True, "gamma": 0.05},
            "toxic_pairs": {"enabled": True, "sparsity": 0.20, "lognormal_sigma": 1.2},
            "burst": {"enabled": True, "p_burst": 0.005},
        },
        "drifting": {
            "description": "Drifting interference matrix",
            "drift": {"type": "linear", "a_us_per_step": 200.0},
            "interactions": {"enabled": False},
            "toxic_pairs": {"enabled": True, "sparsity": 0.05},
            "burst": {"enabled": False},
        },
        "partial_obs": {
            "description": "Partial observability (sparse probes)",
            "drift": {"type": "linear", "a_us_per_step": 50.0},
            "interactions": {"enabled": False},
            "toxic_pairs": {"enabled": True, "sparsity": 0.03},
            "burst": {"enabled": True, "p_burst": 0.002},
        },
    }


def build_load_configs():
    """Generate load regime configurations."""
    return {
        "low": {"queue": {"concurrency": 8, "think_time_us": 100.0}, "load_label": "20-40%"},
        "medium": {"queue": {"concurrency": 16, "think_time_us": 50.0}, "load_label": "50-70%"},
        "high": {"queue": {"concurrency": 24, "think_time_us": 30.0}, "load_label": "80-95%"},
        "saturation": {"queue": {"concurrency": 32, "think_time_us": 15.0}, "load_label": "95-105%"},
    }


def build_job_mix_configs():
    """Generate job mix configurations."""
    return {
        "homogeneous": {"cv_range": (0.15, 0.25), "base_range": (100, 200)},
        "hetero_compute": {"cv_range": (0.20, 0.40), "base_range": (50, 400)},
        "hetero_memory": {"cv_range": (0.15, 0.35), "base_range": (80, 300)},
        "mixed_service": {"cv_range": (0.10, 0.50), "base_range": (50, 500)},
    }


def run_single_evaluation(
    policy_name: str,
    world: World,
    config: Dict[str, Any],
    n_episodes: int,
    seed: int,
) -> pd.DataFrame:
    """Run a single policy evaluation across episodes.

    Returns a DataFrame with one row per episode containing all metrics.
    """
    rng = np.random.RandomState(seed)
    policy_fn = get_policy(policy_name)
    slo_us = config.get("slo_us", 500000)
    n_samples = config.get("sim", {}).get("n_samples_per_micro_run", 2000)

    rows = []
    t_start = time.time()

    for episode_id in range(n_episodes):
        # Select target and regime deterministically
        target = world.targets[episode_id % len(world.targets)]
        regime = world.regimes[episode_id % len(world.regimes)]

        # Get scheduling decision
        decision_start = time.time()
        decision = policy_fn(
            target, world, regime, config, rng, trial_id=episode_id,
        )
        decision_time_us = (time.time() - decision_start) * 1e6

        # Simulate micro-run
        t_index = episode_id % 20  # time index for drift
        mr = simulate_micro_run(
            world, target.workload_id, decision.spectator_ids,
            regime.regime_id, t_index, n_samples, rng,
        )

        latencies = mr.latencies_us
        stats = compute_all_tail_stats(latencies, slo_us=slo_us, alpha=0.99)

        # Compute all metrics
        violation_rate = float(np.mean(latencies > slo_us))
        goodput = 1.0 - violation_rate
        mean_lat = float(np.mean(latencies))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        p999 = float(np.percentile(latencies, 99.9))
        cvar95 = float(np.mean(latencies[latencies >= np.percentile(latencies, 95)])) if len(latencies[latencies >= np.percentile(latencies, 95)]) > 0 else p95
        cvar99 = stats.get("cvar99", p99)
        cvar999 = float(np.mean(latencies[latencies >= np.percentile(latencies, 99.9)])) if len(latencies[latencies >= np.percentile(latencies, 99.9)]) > 0 else p999

        # Queue metrics
        queue_lats = mr.queue_us
        queue_var = float(np.var(queue_lats)) if len(queue_lats) > 0 else 0.0
        mean_backlog = float(np.mean(queue_lats)) if len(queue_lats) > 0 else 0.0
        max_backlog = float(np.max(queue_lats)) if len(queue_lats) > 0 else 0.0

        # Catastrophe detection
        catastrophe = violation_rate > 0.10
        n_spectators = len(decision.spectator_ids)

        rows.append({
            "policy": policy_name,
            "episode_id": episode_id,
            "seed": seed,
            "target_id": target.workload_id,
            "regime_id": regime.regime_id,
            "n_spectators": n_spectators,
            "mean_latency_us": mean_lat,
            "p95_latency_us": p95,
            "p99_latency_us": p99,
            "p999_latency_us": p999,
            "cvar95_us": cvar95,
            "cvar99_us": cvar99,
            "cvar999_us": cvar999,
            "slo_us": slo_us,
            "violation_rate": violation_rate,
            "goodput": goodput,
            "throughput_inv_us": 1.0 / mean_lat if mean_lat > 0 else 0.0,
            "queue_variance": queue_var,
            "mean_backlog_us": mean_backlog,
            "max_backlog_us": max_backlog,
            "catastrophe": catastrophe,
            "safety_pass": decision.safety_pass,
            "decision_time_us": decision_time_us,
            "n_samples": n_samples,
        })

    elapsed = time.time() - t_start
    logger.info(
        "Policy %s: %d episodes in %.1fs (%.1f eps/s)",
        policy_name, n_episodes, elapsed, n_episodes / elapsed if elapsed > 0 else 0,
    )

    return pd.DataFrame(rows)


def run_full_evaluation(
    config: Dict[str, Any],
    n_episodes_per_config: int = 200,
    policies: List[str] = None,
    output_dir: str = "results/bench",
) -> pd.DataFrame:
    """Run the full execution matrix across all policies and configurations.

    Args:
        config: Base configuration dict.
        n_episodes_per_config: Episodes per policy/regime combination.
        policies: List of policy names (None = all).
        output_dir: Directory for output files.

    Returns:
        Combined DataFrame with all results.
    """
    os.makedirs(output_dir, exist_ok=True)

    if policies is None:
        policies = list(POLICY_REGISTRY.keys())

    base_seed = config.get("seed", 123)
    interference_regimes = build_regime_configs()
    load_regimes = build_load_configs()

    all_results = []
    total_runs = 0

    for ir_name, ir_config in interference_regimes.items():
        for lr_name, lr_config in load_regimes.items():
            # Build world with this regime configuration
            regime_config = _build_regime_config(config, ir_config, lr_config)
            rng = np.random.RandomState(base_seed)
            world = build_world(regime_config, rng)

            for policy_name in policies:
                # Use deterministic seed per combination
                combo_seed = base_seed + hash(f"{ir_name}_{lr_name}_{policy_name}") % (2**31)
                combo_seed = abs(combo_seed) % (2**31)

                logger.info(
                    "Running: policy=%s, interference=%s, load=%s",
                    policy_name, ir_name, lr_name,
                )

                result_df = run_single_evaluation(
                    policy_name, world, regime_config,
                    n_episodes_per_config, combo_seed,
                )

                # Add regime metadata
                result_df["interference_regime"] = ir_name
                result_df["load_regime"] = lr_name
                result_df["load_label"] = lr_config.get("load_label", lr_name)

                all_results.append(result_df)
                total_runs += len(result_df)

    combined_df = pd.concat(all_results, ignore_index=True)

    # Save raw results
    raw_path = os.path.join(output_dir, "full_results.csv")
    combined_df.to_csv(raw_path, index=False)
    logger.info("Full evaluation complete: %d total runs saved to %s", total_runs, raw_path)

    return combined_df


def _build_regime_config(base_config, ir_config, lr_config):
    """Merge base config with interference and load regime overrides."""
    import copy
    config = copy.deepcopy(base_config)

    sim = config.setdefault("sim", {})

    # Apply interference regime
    if "drift" in ir_config:
        sim["drift"] = ir_config["drift"]
    if "interactions" in ir_config:
        sim["interactions"] = ir_config["interactions"]
    if "toxic_pairs" in ir_config:
        tp = sim.setdefault("toxic_pairs", {})
        tp.update(ir_config["toxic_pairs"])
    if "burst" in ir_config:
        sim["burst"] = ir_config["burst"]

    # Apply load regime
    if "queue" in lr_config:
        q = sim.setdefault("queue", {"enabled": True})
        q.update(lr_config["queue"])
        q["enabled"] = True

    return config
