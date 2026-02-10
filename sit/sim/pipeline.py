"""Simulation pipeline: runs trials, IRBS measurement, and CI coverage.

Integrates the simulator, measurement layer, and evaluation modules
to produce all required raw and derived datasets.
"""

import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.core.schema import SCHEMA_VERSION
from sit.core.units import CANONICAL_LATENCY_UNIT
from sit.eval.coverage import evaluate_ci_coverage
from sit.measure.bootstrap import bootstrap_ci
from sit.measure.irbs import (
    MeasurementEvent,
    irbs_ctc_estimate,
    naive_ab_estimate,
)
from sit.measure.tail import compute_all_tail_stats, estimate_var
from sit.sim.world import (
    MicroRunResult,
    World,
    build_world,
    ground_truth_to_dataframe,
    simulate_micro_run,
)

logger = get_logger("sim.pipeline")


def run_sim_trials(
    world: World,
    config: Dict[str, Any],
    run_id: str,
    config_hash: str,
    git_commit: str,
    timestamp: str,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Run simulated trials: pick target/spectators/regime, simulate, measure.

    Returns a trials DataFrame conforming to the canonical schema.
    """
    n_trials = config["n_trials"]
    slo_us = config["slo_us"]
    sim_cfg = config.get("sim", {})
    n_samples = sim_cfg.get("n_samples_per_micro_run", 2000)

    trial_rows = []

    for trial_id in range(n_trials):
        # Pick target, spectators, regime deterministically
        target = world.targets[trial_id % len(world.targets)]
        regime = world.regimes[trial_id % len(world.regimes)]

        # Pick 1-4 spectators
        n_specs = rng.randint(1, min(5, len(world.spectators) + 1))
        spec_indices = rng.choice(len(world.spectators), size=n_specs, replace=False)
        spec_ids = sorted([world.spectators[i].workload_id for i in spec_indices])

        t_index = trial_id * 2  # Space out time indices

        result = simulate_micro_run(
            world=world,
            target_id=target.workload_id,
            spectator_ids=spec_ids,
            regime_id=regime.regime_id,
            t_index=t_index,
            n_samples=n_samples,
            rng=rng,
        )

        stats = compute_all_tail_stats(result.latencies_us, slo_us)

        trial_rows.append({
            "run_id": run_id,
            "config_hash": config_hash,
            "git_commit": git_commit,
            "seed": config["seed"],
            "trial_id": trial_id,
            "scheduler_name": "measurement_harness",
            "target_id": target.workload_id,
            "spectators": json.dumps(spec_ids),
            "regime_id": regime.regime_id,
            "n_samples": stats["n_samples"],
            "mean_latency_us": round(stats["mean_latency_us"], 6),
            "p95_latency_us": round(stats["p95_latency_us"], 6),
            "p99_latency_us": round(stats["p99_latency_us"], 6),
            "cvar99_latency_us": round(stats["cvar99_latency_us"], 6),
            "slo_us": float(slo_us),
            "violation_rate": round(stats["violation_rate"], 6),
            "notes": "",
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": timestamp,
            "units_latency": CANONICAL_LATENCY_UNIT,
        })

    return pd.DataFrame(trial_rows)


def run_sim_decisions(
    world: World,
    config: Dict[str, Any],
    run_id: str,
    config_hash: str,
    git_commit: str,
    timestamp: str,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Generate decision rows for the measurement harness.

    Each trial gets a corresponding decision entry.
    """
    n_trials = config["n_trials"]
    decision_rows = []

    for dec_id in range(n_trials):
        target = world.targets[dec_id % len(world.targets)]
        n_specs = rng.randint(1, min(5, len(world.spectators) + 1))
        spec_indices = rng.choice(len(world.spectators), size=n_specs, replace=False)
        spec_ids = sorted([world.spectators[i].workload_id for i in spec_indices])

        # Build a few candidate sets
        candidates = []
        for _ in range(rng.randint(2, 4)):
            nc = rng.randint(1, min(5, len(world.spectators) + 1))
            ci = rng.choice(len(world.spectators), size=nc, replace=False)
            candidates.append(sorted([world.spectators[j].workload_id for j in ci]))

        decision_rows.append({
            "run_id": run_id,
            "config_hash": config_hash,
            "git_commit": git_commit,
            "seed": config["seed"],
            "decision_id": dec_id,
            "scheduler_name": "measurement_harness",
            "time_index": dec_id * 2,
            "target_id": target.workload_id,
            "candidate_sets": json.dumps(candidates),
            "chosen_set": json.dumps(spec_ids),
            "score_components_json": json.dumps({
                "risk": round(rng.uniform(0, 1), 4),
                "diversity": round(rng.uniform(0, 1), 4),
                "uncertainty": round(rng.uniform(0, 1), 4),
            }),
            "predicted_metrics_json": json.dumps({"placeholder": True}),
            "realized_metrics_json": json.dumps(None),
            "safety_pass": True,
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": timestamp,
        })

    return pd.DataFrame(decision_rows)


def run_irbs_evaluation(
    world: World,
    config: Dict[str, Any],
    rng: np.random.RandomState,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run IRBS drift-canceling evaluation on selected pairs.

    For each selected (target, spectator) pair:
    1. Run naive A/B at separated time points
    2. Run IRBS C-T-C sandwich
    3. Compare to ground truth

    Returns:
        Tuple of (measurement_events_df, irbs_estimates_df).
    """
    meas_cfg = config.get("measurement", {})
    sim_cfg = config.get("sim", {})
    n_samples = sim_cfg.get("n_samples_per_micro_run", 2000)
    n_pairs = meas_cfg.get("n_irbs_pairs", 5)
    stat_alpha = meas_cfg.get("alpha", 0.99)

    regime = world.regimes[0]
    gt = world.ground_truth.get(regime.regime_id, {})

    # Select pairs to evaluate
    pairs = []
    for t in world.targets[:min(n_pairs, len(world.targets))]:
        # Pick spectator with highest ground truth interference
        spec_gt = [(sid, gt.get((t.workload_id, s.workload_id), 0.0))
                    for sid_idx, s in enumerate(world.spectators)
                    for sid in [s.workload_id]]
        spec_gt.sort(key=lambda x: -x[1])
        if spec_gt:
            pairs.append((t.workload_id, spec_gt[0][0]))

    # Fill remaining pairs randomly if needed
    while len(pairs) < n_pairs and len(world.targets) > 0 and len(world.spectators) > 0:
        ti = rng.randint(len(world.targets))
        si = rng.randint(len(world.spectators))
        pair = (world.targets[ti].workload_id, world.spectators[si].workload_id)
        if pair not in pairs:
            pairs.append(pair)

    event_rows = []
    estimate_rows = []
    event_id = 0

    for pair_idx, (tid, sid) in enumerate(pairs):
        true_delta = gt.get((tid, sid), 0.0)

        # --- Naive A/B ---
        # Control at t=0
        t_base = pair_idx * 20
        ctrl_result = simulate_micro_run(
            world, tid, [], regime.regime_id, t_base, n_samples, rng,
        )
        ctrl_mean = float(np.mean(ctrl_result.latencies_us))

        # Treatment at t=10 (significant time gap for drift)
        treat_result = simulate_micro_run(
            world, tid, [sid], regime.regime_id, t_base + 10, n_samples, rng,
        )
        treat_mean = float(np.mean(treat_result.latencies_us))

        naive_est = naive_ab_estimate(ctrl_mean, treat_mean)

        event_rows.append({
            "event_id": event_id,
            "t_index": t_base,
            "target_id": tid,
            "spectator_set": json.dumps([]),
            "is_control": True,
            "regime_id": regime.regime_id,
            "stat_type": "mean",
            "stat_value_us": ctrl_mean,
            "drift_value_us": float(world.drift_params.a_us_per_step * t_base)
                if world.drift_params.drift_type == "linear" else 0.0,
        })
        event_id += 1

        event_rows.append({
            "event_id": event_id,
            "t_index": t_base + 10,
            "target_id": tid,
            "spectator_set": json.dumps([sid]),
            "is_control": False,
            "regime_id": regime.regime_id,
            "stat_type": "mean",
            "stat_value_us": treat_mean,
            "drift_value_us": float(world.drift_params.a_us_per_step * (t_base + 10))
                if world.drift_params.drift_type == "linear" else 0.0,
        })
        event_id += 1

        # --- IRBS C-T-C ---
        # C1 at t=8, T at t=10, C2 at t=12
        c1_result = simulate_micro_run(
            world, tid, [], regime.regime_id, t_base + 8, n_samples, rng,
        )
        c1_mean = float(np.mean(c1_result.latencies_us))

        t_result = simulate_micro_run(
            world, tid, [sid], regime.regime_id, t_base + 10, n_samples, rng,
        )
        t_mean = float(np.mean(t_result.latencies_us))

        c2_result = simulate_micro_run(
            world, tid, [], regime.regime_id, t_base + 12, n_samples, rng,
        )
        c2_mean = float(np.mean(c2_result.latencies_us))

        irbs_est = irbs_ctc_estimate(c1_mean, t_mean, c2_mean)

        for (eid_t, eid_ctrl, eid_mean, eid_specs) in [
            (t_base + 8, True, c1_mean, []),
            (t_base + 10, False, t_mean, [sid]),
            (t_base + 12, True, c2_mean, []),
        ]:
            event_rows.append({
                "event_id": event_id,
                "t_index": eid_t,
                "target_id": tid,
                "spectator_set": json.dumps(eid_specs),
                "is_control": eid_ctrl,
                "regime_id": regime.regime_id,
                "stat_type": "mean",
                "stat_value_us": eid_mean,
                "drift_value_us": float(world.drift_params.a_us_per_step * eid_t)
                    if world.drift_params.drift_type == "linear" else 0.0,
            })
            event_id += 1

        # Compute bias relative to ground truth
        bias_naive = naive_est - true_delta
        bias_irbs = irbs_est - true_delta

        estimate_rows.append({
            "estimate_id": pair_idx,
            "target_id": tid,
            "spectator_id": sid,
            "regime_id": regime.regime_id,
            "drift_order": 1,
            "method_name": "CTC",
            "ground_truth_us": true_delta,
            "naive_estimate_us": naive_est,
            "irbs_estimate_us": irbs_est,
            "bias_naive_us": bias_naive,
            "bias_irbs_us": bias_irbs,
            "absolute_error_naive_us": abs(bias_naive),
            "absolute_error_irbs_us": abs(bias_irbs),
        })

    events_df = pd.DataFrame(event_rows)
    estimates_df = pd.DataFrame(estimate_rows)

    logger.info(
        "IRBS evaluation: %d pairs, %d events, %d estimates",
        len(pairs), len(event_rows), len(estimate_rows),
    )

    return events_df, estimates_df


def run_ci_coverage_evaluation(
    config: Dict[str, Any],
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Run a small CI coverage evaluation across synthetic worlds.

    Uses lognormal samples to evaluate bootstrap CI coverage.
    """
    meas_cfg = config.get("measurement", {})
    n_resamples = meas_cfg.get("ci_resamples", 200)
    block_size = meas_cfg.get("block_size", None)
    use_block = meas_cfg.get("block_bootstrap", False)

    # LogNormal distribution: known true p99
    mu_ln = 5.0  # ln(us)
    sigma_ln = 0.5

    def gen_samples(r: np.random.RandomState, n: int) -> np.ndarray:
        return r.lognormal(mean=mu_ln, sigma=sigma_ln, size=n)

    def stat_fn(x: np.ndarray) -> float:
        return float(np.quantile(x, 0.99, method="linear"))

    # True p99 of lognormal: exp(mu + sigma * Phi^{-1}(0.99))
    from scipy.stats import norm
    true_p99 = float(np.exp(mu_ln + sigma_ln * norm.ppf(0.99)))

    def true_stat_fn(r: np.random.RandomState) -> float:
        return true_p99

    coverage_df = evaluate_ci_coverage(
        generate_samples_fn=gen_samples,
        stat_fn=stat_fn,
        true_stat_fn=true_stat_fn,
        n_worlds=20,
        n_samples=2000,
        n_resamples=n_resamples,
        nominal_alpha=0.05,
        block_size=block_size if use_block else None,
        base_seed=rng.randint(0, 100000),
    )

    return coverage_df
