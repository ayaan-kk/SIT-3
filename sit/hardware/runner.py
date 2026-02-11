"""Hardware validation runner: orchestrates the full hardware experiment.

Executes real workloads, measures drift, runs IRBS on hardware,
ranks spectator interference, and exports all artifacts.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.core.registry import RunContext, register_artifact
from sit.core.units import CANONICAL_LATENCY_UNIT
from sit.data.io import write_dataframe
from sit.hardware.calibration import measure_timer_resolution_ns, warmup_stabilize
from sit.hardware.isolation import log_isolation_context, try_set_affinity
from sit.hardware.timing import measure_baseline_drift, measure_batch
from sit.hardware.workloads import (
    SpectatorWorkload,
    create_spectator,
    rpc_loop_target,
)
from sit.measure.irbs import irbs_ctc_estimate, naive_ab_estimate

logger = get_logger("hardware.runner")


def run_hardware_pipeline(
    ctx: RunContext,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the complete hardware validation pipeline.

    Steps:
    1. Log isolation context
    2. Calibrate timer
    3. Measure baseline drift (Gate H1)
    4. Run IRBS experiment per spectator (Gate H2)
    5. Rank spectators by interference (for Gate H3)
    6. Export all artifacts

    Args:
        ctx: Current RunContext.
        config: Full config dict with 'hardware' section.

    Returns:
        Dict with gate results and artifact paths.
    """
    hw_cfg = config.get("hardware", {})
    fmt = config.get("export_format", "parquet")
    slo_us = config.get("slo_us", 500000.0)
    seed = config.get("seed", 123)

    raw_base = ctx.raw_path
    derived_base = ctx.derived_path
    tables_base = ctx.tables_path
    os.makedirs(raw_base, exist_ok=True)
    os.makedirs(derived_base, exist_ok=True)
    os.makedirs(tables_base, exist_ok=True)

    results = {"artifacts": [], "gates": {}}

    # --- Step 0: Isolation context ---
    isolation = log_isolation_context()

    # --- Step 1: Timer calibration ---
    timer_res_ns = measure_timer_resolution_ns()
    logger.info("Timer resolution: %.1f ns", timer_res_ns)

    # --- Step 2: Parse config ---
    target_cfg = hw_cfg.get("target", {})
    target_type = target_cfg.get("type", "rpc_loop")
    duration_s = target_cfg.get("duration_s", 5)
    warmup_s = target_cfg.get("warmup_s", 1)
    work_size = target_cfg.get("work_size", 64)
    n_requests = target_cfg.get("n_requests", 1000)
    warmup_requests = target_cfg.get("warmup_requests", 200)

    spectators_cfg = hw_cfg.get("spectators", [
        {"type": "cpu_burn", "intensity": 0.8},
        {"type": "mem_burn", "intensity": 0.8},
        {"type": "cache_burn", "intensity": 0.8},
    ])
    repeats = hw_cfg.get("repeats_per_condition", 3)

    irbs_cfg = hw_cfg.get("irbs", {})
    irbs_enabled = irbs_cfg.get("enabled", True)
    stat_type = irbs_cfg.get("stat", "p99")

    # --- Step 3: Warmup and stabilize ---
    n_warmup, warmup_mean = warmup_stabilize(
        target_fn=rpc_loop_target,
        work_size=work_size,
        max_warmup=max(warmup_requests, 200),
    )
    logger.info("Warmup: %d requests, mean=%.2f us", n_warmup, warmup_mean)

    # --- Step 4: Baseline drift measurement ---
    drift_trials_df, drift_summary = measure_baseline_drift(
        target_fn=rpc_loop_target,
        n_batches=max(3, repeats),
        n_requests_per_batch=n_requests,
        inter_batch_sleep_s=0.5,
        work_size=work_size,
        warmup_requests=warmup_requests,
        slo_us=slo_us,
        run_id=ctx.run_id,
    )

    # --- Step 5: IRBS experiments per spectator ---
    all_trials = [drift_trials_df]
    measurement_events = []
    irbs_estimates = []
    event_id = 0

    # Create spectator instances
    spectator_instances = []
    for spec_cfg in spectators_cfg:
        spec = create_spectator(
            spec_type=spec_cfg.get("type", "cpu_burn"),
            intensity=spec_cfg.get("intensity", 0.8),
        )
        spectator_instances.append(spec)

    for spec in spectator_instances:
        for rep in range(repeats):
            # C-T-C sandwich
            # Control 1: target alone
            c1 = measure_batch(
                target_fn=rpc_loop_target,
                batch_index=event_id,
                n_requests=n_requests,
                work_size=work_size,
                warmup_requests=warmup_requests // 2,
                slo_us=slo_us,
                target_id="rpc_loop",
                spectator_set=[],
                run_id=ctx.run_id,
                notes=f"control_1_for_{spec.workload_id}_rep{rep}",
            )
            measurement_events.append({
                "event_id": event_id,
                "type": "control",
                "target_id": "rpc_loop",
                "spectator_id": "",
                "stat_type": stat_type,
                "stat_value_us": _extract_stat(c1, stat_type),
                "batch_index": rep,
                "timestamp": c1["timestamp_start"],
                "repeat": rep,
            })
            event_id += 1

            # Treatment: target + spectator
            spec.start()
            time.sleep(0.2)  # let spectator ramp up

            t = measure_batch(
                target_fn=rpc_loop_target,
                batch_index=event_id,
                n_requests=n_requests,
                work_size=work_size,
                warmup_requests=warmup_requests // 2,
                slo_us=slo_us,
                target_id="rpc_loop",
                spectator_set=[spec.workload_id],
                run_id=ctx.run_id,
                notes=f"treatment_{spec.workload_id}_rep{rep}",
            )
            measurement_events.append({
                "event_id": event_id,
                "type": "treatment",
                "target_id": "rpc_loop",
                "spectator_id": spec.workload_id,
                "stat_type": stat_type,
                "stat_value_us": _extract_stat(t, stat_type),
                "batch_index": rep,
                "timestamp": t["timestamp_start"],
                "repeat": rep,
            })
            event_id += 1

            spec.stop()
            time.sleep(0.2)  # let system settle

            # Control 2: target alone
            c2 = measure_batch(
                target_fn=rpc_loop_target,
                batch_index=event_id,
                n_requests=n_requests,
                work_size=work_size,
                warmup_requests=warmup_requests // 2,
                slo_us=slo_us,
                target_id="rpc_loop",
                spectator_set=[],
                run_id=ctx.run_id,
                notes=f"control_2_for_{spec.workload_id}_rep{rep}",
            )
            measurement_events.append({
                "event_id": event_id,
                "type": "control",
                "target_id": "rpc_loop",
                "spectator_id": "",
                "stat_type": stat_type,
                "stat_value_us": _extract_stat(c2, stat_type),
                "batch_index": rep,
                "timestamp": c2["timestamp_start"],
                "repeat": rep,
            })
            event_id += 1

            # Compute estimates
            c1_val = _extract_stat(c1, stat_type)
            t_val = _extract_stat(t, stat_type)
            c2_val = _extract_stat(c2, stat_type)

            naive_delta = naive_ab_estimate(c1_val, t_val)
            irbs_delta = irbs_ctc_estimate(c1_val, t_val, c2_val)
            drift_est = (c2_val - c1_val) / 2.0

            irbs_estimates.append({
                "spectator_id": spec.workload_id,
                "repeat": rep,
                "naive_delta_us": naive_delta,
                "irbs_delta_us": irbs_delta,
                "absolute_difference": abs(naive_delta - irbs_delta),
                "drift_estimate_us": drift_est,
                "stat_type": stat_type,
                "c1_value_us": c1_val,
                "t_value_us": t_val,
                "c2_value_us": c2_val,
            })

            # Trial row for treatment
            trial_row = {k: v for k, v in t.items() if k != "latencies_us"}
            trial_row["spectator_set"] = json.dumps([spec.workload_id])
            all_trials.append(pd.DataFrame([trial_row]))

    # --- Step 6: Build DataFrames ---
    hw_trials_df = pd.concat(all_trials, ignore_index=True)
    events_df = pd.DataFrame(measurement_events)
    irbs_df = pd.DataFrame(irbs_estimates)

    # --- Step 7: Build interference ranking ---
    ranking_df = _build_ranking(irbs_df)

    # --- Step 8: Export artifacts ---
    # hw_trials
    trials_path = os.path.join(raw_base, f"hw_trials.{fmt}")
    write_dataframe(hw_trials_df, trials_path, fmt)
    register_artifact(ctx, trials_path, "raw")
    results["artifacts"].append(trials_path)

    # hw_measurement_events
    events_path = os.path.join(raw_base, f"hw_measurement_events.{fmt}")
    write_dataframe(events_df, events_path, fmt)
    register_artifact(ctx, events_path, "raw")
    results["artifacts"].append(events_path)

    # hw_irbs_estimates
    irbs_path = os.path.join(derived_base, f"hw_irbs_estimates.{fmt}")
    write_dataframe(irbs_df, irbs_path, fmt)
    register_artifact(ctx, irbs_path, "derived")
    results["artifacts"].append(irbs_path)

    # hw_interference_ranking
    ranking_path = os.path.join(derived_base, f"hw_interference_ranking.{fmt}")
    write_dataframe(ranking_df, ranking_path, fmt)
    register_artifact(ctx, ranking_path, "derived")
    results["artifacts"].append(ranking_path)

    # --- Step 9: Compute transfer correlations ---
    from sit.eval.hardware import compute_transfer_correlations
    sim_ranking = _get_sim_ranking(config, seed)
    corr_df = compute_transfer_correlations(ranking_df, sim_ranking)

    corr_path = os.path.join(derived_base, f"hw_counter_correlations.{fmt}")
    write_dataframe(corr_df, corr_path, fmt)
    register_artifact(ctx, corr_path, "derived")
    results["artifacts"].append(corr_path)

    # --- Step 10: Run gates ---
    from sit.eval.hardware import (
        gate_h1_drift_reality,
        gate_h2_irbs_effectiveness,
        gate_h3_ranking_transfer,
    )

    results["gates"]["H1"] = gate_h1_drift_reality(drift_summary)
    results["gates"]["H2"] = gate_h2_irbs_effectiveness(irbs_df)
    results["gates"]["H3"] = gate_h3_ranking_transfer(corr_df)

    # --- Step 11: Summary report ---
    summary_df = _build_summary(
        drift_summary, irbs_df, ranking_df, corr_df,
        results["gates"], isolation,
    )
    summary_path = os.path.join(tables_base, "hardware_validation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    register_artifact(ctx, summary_path, "report")
    results["artifacts"].append(summary_path)

    # Global copy
    global_tables = os.path.join("results", "tables")
    os.makedirs(global_tables, exist_ok=True)
    global_path = os.path.join(global_tables, "hardware_validation_summary.csv")
    summary_df.to_csv(global_path, index=False)

    return results


def _extract_stat(batch_result: Dict[str, Any], stat_type: str) -> float:
    """Extract the requested stat from a batch measurement result."""
    stat_map = {
        "p99": "p99_latency_us",
        "p95": "p95_latency_us",
        "mean": "mean_latency_us",
        "cvar99": "cvar99_latency_us",
    }
    key = stat_map.get(stat_type, "p99_latency_us")
    return float(batch_result[key])


def _build_ranking(irbs_df: pd.DataFrame) -> pd.DataFrame:
    """Build interference ranking from IRBS estimates.

    Ranks spectators by median IRBS delta (descending = most interfering first).
    """
    if len(irbs_df) == 0:
        return pd.DataFrame(columns=[
            "spectator_id", "rank_irbs", "rank_naive",
            "irbs_delta_us", "naive_delta_us",
        ])

    agg = irbs_df.groupby("spectator_id").agg(
        irbs_delta_us=("irbs_delta_us", "median"),
        naive_delta_us=("naive_delta_us", "median"),
    ).reset_index()

    agg["rank_irbs"] = agg["irbs_delta_us"].rank(ascending=False, method="min").astype(int)
    agg["rank_naive"] = agg["naive_delta_us"].rank(ascending=False, method="min").astype(int)

    return agg.sort_values("rank_irbs")


def _get_sim_ranking(config: Dict[str, Any], seed: int) -> Optional[pd.DataFrame]:
    """Get simulator-predicted interference ranking for comparison.

    Runs a small simulator instance with matching spectator types
    and returns predicted rankings.
    """
    if "sim" not in config:
        logger.info("No sim config found, skipping simulator ranking")
        return None

    try:
        from sit.sim.world import build_world

        rng = np.random.RandomState(seed)
        world = build_world(config, rng)

        # Get ground truth interference for first target, first regime
        if not world.targets or not world.regimes:
            return None

        target = world.targets[0]
        regime = world.regimes[0]
        gt = world.ground_truth.get(regime.regime_id, {})

        rows = []
        for spec in world.spectators:
            interference = gt.get((target.workload_id, spec.workload_id), 0.0)
            rows.append({
                "spectator_id": spec.workload_id,
                "sim_interference_us": interference,
            })

        if not rows:
            return None

        sim_df = pd.DataFrame(rows)
        sim_df["sim_rank"] = sim_df["sim_interference_us"].rank(
            ascending=False, method="min",
        ).astype(int)

        return sim_df.sort_values("sim_rank")
    except Exception as e:
        logger.warning("Failed to build sim ranking: %s", e)
        return None


def _build_summary(
    drift_summary: Dict[str, Any],
    irbs_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    gate_results: Dict[str, Any],
    isolation: Dict[str, Any],
) -> pd.DataFrame:
    """Build hardware validation summary table."""
    sections = []

    # Gates
    for gate_name, gate_result in gate_results.items():
        if isinstance(gate_result, dict):
            sections.append({
                "section": "GATES",
                "item": gate_name,
                "metric": "status",
                "value": str(gate_result.get("status", "N/A")),
                "detail": json.dumps({k: v for k, v in gate_result.items()
                                      if k != "status"}),
            })

    # Drift
    sections.append({
        "section": "DRIFT",
        "item": "baseline",
        "metric": "drift_magnitude_us",
        "value": f"{drift_summary.get('drift_magnitude_mean_us', 0):.4f}",
        "detail": f"drift_pct={drift_summary.get('drift_pct_of_mean', 0):.2f}%",
    })

    # IRBS
    if len(irbs_df) > 0:
        median_naive = float(irbs_df["naive_delta_us"].abs().median())
        median_irbs = float(irbs_df["irbs_delta_us"].abs().median())
        sections.append({
            "section": "IRBS",
            "item": "summary",
            "metric": "median_naive_abs_delta_us",
            "value": f"{median_naive:.4f}",
            "detail": "",
        })
        sections.append({
            "section": "IRBS",
            "item": "summary",
            "metric": "median_irbs_abs_delta_us",
            "value": f"{median_irbs:.4f}",
            "detail": "",
        })

    # Ranking
    if len(ranking_df) > 0:
        for _, row in ranking_df.iterrows():
            sections.append({
                "section": "RANKING",
                "item": str(row["spectator_id"]),
                "metric": "irbs_delta_us",
                "value": f"{row['irbs_delta_us']:.4f}",
                "detail": f"rank={row['rank_irbs']}",
            })

    # Transfer correlations
    if len(corr_df) > 0:
        for _, row in corr_df.iterrows():
            sections.append({
                "section": "TRANSFER",
                "item": str(row.get("metric", "")),
                "metric": "value",
                "value": f"{row.get('value', 0):.4f}",
                "detail": f"k={row.get('k', 'N/A')}",
            })

    # System info
    sys_info = isolation.get("system", {})
    sections.append({
        "section": "SYSTEM",
        "item": "platform",
        "metric": "info",
        "value": str(sys_info.get("platform", "unknown")),
        "detail": f"cores={sys_info.get('cpu_count_logical', '?')}",
    })

    return pd.DataFrame(sections)
