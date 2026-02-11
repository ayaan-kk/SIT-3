"""Load sweep experiment orchestration.

Sweeps over offered load levels, running each scheduler at each level,
simulating serving with the open-loop or closed-loop model, and
aggregating metrics for Pareto and admission analysis.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.core.units import CANONICAL_LATENCY_UNIT
from sit.eval.scheduling import build_interference_lookup
from sit.load.admission import compute_admission_curve
from sit.load.diagnostics import compute_queue_diagnostics
from sit.load.model import (
    ServingConfig,
    simulate_closed_loop_host,
    simulate_open_loop_host,
)
from sit.load.pareto import construct_pareto_data
from sit.schedule.constraints import SafetyConfig
from sit.schedule.policies import ScheduleContext, run_scheduler
from sit.schedule.search import local_search_improve
from sit.schedule.state import Episode, Job, Placement, generate_episodes
from sit.sim.world import World, simulate_micro_run

logger = get_logger("load.sweep")


def _get_service_times_for_target(
    world: World,
    target_id: str,
    spectator_wids: List[str],
    regime_id: str,
    t_index: int,
    n_samples: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Get service time distribution for a target with given co-tenants.

    Calls simulate_micro_run and extracts service_us (pre-queue).
    """
    result = simulate_micro_run(
        world, target_id, spectator_wids, regime_id, t_index,
        n_samples, rng,
    )
    return result.service_us


def _get_co_tenant_spectator_wids(
    placement: Placement,
    target_job: Job,
    episode: Episode,
) -> List[str]:
    """Get spectator workload IDs co-located with a target."""
    jobs_by_id = {j.job_id: j for j in episode.jobs}
    co_jids = placement.co_tenants(target_job.job_id)
    return [
        jobs_by_id[jid].workload_id
        for jid in co_jids
        if jobs_by_id.get(jid) and jobs_by_id[jid].role == "spectator"
    ]


def run_load_sweep(
    world: World,
    config: Dict[str, Any],
    ctx: Any,
    rng: np.random.RandomState,
    tomo_data: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full load sweep experiment.

    For each load level in the grid:
    - Reuses episodes generated for scheduling
    - For each scheduler, gets placement
    - Pre-computes service time distributions per target
    - Simulates open/closed-loop serving at the given load
    - Collects per-target metrics

    Args:
        world: Simulation world.
        config: Full config dict with 'load' and 'scheduling' sections.
        ctx: RunContext.
        rng: Random state.
        tomo_data: Optional tomography estimates.

    Returns:
        Tuple of DataFrames:
        (sweep_df, pareto_df, admission_df, diagnostics_df, summary_df)
    """
    load_cfg = config.get("load", {})
    sched_cfg = config.get("scheduling", {})
    slo_us = float(config.get("slo_us", 500_000.0))
    base_seed = config.get("seed", 42)

    # Load sweep parameters
    load_grid = load_cfg.get("grid", [100, 200, 300, 400, 500, 550, 600])
    serving_config = ServingConfig.from_config(config)
    v_target = float(load_cfg.get("v_target", 0.02))
    n_episodes = int(load_cfg.get("n_episodes", sched_cfg.get("episodes", 10)))
    n_service_samples = serving_config.n_service_samples

    # Override episode count for load sweep (use smaller)
    load_sched_cfg = dict(sched_cfg)
    load_sched_cfg["episodes"] = n_episodes
    load_config = dict(config)
    load_config["scheduling"] = load_sched_cfg

    # Schedulers for load sweep
    schedulers = load_cfg.get("schedulers", sched_cfg.get("schedulers", [
        "random", "static_partition", "sit_safe_ucb",
    ]))

    # Build interference lookup
    x_hat, sigma = build_interference_lookup(world, tomo_data)
    safety_config = SafetyConfig.from_config(config)
    lambda_div = float(sched_cfg.get("sit_params", {}).get("lambda_div", 0.5))

    # Build kernel
    from sit.probe.features import build_feature_matrix
    from sit.probe.diversity import build_kernel

    spectator_ids = sorted([s.workload_id for s in world.spectators])
    spectator_id_to_idx = {sid: i for i, sid in enumerate(spectator_ids)}

    kernel_cfg = sched_cfg.get("kernel", config.get("probing", {}).get("kernel", {}))
    k_type = kernel_cfg.get("type", "rbf")
    k_sigma = float(kernel_cfg.get("sigma", 0.3))

    F = build_feature_matrix(world.spectators, mode="workload")
    K = build_kernel(F, kernel_type=k_type, sigma=k_sigma)

    # Generate episodes
    ep_rng = np.random.RandomState(base_seed + 200)
    episodes = generate_episodes(world, load_config, ep_rng)
    logger.info("Load sweep: %d episodes, %d load levels, %d schedulers",
                len(episodes), len(load_grid), len(schedulers))

    # For each scheduler + episode, pre-compute placements and service times
    # This avoids redundant computation across load levels
    all_sweep_rows = []

    for sched_name in schedulers:
        logger.info("Load sweep: scheduler=%s", sched_name)

        for ep in episodes:
            # Run scheduler to get placement
            ep_seed = base_seed + ep.episode_id * 100 + hash(sched_name) % 10000
            ep_rng_sched = np.random.RandomState(ep_seed & 0x7FFFFFFF)

            sched_ctx = ScheduleContext(
                episode=ep,
                x_hat=x_hat,
                sigma=sigma,
                safety_config=safety_config,
                K=K,
                spectator_id_to_idx=spectator_id_to_idx,
                lambda_div=lambda_div,
                slo_us=slo_us,
                rng=ep_rng_sched,
            )

            placement, decisions = run_scheduler(sched_name, sched_ctx)

            # Apply local search for SIT-safe
            if sched_name == "sit_safe_ucb":
                placement, _ = local_search_improve(
                    placement, ep, x_hat, sigma, safety_config,
                    K, spectator_id_to_idx, lambda_div,
                    max_iters=20,
                )

            # Pre-compute service time distributions per target
            target_service_dists = {}
            target_host_map = {}  # target_wid -> host_id

            for target_job in ep.target_jobs:
                host_id = placement.mapping.get(target_job.job_id)
                if host_id is None:
                    continue

                target_host_map[target_job.workload_id] = host_id
                spec_wids = _get_co_tenant_spectator_wids(placement, target_job, ep)

                svc_rng = np.random.RandomState(
                    (base_seed + ep.episode_id * 10 + hash(target_job.workload_id)) & 0x7FFFFFFF
                )
                svc_times = _get_service_times_for_target(
                    world, target_job.workload_id, spec_wids,
                    ep.regime_id, ep.t_index, n_service_samples, svc_rng,
                )
                target_service_dists[target_job.workload_id] = svc_times

            # Group targets by host
            host_targets = {}  # host_id -> {target_wid: service_dist}
            for twid, host_id in target_host_map.items():
                if host_id not in host_targets:
                    host_targets[host_id] = {}
                host_targets[host_id][twid] = target_service_dists[twid]

            # For each load level, simulate serving
            for lambda_rps in load_grid:
                load_rng = np.random.RandomState(
                    (base_seed + ep.episode_id * 1000 + int(lambda_rps) + hash(sched_name)) & 0x7FFFFFFF
                )

                for host_id, host_svc_dists in host_targets.items():
                    if serving_config.mode == "closed_loop":
                        # Convert lambda to concurrency
                        mean_svc = np.mean([
                            np.mean(d) for d in host_svc_dists.values()
                        ])
                        concurrency = max(1, int(lambda_rps * mean_svc / 1e6 * 2))
                        host_metrics = simulate_closed_loop_host(
                            host_svc_dists, concurrency,
                            think_time_us=50.0,
                            window_us=serving_config.window_us,
                            warmup_fraction=serving_config.warmup_fraction,
                            slo_us=slo_us,
                            rng=load_rng,
                        )
                    else:
                        host_metrics = simulate_open_loop_host(
                            host_svc_dists,
                            lambda_rps_per_target=lambda_rps,
                            window_us=serving_config.window_us,
                            warmup_fraction=serving_config.warmup_fraction,
                            slo_us=slo_us,
                            rng=load_rng,
                        )

                    for tid, metrics in host_metrics.items():
                        all_sweep_rows.append({
                            "load_level": lambda_rps,
                            "episode_id": ep.episode_id,
                            "scheduler_name": sched_name,
                            "target_id": tid,
                            "host_id": host_id,
                            "regime_id": ep.regime_id,
                            "n_targets_on_host": len(host_svc_dists),
                            "unit": CANONICAL_LATENCY_UNIT,
                            **metrics,
                        })

    sweep_df = pd.DataFrame(all_sweep_rows)
    if sweep_df.empty:
        logger.warning("Load sweep produced no results")
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty

    logger.info("Load sweep: %d result rows", len(sweep_df))

    # Construct Pareto data
    pareto_df = construct_pareto_data(sweep_df)

    # Compute admission curves
    admission_df = compute_admission_curve(sweep_df, slo_us, v_target)

    # Compute queue diagnostics
    diagnostics_df = compute_queue_diagnostics(sweep_df, slo_us)

    # Build summary
    summary_df = _compute_load_sweep_summary(sweep_df, pareto_df, admission_df)

    return sweep_df, pareto_df, admission_df, diagnostics_df, summary_df


def _compute_load_sweep_summary(
    sweep_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    admission_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute summary table for load sweep."""
    rows = []
    schedulers = sweep_df["scheduler_name"].unique()

    for sched in schedulers:
        sched_df = sweep_df[sweep_df["scheduler_name"] == sched]

        mean_goodput = float(sched_df["goodput_rps"].mean())
        mean_cvar = float(sched_df["cvar99_latency_us"].mean())
        mean_viol = float(sched_df["violation_rate"].mean())
        mean_throughput = float(sched_df["throughput_rps"].mean())

        # Max feasible load from admission
        max_load = 0.0
        if not admission_df.empty:
            adm_row = admission_df[
                (admission_df["scheduler_name"] == sched) &
                (admission_df["max_feasible_load_rps"] > 0)
            ]
            if not adm_row.empty:
                max_load = float(adm_row.iloc[0]["max_feasible_load_rps"])

        # On Pareto frontier?
        on_frontier = False
        if not pareto_df.empty:
            pf = pareto_df[
                (pareto_df["scheduler_name"] == sched) &
                (pareto_df["on_pareto_frontier"] == True)
            ]
            on_frontier = len(pf) > 0

        rows.append({
            "scheduler_name": sched,
            "mean_goodput_rps": mean_goodput,
            "mean_cvar99_us": mean_cvar,
            "mean_violation_rate": mean_viol,
            "mean_throughput_rps": mean_throughput,
            "max_feasible_load_rps": max_load,
            "on_pareto_frontier": on_frontier,
            "unit": CANONICAL_LATENCY_UNIT,
        })

    return pd.DataFrame(rows)
