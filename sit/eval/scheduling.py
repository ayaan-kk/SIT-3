"""Scheduling evaluation harness.

Orchestrates episode generation, scheduler execution, realized
performance evaluation via simulation, and metric aggregation.

Produces:
- schedule_decisions.parquet (per-decision logs)
- schedule_trials.parquet (per-target per-episode realized metrics)
- schedule_episode_metrics.parquet (per-episode aggregated metrics)
- schedule_failure_audit.parquet (worst realized events)
- schedule_regime_heatmap.parquet (regime breakdown)
- scheduling_summary.csv (summary table)
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.core.units import CANONICAL_LATENCY_UNIT
from sit.measure.tail import compute_all_tail_stats
from sit.schedule.constraints import SafetyConfig, is_catastrophe
from sit.schedule.objectives import predicted_risk, predicted_uncertainty, robust_risk
from sit.schedule.policies import (
    SCHEDULER_REGISTRY,
    ScheduleContext,
    ScheduleDecision,
    run_scheduler,
)
from sit.schedule.search import local_search_improve
from sit.schedule.state import (
    Episode,
    EpisodeResult,
    Placement,
    generate_episodes,
)
from sit.sim.world import World, simulate_micro_run

logger = get_logger("eval.scheduling")


def build_interference_lookup(
    world: World,
    tomo_data: Optional[Dict] = None,
) -> Tuple[Dict, Dict]:
    """Build x_hat and sigma lookup dicts from ground truth or tomography.

    For scheduling, we use ground truth as x_hat with added noise to
    simulate reconstruction uncertainty. If tomo_data is available,
    use those estimates directly.

    Args:
        world: Simulation world.
        tomo_data: Optional dict with 'x_hat' and 'sigma' entries.

    Returns:
        Tuple of (x_hat, sigma) dicts keyed by (target_id, spectator_id, regime_id).
    """
    x_hat = {}
    sigma = {}

    for regime in world.regimes:
        gt = world.ground_truth.get(regime.regime_id, {})
        for (t_id, s_id), val in gt.items():
            key = (t_id, s_id, regime.regime_id)
            # Use ground truth as x_hat (best case for scheduling)
            x_hat[key] = val
            # Sigma proportional to x_hat magnitude (uncertainty model)
            sigma[key] = max(0.1, abs(val) * 0.15)

    # Override with tomo data if available
    if tomo_data:
        for key, val in tomo_data.get("x_hat", {}).items():
            x_hat[key] = val
        for key, val in tomo_data.get("sigma", {}).items():
            sigma[key] = val

    return x_hat, sigma


def evaluate_placement(
    world: World,
    episode: Episode,
    placement: Placement,
    scheduler_name: str,
    n_samples: int,
    rng: np.random.RandomState,
    slo_us: float,
) -> EpisodeResult:
    """Evaluate a placement by running micro-simulations.

    For each target, simulates the realized latency distribution
    with its co-tenants and computes tail metrics.

    Args:
        world: Simulation world.
        episode: The episode.
        placement: The placement to evaluate.
        scheduler_name: Name of the scheduler.
        n_samples: Samples per micro-run.
        rng: Random state.
        slo_us: SLO threshold.

    Returns:
        EpisodeResult with per-target and aggregate metrics.
    """
    jobs_by_id = {j.job_id: j for j in episode.jobs}
    target_metrics = {}

    for target_job in episode.target_jobs:
        # Get spectator workload IDs co-located with this target
        co_tenant_jids = placement.co_tenants(target_job.job_id)
        spectator_wids = []
        for jid in co_tenant_jids:
            j = jobs_by_id.get(jid)
            if j and j.role == "spectator":
                spectator_wids.append(j.workload_id)

        # Run micro-simulation
        result = simulate_micro_run(
            world,
            target_job.workload_id,
            spectator_wids,
            episode.regime_id,
            episode.t_index,
            n_samples,
            rng,
        )

        # Compute tail statistics
        stats = compute_all_tail_stats(result.latencies_us, slo_us)
        target_metrics[target_job.workload_id] = stats

    # Aggregate episode metrics
    if target_metrics:
        cvar_vals = [m["cvar99_latency_us"] for m in target_metrics.values()]
        p99_vals = [m["p99_latency_us"] for m in target_metrics.values()]
        viol_vals = [m["violation_rate"] for m in target_metrics.values()]

        episode_metrics = {
            "mean_cvar99": float(np.mean(cvar_vals)),
            "max_cvar99": float(np.max(cvar_vals)),
            "mean_p99": float(np.mean(p99_vals)),
            "max_p99": float(np.max(p99_vals)),
            "mean_violation_rate": float(np.mean(viol_vals)),
            "max_violation_rate": float(np.max(viol_vals)),
            "goodput": float(np.mean([1.0 - v for v in viol_vals])),
        }
    else:
        episode_metrics = {}

    return EpisodeResult(
        episode_id=episode.episode_id,
        scheduler_name=scheduler_name,
        placement=placement,
        target_metrics=target_metrics,
        episode_metrics=episode_metrics,
    )


def run_scheduling_evaluation(
    world: World,
    config: Dict[str, Any],
    ctx: Any,
    rng: np.random.RandomState,
    tomo_data: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full scheduling evaluation pipeline.

    Args:
        world: Simulation world.
        config: Full config dict.
        ctx: RunContext.
        rng: Random state.
        tomo_data: Optional tomography estimates.

    Returns:
        Tuple of DataFrames:
        (decisions_df, trials_df, episode_metrics_df,
         failure_audit_df, regime_heatmap_df, summary_df)
    """
    sched_cfg = config.get("scheduling", {})
    schedulers = sched_cfg.get("schedulers", list(SCHEDULER_REGISTRY.keys()))
    sit_params = sched_cfg.get("sit_params", {})
    slo_us = config.get("slo_us", 500000.0)
    n_samples = config.get("sim", {}).get("n_samples_per_micro_run", 1000)
    base_seed = config.get("seed", 42)

    safety_config = SafetyConfig.from_config(config)
    lambda_div = float(sit_params.get("lambda_div", 0.5))

    # Build interference lookup
    x_hat, sigma = build_interference_lookup(world, tomo_data)

    # Build spectator kernel for diversity-based policies
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
    ep_rng = np.random.RandomState(base_seed + 100)
    episodes = generate_episodes(world, config, ep_rng)
    logger.info("Generated %d scheduling episodes", len(episodes))

    # Run each scheduler on each episode
    all_decisions = []
    all_trials = []
    all_episode_metrics = []
    all_results_by_scheduler = {}  # scheduler -> list of EpisodeResult

    for sched_name in schedulers:
        logger.info("Running scheduler: %s", sched_name)
        sched_results = []

        for ep in episodes:
            # Create context with deterministic RNG per episode+scheduler
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

            # Run scheduler
            placement, decisions = run_scheduler(sched_name, sched_ctx)

            # For SIT-safe, apply local search improvement
            if sched_name == "sit_safe_ucb":
                placement, n_swaps = local_search_improve(
                    placement, ep, x_hat, sigma, safety_config,
                    K, spectator_id_to_idx, lambda_div,
                    max_iters=20,
                )

            # Log decisions
            for dec in decisions:
                all_decisions.append({
                    "episode_id": dec.episode_id,
                    "step_index": dec.step_index,
                    "scheduler_name": dec.scheduler_name,
                    "target_id": dec.target_id,
                    "candidate_hosts": json.dumps(dec.candidate_hosts),
                    "chosen_host": dec.chosen_host,
                    "safety_pass": dec.safety_pass,
                    "fallback_used": dec.fallback_used,
                    "fallback_type": dec.fallback_type,
                    "score_breakdown": json.dumps(dec.score_breakdown),
                })

            # Evaluate realized performance
            eval_rng = np.random.RandomState((base_seed + ep.episode_id + 999) & 0x7FFFFFFF)
            ep_result = evaluate_placement(
                world, ep, placement, sched_name,
                n_samples, eval_rng, slo_us,
            )
            sched_results.append(ep_result)

            # Log per-target trials
            jobs_by_id = {j.job_id: j for j in ep.jobs}
            for target_id, metrics in ep_result.target_metrics.items():
                # Compute predicted metrics for comparison
                co_jids = placement.co_tenants(
                    next(j.job_id for j in ep.target_jobs if j.workload_id == target_id)
                )
                spec_wids = [jobs_by_id[jid].workload_id for jid in co_jids
                             if jobs_by_id.get(jid) and jobs_by_id[jid].role == "spectator"]

                pred_risk = predicted_risk(target_id, spec_wids, ep.regime_id, x_hat)
                pred_unc = predicted_uncertainty(target_id, spec_wids, ep.regime_id, sigma)
                r_ucb = pred_risk + safety_config.beta_ucb * pred_unc

                all_trials.append({
                    "episode_id": ep.episode_id,
                    "scheduler_name": sched_name,
                    "target_id": target_id,
                    "regime_id": ep.regime_id,
                    "n_co_tenants": len(spec_wids),
                    "predicted_risk_us": pred_risk,
                    "predicted_uncertainty_us": pred_unc,
                    "robust_risk_us": r_ucb,
                    **metrics,
                    "is_catastrophe": is_catastrophe(metrics, safety_config),
                    "unit": CANONICAL_LATENCY_UNIT,
                })

            # Log episode-level metrics
            all_episode_metrics.append({
                "episode_id": ep.episode_id,
                "scheduler_name": sched_name,
                "regime_id": ep.regime_id,
                **ep_result.episode_metrics,
                "n_targets": len(ep.target_jobs),
                "n_spectators": len(ep.spectator_jobs),
                "fallback_used": any(d.fallback_used for d in decisions),
            })

        all_results_by_scheduler[sched_name] = sched_results

    # Build DataFrames
    decisions_df = pd.DataFrame(all_decisions)
    trials_df = pd.DataFrame(all_trials)
    episode_metrics_df = pd.DataFrame(all_episode_metrics)

    # Build failure audit: top worst events per scheduler
    failure_rows = []
    for sched_name, results in all_results_by_scheduler.items():
        worst_events = []
        for er in results:
            for target_id, metrics in er.target_metrics.items():
                worst_events.append({
                    "episode_id": er.episode_id,
                    "scheduler_name": sched_name,
                    "target_id": target_id,
                    "cvar99_latency_us": metrics.get("cvar99_latency_us", 0),
                    "p99_latency_us": metrics.get("p99_latency_us", 0),
                    "violation_rate": metrics.get("violation_rate", 0),
                    "is_catastrophe": is_catastrophe(metrics, safety_config),
                })

        # Sort by cvar99 descending, take top 20
        worst_events.sort(key=lambda x: x["cvar99_latency_us"], reverse=True)
        for evt in worst_events[:20]:
            # Add predicted vs realized comparison
            failure_rows.append(evt)

    failure_audit_df = pd.DataFrame(failure_rows) if failure_rows else pd.DataFrame()

    # Build regime heatmap
    heatmap_rows = _compute_regime_heatmap(
        episode_metrics_df, trials_df, schedulers,
    )
    regime_heatmap_df = pd.DataFrame(heatmap_rows)

    # Build summary
    summary_rows = _compute_scheduling_summary(
        episode_metrics_df, trials_df, safety_config, schedulers,
    )
    summary_df = pd.DataFrame(summary_rows)

    logger.info(
        "Scheduling evaluation complete: %d episodes x %d schedulers = %d trials",
        len(episodes), len(schedulers), len(trials_df),
    )

    return (decisions_df, trials_df, episode_metrics_df,
            failure_audit_df, regime_heatmap_df, summary_df)


def _compute_regime_heatmap(
    episode_metrics_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    schedulers: List[str],
) -> List[Dict]:
    """Compute regime breakdown for heatmap."""
    rows = []
    if trials_df.empty or episode_metrics_df.empty:
        return rows

    regimes = trials_df["regime_id"].unique()

    # Find baseline (random) metrics per regime
    for regime_id in regimes:
        regime_trials = trials_df[trials_df["regime_id"] == regime_id]
        regime_eps = episode_metrics_df[episode_metrics_df["regime_id"] == regime_id]

        baseline_name = "random"
        baseline_eps = regime_eps[regime_eps["scheduler_name"] == baseline_name]
        if baseline_eps.empty:
            continue

        baseline_cvar = float(baseline_eps["mean_cvar99"].mean())
        baseline_p99 = float(baseline_eps["mean_p99"].mean())
        baseline_goodput = float(baseline_eps["goodput"].mean())

        for sched in schedulers:
            sched_eps = regime_eps[regime_eps["scheduler_name"] == sched]
            sched_trials = regime_trials[regime_trials["scheduler_name"] == sched]

            if sched_eps.empty:
                continue

            sched_cvar = float(sched_eps["mean_cvar99"].mean())
            sched_p99 = float(sched_eps["mean_p99"].mean())
            sched_goodput = float(sched_eps["goodput"].mean())

            n_catastrophes = int(sched_trials["is_catastrophe"].sum()) if "is_catastrophe" in sched_trials.columns else 0

            rows.append({
                "regime_id": regime_id,
                "scheduler_name": sched,
                "mean_cvar99": sched_cvar,
                "mean_p99": sched_p99,
                "goodput": sched_goodput,
                "delta_cvar99": sched_cvar - baseline_cvar,
                "delta_p99": sched_p99 - baseline_p99,
                "delta_goodput": sched_goodput - baseline_goodput,
                "catastrophe_count": n_catastrophes,
                "n_episodes": len(sched_eps),
            })

    return rows


def _compute_scheduling_summary(
    episode_metrics_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    safety_config: SafetyConfig,
    schedulers: List[str],
) -> List[Dict]:
    """Compute summary statistics per scheduler."""
    rows = []

    for sched in schedulers:
        sched_eps = episode_metrics_df[episode_metrics_df["scheduler_name"] == sched]
        sched_trials = trials_df[trials_df["scheduler_name"] == sched]

        if sched_eps.empty:
            continue

        n_catastrophes = 0
        if not sched_trials.empty and "is_catastrophe" in sched_trials.columns:
            n_catastrophes = int(sched_trials["is_catastrophe"].sum())

        rows.append({
            "scheduler_name": sched,
            "mean_cvar99_us": float(sched_eps["mean_cvar99"].mean()),
            "max_cvar99_us": float(sched_eps["max_cvar99"].max()),
            "mean_p99_us": float(sched_eps["mean_p99"].mean()),
            "mean_goodput": float(sched_eps["goodput"].mean()),
            "n_catastrophes": n_catastrophes,
            "n_episodes": len(sched_eps),
            "n_fallbacks": int(sched_eps["fallback_used"].sum()) if "fallback_used" in sched_eps.columns else 0,
            "unit": CANONICAL_LATENCY_UNIT,
        })

    return rows
