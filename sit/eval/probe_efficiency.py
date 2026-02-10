"""Probe efficiency evaluation pipeline.

Runs the active probing loop for all policies, collects step-by-step
metrics, computes efficiency summaries, and exports all required
artifacts (probe_plan, step_metrics, efficiency_summary, selection_overlap).
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.core.registry import RunContext, register_artifact
from sit.data.io import write_dataframe
from sit.probe.budget import ProbeBudget
from sit.probe.diversity import (
    build_kernel,
    coverage_entropy,
    logdet_diversity,
    mean_pairwise_similarity,
)
from sit.probe.features import build_feature_matrix
from sit.probe.policies import (
    POLICY_REGISTRY,
    ProbeChoice,
    ProbeState,
    select_next_probe,
)
from sit.tomography.design import (
    build_design_matrix,
    collect_ground_truth_vector,
    collect_measurements,
)
from sit.tomography.diagnostics import compute_diagnostics
from sit.tomography.metrics import compute_all_recovery_metrics
from sit.tomography.solvers import solve_nonneg_elastic_net
from sit.tomography.uncertainty import bootstrap_x_hat_ci

logger = get_logger("eval.probe_efficiency")


def _compute_sigma_from_bootstrap(
    A: np.ndarray,
    y: np.ndarray,
    lambda_1: float,
    lambda_2: float,
    n_resamples: int = 50,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute x_hat and per-component uncertainty sigma via bootstrap.

    Uses a fast bootstrap with fewer resamples for online use.

    Returns:
        Tuple of (x_hat, sigma) where sigma = std of bootstrap samples.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    m, n = A.shape
    x_hat = solve_nonneg_elastic_net(A, y, lambda_1, lambda_2)

    boot_samples = np.zeros((n_resamples, n), dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.randint(0, m, size=m)
        boot_samples[b] = solve_nonneg_elastic_net(
            A[idx], y[idx], lambda_1, lambda_2,
            max_iter=2000, tol=1e-6, warm_start=x_hat,
        )

    sigma = np.std(boot_samples, axis=0)
    return x_hat, sigma


def run_probe_efficiency(
    world: Any,
    config: Dict[str, Any],
    ctx: RunContext,
    rng: np.random.RandomState,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full probe efficiency evaluation pipeline.

    For each policy in the config, runs the adaptive probing loop
    and collects metrics at checkpoints.

    Args:
        world: Simulation World object.
        config: Full configuration dict.
        ctx: Run context for artifact registration.
        rng: Random state.

    Returns:
        Tuple of (probe_plan_df, step_metrics_df, efficiency_summary_df, overlap_df).
    """
    probe_cfg = config.get("probing", {})
    tomo_cfg = config.get("tomography", {})
    sim_cfg = config.get("sim", {})

    m_total = probe_cfg.get("m_total", 200)
    checkpoints_every = probe_cfg.get("checkpoints_every", 10)
    policies = probe_cfg.get("policies", ["random", "sit_active"])
    set_size = tomo_cfg.get("set_size", probe_cfg.get("set_size", 3))

    kernel_cfg = probe_cfg.get("kernel", {})
    kernel_type = kernel_cfg.get("type", "rbf")
    kernel_sigma = kernel_cfg.get("sigma", 1.0)
    kernel_epsilon = kernel_cfg.get("epsilon", 1e-6)

    hybrid_lambda = probe_cfg.get("hybrid_lambda", 0.5)
    max_repeats = probe_cfg.get("max_repeats_per_spectator", 20)

    solver_cfg = tomo_cfg.get("solver", {})
    lambda_1 = solver_cfg.get("lambda1", solver_cfg.get("lambda_1", 0.05))
    lambda_2 = solver_cfg.get("lambda2", solver_cfg.get("lambda_2", 0.01))

    stat = tomo_cfg.get("stat", config.get("measurement", {}).get("stat", "mean"))
    use_irbs = tomo_cfg.get("use_irbs", config.get("measurement", {}).get("use_irbs", True))
    n_samples = sim_cfg.get("n_samples_per_micro_run", tomo_cfg.get("n_samples_tomo", 5000))

    # Checkpoint list
    checkpoints = set()
    for cp in range(checkpoints_every, m_total + 1, checkpoints_every):
        checkpoints.add(cp)
    checkpoints.add(m_total)

    spectator_ids = [s.workload_id for s in world.spectators]
    n_spectators = len(spectator_ids)

    # Build feature matrix and kernel
    F = build_feature_matrix(world.spectators, mode="workload")
    K = build_kernel(F, kernel_type=kernel_type, sigma=kernel_sigma)

    # Collect results
    all_plan_rows = []
    all_step_rows = []
    all_policy_results = {}  # policy -> {target_id -> {regime_id -> list of chosen sets}}

    # For each target and regime
    targets = [t.workload_id for t in world.targets]
    regimes = [r.regime_id for r in world.regimes]

    for target_id in targets:
        for regime_id in regimes:
            logger.info(
                "Probing target=%s, regime=%s", target_id, regime_id,
            )

            # Get ground truth
            x_true = collect_ground_truth_vector(world, target_id, regime_id, spectator_ids)

            for policy_name in policies:
                logger.info("  Policy: %s", policy_name)

                # Initialize state
                policy_seed = config["seed"] + hash(policy_name) % 10000
                policy_rng = np.random.RandomState(policy_seed)

                coverage_counts = {sid: 0 for sid in spectator_ids}
                chosen_so_far = []
                all_probe_sets = []
                all_measurements = []

                # Per-step sigma/x_hat history for replay
                sigma_history: Dict[int, np.ndarray] = {}
                x_hat_history: Dict[int, np.ndarray] = {}

                current_x_hat = None
                current_sigma = None

                # Minimum probes before first sigma estimate
                sigma_warmup = max(set_size + 2, 5)
                # Update sigma every N steps (more frequent than checkpoints)
                sigma_update_every = max(5, checkpoints_every // 2)

                for step_m in range(1, m_total + 1):
                    # Build ProbeState
                    state = ProbeState(
                        target_id=target_id,
                        regime_id=regime_id,
                        spectator_ids=spectator_ids,
                        chosen_so_far=chosen_so_far,
                        coverage_counts=dict(coverage_counts),
                        x_hat=current_x_hat,
                        sigma=current_sigma,
                        K=K,
                        set_size=set_size,
                        max_repeats=max_repeats,
                        hybrid_lambda=hybrid_lambda,
                        epsilon=kernel_epsilon,
                        rng=np.random.RandomState(policy_seed + step_m),
                    )

                    # Select next probe
                    choice = select_next_probe(state, policy_name)

                    # Record coverage before/after
                    cov_before = dict(coverage_counts)
                    for sid in choice.chosen_set:
                        coverage_counts[sid] = coverage_counts.get(sid, 0) + 1
                    cov_after = dict(coverage_counts)

                    # Collect measurement
                    probe_set = choice.chosen_set
                    all_probe_sets.append(probe_set)
                    chosen_so_far.append(choice.chosen_indices)

                    # Run micro-run to collect y
                    y_val = collect_measurements(
                        world, target_id, regime_id,
                        [probe_set], n_samples, policy_rng,
                        stat=stat, use_irbs=use_irbs,
                    )
                    all_measurements.append(float(y_val[0]))

                    # Compute pairwise similarity of this chosen set
                    sim_val = mean_pairwise_similarity(choice.chosen_indices, K)

                    # Hash candidate pool
                    import hashlib
                    pool_hash = hashlib.sha256(
                        json.dumps(sorted(spectator_ids)).encode()
                    ).hexdigest()[:16]

                    # Log probe plan row
                    plan_row = {
                        "run_id": ctx.run_id,
                        "config_hash": ctx.config_hash,
                        "git_commit": ctx.git_commit,
                        "seed": policy_seed,
                        "policy_name": policy_name,
                        "step_m": step_m,
                        "target_id": target_id,
                        "regime_id": regime_id,
                        "chosen_set": json.dumps(probe_set),
                        "candidate_pool_hash": pool_hash,
                        "score_breakdown_json": json.dumps(
                            choice.score_breakdown, default=str
                        ),
                        "coverage_before": json.dumps(cov_before),
                        "coverage_after": json.dumps(cov_after),
                        "mean_pairwise_similarity_of_set": sim_val,
                    }
                    all_plan_rows.append(plan_row)

                    # Intermediate sigma updates for active policies
                    needs_sigma = policy_name in ("uncertainty", "sit_active")
                    is_sigma_update = (
                        needs_sigma
                        and step_m >= sigma_warmup
                        and step_m % sigma_update_every == 0
                        and step_m not in checkpoints
                    )
                    if is_sigma_update:
                        A_tmp = build_design_matrix(all_probe_sets, spectator_ids)
                        y_tmp = np.array(all_measurements, dtype=np.float64)
                        current_x_hat = solve_nonneg_elastic_net(
                            A_tmp, y_tmp, lambda_1, lambda_2,
                        )
                        boot_rng_tmp = np.random.RandomState(
                            policy_seed + step_m + 2000
                        )
                        _, current_sigma = _compute_sigma_from_bootstrap(
                            A_tmp, y_tmp, lambda_1, lambda_2,
                            n_resamples=15, rng=boot_rng_tmp,
                        )

                    # At checkpoints, reconstruct and compute metrics
                    if step_m in checkpoints:
                        A = build_design_matrix(all_probe_sets, spectator_ids)
                        y_vec = np.array(all_measurements, dtype=np.float64)

                        diag = compute_diagnostics(A)

                        # Solve
                        x_hat = solve_nonneg_elastic_net(
                            A, y_vec, lambda_1, lambda_2,
                        )

                        # Bootstrap uncertainty (fast, fewer resamples for online)
                        boot_rng = np.random.RandomState(policy_seed + step_m + 1000)
                        _, sigma = _compute_sigma_from_bootstrap(
                            A, y_vec, lambda_1, lambda_2,
                            n_resamples=30, rng=boot_rng,
                        )

                        current_x_hat = x_hat
                        current_sigma = sigma
                        sigma_history[step_m] = sigma.copy()
                        x_hat_history[step_m] = x_hat.copy()

                        # Recovery metrics
                        metrics = compute_all_recovery_metrics(x_hat, x_true)

                        # Compute probe diversity across all chosen sets so far
                        all_chosen_flat = []
                        for cs in chosen_so_far:
                            all_chosen_flat.extend(cs)
                        unique_chosen = list(set(all_chosen_flat))

                        step_row = {
                            "step_m": step_m,
                            "policy_name": policy_name,
                            "target_id": target_id,
                            "regime_id": regime_id,
                            "rank_eff": diag["rank_eff"],
                            "cond_est": diag["cond_est"],
                            "rel_L2": metrics["relative_l2"],
                            "recall_at_k": metrics["topk_recall"],
                            "ndcg_at_k": metrics["ndcg_at_k"],
                            "mean_sigma": float(np.mean(sigma)),
                            "max_sigma": float(np.max(sigma)),
                            "mean_pairwise_sim": _mean_probe_set_similarity(
                                chosen_so_far, K
                            ),
                            "l1_error": metrics["l1_error"],
                            "l2_error": metrics["l2_error"],
                            "sign_consistency": metrics["sign_consistency"],
                        }
                        all_step_rows.append(step_row)

                # Store for overlap computation
                key = (policy_name, target_id, regime_id)
                all_policy_results[key] = all_probe_sets

    # Build DataFrames
    probe_plan_df = pd.DataFrame(all_plan_rows)
    step_metrics_df = pd.DataFrame(all_step_rows)

    # Compute efficiency summary
    efficiency_summary_df = _compute_efficiency_summary(
        step_metrics_df, config,
    )

    # Compute selection overlap between policies
    overlap_df = _compute_selection_overlap(all_policy_results, spectator_ids)

    # Export artifacts
    fmt = config.get("export_format", "parquet")
    derived_base = ctx.derived_path
    os.makedirs(derived_base, exist_ok=True)
    tables_base = os.path.join("results", "tables")
    os.makedirs(tables_base, exist_ok=True)

    for name, df in [
        ("probe_plan", probe_plan_df),
        ("probe_step_metrics", step_metrics_df),
        ("probe_efficiency_summary", efficiency_summary_df),
        ("selection_overlap", overlap_df),
    ]:
        path = os.path.join(derived_base, f"{name}.{fmt}")
        write_dataframe(df, path, fmt)
        register_artifact(ctx, path, "derived")

    # CSV export for tables
    csv_path = os.path.join(tables_base, "probe_efficiency.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    efficiency_summary_df.to_csv(csv_path, index=False)

    # Also write decisions.parquet rows for probe selections
    decisions_df = _build_decisions_df(probe_plan_df, ctx)

    logger.info(
        "Probe efficiency evaluation complete: %d plan rows, %d step rows, %d summary rows",
        len(probe_plan_df), len(step_metrics_df), len(efficiency_summary_df),
    )

    return probe_plan_df, step_metrics_df, efficiency_summary_df, overlap_df, decisions_df


def _mean_probe_set_similarity(
    chosen_so_far: List[List[int]],
    K: np.ndarray,
) -> float:
    """Compute mean pairwise similarity across all probe sets."""
    if len(chosen_so_far) == 0:
        return 0.0

    sims = []
    for cs in chosen_so_far:
        s = mean_pairwise_similarity(cs, K)
        sims.append(s)

    return float(np.mean(sims))


def _compute_efficiency_summary(
    step_metrics_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Compute probes-to-threshold efficiency summary.

    For each policy/target/regime, find minimal m such that
    recall@k >= 0.95 AND ndcg@k >= 0.95.
    """
    gates_cfg = config.get("gates", {})
    recall_thresh = gates_cfg.get("recall", 0.95)
    ndcg_thresh = gates_cfg.get("ndcg", 0.95)

    rows = []
    policies = step_metrics_df["policy_name"].unique()

    for policy in policies:
        pdf = step_metrics_df[step_metrics_df["policy_name"] == policy]
        targets = pdf["target_id"].unique()
        regimes = pdf["regime_id"].unique()

        for target in targets:
            for regime in regimes:
                subset = pdf[
                    (pdf["target_id"] == target) & (pdf["regime_id"] == regime)
                ].sort_values("step_m")

                m_needed = None
                for _, row in subset.iterrows():
                    if (row["recall_at_k"] >= recall_thresh and
                            row["ndcg_at_k"] >= ndcg_thresh):
                        m_needed = int(row["step_m"])
                        break

                rows.append({
                    "policy_name": policy,
                    "target_id": target,
                    "regime_id": regime,
                    "m_needed": m_needed if m_needed is not None else -1,
                    "final_recall": float(subset.iloc[-1]["recall_at_k"]) if len(subset) > 0 else 0.0,
                    "final_ndcg": float(subset.iloc[-1]["ndcg_at_k"]) if len(subset) > 0 else 0.0,
                    "final_rel_l2": float(subset.iloc[-1]["rel_L2"]) if len(subset) > 0 else 1.0,
                })

    summary_df = pd.DataFrame(rows)

    # Add reduction ratios
    if "random" in policies:
        random_rows = summary_df[summary_df["policy_name"] == "random"]
        for policy in policies:
            if policy == "random":
                continue
            policy_rows = summary_df[summary_df["policy_name"] == policy]
            for idx, prow in policy_rows.iterrows():
                rrow = random_rows[
                    (random_rows["target_id"] == prow["target_id"]) &
                    (random_rows["regime_id"] == prow["regime_id"])
                ]
                if len(rrow) > 0:
                    m_random = rrow.iloc[0]["m_needed"]
                    m_policy = prow["m_needed"]
                    if m_random > 0 and m_policy > 0:
                        summary_df.loc[idx, "reduction_ratio"] = m_policy / m_random
                    else:
                        summary_df.loc[idx, "reduction_ratio"] = np.nan

    return summary_df


def _compute_selection_overlap(
    all_policy_results: Dict[Tuple, List[List[str]]],
    spectator_ids: List[str],
) -> pd.DataFrame:
    """Compute pairwise Jaccard overlap between policies' probe selections."""
    rows = []
    keys = list(all_policy_results.keys())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            p1, t1, r1 = keys[i]
            p2, t2, r2 = keys[j]

            if t1 != t2 or r1 != r2:
                continue

            sets1 = [frozenset(s) for s in all_policy_results[keys[i]]]
            sets2 = [frozenset(s) for s in all_policy_results[keys[j]]]

            # Jaccard of unique probe sets
            unique1 = set(sets1)
            unique2 = set(sets2)
            intersection = len(unique1 & unique2)
            union = len(unique1 | unique2)
            jaccard = intersection / max(union, 1)

            rows.append({
                "policy_1": p1,
                "policy_2": p2,
                "target_id": t1,
                "regime_id": r1,
                "jaccard_overlap": jaccard,
                "unique_sets_1": len(unique1),
                "unique_sets_2": len(unique2),
                "intersection": intersection,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "policy_1", "policy_2", "target_id", "regime_id",
        "jaccard_overlap", "unique_sets_1", "unique_sets_2", "intersection",
    ])


def _build_decisions_df(
    probe_plan_df: pd.DataFrame,
    ctx: RunContext,
) -> pd.DataFrame:
    """Build canonical decisions DataFrame from probe plan rows."""
    if len(probe_plan_df) == 0:
        return pd.DataFrame(columns=[
            "run_id", "config_hash", "git_commit", "seed",
            "decision_id", "scheduler_name", "time_index",
            "target_id", "candidate_sets", "chosen_set",
            "score_components_json", "predicted_metrics_json",
            "realized_metrics_json", "safety_pass",
        ])

    decisions = []
    for i, (_, row) in enumerate(probe_plan_df.iterrows()):
        decisions.append({
            "run_id": ctx.run_id,
            "config_hash": ctx.config_hash,
            "git_commit": ctx.git_commit,
            "seed": int(row["seed"]),
            "decision_id": i,
            "scheduler_name": f"probe_{row['policy_name']}",
            "time_index": int(row["step_m"]),
            "target_id": row["target_id"],
            "candidate_sets": row["candidate_pool_hash"],
            "chosen_set": row["chosen_set"],
            "score_components_json": row["score_breakdown_json"],
            "predicted_metrics_json": "{}",
            "realized_metrics_json": "{}",
            "safety_pass": True,
        })

    return pd.DataFrame(decisions)
