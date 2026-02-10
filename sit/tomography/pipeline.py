"""Tomography pipeline: probe design, solve, diagnose, evaluate.

Orchestrates the full tomography workflow:
1. Generate probe sets (uniform or coverage-aware)
2. Build design matrix
3. Collect measurements via simulated micro-runs
4. Run diagnostics and mitigate if needed
5. Solve nonneg elastic net
6. Compute recovery metrics against ground truth
7. Optionally compute bootstrap CIs for x_hat
"""

import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.core.units import CANONICAL_LATENCY_UNIT
from sit.probe.budget import ProbeBudget
from sit.probe.selection import (
    coverage_aware_probes,
    compute_coverage,
    uniform_random_probes,
)
from sit.tomography.design import (
    build_design_matrix,
    collect_ground_truth_vector,
    collect_measurements,
)
from sit.tomography.diagnostics import (
    compute_diagnostics,
    is_well_conditioned,
    mitigate_design,
)
from sit.tomography.metrics import compute_all_recovery_metrics
from sit.tomography.solvers import (
    cross_validate_lambda,
    solve_nonneg_elastic_net,
)
from sit.tomography.uncertainty import (
    bootstrap_x_hat_ci,
    evaluate_x_hat_coverage,
)

logger = get_logger("tomography.pipeline")


def run_tomography(
    world: Any,
    config: Dict[str, Any],
    rng: np.random.RandomState,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full tomography pipeline.

    For each target, generates probes, builds design matrix, collects
    measurements, solves the inverse problem, and evaluates recovery.

    Args:
        world: Simulation World object.
        config: Full config dict with 'tomography' section.
        rng: Random state.

    Returns:
        Tuple of (recovery_df, diagnostics_df, uncertainty_df).
    """
    tomo_cfg = config.get("tomography", {})
    sim_cfg = config.get("sim", {})
    # Use tomo-specific sample count if provided (higher for lower noise)
    n_samples = int(tomo_cfg.get(
        "n_samples_tomo",
        sim_cfg.get("n_samples_per_micro_run", 1000),
    ))

    # Probe config
    probe_cfg = tomo_cfg.get("probe", {})
    strategy = probe_cfg.get("strategy", "coverage_aware")
    m_probes = probe_cfg.get("m_probes", 60)
    set_size = probe_cfg.get("set_size", 3)
    max_repeats = probe_cfg.get("max_repeats", 20)

    # Solver config
    solver_cfg = tomo_cfg.get("solver", {})
    lambda_1 = float(solver_cfg.get("lambda_1", 0.01))
    lambda_2 = float(solver_cfg.get("lambda_2", 0.01))
    max_iter = int(solver_cfg.get("max_iter", 5000))
    do_cv = solver_cfg.get("cross_validate", False)
    cv_folds = int(solver_cfg.get("cv_folds", 5))

    # Diagnostics config
    diag_cfg = tomo_cfg.get("diagnostics", {})
    diag_enabled = diag_cfg.get("enabled", True)
    max_cond = float(diag_cfg.get("max_cond", 100.0))
    min_coverage = float(diag_cfg.get("min_coverage", 1.0))
    max_coherence = float(diag_cfg.get("max_coherence", 0.99))
    do_mitigate = diag_cfg.get("mitigate", True)
    extra_probes = int(diag_cfg.get("extra_probes", 20))

    # Uncertainty config
    unc_cfg = tomo_cfg.get("uncertainty", {})
    unc_enabled = unc_cfg.get("enabled", True)
    n_resamples = int(unc_cfg.get("n_resamples", 100))
    unc_alpha = float(unc_cfg.get("alpha", 0.05))

    # Measurement config
    stat = tomo_cfg.get("stat", "cvar99")
    use_irbs = tomo_cfg.get("use_irbs", True)

    spectator_ids = sorted([s.workload_id for s in world.spectators])

    budget = ProbeBudget(m_probes=m_probes, set_size=set_size, max_repeats=max_repeats)

    recovery_rows = []
    diag_rows = []
    uncertainty_rows = []

    regime = world.regimes[0]

    for target in world.targets:
        tid = target.workload_id
        logger.info("=== Tomography for target: %s ===", tid)

        # 1. Generate probe sets
        probe_rng = np.random.RandomState(rng.randint(0, 2**31))
        if strategy == "coverage_aware":
            probe_sets = coverage_aware_probes(spectator_ids, budget, probe_rng)
        else:
            probe_sets = uniform_random_probes(spectator_ids, budget, probe_rng)

        # 2. Build design matrix
        A = build_design_matrix(probe_sets, spectator_ids)

        # 3. Diagnostics
        if diag_enabled:
            diag = compute_diagnostics(A)
            diag_rows.append({
                "target_id": tid,
                "regime_id": regime.regime_id,
                "stage": "initial",
                **diag,
            })

            if not is_well_conditioned(diag, max_cond, min_coverage, max_coherence):
                if do_mitigate:
                    A = mitigate_design(
                        A, spectator_ids, probe_rng,
                        extra_probes=extra_probes,
                        target_coverage_min=min_coverage,
                    )
                    diag_post = compute_diagnostics(A)
                    diag_rows.append({
                        "target_id": tid,
                        "regime_id": regime.regime_id,
                        "stage": "mitigated",
                        **diag_post,
                    })

        # 4. Collect measurements
        meas_rng = np.random.RandomState(rng.randint(0, 2**31))

        # If A was augmented, we need probe sets for all rows
        # Re-derive probe sets from A
        actual_m = A.shape[0]
        if actual_m > len(probe_sets):
            # Reconstruct augmented probe sets from A
            for row_idx in range(len(probe_sets), actual_m):
                extra_probe = [spectator_ids[j] for j in range(len(spectator_ids))
                               if A[row_idx, j] > 0.5]
                probe_sets.append(sorted(extra_probe))

        y = collect_measurements(
            world, tid, regime.regime_id, probe_sets,
            n_samples, meas_rng, stat=stat, use_irbs=use_irbs,
        )

        # 5. Cross-validate lambda if requested
        if do_cv:
            cv_rng = np.random.RandomState(rng.randint(0, 2**31))
            lambda_1, _ = cross_validate_lambda(
                A, y, n_folds=cv_folds, lambda_2=lambda_2, rng=cv_rng,
            )

        # 6. Solve
        x_hat = solve_nonneg_elastic_net(
            A, y, lambda_1=lambda_1, lambda_2=lambda_2, max_iter=max_iter,
        )

        # 7. Get ground truth and compute metrics
        x_true = collect_ground_truth_vector(
            world, tid, regime.regime_id, spectator_ids,
        )

        metrics = compute_all_recovery_metrics(x_hat, x_true)

        recovery_rows.append({
            "target_id": tid,
            "regime_id": regime.regime_id,
            "n_spectators": len(spectator_ids),
            "n_probes": actual_m,
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "stat": stat,
            "use_irbs": use_irbs,
            "strategy": strategy,
            **metrics,
            "unit": CANONICAL_LATENCY_UNIT,
        })

        # 8. Uncertainty quantification
        if unc_enabled:
            unc_rng = np.random.RandomState(rng.randint(0, 2**31))
            _, ci_lo, ci_hi, _ = bootstrap_x_hat_ci(
                A, y, lambda_1, lambda_2,
                n_resamples=n_resamples, alpha=unc_alpha, rng=unc_rng,
            )

            cov_stats = evaluate_x_hat_coverage(x_true, ci_lo, ci_hi)

            uncertainty_rows.append({
                "target_id": tid,
                "regime_id": regime.regime_id,
                "n_resamples": n_resamples,
                "nominal_alpha": unc_alpha,
                "nominal_coverage": 1.0 - unc_alpha,
                **cov_stats,
            })

    recovery_df = pd.DataFrame(recovery_rows)
    diagnostics_df = pd.DataFrame(diag_rows)
    uncertainty_df = pd.DataFrame(uncertainty_rows)

    logger.info(
        "Tomography complete: %d targets, %d recovery rows, "
        "%d diagnostic rows, %d uncertainty rows",
        len(world.targets), len(recovery_rows),
        len(diag_rows), len(uncertainty_rows),
    )

    return recovery_df, diagnostics_df, uncertainty_df
