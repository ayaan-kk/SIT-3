"""Stats pipeline: orchestrates CI calibration, EVT, effects, FDR, and robustness.

This module implements the 'stats' run mode for `sit run --config`.
It runs the load pipeline (or reuses cached results), then executes
all statistical validation steps and gates.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.core.registry import RunContext, register_artifact
from sit.data.io import write_dataframe

logger = get_logger("stats.pipeline")


def run_stats_pipeline(
    ctx: RunContext,
    config: Dict[str, Any],
    trials_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Run the full statistical validation pipeline.

    Steps:
    1. CI calibration harness
    2. EVT diagnostics (if enabled)
    3. Regime-wise effect comparisons
    4. FDR correction
    5. Robustness sweeps
    6. Generate summary report
    7. Run gates

    Args:
        ctx: Current RunContext.
        config: Full config dict.
        trials_df: Trials DataFrame from the load pipeline.

    Returns:
        Dict with gate results and artifact paths.
    """
    stats_cfg = config.get("stats", {})
    gates_cfg = config.get("gates", {})
    fmt = config.get("export_format", "parquet")
    seed = config.get("seed", 0)

    derived_base = ctx.derived_path
    os.makedirs(derived_base, exist_ok=True)

    tables_base = ctx.tables_path
    os.makedirs(tables_base, exist_ok=True)

    results = {
        "artifacts": [],
        "gates": {},
    }

    # --- Step 1: CI Calibration ---
    ci_calibration_df = _run_ci_calibration(stats_cfg, seed)
    if ci_calibration_df is not None and len(ci_calibration_df) > 0:
        ci_path = os.path.join(derived_base, f"ci_calibration.{fmt}")
        write_dataframe(ci_calibration_df, ci_path, fmt)
        register_artifact(ctx, ci_path, "derived")
        results["artifacts"].append(ci_path)

    # --- Step 2: EVT diagnostics ---
    evt_df = _run_evt_diagnostics(stats_cfg, trials_df)
    if evt_df is not None and len(evt_df) > 0:
        evt_path = os.path.join(derived_base, f"tail_validity_evt.{fmt}")
        write_dataframe(evt_df, evt_path, fmt)
        register_artifact(ctx, evt_path, "derived")
        results["artifacts"].append(evt_path)

    # --- Step 3: Regime-wise effects ---
    effects_df = _run_effects(stats_cfg, trials_df, seed)
    if effects_df is not None and len(effects_df) > 0:
        eff_path = os.path.join(derived_base, f"regime_effects.{fmt}")
        write_dataframe(effects_df, eff_path, fmt)
        register_artifact(ctx, eff_path, "derived")
        results["artifacts"].append(eff_path)

    # --- Step 4: FDR correction ---
    fdr_df = _run_fdr(stats_cfg, effects_df)
    if fdr_df is not None and len(fdr_df) > 0:
        fdr_path = os.path.join(derived_base, f"fdr_results.{fmt}")
        write_dataframe(fdr_df, fdr_path, fmt)
        register_artifact(ctx, fdr_path, "derived")
        results["artifacts"].append(fdr_path)

    # --- Step 5: Robustness sweeps ---
    robustness_df = _run_robustness(stats_cfg, config, seed)
    if robustness_df is not None and len(robustness_df) > 0:
        rob_path = os.path.join(derived_base, f"robustness_sweeps.{fmt}")
        write_dataframe(robustness_df, rob_path, fmt)
        register_artifact(ctx, rob_path, "derived")
        results["artifacts"].append(rob_path)

    # --- Step 6: Gates ---
    gate_results = _run_gates(
        gates_cfg, ci_calibration_df, evt_df, fdr_df,
    )
    results["gates"] = gate_results

    # --- Step 7: Summary report ---
    summary_path = os.path.join(tables_base, "stats_summary.csv")
    from sit.stats.report import generate_stats_summary
    generate_stats_summary(
        ci_calibration_df=ci_calibration_df,
        evt_df=evt_df,
        fdr_df=fdr_df,
        robustness_df=robustness_df,
        gate_results=gate_results,
        output_path=summary_path,
    )
    register_artifact(ctx, summary_path, "report")
    results["artifacts"].append(summary_path)

    # Also write to results/tables/ (global, not run-specific)
    global_tables = os.path.join("results", "tables")
    os.makedirs(global_tables, exist_ok=True)
    global_summary_path = os.path.join(global_tables, "stats_summary.csv")
    from sit.stats.report import generate_stats_summary as gen_summary
    gen_summary(
        ci_calibration_df=ci_calibration_df,
        evt_df=evt_df,
        fdr_df=fdr_df,
        robustness_df=robustness_df,
        gate_results=gate_results,
        output_path=global_summary_path,
    )

    return results


def _run_ci_calibration(
    stats_cfg: Dict[str, Any],
    seed: int,
) -> Optional[pd.DataFrame]:
    """Run CI calibration step."""
    ci_cfg = stats_cfg.get("ci", {})
    if not ci_cfg.get("enabled", True):
        logger.info("CI calibration disabled, skipping")
        return None

    from sit.stats.coverage import run_ci_calibration

    n_repeats = ci_cfg.get("repeats", 30)
    n_resamples = ci_cfg.get("bootstrap_resamples", 200)
    ref_mult = ci_cfg.get("reference_multiplier", 5)
    nominal = ci_cfg.get("nominal", 0.95)
    n_samples_base = ci_cfg.get("n_samples_base", 500)

    logger.info("Running CI calibration: %d repeats, %d resamples", n_repeats, n_resamples)

    df = run_ci_calibration(
        n_repeats=n_repeats,
        n_bootstrap_resamples=n_resamples,
        reference_multiplier=ref_mult,
        nominal_alpha=1.0 - nominal,
        base_seed=seed,
        n_samples_base=n_samples_base,
    )

    return df


def _run_evt_diagnostics(
    stats_cfg: Dict[str, Any],
    trials_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Run EVT tail diagnostics."""
    evt_cfg = stats_cfg.get("evt", {})
    if not evt_cfg.get("enabled", True):
        logger.info("EVT diagnostics disabled, skipping")
        return None

    threshold_quantiles = evt_cfg.get("threshold_quantiles", [0.90, 0.92, 0.94, 0.95])
    min_exceedances = evt_cfg.get("min_exceedances", 200)

    from sit.stats.evt import run_evt_analysis

    # Run EVT on the trial-level latency data
    group_cols = ["scheduler_name"]
    if "regime_id" in trials_df.columns:
        group_cols.append("regime_id")

    logger.info("Running EVT diagnostics with thresholds: %s", threshold_quantiles)

    # Use p99 for tail analysis
    latency_col = "p99_latency_us"
    if latency_col not in trials_df.columns:
        latency_col = "cvar99_latency_us"
    if latency_col not in trials_df.columns:
        logger.warning("No tail latency column found, skipping EVT")
        return None

    df = run_evt_analysis(
        trials_df=trials_df,
        group_cols=group_cols,
        latency_col=latency_col,
        threshold_quantiles=threshold_quantiles,
        min_exceedances=min_exceedances,
    )

    return df


def _run_effects(
    stats_cfg: Dict[str, Any],
    trials_df: pd.DataFrame,
    seed: int,
) -> Optional[pd.DataFrame]:
    """Run regime-wise effect comparisons."""
    effects_cfg = stats_cfg.get("effects", {})
    if not effects_cfg:
        logger.info("Effects config not found, skipping")
        return None

    from sit.stats.effects import compute_all_pairwise_effects

    baseline = effects_cfg.get("baseline", "measurement_harness")
    schedulers = trials_df["scheduler_name"].unique().tolist()

    if len(schedulers) < 2:
        # Single scheduler: create a synthetic comparison using episode splits
        logger.info("Single scheduler found, creating synthetic split comparison")
        return _synthetic_split_effects(trials_df, baseline, seed)

    metric_cols = ["cvar99_latency_us", "p99_latency_us"]
    available_metrics = [m for m in metric_cols if m in trials_df.columns]
    if not available_metrics:
        return None

    bucket_cols = ["regime_id"] if "regime_id" in trials_df.columns else []
    if not bucket_cols:
        return None

    df = compute_all_pairwise_effects(
        trials_df=trials_df,
        schedulers=schedulers,
        baseline=baseline,
        metric_cols=available_metrics,
        bucket_cols=bucket_cols,
        n_bootstrap=200,
        seed=seed,
    )

    return df


def _synthetic_split_effects(
    trials_df: pd.DataFrame,
    scheduler_name: str,
    seed: int,
) -> pd.DataFrame:
    """Create synthetic paired comparison by splitting trials.

    Splits the trials for a single scheduler into two halves and
    compares them. This serves as a null-effect baseline.
    """
    from sit.stats.effects import compute_regime_effects

    n = len(trials_df)
    if n < 6:
        return pd.DataFrame()

    # Split into A and B halves
    half = n // 2
    df_a = trials_df.iloc[:half].copy()
    df_b = trials_df.iloc[half:2 * half].copy()
    df_a["scheduler_name"] = f"{scheduler_name}_split_A"
    df_b["scheduler_name"] = f"{scheduler_name}_split_B"

    combined = pd.concat([df_a, df_b], ignore_index=True)

    metric_cols = ["cvar99_latency_us", "p99_latency_us"]
    available_metrics = [m for m in metric_cols if m in combined.columns]
    if not available_metrics:
        return pd.DataFrame()

    bucket_cols = ["regime_id"] if "regime_id" in combined.columns else []
    if not bucket_cols:
        return pd.DataFrame()

    return compute_regime_effects(
        trials_df=combined,
        scheduler_a=f"{scheduler_name}_split_A",
        scheduler_b=f"{scheduler_name}_split_B",
        metric_cols=available_metrics,
        bucket_cols=bucket_cols,
        n_bootstrap=200,
        seed=seed,
    )


def _run_fdr(
    stats_cfg: Dict[str, Any],
    effects_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Apply FDR correction to effects."""
    if effects_df is None or len(effects_df) == 0:
        return None

    effects_cfg = stats_cfg.get("effects", {})
    q_threshold = effects_cfg.get("fdr_q", 0.10)

    from sit.stats.fdr import apply_fdr_to_effects

    return apply_fdr_to_effects(effects_df, q_threshold=q_threshold)


def _run_robustness(
    stats_cfg: Dict[str, Any],
    config: Dict[str, Any],
    seed: int,
) -> Optional[pd.DataFrame]:
    """Run robustness parameter sweeps."""
    rob_cfg = stats_cfg.get("robustness", {})
    if not rob_cfg.get("enabled", False):
        logger.info("Robustness sweeps disabled, skipping")
        return None

    from sit.stats.robustness import run_robustness_sweep

    beta_grid = rob_cfg.get("beta_grid", [0.5, 1.0, 2.0])
    lambda_grid = rob_cfg.get("lambda_grid", [0.2, 0.5, 1.0])
    tau_grid_us = rob_cfg.get("tau_grid_us", [1500000, 2000000, 3000000])
    slo_us = config.get("slo_us", 500000.0)

    logger.info(
        "Running robustness sweep: %d beta x %d lambda x %d tau = %d configs",
        len(beta_grid), len(lambda_grid), len(tau_grid_us),
        len(beta_grid) * len(lambda_grid) * len(tau_grid_us),
    )

    df = run_robustness_sweep(
        beta_grid=beta_grid,
        lambda_grid=lambda_grid,
        tau_grid_us=tau_grid_us,
        slo_us=slo_us,
        seed=seed,
    )

    return df


def _run_gates(
    gates_cfg: Dict[str, Any],
    ci_calibration_df: Optional[pd.DataFrame],
    evt_df: Optional[pd.DataFrame],
    fdr_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """Run all statistical gates."""
    if not gates_cfg.get("enable", True):
        return {"status": "disabled"}

    gate_results = {}

    # Gate R1: CI Calibration
    if ci_calibration_df is not None and len(ci_calibration_df) > 0:
        from sit.stats.coverage import gate_r1_ci_calibration
        min_coverage = gates_cfg.get("min_coverage", 0.90)
        gate_results["R1"] = gate_r1_ci_calibration(
            ci_calibration_df, min_coverage=min_coverage,
        )

    # Gate R2: Tail Validity
    if evt_df is not None and len(evt_df) > 0:
        from sit.stats.evt import gate_r2_tail_validity
        evt_required = gates_cfg.get("evt_required", False)
        gate_results["R2"] = gate_r2_tail_validity(
            evt_df, evt_required=evt_required,
        )

    # Gate R3: No P-hacking
    if fdr_df is not None and len(fdr_df) > 0:
        from sit.stats.fdr import gate_r3_no_phacking
        fdr_q = gates_cfg.get("fdr_q", 0.10)
        gate_results["R3"] = gate_r3_no_phacking(
            fdr_df, q_threshold=fdr_q,
        )

    return gate_results
