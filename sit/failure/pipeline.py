"""Failure pipeline: orchestrates injection, detection, taxonomy, mitigation, ablation.

This is the main entry point for `sit run --config configs/failure_smoke.yaml`.
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.core.registry import RunContext, register_artifact
from sit.data.io import write_dataframe

logger = get_logger("failure.pipeline")


def run_failure_pipeline(
    ctx: RunContext,
    config: Dict[str, Any],
    trials_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Run the full failure modes and ablation pipeline.

    Steps:
    1. Run failure injections (controlled assumption violations)
    2. Run detectors on each injection
    3. Build failure taxonomy
    4. Apply mitigations
    5. Run ablation study
    6. Audit for silent catastrophes
    7. Evaluate gates F1, F2, F3
    8. Generate summary report

    Args:
        ctx: Current RunContext.
        config: Full config dict with 'failure' section.
        trials_df: Optional baseline trials DataFrame.

    Returns:
        Dict with gate results, artifact paths, and pipeline data.
    """
    failure_cfg = config.get("failure", config.get("injectors", {}))
    ablation_cfg = config.get("ablations", {})
    gates_cfg = config.get("gates", {})
    fmt = config.get("export_format", "parquet")
    seed = config.get("seed", 0)

    derived_base = ctx.derived_path
    os.makedirs(derived_base, exist_ok=True)

    results = {
        "artifacts": [],
        "gates": {},
    }

    rng = np.random.RandomState(seed + 100)

    # --- Step 1: Run failure injections ---
    logger.info("=== Step 1: Running failure injections ===")
    from sit.failure.injectors import run_all_injections

    # Build enabled map from config
    injector_cfg = config.get("injectors", {})
    if isinstance(injector_cfg, dict):
        enabled_map = {k: v for k, v in injector_cfg.items() if isinstance(v, bool)}
    else:
        enabled_map = None

    injection_results = run_all_injections(config, rng, enabled=enabled_map)
    logger.info("Completed %d injections", len(injection_results))

    # --- Step 2: Run detectors on each injection ---
    logger.info("=== Step 2: Running failure detectors ===")
    from sit.failure.detectors import DetectorEngine

    detector = DetectorEngine()
    injected_assumption_ids = []

    for inj_result in injection_results:
        injected_assumption_ids.append(inj_result.injected_assumption_id)

        # Run detectors on the injection state
        new_events = detector.run_all_detectors(
            inj_result.state,
            stage="load",
        )
        logger.info(
            "Injection %s: %d events detected",
            inj_result.injection_name, len(new_events),
        )

    failure_events_df = detector.to_dataframe()
    detected_ids = detector.detected_assumption_ids()

    # Write failure_events.parquet
    fe_path = os.path.join(derived_base, f"failure_events.{fmt}")
    write_dataframe(failure_events_df, fe_path, fmt)
    register_artifact(ctx, fe_path, "derived")
    results["artifacts"].append(fe_path)
    logger.info("Wrote %d failure events to %s", len(failure_events_df), fe_path)

    # --- Step 3: Build failure taxonomy ---
    logger.info("=== Step 3: Building failure taxonomy ===")
    from sit.failure.taxonomy import build_taxonomy, compute_system_action

    taxonomy_df = build_taxonomy(failure_events_df)
    system_action = compute_system_action(taxonomy_df)
    logger.info("System action: %s", system_action)

    tax_path = os.path.join(derived_base, f"failure_taxonomy.{fmt}")
    write_dataframe(taxonomy_df, tax_path, fmt)
    register_artifact(ctx, tax_path, "derived")
    results["artifacts"].append(tax_path)

    # --- Step 4: Apply mitigations ---
    logger.info("=== Step 4: Applying mitigations ===")
    from sit.failure.mitigations import apply_mitigations

    # Collect all trials from injections for mitigation
    all_injection_trials = pd.concat(
        [inj.trials_df for inj in injection_results if len(inj.trials_df) > 0],
        ignore_index=True,
    ) if injection_results else pd.DataFrame()

    mitigation_df, mitigation_actions = apply_mitigations(
        failure_events_df, all_injection_trials,
    )

    mit_path = os.path.join(derived_base, f"mitigation_actions.{fmt}")
    write_dataframe(mitigation_df, mit_path, fmt)
    register_artifact(ctx, mit_path, "derived")
    results["artifacts"].append(mit_path)
    logger.info("Applied %d mitigations", len(mitigation_actions))

    # --- Step 5: Run ablation study ---
    logger.info("=== Step 5: Running ablation study ===")
    from sit.eval.ablations import run_ablation_study

    abl_enabled = ablation_cfg.get("enabled", True) if isinstance(ablation_cfg, dict) else True
    abl_list = ablation_cfg.get("list", None) if isinstance(ablation_cfg, dict) else None

    if abl_enabled:
        abl_rng = np.random.RandomState(seed + 200)
        ablation_df = run_ablation_study(
            config, abl_rng,
            ablation_list=abl_list,
            baseline_trials_df=trials_df,
        )
    else:
        ablation_df = pd.DataFrame()

    abl_path = os.path.join(derived_base, f"ablation_results.{fmt}")
    write_dataframe(ablation_df, abl_path, fmt)
    register_artifact(ctx, abl_path, "derived")
    results["artifacts"].append(abl_path)
    logger.info("Ablation study: %d ablations", len(ablation_df))

    # --- Step 6: Silent catastrophe audit ---
    logger.info("=== Step 6: Auditing for silent catastrophes ===")
    from sit.eval.ablations import audit_silent_catastrophes

    slo_us = config.get("slo_us", 500000.0)
    violation_report_df = audit_silent_catastrophes(
        all_injection_trials, failure_events_df, mitigation_df,
        slo_us=slo_us,
    )

    avr_path = os.path.join(derived_base, f"assumption_violation_report.{fmt}")
    write_dataframe(violation_report_df, avr_path, fmt)
    register_artifact(ctx, avr_path, "derived")
    results["artifacts"].append(avr_path)
    logger.info("Violation report: %d entries", len(violation_report_df))

    # --- Step 7: Evaluate gates ---
    logger.info("=== Step 7: Evaluating failure gates ===")
    from sit.eval.gates import (
        gate_f1_detection_completeness,
        gate_f2_graceful_degradation,
        gate_f3_no_silent_catastrophes,
    )

    gate_results = {}

    if gates_cfg.get("require_detection", True):
        gate_results["F1"] = gate_f1_detection_completeness(
            injected_assumption_ids, detected_ids,
        )

    if gates_cfg.get("require_graceful_degradation", True):
        gate_results["F2"] = gate_f2_graceful_degradation(
            ablation_df,
            cvar_tolerance=gates_cfg.get("cvar_tolerance", 0.10),
        )

    if gates_cfg.get("require_no_silent_catastrophes", True):
        gate_results["F3"] = gate_f3_no_silent_catastrophes(violation_report_df)

    results["gates"] = gate_results

    # --- Step 8: Generate summary report ---
    logger.info("=== Step 8: Generating summary report ===")
    summary_path = _generate_summary(
        ctx, config, failure_events_df, taxonomy_df,
        mitigation_df, ablation_df, violation_report_df,
        gate_results,
    )
    results["artifacts"].append(summary_path)

    # Log final gate results
    logger.info("=== Failure Pipeline Complete ===")
    for gate_name, gate_result in gate_results.items():
        status = "PASS" if gate_result.get("passed", False) else "FAIL"
        logger.info("  %s: %s - %s", gate_name, status, gate_result.get("details", ""))

    return results


def _generate_summary(
    ctx: RunContext,
    config: Dict[str, Any],
    failure_events_df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    mitigation_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    violation_report_df: pd.DataFrame,
    gate_results: Dict[str, Any],
) -> str:
    """Generate failure_and_limits_summary.csv."""
    rows = []

    # Section: Gates
    rows.append({"section": "gates", "key": "---", "value": "---"})
    for gate_name, result in gate_results.items():
        status = "PASS" if result.get("passed", False) else "FAIL"
        rows.append({"section": "gates", "key": gate_name, "value": status})
        if "details" in result:
            rows.append({
                "section": "gates", "key": f"{gate_name}_details",
                "value": str(result["details"]),
            })

    # Section: Failure events
    rows.append({"section": "failure_events", "key": "---", "value": "---"})
    rows.append({
        "section": "failure_events", "key": "total_events",
        "value": str(len(failure_events_df)),
    })
    if len(failure_events_df) > 0:
        for aid in failure_events_df["assumption_id"].unique():
            count = len(failure_events_df[failure_events_df["assumption_id"] == aid])
            rows.append({
                "section": "failure_events", "key": f"events_{aid}",
                "value": str(count),
            })

    # Section: Taxonomy
    rows.append({"section": "taxonomy", "key": "---", "value": "---"})
    if len(taxonomy_df) > 0:
        for _, row in taxonomy_df.iterrows():
            rows.append({
                "section": "taxonomy",
                "key": f"{row['category']}_{row['assumption_id']}",
                "value": f"severity={row['max_severity']}, action={row['action']}",
            })

    # Section: Mitigations
    rows.append({"section": "mitigations", "key": "---", "value": "---"})
    rows.append({
        "section": "mitigations", "key": "total_mitigations",
        "value": str(len(mitigation_df)),
    })

    # Section: Ablations
    rows.append({"section": "ablations", "key": "---", "value": "---"})
    if len(ablation_df) > 0:
        for _, row in ablation_df.iterrows():
            rows.append({
                "section": "ablations",
                "key": row["ablation_name"],
                "value": f"delta_cvar={row['delta_cvar']:.2f}, delta_goodput={row['delta_goodput']:.4f}",
            })

    # Section: Silent catastrophes
    rows.append({"section": "silent_catastrophes", "key": "---", "value": "---"})
    n_silent = int(violation_report_df["silent"].sum()) if len(violation_report_df) > 0 and "silent" in violation_report_df.columns else 0
    rows.append({
        "section": "silent_catastrophes", "key": "count",
        "value": str(n_silent),
    })

    summary_df = pd.DataFrame(rows)

    # Write to run-specific tables dir
    tables_base = ctx.tables_path
    os.makedirs(tables_base, exist_ok=True)
    summary_path = os.path.join(tables_base, "failure_and_limits_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    register_artifact(ctx, summary_path, "report")

    # Also write to global results/tables/
    global_tables = os.path.join("results", "tables")
    os.makedirs(global_tables, exist_ok=True)
    global_path = os.path.join(global_tables, "failure_and_limits_summary.csv")
    summary_df.to_csv(global_path, index=False)

    logger.info("Summary written to %s", summary_path)
    return summary_path
