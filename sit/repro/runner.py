"""One-command reproducibility runner.

Entry point: run_repro_pipeline(config)
Sequentially executes all SIT pipeline stages, captures run_ids,
builds manifests, runs replays, generates docs, and checks gates.
"""

import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sit.core.config import load_config
from sit.core.hashing import hash_config, sha256_file
from sit.core.logging import get_logger
from sit.core.registry import (
    RunContext,
    create_run_id,
    finalize_run,
    register_artifact,
    start_run,
)
from sit.data.io import write_dataframe
from sit.data.paths import (
    artifact_path,
    decision_path,
    derived_dir,
    raw_dir,
    trial_path,
)
from sit.repro.env import capture_environment, set_deterministic_env
from sit.repro.manifest import build_artifact_manifest

logger = get_logger("repro.runner")

# Canonical stage order
STAGE_ORDER = [
    "probe",
    "tomography",
    "scheduling",
    "load",
    "stats",
    "failure",
    "hardware",
]


def run_repro_pipeline(repro_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full reproducibility pipeline.

    Executes all enabled stages sequentially, captures metadata,
    builds manifests, runs replay verification, generates reports,
    and evaluates reproducibility gates.

    Args:
        repro_config: Loaded repro.yaml configuration.

    Returns:
        Dict with master_run_id, stage_results, manifest, gates, env.
    """
    master_run_id = create_run_id()
    seed = repro_config.get("seed", 123)

    # Set deterministic environment
    set_deterministic_env(seed)

    # Capture environment
    env_snapshot = capture_environment()

    logger.info("Starting repro pipeline: master_run_id=%s", master_run_id)

    stages_config = repro_config.get("stages", {})
    repro_settings = repro_config.get("repro", {})
    base_output_dir = repro_config.get("output_dir", "data")

    stage_results: Dict[str, Dict[str, Any]] = {}
    all_trials_df = None
    all_decisions_df = None
    first_run_id = None

    for stage_name in STAGE_ORDER:
        stage_cfg = stages_config.get(stage_name)
        if stage_cfg is None:
            continue

        # Hardware stage may be disabled
        if stage_name == "hardware":
            if not stage_cfg.get("enabled", False):
                stage_results[stage_name] = {
                    "run_id": "",
                    "config_hash": "",
                    "status": "skipped",
                    "reason": "hardware.enabled=false",
                }
                continue

        stage_config_path = stage_cfg.get("config", "")
        if not stage_config_path:
            continue

        logger.info("Running stage: %s (config: %s)", stage_name, stage_config_path)

        try:
            result = _run_stage(
                stage_name, stage_config_path, repro_config, base_output_dir,
            )
            stage_results[stage_name] = result

            # Capture trials/decisions and their run_id for replay
            if result.get("trials_df") is not None:
                all_trials_df = result["trials_df"]
                first_run_id = result["run_id"]
            if result.get("decisions_df") is not None:
                all_decisions_df = result["decisions_df"]

            logger.info(
                "Stage %s completed: run_id=%s, status=%s",
                stage_name, result.get("run_id", ""), result.get("status", ""),
            )

        except Exception as e:
            logger.error("Stage %s FAILED: %s", stage_name, str(e))
            stage_results[stage_name] = {
                "run_id": "",
                "config_hash": "",
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            # Stop on failure
            break

    # Build artifact manifest
    manifest_df = build_artifact_manifest(base_output_dir, master_run_id, stage_results)

    # Write manifest
    manifest_dir = os.path.join(base_output_dir, "derived", master_run_id)
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, "artifact_manifest.parquet")
    write_dataframe(manifest_df, manifest_path, "parquet")

    # Run replay verification if data available
    replay_results = None
    if all_trials_df is not None and all_decisions_df is not None and first_run_id:
        from sit.repro.replay import run_replay_verification

        replay_results = run_replay_verification(
            config=repro_config,
            trials_df=all_trials_df,
            decisions_df=all_decisions_df,
            run_id=first_run_id,
            n_probe_samples=repro_settings.get("n_probe_samples", 5),
            n_sched_samples=repro_settings.get("n_sched_samples", 5),
            n_load_samples=repro_settings.get("n_load_samples", 3),
        )

    # Generate reports
    _generate_reports(repro_config, stage_results, manifest_df, env_snapshot, master_run_id)

    # Generate docs
    _generate_docs(repro_config, stage_results, manifest_df, env_snapshot, master_run_id)

    # Evaluate gates
    gates = _evaluate_gates(
        repro_config, stage_results, manifest_df, replay_results,
    )

    return {
        "master_run_id": master_run_id,
        "stage_results": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("trials_df", "decisions_df")}
            for k, v in stage_results.items()
        },
        "manifest_path": manifest_path,
        "manifest_count": len(manifest_df),
        "gates": gates,
        "env": env_snapshot,
        "replay": replay_results,
    }


def _run_stage(
    stage_name: str,
    config_path: str,
    repro_config: Dict[str, Any],
    base_output_dir: str,
) -> Dict[str, Any]:
    """Run a single pipeline stage.

    Loads the stage config, runs the appropriate pipeline, and returns results.
    """
    from sit.core.config import load_config
    from sit.core.logging import reset_logging

    reset_logging()

    config = load_config(config_path)
    # Override seed from repro config for consistency
    config["seed"] = repro_config.get("seed", config["seed"])
    config["output_dir"] = base_output_dir

    ctx = start_run(config)
    config_hash_val = ctx.config_hash
    fmt = config["export_format"]
    timestamp = ctx.created_at_utc

    trials_df = None
    decisions_df = None
    gate_results = {}

    if stage_name in ("probe", "scheduling", "load"):
        # These stages run the sim pipeline
        trials_df, decisions_df = _run_sim_stage(ctx, config, fmt, timestamp)

    elif stage_name == "tomography":
        # Tomo runs sim + tomography
        trials_df, decisions_df = _run_sim_stage(ctx, config, fmt, timestamp)

        if "tomography" in config:
            from sit.tomography.pipeline import run_tomography
            from sit.sim.world import build_world

            rng = np.random.RandomState(config["seed"])
            world = build_world(config, rng)
            tomo_rng = np.random.RandomState(config["seed"] + 5)
            recovery_df, diag_df, unc_df = run_tomography(world, config, tomo_rng)

            derived_base = ctx.derived_path
            os.makedirs(derived_base, exist_ok=True)

            for name, df in [("tomo_recovery", recovery_df), ("tomo_diagnostics", diag_df), ("tomo_uncertainty", unc_df)]:
                if df is not None and len(df) > 0:
                    p = os.path.join(derived_base, f"{name}.{fmt}")
                    write_dataframe(df, p, fmt)
                    register_artifact(ctx, p, "derived")

    elif stage_name == "stats":
        # Stats needs sim trials first
        trials_df, decisions_df = _run_sim_stage(ctx, config, fmt, timestamp)

        if "stats" in config:
            from sit.stats.pipeline import run_stats_pipeline

            stats_results = run_stats_pipeline(ctx, config, trials_df)
            gate_results.update(stats_results.get("gates", {}))

    elif stage_name == "failure":
        # Failure mode
        from sit.failure.pipeline import run_failure_pipeline

        baseline_trials = None
        if "sim" in config:
            from sit.sim.pipeline import run_sim_trials, run_sim_decisions
            from sit.sim.world import build_world

            rng = np.random.RandomState(config["seed"])
            world = build_world(config, rng)
            trial_rng = np.random.RandomState(config["seed"] + 1)
            baseline_trials = run_sim_trials(
                world, config, ctx.run_id, ctx.config_hash,
                ctx.git_commit, timestamp, trial_rng,
            )
            decision_rng = np.random.RandomState(config["seed"] + 2)
            decisions_df = run_sim_decisions(
                world, config, ctx.run_id, ctx.config_hash,
                ctx.git_commit, timestamp, decision_rng,
            )

            from sit.core.schema import validate_trials, validate_decisions
            baseline_trials = validate_trials(baseline_trials)
            decisions_df = validate_decisions(decisions_df)

            t_path = trial_path(ctx.run_id, ctx.output_dir, fmt)
            d_path = decision_path(ctx.run_id, ctx.output_dir, fmt)
            write_dataframe(baseline_trials, t_path, fmt)
            write_dataframe(decisions_df, d_path, fmt)
            register_artifact(ctx, t_path, "raw")
            register_artifact(ctx, d_path, "raw")

            trials_df = baseline_trials

        failure_results = run_failure_pipeline(ctx, config, baseline_trials)
        gate_results.update(failure_results.get("gates", {}))

    elif stage_name == "hardware":
        from sit.hardware.runner import run_hardware_pipeline

        hw_results = run_hardware_pipeline(ctx, config)
        gate_results.update(hw_results.get("gates", {}))

    # Finalize
    artifacts_df = finalize_run(ctx)
    a_path = artifact_path(ctx.run_id, ctx.output_dir, fmt)
    write_dataframe(artifacts_df, a_path, fmt)

    return {
        "run_id": ctx.run_id,
        "config_hash": config_hash_val,
        "status": "completed",
        "n_artifacts": len(artifacts_df),
        "gates": gate_results,
        "trials_df": trials_df,
        "decisions_df": decisions_df,
    }


def _run_sim_stage(ctx, config, fmt, timestamp):
    """Run the sim pipeline (trials + decisions + derived data)."""
    from sit.core.schema import validate_trials, validate_decisions
    from sit.sim.pipeline import (
        run_sim_trials, run_sim_decisions,
        run_irbs_evaluation, run_ci_coverage_evaluation,
    )
    from sit.sim.world import build_world, ground_truth_to_dataframe

    rng = np.random.RandomState(config["seed"])
    world = build_world(config, rng)

    trial_rng = np.random.RandomState(config["seed"] + 1)
    decision_rng = np.random.RandomState(config["seed"] + 2)
    irbs_rng = np.random.RandomState(config["seed"] + 3)
    ci_rng = np.random.RandomState(config["seed"] + 4)

    trials_df = run_sim_trials(
        world, config, ctx.run_id, ctx.config_hash,
        ctx.git_commit, timestamp, trial_rng,
    )
    decisions_df = run_sim_decisions(
        world, config, ctx.run_id, ctx.config_hash,
        ctx.git_commit, timestamp, decision_rng,
    )

    gt_df = ground_truth_to_dataframe(world)
    events_df, irbs_df = run_irbs_evaluation(world, config, irbs_rng)
    ci_coverage_df = run_ci_coverage_evaluation(config, ci_rng)

    # Validate
    trials_df = validate_trials(trials_df)
    decisions_df = validate_decisions(decisions_df)

    # Write raw
    t_path = trial_path(ctx.run_id, ctx.output_dir, fmt)
    d_path = decision_path(ctx.run_id, ctx.output_dir, fmt)
    write_dataframe(trials_df, t_path, fmt)
    write_dataframe(decisions_df, d_path, fmt)
    register_artifact(ctx, t_path, "raw")
    register_artifact(ctx, d_path, "raw")

    # Write derived
    derived_base = ctx.derived_path
    os.makedirs(derived_base, exist_ok=True)

    for name, df in [
        ("ground_truth", gt_df),
        ("measurement_events", events_df),
        ("irbs_estimates", irbs_df),
        ("tail_ci_coverage", ci_coverage_df),
    ]:
        if df is not None and len(df) > 0:
            p = os.path.join(derived_base, f"{name}.{fmt}")
            write_dataframe(df, p, fmt)
            register_artifact(ctx, p, "derived")

    return trials_df, decisions_df


def _generate_reports(repro_config, stage_results, manifest_df, env_snapshot, master_run_id):
    """Generate summary tables and reports."""
    from sit.report.summary import generate_narrative_summary
    from sit.report.tables import generate_final_summary_table

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/reports", exist_ok=True)

    # Final summary table
    summary_df = generate_final_summary_table(stage_results)
    summary_df.to_csv("results/tables/final_summary.csv", index=False)

    # Narrative summary
    narrative = generate_narrative_summary(stage_results, env_snapshot, master_run_id)
    with open("results/reports/summary.md", "w") as f:
        f.write(narrative)


def _generate_docs(repro_config, stage_results, manifest_df, env_snapshot, master_run_id):
    """Generate reproducibility docs."""
    from sit.report.figures import generate_figure_manifest
    from sit.report.faq import generate_interview_doc

    os.makedirs("docs", exist_ok=True)

    # Repro binder
    _write_repro_binder(repro_config, stage_results, manifest_df, env_snapshot, master_run_id)

    # Figure manifest
    fig_md = generate_figure_manifest(stage_results, manifest_df)
    with open("docs/figures.md", "w") as f:
        f.write(fig_md)

    # Interview armor
    interview_md = generate_interview_doc(stage_results, manifest_df)
    with open("docs/interview.md", "w") as f:
        f.write(interview_md)


def _write_repro_binder(repro_config, stage_results, manifest_df, env_snapshot, master_run_id):
    """Generate docs/repro.md - the reproducibility binder."""
    lines = []
    lines.append("# SIT Reproducibility Binder")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append("SIT (Spectator Interference Tomography) is a system for measuring,")
    lines.append("modeling, and mitigating tail-latency interference in shared computing")
    lines.append("environments. This binder documents a complete, reproducible pipeline run")
    lines.append("with all artifacts, hashes, and verification steps.")
    lines.append("")
    lines.append("This binder contains: environment snapshot, pipeline stage results,")
    lines.append("artifact manifest with SHA-256 hashes, decision replay verification,")
    lines.append("and a verification checklist for external reviewers.")
    lines.append("")

    lines.append("## 2. Exact Rerun Command")
    lines.append("")
    lines.append("```bash")
    lines.append("sit repro --config configs/repro.yaml")
    lines.append("```")
    lines.append("")

    lines.append("## 3. Environment")
    lines.append("")
    lines.append(f"- **OS:** {env_snapshot.get('os', 'N/A')} ({env_snapshot.get('os_version', 'N/A')})")
    lines.append(f"- **Python:** {env_snapshot.get('python_version', 'N/A').split()[0]}")
    lines.append(f"- **CPU:** {env_snapshot.get('cpu', 'N/A')}")
    lines.append(f"- **Platform:** {env_snapshot.get('platform', 'N/A')}")
    libs = env_snapshot.get("libraries", {})
    for lib_name, lib_ver in sorted(libs.items()):
        lines.append(f"- **{lib_name}:** {lib_ver}")
    lines.append("")
    lines.append("Note: Hardware performance counters are optional and may not be")
    lines.append("available on all platforms.")
    lines.append("")

    lines.append("## 4. Pipeline Stages")
    lines.append("")
    lines.append("| Stage | Run ID | Config Hash | Status |")
    lines.append("|-------|--------|-------------|--------|")
    for stage_name in STAGE_ORDER:
        info = stage_results.get(stage_name, {})
        run_id = info.get("run_id", "N/A")
        cfg_hash = info.get("config_hash", "N/A")
        status = info.get("status", "not run")
        if run_id and len(run_id) > 12:
            run_id = run_id[:12] + "..."
        if cfg_hash and len(cfg_hash) > 12:
            cfg_hash = cfg_hash[:12] + "..."
        lines.append(f"| {stage_name} | {run_id} | {cfg_hash} | {status} |")
    lines.append("")

    lines.append("## 5. Artifacts")
    lines.append("")
    lines.append(f"Total artifacts: {len(manifest_df)}")
    lines.append("")
    lines.append(f"Manifest location: `data/derived/{master_run_id}/artifact_manifest.parquet`")
    lines.append("")
    if len(manifest_df) > 0:
        lines.append("| Type | Count |")
        lines.append("|------|-------|")
        for atype, count in manifest_df["artifact_type"].value_counts().items():
            lines.append(f"| {atype} | {count} |")
        lines.append("")

    lines.append("## 6. Known Limitations")
    lines.append("")
    lines.append("- Hardware validation requires physical access to bare-metal servers")
    lines.append("  with performance counters enabled. The hardware stage is optional.")
    lines.append("- Floating-point results may vary across platforms due to differences")
    lines.append(f"  in math library implementations. Tolerance: epsilon={repro_config.get('repro', {}).get('epsilon_float', 1e-8)}")
    lines.append("- Parquet file hashes are byte-exact; numeric comparisons use epsilon tolerance.")
    lines.append("")

    lines.append("## 7. Verification Checklist")
    lines.append("")
    lines.append("For a reviewer to verify results:")
    lines.append("")
    lines.append("1. Clone the repository and install dependencies")
    lines.append("2. Run: `sit repro --config configs/repro.yaml`")
    lines.append("3. Check that all stages report 'completed' status")
    lines.append("4. Verify the artifact manifest exists and all hashes match")
    lines.append("5. Check that gates R0 (reproducibility), R1 (manifest), R2 (replay) pass")
    lines.append("6. Review results/tables/final_summary.csv for key metrics")
    lines.append("7. Review results/reports/summary.md for narrative interpretation")
    lines.append("")
    lines.append(f"Master run ID: `{master_run_id}`")
    lines.append("")
    lines.append(f"Generated at: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    with open("docs/repro.md", "w") as f:
        f.write("\n".join(lines))


def _evaluate_gates(repro_config, stage_results, manifest_df, replay_results):
    """Evaluate reproducibility gates R0, R1, R2."""
    from sit.eval.gates import gate_r0_reproducibility, gate_r1_manifest_completeness, gate_r2_replay_integrity

    gates = {}

    # R0: Reproducibility (we verify manifest is non-empty and stages completed)
    gates["R0"] = gate_r0_reproducibility(stage_results, manifest_df)

    # R1: Manifest completeness
    gates["R1"] = gate_r1_manifest_completeness(manifest_df)

    # R2: Replay integrity
    gates["R2"] = gate_r2_replay_integrity(replay_results)

    return gates
