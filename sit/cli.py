"""CLI entry point for SIT.

Commands:
    sit run       --config <path>     Run pipeline (smoke or full)
    sit validate  --run-id <id>       Validate raw data for a run
    sit info      --run-id <id>       Show run metadata and counts
"""

import json
import os
import sys
from datetime import datetime, timezone

import click

from sit.core.config import load_config, config_hash
from sit.core.logging import setup_logging, get_logger, reset_logging
from sit.core.registry import start_run, register_artifact, finalize_run
from sit.core.schema import (
    validate_trials,
    validate_decisions,
    validate_artifacts,
    SCHEMA_VERSION,
)
from sit.core.units import CANONICAL_LATENCY_UNIT
from sit.core.validation import (
    generate_smoke_trials_and_decisions,
    validate_trial_metrics,
)
from sit.data.io import write_dataframe, read_dataframe
from sit.data.paths import (
    raw_dir,
    trial_path,
    decision_path,
    artifact_path,
    find_run_ids,
)


@click.group()
def cli():
    """SIT: Spectator Interference Tomography."""
    pass


@cli.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML configuration file.",
)
def run(config_path: str):
    """Run the SIT pipeline.

    Creates a new run, generates data (smoke or sim), validates,
    writes raw tables, and produces an artifacts manifest.

    If the config contains a 'sim' section, runs the full simulator.
    Otherwise, runs the placeholder smoke generator.
    """
    # Reset logging for fresh run
    reset_logging()

    import numpy as np

    # Load config
    config = load_config(config_path)

    # Start run
    ctx = start_run(config)
    logger = get_logger("cli.run")
    logger.info("Starting SIT run with config: %s", config_path)

    fmt = config["export_format"]
    timestamp = ctx.created_at_utc
    has_sim = "sim" in config
    has_tomo = "tomography" in config
    has_hw = config.get("run_mode") == "hardware"
    has_failure = config.get("run_mode") == "failure"

    # --- Failure mode (separate path) ---
    if has_failure:
        from sit.failure.pipeline import run_failure_pipeline

        click.echo("Running failure modes and ablation pipeline...")

        # Run baseline sim trials first if sim section present
        baseline_trials = None
        if has_sim:
            from sit.sim.pipeline import run_sim_trials, run_sim_decisions
            from sit.sim.world import build_world

            import numpy as np
            rng = np.random.RandomState(config["seed"])
            world = build_world(config, rng)
            trial_rng = np.random.RandomState(config["seed"] + 1)
            baseline_trials = run_sim_trials(
                world, config, ctx.run_id, ctx.config_hash,
                ctx.git_commit, timestamp, trial_rng,
            )

            # Also generate decisions for schema compliance
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

        failure_results = run_failure_pipeline(ctx, config, baseline_trials)

        # Finalize
        artifacts_df = finalize_run(ctx)
        a_path = artifact_path(ctx.run_id, ctx.output_dir, fmt)
        write_dataframe(artifacts_df, a_path, fmt)

        click.echo("")
        click.echo("=" * 60)
        click.echo("SIT Failure & Ablation Run Complete")
        click.echo("=" * 60)
        click.echo(f"  Run ID:       {ctx.run_id}")
        click.echo(f"  Config hash:  {ctx.config_hash}")
        click.echo(f"  Git commit:   {ctx.git_commit}")
        click.echo(f"  Mode:         failure")
        click.echo(f"  Artifacts:    {len(artifacts_df)}")
        click.echo(f"  Output dir:   {ctx.raw_path}")

        gate_results = failure_results.get("gates", {})
        if gate_results:
            click.echo("")
            click.echo("Failure gates:")
            for gate_name, gate_result in gate_results.items():
                if isinstance(gate_result, dict):
                    status = "PASS" if gate_result.get("passed", False) else "FAIL"
                    details = gate_result.get("details", "")
                    click.echo(f"  {gate_name}: {status} ({details})")

        click.echo("=" * 60)
        return

    # --- Hardware run mode (separate path) ---
    if has_hw:
        from sit.hardware.runner import run_hardware_pipeline

        click.echo("Running hardware validation pipeline...")
        hw_results = run_hardware_pipeline(ctx, config)

        # Finalize
        artifacts_df = finalize_run(ctx)
        a_path = artifact_path(ctx.run_id, ctx.output_dir, fmt)
        write_dataframe(artifacts_df, a_path, fmt)

        click.echo("")
        click.echo("=" * 60)
        click.echo("SIT Hardware Run Complete")
        click.echo("=" * 60)
        click.echo(f"  Run ID:       {ctx.run_id}")
        click.echo(f"  Config hash:  {ctx.config_hash}")
        click.echo(f"  Git commit:   {ctx.git_commit}")
        click.echo(f"  Mode:         hardware")
        click.echo(f"  Artifacts:    {len(artifacts_df)}")
        click.echo(f"  Output dir:   {ctx.raw_path}")

        gate_results = hw_results.get("gates", {})
        if gate_results:
            click.echo("")
            click.echo("Hardware gates:")
            for gate_name, gate_result in gate_results.items():
                if isinstance(gate_result, dict):
                    status = gate_result.get("status", "N/A")
                    click.echo(f"  {gate_name}: {status}")

        click.echo("=" * 60)
        return

    if has_sim:
        # Full simulator pipeline
        from sit.sim.pipeline import (
            run_sim_trials,
            run_sim_decisions,
            run_irbs_evaluation,
            run_ci_coverage_evaluation,
        )
        from sit.sim.world import build_world, ground_truth_to_dataframe

        rng = np.random.RandomState(config["seed"])
        world = build_world(config, rng)

        # Use separate RNG streams for each phase (deterministic)
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

        # Ground truth
        gt_df = ground_truth_to_dataframe(world)

        # IRBS evaluation
        events_df, irbs_df = run_irbs_evaluation(world, config, irbs_rng)

        # CI coverage evaluation
        ci_coverage_df = run_ci_coverage_evaluation(config, ci_rng)

        # Tomography pipeline (if configured)
        tomo_recovery_df = None
        tomo_diagnostics_df = None
        tomo_uncertainty_df = None

        if has_tomo:
            from sit.tomography.pipeline import run_tomography

            tomo_rng = np.random.RandomState(config["seed"] + 5)
            tomo_recovery_df, tomo_diagnostics_df, tomo_uncertainty_df = \
                run_tomography(world, config, tomo_rng)

    else:
        # Smoke generator (no sim section)
        trials_df, decisions_df = generate_smoke_trials_and_decisions(
            config=config,
            run_id=ctx.run_id,
            config_hash=ctx.config_hash,
            git_commit=ctx.git_commit,
        )
        trials_df["created_at_utc"] = timestamp
        decisions_df["created_at_utc"] = timestamp
        gt_df = None
        events_df = None
        irbs_df = None
        ci_coverage_df = None
        tomo_recovery_df = None
        tomo_diagnostics_df = None
        tomo_uncertainty_df = None

    # Validate schema
    trials_df = validate_trials(trials_df)
    decisions_df = validate_decisions(decisions_df)

    # Validate metrics
    report = validate_trial_metrics(
        trials_df,
        strict_mode=config["strict_mode"],
        latency_unit=config["latency_unit"],
    )

    # Write raw data
    t_path = trial_path(ctx.run_id, ctx.output_dir, fmt)
    d_path = decision_path(ctx.run_id, ctx.output_dir, fmt)
    write_dataframe(trials_df, t_path, fmt)
    write_dataframe(decisions_df, d_path, fmt)

    # Register artifacts
    register_artifact(ctx, t_path, "raw")
    register_artifact(ctx, d_path, "raw")

    # Write derived data if sim mode
    derived_count = 0
    if has_sim:
        import os
        derived_base = ctx.derived_path
        os.makedirs(derived_base, exist_ok=True)

        if gt_df is not None and len(gt_df) > 0:
            gt_path = os.path.join(derived_base, f"ground_truth.{fmt}")
            write_dataframe(gt_df, gt_path, fmt)
            register_artifact(ctx, gt_path, "derived")
            derived_count += 1

        if events_df is not None and len(events_df) > 0:
            ev_path = os.path.join(derived_base, f"measurement_events.{fmt}")
            write_dataframe(events_df, ev_path, fmt)
            register_artifact(ctx, ev_path, "derived")
            derived_count += 1

        if irbs_df is not None and len(irbs_df) > 0:
            ir_path = os.path.join(derived_base, f"irbs_estimates.{fmt}")
            write_dataframe(irbs_df, ir_path, fmt)
            register_artifact(ctx, ir_path, "derived")
            derived_count += 1

        if ci_coverage_df is not None and len(ci_coverage_df) > 0:
            ci_path = os.path.join(derived_base, f"tail_ci_coverage.{fmt}")
            write_dataframe(ci_coverage_df, ci_path, fmt)
            register_artifact(ctx, ci_path, "derived")
            derived_count += 1

        if tomo_recovery_df is not None and len(tomo_recovery_df) > 0:
            tr_path = os.path.join(derived_base, f"tomo_recovery.{fmt}")
            write_dataframe(tomo_recovery_df, tr_path, fmt)
            register_artifact(ctx, tr_path, "derived")
            derived_count += 1

        if tomo_diagnostics_df is not None and len(tomo_diagnostics_df) > 0:
            td_path = os.path.join(derived_base, f"tomo_diagnostics.{fmt}")
            write_dataframe(tomo_diagnostics_df, td_path, fmt)
            register_artifact(ctx, td_path, "derived")
            derived_count += 1

        if tomo_uncertainty_df is not None and len(tomo_uncertainty_df) > 0:
            tu_path = os.path.join(derived_base, f"tomo_uncertainty.{fmt}")
            write_dataframe(tomo_uncertainty_df, tu_path, fmt)
            register_artifact(ctx, tu_path, "derived")
            derived_count += 1

    # Run stats pipeline if configured
    has_stats = config.get("run_mode") == "stats" or "stats" in config
    stats_results = None
    if has_stats and has_sim:
        from sit.stats.pipeline import run_stats_pipeline

        click.echo("")
        click.echo("Running statistical validation pipeline...")
        stats_results = run_stats_pipeline(ctx, config, trials_df)
        derived_count += len(stats_results.get("artifacts", []))

    # Finalize: write artifacts manifest
    artifacts_df = finalize_run(ctx)
    a_path = artifact_path(ctx.run_id, ctx.output_dir, fmt)
    write_dataframe(artifacts_df, a_path, fmt)

    # Print summary
    click.echo("")
    click.echo("=" * 60)
    click.echo("SIT Run Complete")
    click.echo("=" * 60)
    click.echo(f"  Run ID:       {ctx.run_id}")
    click.echo(f"  Config hash:  {ctx.config_hash}")
    click.echo(f"  Git commit:   {ctx.git_commit}")
    click.echo(f"  Seed:         {ctx.seed}")
    mode = "stats" if has_stats else ("tomography" if has_tomo else ("simulator" if has_sim else "smoke"))
    click.echo(f"  Mode:         {mode}")
    click.echo(f"  Trials:       {len(trials_df)}")
    click.echo(f"  Decisions:    {len(decisions_df)}")
    click.echo(f"  Artifacts:    {len(artifacts_df)}")
    if has_sim:
        click.echo(f"  Derived:      {derived_count} files")
    click.echo(f"  Output dir:   {ctx.raw_path}")
    click.echo(f"  Latency unit: {CANONICAL_LATENCY_UNIT}")
    click.echo(f"  Schema ver:   {SCHEMA_VERSION}")
    click.echo("")
    click.echo("Validation:")
    click.echo(f"  {report.summary()}")

    # Print stats gates if applicable
    if stats_results is not None:
        gate_results = stats_results.get("gates", {})
        if gate_results:
            click.echo("")
            click.echo("Statistical gates:")
            for gate_name, gate_result in gate_results.items():
                if isinstance(gate_result, dict):
                    status = gate_result.get("overall", "N/A")
                    click.echo(f"  {gate_name}: {status}")

    # Run tomography gates if applicable
    if has_tomo and tomo_recovery_df is not None and len(tomo_recovery_df) > 0:
        from sit.eval.gates import gate_topk_recovery, gate_relative_error
        tomo_gates_cfg = config.get("tomography", {}).get("gates", {})

        click.echo("")
        click.echo("Tomography gates:")
        try:
            topk_pass = gate_topk_recovery(
                tomo_recovery_df,
                target_recall=tomo_gates_cfg.get("target_topk_recall", 0.95),
                target_ndcg=tomo_gates_cfg.get("target_ndcg", 0.95),
            )
            click.echo(f"  top-k recovery: {'PASS' if topk_pass else 'FAIL'}")
        except Exception as e:
            click.echo(f"  top-k recovery: ERROR ({e})")

        try:
            rel_pass = gate_relative_error(
                tomo_recovery_df,
                max_rel_l2=tomo_gates_cfg.get("max_relative_l2", 0.10),
            )
            click.echo(f"  relative L2:    {'PASS' if rel_pass else 'FAIL'}")
        except Exception as e:
            click.echo(f"  relative L2:    ERROR ({e})")

    click.echo("=" * 60)


@cli.command()
@click.option("--run-id", required=True, help="Run ID to validate.")
@click.option("--base-dir", default="data", help="Base data directory.")
def validate(run_id: str, base_dir: str):
    """Validate raw data for a completed run.

    Loads trials, decisions, and artifacts tables and runs schema
    and metric validation checks.
    """
    setup_logging(level="INFO")
    logger = get_logger("cli.validate")

    # Find data files
    raw = raw_dir(run_id, base_dir)
    if not os.path.isdir(raw):
        click.echo(f"Error: No raw data directory found at {raw}", err=True)
        sys.exit(1)

    # Detect format
    fmt = "parquet"
    t_path = trial_path(run_id, base_dir, fmt)
    if not os.path.exists(t_path):
        fmt = "csv"
        t_path = trial_path(run_id, base_dir, fmt)

    if not os.path.exists(t_path):
        click.echo(f"Error: No trials file found in {raw}", err=True)
        sys.exit(1)

    d_path = decision_path(run_id, base_dir, fmt)
    a_path = artifact_path(run_id, base_dir, fmt)

    click.echo(f"Validating run: {run_id}")
    click.echo(f"  Format: {fmt}")
    click.echo("")

    # Load and validate trials
    trials_df = read_dataframe(t_path)
    trials_df = validate_trials(trials_df)
    report = validate_trial_metrics(trials_df, strict_mode=False)

    click.echo("Trials validation:")
    click.echo(f"  {report.summary()}")
    click.echo("")

    # Load and validate decisions
    if os.path.exists(d_path):
        decisions_df = read_dataframe(d_path)
        decisions_df = validate_decisions(decisions_df)
        click.echo(f"Decisions: {len(decisions_df)} rows - schema OK")
    else:
        click.echo("Decisions: file not found (skipping)")

    click.echo("")

    # Load and validate artifacts
    if os.path.exists(a_path):
        artifacts_df = read_dataframe(a_path)
        artifacts_df = validate_artifacts(artifacts_df)
        click.echo(f"Artifacts: {len(artifacts_df)} rows - schema OK")
    else:
        click.echo("Artifacts: file not found (skipping)")

    click.echo("")
    if report.is_clean:
        click.echo("RESULT: All validations passed.")
    else:
        click.echo("RESULT: Validation issues found (see above).")
        sys.exit(1)


@cli.command()
@click.option("--run-id", required=True, help="Run ID to inspect.")
@click.option("--base-dir", default="data", help="Base data directory.")
def info(run_id: str, base_dir: str):
    """Show metadata and counts for a completed run.

    Displays config hash, seed, git commit, and row counts.
    """
    setup_logging(level="WARNING")

    raw = raw_dir(run_id, base_dir)
    if not os.path.isdir(raw):
        click.echo(f"Error: No raw data directory found at {raw}", err=True)
        sys.exit(1)

    # Detect format
    fmt = "parquet"
    t_path = trial_path(run_id, base_dir, fmt)
    if not os.path.exists(t_path):
        fmt = "csv"
        t_path = trial_path(run_id, base_dir, fmt)

    click.echo(f"Run ID:    {run_id}")
    click.echo(f"Format:    {fmt}")
    click.echo(f"Raw dir:   {raw}")

    if os.path.exists(t_path):
        trials_df = read_dataframe(t_path)
        # Extract metadata from first row
        if len(trials_df) > 0:
            row = trials_df.iloc[0]
            click.echo(f"Config hash: {row.get('config_hash', 'N/A')}")
            click.echo(f"Git commit:  {row.get('git_commit', 'N/A')}")
            click.echo(f"Seed:        {row.get('seed', 'N/A')}")
        click.echo(f"Trials:      {len(trials_df)} rows")
    else:
        click.echo("Trials:    not found")

    d_path = decision_path(run_id, base_dir, fmt)
    if os.path.exists(d_path):
        decisions_df = read_dataframe(d_path)
        click.echo(f"Decisions:   {len(decisions_df)} rows")
    else:
        click.echo("Decisions: not found")

    a_path = artifact_path(run_id, base_dir, fmt)
    if os.path.exists(a_path):
        artifacts_df = read_dataframe(a_path)
        click.echo(f"Artifacts:   {len(artifacts_df)} rows")
    else:
        click.echo("Artifacts: not found")


@cli.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to repro YAML configuration file.",
)
def repro(config_path: str):
    """Run the full reproducibility pipeline.

    Executes all SIT pipeline stages sequentially, builds artifact manifest,
    runs decision replay, generates docs, and evaluates repro gates.

    Usage: sit repro --config configs/repro.yaml
    """
    reset_logging()

    from sit.repro.runner import run_repro_pipeline

    config = load_config(config_path)

    click.echo("Running reproducibility pipeline...")
    click.echo("This will execute all stages sequentially.")
    click.echo("")

    repro_results = run_repro_pipeline(config)

    click.echo("")
    click.echo("=" * 60)
    click.echo("SIT Reproducibility Pipeline Complete")
    click.echo("=" * 60)
    click.echo(f"  Master Run ID: {repro_results['master_run_id']}")
    click.echo(f"  Manifest:      {repro_results['manifest_count']} artifacts")
    click.echo(f"  Manifest path: {repro_results['manifest_path']}")

    # Print stage statuses
    click.echo("")
    click.echo("Stage results:")
    for stage_name, stage_info in repro_results.get("stage_results", {}).items():
        status = stage_info.get("status", "N/A")
        n_art = stage_info.get("n_artifacts", 0)
        click.echo(f"  {stage_name}: {status} ({n_art} artifacts)")

    # Print gate results
    gate_results = repro_results.get("gates", {})
    if gate_results:
        click.echo("")
        click.echo("Reproducibility gates:")
        for gate_name, gate_result in gate_results.items():
            if isinstance(gate_result, dict):
                status = "PASS" if gate_result.get("passed", False) else "FAIL"
                details = gate_result.get("details", "")
                click.echo(f"  {gate_name}: {status} ({details})")

    # Print replay summary
    replay = repro_results.get("replay")
    if replay:
        click.echo("")
        click.echo("Decision replay:")
        click.echo(f"  Probe replays:  {len(replay.get('probe_replays', []))}")
        click.echo(f"  Sched replays:  {len(replay.get('sched_replays', []))}")
        click.echo(f"  Load replays:   {len(replay.get('load_replays', []))}")
        click.echo(f"  All match:      {replay.get('all_match', 'N/A')}")

    click.echo("")
    click.echo("Generated docs:")
    click.echo("  docs/repro.md      - Reproducibility binder")
    click.echo("  docs/figures.md    - Figure manifest")
    click.echo("  docs/interview.md  - Interview preparation")
    click.echo("  results/tables/final_summary.csv")
    click.echo("  results/reports/summary.md")
    click.echo("=" * 60)


def main():
    """Entry point for the SIT CLI."""
    cli()


if __name__ == "__main__":
    main()
