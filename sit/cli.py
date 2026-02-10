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

    Creates a new run, generates data (smoke or full), validates,
    writes raw tables, and produces an artifacts manifest.
    """
    # Reset logging for fresh run
    reset_logging()

    # Load config
    config = load_config(config_path)

    # Start run
    ctx = start_run(config)
    logger = get_logger("cli.run")
    logger.info("Starting SIT run with config: %s", config_path)

    fmt = config["export_format"]
    timestamp = ctx.created_at_utc

    # Generate smoke data
    trials_df, decisions_df = generate_smoke_trials_and_decisions(
        config=config,
        run_id=ctx.run_id,
        config_hash=ctx.config_hash,
        git_commit=ctx.git_commit,
    )

    # Fill in timestamps
    trials_df["created_at_utc"] = timestamp
    decisions_df["created_at_utc"] = timestamp

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

    # Finalize: write artifacts manifest
    artifacts_df = finalize_run(ctx)
    a_path = artifact_path(ctx.run_id, ctx.output_dir, fmt)
    write_dataframe(artifacts_df, a_path, fmt)

    # Re-register the manifest itself (chicken-and-egg: manifest doesn't include itself)
    # This is intentional - the manifest records pre-manifest artifacts only.

    # Print summary
    click.echo("")
    click.echo("=" * 60)
    click.echo("SIT Run Complete")
    click.echo("=" * 60)
    click.echo(f"  Run ID:       {ctx.run_id}")
    click.echo(f"  Config hash:  {ctx.config_hash}")
    click.echo(f"  Git commit:   {ctx.git_commit}")
    click.echo(f"  Seed:         {ctx.seed}")
    click.echo(f"  Trials:       {len(trials_df)}")
    click.echo(f"  Decisions:    {len(decisions_df)}")
    click.echo(f"  Artifacts:    {len(artifacts_df)}")
    click.echo(f"  Output dir:   {ctx.raw_path}")
    click.echo(f"  Latency unit: {CANONICAL_LATENCY_UNIT}")
    click.echo(f"  Schema ver:   {SCHEMA_VERSION}")
    click.echo("")
    click.echo("Validation:")
    click.echo(f"  {report.summary()}")
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


def main():
    """Entry point for the SIT CLI."""
    cli()


if __name__ == "__main__":
    main()
