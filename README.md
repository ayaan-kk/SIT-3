# SIT: Spectator Interference Tomography

A systems software framework for measuring, modeling, and mitigating co-location interference in shared computing environments.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run smoke test
sit run --config configs/smoke.yaml

# Validate a run
sit validate --run-id <RUN_ID>

# Inspect run metadata
sit info --run-id <RUN_ID>
```

## Project Structure

```
sit/
  cli.py              # CLI entry point
  core/
    units.py           # Canonical unit system
    schema.py          # Data schemas and validation
    hashing.py         # Deterministic hashing
    config.py          # Config loading and normalization
    registry.py        # Experiment registry and run lifecycle
    logging.py         # Structured logging
    validation.py      # Metric validation and smoke generator
  data/
    io.py              # DataFrame I/O (Parquet/CSV)
    paths.py           # Output path management
  eval/
    gates.py           # Acceptance gate placeholders
  tests/               # Test suite
configs/
  base.yaml            # Base configuration
  smoke.yaml           # Smoke test configuration
```

## Design Principles

1. **Unit consistency**: All latency in microseconds (us), enforced by assertions
2. **Deterministic reproducibility**: Same seed + config = identical outputs
3. **Auditability**: Every decision logged with full context
4. **Fail-fast**: Metric inconsistencies crash the run, not silently proceed
5. **Raw data first**: Per-trial and per-decision exports, never aggregate-only
