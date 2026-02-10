# SIT Reproducibility Binder

## Overview

Every SIT run produces a complete reproducibility record:

- **run_id**: UUID4 uniquely identifying each run
- **config_hash**: SHA-256 of the normalized configuration
- **git_commit**: Short hash of the code version
- **seed**: Deterministic PRNG seed
- **created_at_utc**: ISO timestamp of run creation

## One-Command Rerun

```bash
sit run --config configs/smoke.yaml
```

## Artifact Registry

Every output file is registered in `artifacts.parquet` with:
- File path
- SHA-256 content hash
- File size in bytes
- Creation timestamp
- Type (raw / derived / figure / report)

## Deterministic Guarantees

With identical seed and config:
1. Trial data is byte-identical
2. Decision logs are byte-identical
3. Artifact hashes match exactly

## Validation

```bash
sit validate --run-id <RUN_ID>
```

Checks metric ordering, finiteness, unit consistency, and suspicious ratios.
