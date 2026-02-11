# SIT Reproducibility Binder

## 1. Overview

SIT (Spectator Interference Tomography) is a system for measuring,
modeling, and mitigating tail-latency interference in shared computing
environments. This binder documents a complete, reproducible pipeline run
with all artifacts, hashes, and verification steps.

This binder contains: environment snapshot, pipeline stage results,
artifact manifest with SHA-256 hashes, decision replay verification,
and a verification checklist for external reviewers.

## 2. Exact Rerun Command

```bash
sit repro --config configs/repro.yaml
```

## 3. Environment

- **OS:** Linux (#1 SMP Sun Jan 10 15:06:54 PST 2016)
- **Python:** 3.11.14
- **CPU:** unknown
- **Platform:** Linux-4.4.0-x86_64-with-glibc2.39
- **click:** 8.3.1
- **numpy:** 2.4.2
- **pandas:** 3.0.0
- **scipy:** 1.17.0
- **yaml:** 6.0.1

Note: Hardware performance counters are optional and may not be
available on all platforms.

## 4. Pipeline Stages

| Stage | Run ID | Config Hash | Status |
|-------|--------|-------------|--------|
| probe | 25513fd8-b1a... | 6e0ee932f6c4... | completed |
| tomography | 22e8adc7-eae... | 0262ee5d7577... | completed |
| scheduling | 055d375d-95f... | 6e0ee932f6c4... | completed |
| load | ab8f1ffb-c07... | 6e0ee932f6c4... | completed |
| stats | 72cfc673-713... | 070759be4c19... | completed |
| failure | 3d8b07f1-eae... | 94a114aee0df... | completed |
| hardware |  |  | skipped |

## 5. Artifacts

Total artifacts: 57

Manifest location: `data/derived/9a9a5e47-8d13-4ddc-81f2-0eca039cfff4/artifact_manifest.parquet`

| Type | Count |
|------|-------|
| derived | 30 |
| raw | 18 |
| report | 9 |

## 6. Known Limitations

- Hardware validation requires physical access to bare-metal servers
  with performance counters enabled. The hardware stage is optional.
- Floating-point results may vary across platforms due to differences
  in math library implementations. Tolerance: epsilon=1e-08
- Parquet file hashes are byte-exact; numeric comparisons use epsilon tolerance.

## 7. Verification Checklist

For a reviewer to verify results:

1. Clone the repository and install dependencies
2. Run: `sit repro --config configs/repro.yaml`
3. Check that all stages report 'completed' status
4. Verify the artifact manifest exists and all hashes match
5. Check that gates R0 (reproducibility), R1 (manifest), R2 (replay) pass
6. Review results/tables/final_summary.csv for key metrics
7. Review results/reports/summary.md for narrative interpretation

Master run ID: `9a9a5e47-8d13-4ddc-81f2-0eca039cfff4`

Generated at: 2026-02-11T04:09:21.298858+00:00
