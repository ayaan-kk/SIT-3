# SIT Pipeline Summary

**Run ID:** 9a9a5e47-8d13-4ddc-81f2-0eca039cfff4
**Generated:** 2026-02-11 04:09 UTC
**Python:** 3.11.14
**Platform:** Linux-4.4.0-x86_64-with-glibc2.39

## Pipeline Stages

- **probe:** completed (6 artifacts)
- **tomography:** completed (9 artifacts)
- **scheduling:** completed (6 artifacts)
- **load:** completed (6 artifacts)
- **stats:** completed (9 artifacts)
- **failure:** completed (8 artifacts)
- **hardware:** skipped (0 artifacts)

**6/7 stages completed, 44 total artifacts**

## Gate Results

- **R1:** FAIL ()
- **F1:** PASS (7/7 detected)
- **F2:** PASS (0 violations across fallback ablations)
- **F3:** PASS (0 silent out of 10 catastrophes)

**3/4 gates passed**

## Key Metrics

- Failure detection rate: 1.0
- Worst CVaR degradation: 1.9776
- Silent catastrophes: 0
- Stats R1: PASS

## Reproducibility

- All stages use seeded RNG for deterministic execution
- Artifact manifest includes SHA-256 hashes for every output
- Decision replay verifies probe, scheduling, and load choices
- Re-running `sit repro` with the same config produces identical results
