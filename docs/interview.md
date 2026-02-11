# SIT Interview Preparation

This document prepares you to defend every design decision in SIT.
Each answer references a specific table, figure, or gate result.

## 1. What problem does SIT solve?

In shared computing environments (cloud, HPC, colocation), workloads
running on adjacent hardware interfere with each other through shared
resources: last-level cache, memory bandwidth, I/O channels, TLB, and
SMT pipelines. This interference causes unpredictable tail-latency
spikes that violate service-level objectives (SLOs).

SIT solves this by: (1) measuring interference using drift-canceling
IRBS measurement (see fig_irbs_bias), (2) decomposing interference
sources via sparse tomography (see fig_tomo_recovery), and (3) making
placement decisions that minimize tail-latency risk. The system is
validated end-to-end with statistical rigor (see fig_ci_coverage).

## 2. Why averages fail and tails matter

Mean latency hides catastrophic tail behavior. A workload with 100us
mean can have 10ms p99 due to interference bursts. SIT targets CVaR99
(expected value in the worst 1% of cases) because this is the metric
that determines SLO compliance. See fig_evt_tail for EVT validation
of tail modeling accuracy. The ablation study (fig_ablation_matrix)
quantifies the CVaR impact of each SIT component.

## 3. Why drift breaks naive measurement

Naive A/B measurement compares control and treatment at different times.
In systems with temporal drift (thermal throttling, garbage collection,
background maintenance), the time gap introduces bias that confounds
the interference signal. IRBS (Interleaved Randomized Benchmarking
Sequences) uses a Control-Treatment-Control (C-T-C) design that cancels
linear drift by construction. See fig_irbs_bias: IRBS reduces estimation
bias by 3x+ compared to naive A/B measurement.

## 4. Why tomography instead of black-box ML

Black-box ML models (random forests, neural networks) can predict
interference but cannot decompose it. SIT uses sparse tomography
(elastic-net with non-negativity constraints) to identify which specific
spectators cause interference. This decomposition enables targeted
mitigation: move the toxic spectator, not the entire neighborhood.
See fig_tomo_recovery: the solver recovers top-k toxic spectators with
>= 0.95 recall and NDCG. The design matrix condition number and
coherence are monitored via diagnostics (tomo_diagnostics.parquet).

## 5. Why active probing matters

Passive observation of production traffic gives biased samples:
schedulers avoid known-bad placements, creating selection bias.
Active probing with coverage-aware design (see tomography config)
ensures every spectator is measured in diverse contexts. The probe
budget (m_probes, set_size, max_repeats) balances measurement cost
against statistical power. Coverage-aware probes achieve better
design matrix conditioning than uniform random probes.

## 6. Why partition is suboptimal

Static partitioning (pinning workloads to dedicated resources) wastes
capacity. SIT enables safe co-location by identifying which spectator
combinations are safe and which are toxic. The scheduler can pack more
workloads per machine while respecting tail-latency SLOs. The ablation
study (fig_ablation_matrix) shows that removing SIT safety constraints
(no_safety ablation) causes catastrophic degradation, confirming that
SIT's interference management adds genuine value beyond partition.

## 7. When SIT fails and how it detects failure

SIT models 7 core assumptions (see fig_failure_taxonomy):

1. **Additivity** (A1): interference is additive across spectators
2. **Sparsity** (A2): only a few spectators cause significant interference
3. **Drift smoothness** (A3): temporal drift is smooth and IRBS-cancellable
4. **Stationarity** (A4): interference distributions are stable within regimes
5. **Coverage** (A5): probes adequately cover the spectator space
6. **Tail validity** (A6): enough samples exist for reliable tail estimation
7. **Feasibility** (A7): safe placements exist under current constraints

Each assumption has an injector (adversarial violation) and detector.
Gate F1 requires 100% detection of injected violations. Gate F3 requires
zero silent catastrophes. See failure_events.parquet and
assumption_violation_report.parquet for detailed evidence.

## 8. What happens when safety constraints bind

When the SIT safety gate rejects all candidate placements, the system
falls back to the safest available option and raises an alert. The
ablation study (fig_ablation_matrix) quantifies this: removing admission
control (no_admission) or safety constraints (no_safety) causes
measurable CVaR degradation. The mitigation layer (mitigation_actions.parquet)
documents before/after metrics for each mitigation strategy applied.
The failure taxonomy (failure_taxonomy.parquet) recommends actions:
monitor, degrade, fallback, or abort based on cumulative severity.

## 9. How to reproduce every figure

Every figure and table traces to raw data through the artifact manifest:

1. Run: `sit repro --config configs/repro.yaml`
2. Check `data/derived/<run_id>/artifact_manifest.parquet` for SHA-256 hashes
3. See `docs/figures.md` for the mapping: figure -> script -> input data
4. Each input dataset has a recorded hash; verify with `sha256sum`
5. Decision replay (`sit repro` gate R2) verifies individual choices

All stages use seeded RNG. Re-running with the same config and seed
produces bit-identical results (within float tolerance epsilon=1e-8).

## 10. One-minute verbal explanation

"SIT measures how co-located workloads interfere with each other's
tail latency. We use drift-canceling measurement to get unbiased
estimates, sparse tomography to identify which specific neighbors
are toxic, and a safety-constrained scheduler to avoid catastrophic
placements. The system detects its own failures -- we inject every
known assumption violation and verify 100% detection. Every result
is reproducible from a single command, every figure traces to raw
data with SHA-256 hashes, and every decision can be replayed and
interrogated. This is not a demo -- it is an auditable scientific
instrument."
