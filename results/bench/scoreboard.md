# SIT Comprehensive Scoreboard

Total episodes: 91200
**Scoreboard: 16/16 passed**

Elapsed: 239.4s

## Probe Layer
- P1 efficiency ratio: 0.3065 (target <= 0.40) **PASS**
- P2 diversity ratio: 0.2851 (target <= 0.70) **PASS**
- P3 replay: True **PASS**

## Tomography Recovery
- Median Recall@k: 1.0 (target >= 0.97) **PASS**
- Median NDCG@k: 0.9997 (target >= 0.97) **PASS**
- Median relative L2: 0.0343 (target <= 0.08) **PASS**
- CI coverage (toxic): 0.945 (target >= 0.93) **PASS**

## Scheduling Safety
- Catastrophes: 0/5000 **PASS**
- Silent catastrophes: 0 **PASS**
- Detection recall: 1.0 (target >= 0.98) **PASS**
- Detection precision: 1.0 (target >= 0.95)

## Load / Queueing
- Goodput improvement (high load): 39.2% (target +20-40%) **PASS**
- Admitted load ratio: 1.385x (target >= 1.30x) **PASS**
- CVaR reduction vs best: 47.5% (target >= 30%) **PASS**
- Pareto frontier (high load): 100.0% (target >= 60%) **PASS**

## Statistical Rigor
- CI coverage: {'cvar99': 0.92, 'mean': 0.96, 'p99': 0.94} **PASS**
- FDR controlled (q <= 0.05): True **PASS**

## Failure Modes
- Detection rate: 1.0 (7/7) **PASS**
- Graceful degradation: **PASS**
