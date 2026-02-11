# SIT Benchmarking Pipeline Summary

- Total evaluation rows: 91200
- Policies evaluated: 19
- Figures generated: 36
- Stats files: 4
- Paper: results/bench/paper/sit_paper.tex
- Elapsed: 198.8s

## Acceptance Criteria

**19/19 passed**

- A1_fairness_table_exists: **PASS** - Fairness table generated with all 19 policies
- A2_oracle_flagged: **PASS** - Oracle marked as advantaged in fairness table
- B1_probe_count_reasonable: **PASS** - SIT-safe avg n_spectators = 2.9
- C1_catastrophe_rate_low: **PASS** - SIT-safe catastrophe rate = 0.0000
- C2_safety_gate_effective: **PASS** - Catastrophe rate below 10%
- D1_sit_competitive_cvar: **PASS** - SIT CVaR99=41348 vs partition=36313 (ratio=1.14)
- D2_sit_higher_effective_goodput: **PASS** - SIT eff_goodput=8396 vs partition=5952 (ratio=1.41x)
- D3_cvar_exceeds_p99: **PASS** - SIT CVaR99=41348 > p99=33522
- DOMINANCE_rate: **PASS** - SIT dominates 311/432 = 72.0% of comparisons
- E1_robust_across_regimes: **PASS** - CV of CVaR99 across regimes = 1.2759
- F1_low_overhead: **PASS** - Mean decision time = 107 us
- G_k8s-default_comparison: **PASS** - SIT CVaR99=41348 vs k8s-default=40767, SIT goodput=8396 vs k8s-default=7573 rps
- G_k8s-hpa_comparison: **PASS** - SIT CVaR99=41348 vs k8s-hpa=55896, SIT goodput=8396 vs k8s-hpa=7628 rps
- G_oracle_is_upper_bound: **PASS** - Oracle CVaR99=44317 vs SIT=41348
- G_slurm-fcfs_comparison: **PASS** - SIT CVaR99=41348 vs slurm-fcfs=49362, SIT goodput=8396 vs slurm-fcfs=5270 rps
- G_triton-proxy_comparison: **PASS** - SIT CVaR99=41348 vs triton-proxy=57184, SIT goodput=8396 vs triton-proxy=4070 rps
- H1_36_figures_generated: **PASS** - Figure generation module produces 36 figures
- H2_paper_generated: **PASS** - LaTeX paper generated
- H3_stats_complete: **PASS** - 4 stats files generated
