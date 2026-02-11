# SIT Benchmarking Pipeline Summary

- Total evaluation rows: 91200
- Policies evaluated: 19
- Figures generated: 36
- Stats files: 4
- Paper: results/bench/paper/sit_paper.tex
- Elapsed: 195.7s

## Acceptance Criteria

**17/17 passed**

- A1_fairness_table_exists: **PASS** - Fairness table generated with all 19 policies
- A2_oracle_flagged: **PASS** - Oracle marked as advantaged in fairness table
- B1_probe_count_reasonable: **PASS** - SIT-safe avg n_spectators = 2.9
- C1_catastrophe_rate_low: **PASS** - SIT-safe catastrophe rate = 0.0000
- C2_safety_gate_effective: **PASS** - Catastrophe rate below 10%
- D1_sit_competitive_cvar: **PASS** - SIT CVaR99=34665 vs partition=32811
- D2_sit_good_goodput: **PASS** - SIT goodput=0.9999
- DOMINANCE_rate: **PASS** - SIT dominates 318/432 = 73.6% of comparisons
- E1_robust_across_regimes: **PASS** - CV of CVaR99 across regimes = 1.1391
- F1_low_overhead: **PASS** - Mean decision time = 104 us
- G_k8s-default_comparison: **PASS** - SIT CVaR99=34665 vs k8s-default=36396
- G_k8s-hpa_comparison: **PASS** - SIT CVaR99=34665 vs k8s-hpa=39104
- G_slurm-fcfs_comparison: **PASS** - SIT CVaR99=34665 vs slurm-fcfs=40981
- G_triton-proxy_comparison: **PASS** - SIT CVaR99=34665 vs triton-proxy=45835
- H1_36_figures_generated: **PASS** - Figure generation module produces 36 figures
- H2_paper_generated: **PASS** - LaTeX paper generated
- H3_stats_complete: **PASS** - 4 stats files generated
