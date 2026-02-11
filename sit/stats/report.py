"""Judge-proof statistical reporting.

Generates results/tables/stats_summary.csv with:
- CI calibration results (coverage)
- EVT validity summary
- Significant regime buckets with effect sizes
- Worst-case buckets with mitigation recommendations

All numbers are absolute, not just percentages.
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from sit.core.logging import get_logger

logger = get_logger("stats.report")


def generate_stats_summary(
    ci_calibration_df: Optional[pd.DataFrame],
    evt_df: Optional[pd.DataFrame],
    fdr_df: Optional[pd.DataFrame],
    robustness_df: Optional[pd.DataFrame],
    gate_results: Dict[str, Any],
    output_path: str,
) -> pd.DataFrame:
    """Generate the comprehensive stats_summary.csv.

    Combines all statistical validation results into a single
    auditable table.

    Args:
        ci_calibration_df: CI calibration results.
        evt_df: EVT diagnostics results.
        fdr_df: FDR-corrected effects.
        robustness_df: Robustness sweep results.
        gate_results: Dict of gate results (R1, R2, R3).
        output_path: Path to write the CSV.

    Returns:
        The summary DataFrame.
    """
    sections = []

    # Section 1: Gate summary
    gate_rows = []
    for gate_name, gate_result in gate_results.items():
        if isinstance(gate_result, dict):
            gate_rows.append({
                "section": "GATES",
                "item": gate_name,
                "metric": "overall",
                "value": str(gate_result.get("overall", "N/A")),
                "detail": str({k: v for k, v in gate_result.items()
                              if k not in ("gate", "overall")}),
            })
    if gate_rows:
        sections.extend(gate_rows)

    # Section 2: CI calibration
    if ci_calibration_df is not None and len(ci_calibration_df) > 0:
        for metric_name, sub in ci_calibration_df.groupby("metric_name"):
            n_total = len(sub)
            n_covered = int(sub["covered"].sum())
            emp_cov = n_covered / n_total if n_total > 0 else 0.0
            nominal = sub["nominal_level"].iloc[0] if "nominal_level" in sub.columns else 0.95
            sections.append({
                "section": "CI_CALIBRATION",
                "item": str(metric_name),
                "metric": "empirical_coverage",
                "value": f"{emp_cov:.3f}",
                "detail": f"n_covered={n_covered}, n_total={n_total}, nominal={nominal:.2f}",
            })

    # Section 3: EVT validity
    if evt_df is not None and len(evt_df) > 0:
        applicable = evt_df[evt_df.get("applicable", pd.Series(dtype=bool)) == True]  # noqa: E712
        if len(applicable) > 0:
            median_ks = float(applicable["ks_p_value"].median()) if "ks_p_value" in applicable.columns else float("nan")
            median_rel = float(applicable["relative_difference_cvar"].median()) if "relative_difference_cvar" in applicable.columns else float("nan")
            sections.append({
                "section": "EVT_VALIDITY",
                "item": "summary",
                "metric": "median_ks_p_value",
                "value": f"{median_ks:.4f}",
                "detail": f"n_applicable={len(applicable)}, n_total={len(evt_df)}",
            })
            sections.append({
                "section": "EVT_VALIDITY",
                "item": "summary",
                "metric": "median_relative_diff_cvar",
                "value": f"{median_rel:.4f}",
                "detail": "",
            })
        else:
            n_na = len(evt_df[evt_df.get("applicable", pd.Series(dtype=bool)) == False])  # noqa: E712
            sections.append({
                "section": "EVT_VALIDITY",
                "item": "summary",
                "metric": "status",
                "value": "NOT_APPLICABLE",
                "detail": f"n_non_applicable={n_na}",
            })

    # Section 4: Significant improvements (q <= threshold)
    if fdr_df is not None and len(fdr_df) > 0 and "significant" in fdr_df.columns:
        sig = fdr_df[fdr_df["significant"] == True]  # noqa: E712
        for _, row in sig.iterrows():
            bucket = row.get("bucket_id", "unknown")
            metric = row.get("metric_name", "unknown")
            q_val = row.get("q_value", float("nan"))
            effect = row.get("mean_difference", row.get("median_difference", float("nan")))
            sections.append({
                "section": "SIGNIFICANT_IMPROVEMENTS",
                "item": f"{bucket}|{metric}",
                "metric": "effect_size",
                "value": f"{effect:.4f}",
                "detail": f"q_value={q_val:.4f}, scheduler_a={row.get('scheduler_a', '')}, scheduler_b={row.get('scheduler_b', '')}",
            })

        # Worst-case buckets (where treatment is worse)
        not_sig = fdr_df[fdr_df["significant"] == False]  # noqa: E712
        if "mean_difference" in fdr_df.columns:
            worse = fdr_df[fdr_df["mean_difference"] > 0].sort_values("mean_difference", ascending=False)
            for _, row in worse.head(5).iterrows():
                bucket = row.get("bucket_id", "unknown")
                metric = row.get("metric_name", "unknown")
                effect = row.get("mean_difference", 0.0)
                sections.append({
                    "section": "WORST_CASE_BUCKETS",
                    "item": f"{bucket}|{metric}",
                    "metric": "mean_difference",
                    "value": f"{effect:.4f}",
                    "detail": "mitigation: consider per-regime tuning or fallback to baseline",
                })

    # Section 5: Robustness
    if robustness_df is not None and len(robustness_df) > 0:
        n_total = len(robustness_df)
        n_safe = int(robustness_df["zero_catastrophes"].sum())
        n_pareto = int(robustness_df["on_pareto"].sum())
        sections.append({
            "section": "ROBUSTNESS",
            "item": "sweep_summary",
            "metric": "safety_fraction",
            "value": f"{n_safe / max(n_total, 1):.3f}",
            "detail": f"n_safe={n_safe}, n_total={n_total}",
        })
        sections.append({
            "section": "ROBUSTNESS",
            "item": "sweep_summary",
            "metric": "pareto_fraction",
            "value": f"{n_pareto / max(n_total, 1):.3f}",
            "detail": f"n_pareto={n_pareto}, n_total={n_total}",
        })
        worst_cat = float(robustness_df["catastrophe_rate"].max())
        sections.append({
            "section": "ROBUSTNESS",
            "item": "sweep_summary",
            "metric": "worst_catastrophe_rate",
            "value": f"{worst_cat:.6f}",
            "detail": "",
        })

    summary_df = pd.DataFrame(sections)

    # Write
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    logger.info("Stats summary written to %s (%d rows)", output_path, len(summary_df))

    return summary_df
