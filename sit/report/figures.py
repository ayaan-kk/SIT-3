"""Figure manifest generation for the SIT reproducibility binder.

Auto-generates docs/figures.md documenting every figure with its
generating script, input datasets, and expected interpretation.
"""

from typing import Any, Dict, List

import pandas as pd

from sit.core.logging import get_logger

logger = get_logger("report.figures")

# Known figure definitions (maps figure_id -> metadata)
FIGURE_REGISTRY = [
    {
        "figure_id": "fig_irbs_bias",
        "description": "IRBS bias reduction vs naive A/B measurement across drift conditions",
        "generating_script": "sit/sim/pipeline.py::run_irbs_evaluation",
        "input_datasets": ["irbs_estimates.parquet"],
        "config_section": "measurement.irbs",
        "interpretation": "Shows that IRBS C-T-C measurement reduces estimation bias by 3x+ vs naive time-separated A/B, validating the drift-canceling measurement layer.",
    },
    {
        "figure_id": "fig_tomo_recovery",
        "description": "Tomography top-k recovery accuracy across targets and regimes",
        "generating_script": "sit/tomography/pipeline.py::run_tomography",
        "input_datasets": ["tomo_recovery.parquet"],
        "config_section": "tomography",
        "interpretation": "Demonstrates that elastic-net tomography recovers the top-k most interfering spectators with recall >= 0.95 and NDCG >= 0.95.",
    },
    {
        "figure_id": "fig_ci_coverage",
        "description": "Bootstrap confidence interval calibration across sample sizes",
        "generating_script": "sit/stats/pipeline.py::run_stats_pipeline (CI calibration)",
        "input_datasets": ["ci_calibration.parquet"],
        "config_section": "stats.ci",
        "interpretation": "Validates that bootstrap CIs achieve nominal coverage (>= 0.90 at alpha=0.05), confirming statistical reliability of uncertainty estimates.",
    },
    {
        "figure_id": "fig_evt_tail",
        "description": "EVT GPD fit diagnostics by threshold quantile",
        "generating_script": "sit/stats/pipeline.py::run_stats_pipeline (EVT)",
        "input_datasets": ["tail_validity_evt.parquet"],
        "config_section": "stats.evt",
        "interpretation": "Shows GPD fit quality across threshold choices, identifying the optimal threshold for tail extrapolation.",
    },
    {
        "figure_id": "fig_ablation_matrix",
        "description": "Ablation study: CVaR impact of removing each SIT component",
        "generating_script": "sit/failure/pipeline.py::run_failure_pipeline (ablation)",
        "input_datasets": ["ablation_results.parquet"],
        "config_section": "ablations",
        "interpretation": "Quantifies the contribution of each SIT component. Adversarial ablations (no_safety, no_irbs) show large degradation; fallback ablations remain within tolerance.",
    },
    {
        "figure_id": "fig_failure_taxonomy",
        "description": "Failure taxonomy: detected violations by category and severity",
        "generating_script": "sit/failure/pipeline.py::run_failure_pipeline (taxonomy)",
        "input_datasets": ["failure_taxonomy.parquet", "failure_events.parquet"],
        "config_section": "injectors",
        "interpretation": "Maps all 7 assumption violations to structural/statistical/control categories with severity-based action recommendations.",
    },
    {
        "figure_id": "fig_robustness_sweep",
        "description": "Robustness sweep: metric stability across parameter perturbations",
        "generating_script": "sit/stats/pipeline.py::run_stats_pipeline (robustness)",
        "input_datasets": ["robustness_sweeps.parquet"],
        "config_section": "stats.robustness",
        "interpretation": "Demonstrates that SIT metrics remain stable across reasonable parameter variations (beta, lambda, tau), confirming robustness of conclusions.",
    },
]


def generate_figure_manifest(
    stage_results: Dict[str, Dict[str, Any]],
    manifest_df: pd.DataFrame,
) -> str:
    """Generate docs/figures.md - the figure manifest.

    Args:
        stage_results: Dict mapping stage_name -> result dict.
        manifest_df: Full artifact manifest.

    Returns:
        Markdown string documenting all figures.
    """
    lines = []
    lines.append("# SIT Figure Manifest")
    lines.append("")
    lines.append("This document maps every figure and table to its generating script,")
    lines.append("input datasets, and expected interpretation. Every claim in the")
    lines.append("SIT system is traceable to raw data through this manifest.")
    lines.append("")

    for fig in FIGURE_REGISTRY:
        lines.append(f"## {fig['figure_id']}")
        lines.append("")
        lines.append(f"**Description:** {fig['description']}")
        lines.append("")
        lines.append(f"**Generating script:** `{fig['generating_script']}`")
        lines.append("")
        lines.append(f"**Config section:** `{fig['config_section']}`")
        lines.append("")
        lines.append("**Input datasets:**")
        for ds in fig["input_datasets"]:
            # Find hash from manifest
            match = manifest_df[manifest_df["artifact_path"].str.endswith(ds)] if len(manifest_df) > 0 else pd.DataFrame()
            if len(match) > 0:
                ds_hash = match.iloc[0]["sha256"][:16] + "..."
                ds_path = match.iloc[0]["artifact_path"]
                lines.append(f"- `{ds_path}` (sha256: `{ds_hash}`)")
            else:
                lines.append(f"- `{ds}` (not yet generated)")
        lines.append("")
        lines.append(f"**Interpretation:** {fig['interpretation']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Figure ID | Input Dataset | Generated By |")
    lines.append("|-----------|---------------|-------------|")
    for fig in FIGURE_REGISTRY:
        datasets = ", ".join(fig["input_datasets"])
        lines.append(f"| {fig['figure_id']} | {datasets} | {fig['generating_script'].split('::')[0]} |")
    lines.append("")

    md = "\n".join(lines)
    logger.info("Generated figure manifest: %d figures", len(FIGURE_REGISTRY))
    return md
