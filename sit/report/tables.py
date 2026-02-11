"""Auto-generated summary tables for the SIT reproducibility binder.

Produces results/tables/final_summary.csv with key metrics from all stages.
"""

from typing import Any, Dict

import pandas as pd

from sit.core.logging import get_logger

logger = get_logger("report.tables")


def generate_final_summary_table(
    stage_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Generate the final summary table from all stage results.

    Extracts key metrics from each stage and assembles them into
    a single CSV-friendly DataFrame.

    Args:
        stage_results: Dict mapping stage_name -> result dict.

    Returns:
        DataFrame with metric_name, value, stage, gate columns.
    """
    rows = []

    for stage_name, result in stage_results.items():
        status = result.get("status", "not run")
        n_artifacts = result.get("n_artifacts", 0)

        rows.append({
            "metric_name": f"{stage_name}_status",
            "value": status,
            "stage": stage_name,
            "gate": "",
        })
        rows.append({
            "metric_name": f"{stage_name}_n_artifacts",
            "value": str(n_artifacts),
            "stage": stage_name,
            "gate": "",
        })

        # Extract gate results
        gates = result.get("gates", {})
        for gate_name, gate_result in gates.items():
            if isinstance(gate_result, dict):
                passed = gate_result.get("passed", gate_result.get("overall", "N/A"))
                rows.append({
                    "metric_name": f"{stage_name}_{gate_name}",
                    "value": str(passed),
                    "stage": stage_name,
                    "gate": gate_name,
                })

                # Extract numeric details
                for key in ["detection_rate", "worst_cvar_change", "n_silent",
                            "mean_coverage", "mean_recall", "ratio"]:
                    if key in gate_result:
                        rows.append({
                            "metric_name": f"{stage_name}_{gate_name}_{key}",
                            "value": str(gate_result[key]),
                            "stage": stage_name,
                            "gate": gate_name,
                        })

    # Add summary metrics from stage data
    _add_computed_metrics(rows, stage_results)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        df = pd.DataFrame(columns=["metric_name", "value", "stage", "gate"])

    logger.info("Generated final summary table: %d rows", len(df))
    return df


def _add_computed_metrics(rows, stage_results):
    """Add computed metrics from pipeline data."""
    # Count total stages run
    n_completed = sum(
        1 for r in stage_results.values() if r.get("status") == "completed"
    )
    n_total = len(stage_results)
    rows.append({
        "metric_name": "stages_completed",
        "value": f"{n_completed}/{n_total}",
        "stage": "summary",
        "gate": "",
    })

    # Count total artifacts
    total_artifacts = sum(
        r.get("n_artifacts", 0) for r in stage_results.values()
    )
    rows.append({
        "metric_name": "total_artifacts",
        "value": str(total_artifacts),
        "stage": "summary",
        "gate": "",
    })

    # Count gates passed
    n_gates_passed = 0
    n_gates_total = 0
    for result in stage_results.values():
        for gate_result in result.get("gates", {}).values():
            if isinstance(gate_result, dict):
                n_gates_total += 1
                if gate_result.get("passed", False):
                    n_gates_passed += 1
    rows.append({
        "metric_name": "gates_passed",
        "value": f"{n_gates_passed}/{n_gates_total}",
        "stage": "summary",
        "gate": "",
    })
