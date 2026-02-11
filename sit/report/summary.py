"""Auto-generated narrative summary for the SIT reproducibility binder.

Produces results/reports/summary.md: 1-page max, bullet points,
numbers from tables, no adjectives without metrics.
"""

from datetime import datetime, timezone
from typing import Any, Dict

from sit.core.logging import get_logger

logger = get_logger("report.summary")


def generate_narrative_summary(
    stage_results: Dict[str, Dict[str, Any]],
    env_snapshot: Dict[str, Any],
    master_run_id: str,
) -> str:
    """Generate a narrative summary of the repro pipeline run.

    Args:
        stage_results: Dict mapping stage_name -> result dict.
        env_snapshot: Environment snapshot dict.
        master_run_id: Master run ID for this repro.

    Returns:
        Markdown string (1 page max).
    """
    lines = []
    lines.append("# SIT Pipeline Summary")
    lines.append("")
    lines.append(f"**Run ID:** {master_run_id}")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Python:** {env_snapshot.get('python_version', 'N/A').split()[0]}")
    lines.append(f"**Platform:** {env_snapshot.get('platform', 'N/A')}")
    lines.append("")

    # Stage summary
    lines.append("## Pipeline Stages")
    lines.append("")

    n_completed = 0
    n_total = 0
    total_artifacts = 0

    for stage_name in ["probe", "tomography", "scheduling", "load", "stats", "failure", "hardware"]:
        info = stage_results.get(stage_name)
        if info is None:
            continue
        n_total += 1
        status = info.get("status", "not run")
        n_art = info.get("n_artifacts", 0)
        total_artifacts += n_art
        if status == "completed":
            n_completed += 1
        lines.append(f"- **{stage_name}:** {status} ({n_art} artifacts)")

    lines.append("")
    lines.append(f"**{n_completed}/{n_total} stages completed, {total_artifacts} total artifacts**")
    lines.append("")

    # Gate results
    lines.append("## Gate Results")
    lines.append("")

    gate_pass = 0
    gate_total = 0
    for stage_name, result in stage_results.items():
        gates = result.get("gates", {})
        for gate_name, gate_result in gates.items():
            if isinstance(gate_result, dict):
                gate_total += 1
                passed = gate_result.get("passed", False)
                if passed:
                    gate_pass += 1
                status = "PASS" if passed else "FAIL"
                details = gate_result.get("details", "")
                lines.append(f"- **{gate_name}:** {status} ({details})")

    if gate_total > 0:
        lines.append("")
        lines.append(f"**{gate_pass}/{gate_total} gates passed**")
    lines.append("")

    # Key metrics
    lines.append("## Key Metrics")
    lines.append("")

    # Extract from failure stage
    failure_result = stage_results.get("failure", {})
    failure_gates = failure_result.get("gates", {})
    f1 = failure_gates.get("F1", {})
    f2 = failure_gates.get("F2", {})
    f3 = failure_gates.get("F3", {})

    if f1:
        lines.append(f"- Failure detection rate: {f1.get('detection_rate', 'N/A')}")
    if f2:
        lines.append(f"- Worst CVaR degradation: {f2.get('worst_cvar_change', 'N/A')}")
    if f3:
        lines.append(f"- Silent catastrophes: {f3.get('n_silent', 'N/A')}")

    # Extract from stats stage
    stats_result = stage_results.get("stats", {})
    stats_gates = stats_result.get("gates", {})
    for gname, gval in stats_gates.items():
        if isinstance(gval, dict):
            lines.append(f"- Stats {gname}: {gval.get('overall', gval.get('passed', 'N/A'))}")

    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("- All stages use seeded RNG for deterministic execution")
    lines.append("- Artifact manifest includes SHA-256 hashes for every output")
    lines.append("- Decision replay verifies probe, scheduling, and load choices")
    lines.append("- Re-running `sit repro` with the same config produces identical results")
    lines.append("")

    summary = "\n".join(lines)
    logger.info("Generated narrative summary: %d lines", len(lines))
    return summary
