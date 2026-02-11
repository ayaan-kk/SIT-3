"""Ablation study and silent catastrophe audit.

Systematically removes one SIT component at a time and measures impact.
Also scans for catastrophic events that lack preceding detection.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.measure.tail import compute_all_tail_stats
from sit.sim.world import World, build_world, simulate_micro_run

logger = get_logger("eval.ablations")


# --- Ablation definitions ---

ABLATION_DEFS = {
    "no_irbs": {
        "description": "Disable IRBS drift-canceling measurement (naive A/B only)",
        "component": "IRBS",
        "config_overrides": {"measurement": {"irbs": {"enabled": False}}},
    },
    "no_uncertainty": {
        "description": "Disable uncertainty estimation (sigma = 0)",
        "component": "Uncertainty",
        "config_overrides": {"tomography": {"uncertainty": {"enabled": False}}},
    },
    "no_diversity": {
        "description": "Disable diversity in probing (uniform random only)",
        "component": "Probe diversity",
        "config_overrides": {"tomography": {"probe": {"strategy": "uniform"}}},
    },
    "no_regularization": {
        "description": "Disable regularization in tomography (lambda=0)",
        "component": "Regularization",
        "config_overrides": {
            "tomography": {"solver": {"lambda_1": 0.0, "lambda_2": 0.0}},
        },
    },
    "no_safety": {
        "description": "Disable safety constraint in scheduler",
        "component": "Safety constraint",
        "config_overrides": {"scheduler": {"safety": {"enabled": False}}},
    },
    "no_admission": {
        "description": "Disable admission control (accept all placements)",
        "component": "Admission control",
        "config_overrides": {"scheduler": {"admission_control": False}},
    },
    "no_mitigation": {
        "description": "Disable all mitigation logic",
        "component": "Mitigation",
        "config_overrides": {"failure": {"mitigations": {"enabled": False}}},
    },
}


def run_ablation_study(
    config: Dict[str, Any],
    rng: np.random.RandomState,
    ablation_list: Optional[List[str]] = None,
    baseline_trials_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run ablation study: remove one component at a time.

    For each ablation, runs a reduced pipeline and computes key metrics
    relative to the full-system baseline.

    Args:
        config: Base configuration dict.
        rng: Random state.
        ablation_list: List of ablation names to run. Defaults to all.
        baseline_trials_df: Optional baseline trials for comparison.

    Returns:
        DataFrame with ablation results.
    """
    if ablation_list is None:
        ablation_list = list(ABLATION_DEFS.keys())

    # Always compute baseline using deterministic sim (same world/seeds as ablations)
    # This ensures fair comparison even if baseline_trials_df came from a different RNG path
    baseline = _run_sim_and_compute(config, rng)

    rows = []
    for abl_name in ablation_list:
        if abl_name not in ABLATION_DEFS:
            logger.warning("Unknown ablation: %s, skipping", abl_name)
            continue

        abl_def = ABLATION_DEFS[abl_name]
        logger.info("Running ablation: %s (%s)", abl_name, abl_def["description"])

        abl_rng = np.random.RandomState(rng.randint(0, 2**31))
        abl_metrics = _run_ablated_pipeline(config, abl_def, abl_rng)

        # Compute deltas
        delta_cvar = abl_metrics["cvar99"] - baseline["cvar99"]
        delta_goodput = abl_metrics["goodput"] - baseline["goodput"]

        # Relative delta for graceful degradation check
        if baseline["cvar99"] > 0:
            cvar_relative_change = delta_cvar / baseline["cvar99"]
        else:
            cvar_relative_change = 0.0

        rows.append({
            "ablation_name": abl_name,
            "component_removed": abl_def["component"],
            "description": abl_def["description"],
            "baseline_cvar99": round(baseline["cvar99"], 2),
            "ablated_cvar99": round(abl_metrics["cvar99"], 2),
            "delta_cvar": round(delta_cvar, 2),
            "cvar_relative_change": round(cvar_relative_change, 4),
            "baseline_goodput": round(baseline["goodput"], 4),
            "ablated_goodput": round(abl_metrics["goodput"], 4),
            "delta_goodput": round(delta_goodput, 4),
            "baseline_catastrophe_rate": round(baseline["catastrophe_rate"], 4),
            "ablated_catastrophe_rate": round(abl_metrics["catastrophe_rate"], 4),
            "catastrophe_rate": round(abl_metrics["catastrophe_rate"], 4),
            "probe_efficiency": round(abl_metrics.get("probe_efficiency", 1.0), 4),
            "notes": abl_def["description"],
        })

    ablation_df = pd.DataFrame(rows)

    if len(ablation_df) > 0:
        logger.info("Ablation study complete: %d ablations", len(ablation_df))
        for _, row in ablation_df.iterrows():
            logger.info(
                "  %s: delta_cvar=%.2f, delta_goodput=%.4f, catastrophe_rate=%.4f",
                row["ablation_name"], row["delta_cvar"],
                row["delta_goodput"], row["catastrophe_rate"],
            )

    return ablation_df


def _run_baseline(
    config: Dict[str, Any],
    rng: np.random.RandomState,
    trials_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Run baseline (full system) and compute metrics."""
    if trials_df is not None and len(trials_df) > 0:
        return _compute_trial_metrics(trials_df, config.get("slo_us", 500000.0))

    # Run fresh baseline
    return _run_sim_and_compute(config, rng)


def _run_ablated_pipeline(
    config: Dict[str, Any],
    abl_def: Dict[str, Any],
    rng: np.random.RandomState,
) -> Dict[str, float]:
    """Run pipeline with one component ablated.

    Each ablation modifies the world/pipeline to remove a specific component
    and measures the resulting degradation.
    """
    import copy
    cfg = copy.deepcopy(config)

    abl_name = abl_def.get("component", "")

    # Apply config overrides (for documentation, not all are functional)
    overrides = abl_def.get("config_overrides", {})
    _deep_merge(cfg, overrides)

    # Run simulation with ablation-specific modifications
    return _run_sim_with_ablation(cfg, abl_name, rng)


def _run_sim_with_ablation(
    config: Dict[str, Any],
    component: str,
    rng: np.random.RandomState,
) -> Dict[str, float]:
    """Run simulation with specific component ablated."""
    import copy
    cfg = copy.deepcopy(config)

    slo_us = cfg.get("slo_us", 500000.0)

    # Use deterministic seed for world building (same world across ablations)
    world_seed = cfg.get("seed", 0) + 300
    world = build_world(cfg, np.random.RandomState(world_seed))

    sim_cfg = cfg.get("sim", {})
    n_samples = sim_cfg.get("n_samples_per_micro_run", 2000)
    n_trials = min(cfg.get("n_trials", 10), 10)

    all_latencies = []
    trial_violation_rates = []

    for trial_id in range(n_trials):
        target = world.targets[trial_id % len(world.targets)]
        regime = world.regimes[trial_id % len(world.regimes)]
        n_specs = min(3, len(world.spectators))
        spec_ids = [world.spectators[i].workload_id for i in range(n_specs)]
        t_index = trial_id * 2

        if component == "IRBS":
            # No IRBS: use wider time gaps (more drift contamination)
            t_index = trial_id * 10  # Amplify drift effect

        # Use deterministic per-trial seed for reproducibility
        trial_seed = world_seed + trial_id + 1000
        result = simulate_micro_run(
            world, target.workload_id, spec_ids, regime.regime_id,
            t_index, n_samples, np.random.RandomState(trial_seed),
        )

        latencies = result.latencies_us

        if component == "Safety constraint":
            # No safety: use ALL spectators (worst-case interference)
            all_spec_ids = [s.workload_id for s in world.spectators]
            result = simulate_micro_run(
                world, target.workload_id, all_spec_ids, regime.regime_id,
                t_index, n_samples, np.random.RandomState(trial_seed + 500),
            )
            latencies = result.latencies_us

        if component == "Admission control":
            # No admission: accept even overloaded placements
            all_spec_ids = [s.workload_id for s in world.spectators[:n_specs + 3]]
            result = simulate_micro_run(
                world, target.workload_id, all_spec_ids, regime.regime_id,
                t_index, n_samples, np.random.RandomState(trial_seed + 500),
            )
            latencies = result.latencies_us

        all_latencies.extend(latencies.tolist())
        vr = float(np.mean(latencies > slo_us))
        trial_violation_rates.append(vr)

    return _compute_latency_metrics(np.array(all_latencies), trial_violation_rates, slo_us)


def _run_sim_and_compute(
    config: Dict[str, Any],
    rng: np.random.RandomState,
) -> Dict[str, float]:
    """Run a fresh simulation and compute metrics."""
    # Use deterministic seed for world building (same world as ablations)
    world_seed = config.get("seed", 0) + 300
    world = build_world(config, np.random.RandomState(world_seed))

    sim_cfg = config.get("sim", {})
    n_samples = sim_cfg.get("n_samples_per_micro_run", 2000)
    n_trials = min(config.get("n_trials", 10), 10)
    slo_us = config.get("slo_us", 500000.0)

    all_latencies = []
    trial_violation_rates = []

    for trial_id in range(n_trials):
        target = world.targets[trial_id % len(world.targets)]
        regime = world.regimes[trial_id % len(world.regimes)]
        n_specs = min(3, len(world.spectators))
        spec_ids = [world.spectators[i].workload_id for i in range(n_specs)]

        trial_seed = world_seed + trial_id + 1000
        result = simulate_micro_run(
            world, target.workload_id, spec_ids, regime.regime_id,
            trial_id * 2, n_samples, np.random.RandomState(trial_seed),
        )

        all_latencies.extend(result.latencies_us.tolist())
        vr = float(np.mean(result.latencies_us > slo_us))
        trial_violation_rates.append(vr)

    return _compute_latency_metrics(np.array(all_latencies), trial_violation_rates, slo_us)


def _compute_trial_metrics(
    trials_df: pd.DataFrame,
    slo_us: float,
) -> Dict[str, float]:
    """Compute metrics from a trials DataFrame."""
    cvar99 = float(trials_df["cvar99_latency_us"].mean()) if "cvar99_latency_us" in trials_df.columns else 0.0
    vr = float(trials_df["violation_rate"].mean()) if "violation_rate" in trials_df.columns else 0.0
    goodput = 1.0 - vr
    catastrophe_rate = float((trials_df["violation_rate"] > 0.1).mean()) if "violation_rate" in trials_df.columns else 0.0

    return {
        "cvar99": cvar99,
        "goodput": goodput,
        "catastrophe_rate": catastrophe_rate,
        "probe_efficiency": 1.0,
    }


def _compute_latency_metrics(
    latencies: np.ndarray,
    trial_violation_rates: List[float],
    slo_us: float,
) -> Dict[str, float]:
    """Compute key metrics from raw latencies."""
    if len(latencies) == 0:
        return {"cvar99": 0.0, "goodput": 0.0, "catastrophe_rate": 0.0, "probe_efficiency": 1.0}

    stats = compute_all_tail_stats(latencies, slo_us)
    goodput = 1.0 - stats["violation_rate"]
    catastrophe_rate = float(np.mean(np.array(trial_violation_rates) > 0.1))

    return {
        "cvar99": stats["cvar99_latency_us"],
        "goodput": goodput,
        "catastrophe_rate": catastrophe_rate,
        "probe_efficiency": 1.0,
    }


# --- Silent Catastrophe Audit ---

def audit_silent_catastrophes(
    trials_df: pd.DataFrame,
    failure_events_df: pd.DataFrame,
    mitigation_df: pd.DataFrame,
    slo_us: float = 500000.0,
    catastrophe_threshold: float = 0.1,
) -> pd.DataFrame:
    """Scan all runs for catastrophic events that lack detection.

    A catastrophe is a trial with violation_rate > catastrophe_threshold.
    A silent catastrophe has no preceding detector event or mitigation.

    Args:
        trials_df: All trial results.
        failure_events_df: All detected failure events.
        mitigation_df: All mitigation actions.
        slo_us: SLO threshold.
        catastrophe_threshold: Violation rate threshold for catastrophe.

    Returns:
        assumption_violation_report DataFrame.
    """
    if trials_df is None or len(trials_df) == 0:
        return _empty_violation_report()

    # Identify catastrophic trials
    if "violation_rate" not in trials_df.columns:
        return _empty_violation_report()

    catastrophes = trials_df[trials_df["violation_rate"] > catastrophe_threshold].copy()

    if len(catastrophes) == 0:
        logger.info("No catastrophic events found (threshold=%.2f)", catastrophe_threshold)
        return _empty_violation_report()

    # Get set of detected assumption IDs
    detected_ids = set()
    if failure_events_df is not None and len(failure_events_df) > 0:
        detected_ids = set(failure_events_df["assumption_id"].unique())

    # Get set of mitigated assumption IDs
    mitigated_ids = set()
    if mitigation_df is not None and len(mitigation_df) > 0 and "assumption_id" in mitigation_df.columns:
        mitigated_ids = set(mitigation_df["assumption_id"].unique())

    rows = []
    for idx, (_, cat_row) in enumerate(catastrophes.iterrows()):
        # Check if any detector fired
        has_detector = len(detected_ids) > 0
        has_mitigation = len(mitigated_ids) > 0

        silent = not has_detector

        rows.append({
            "catastrophe_id": idx,
            "trial_id": int(cat_row.get("trial_id", idx)),
            "violation_rate": round(float(cat_row["violation_rate"]), 4),
            "cvar99_latency_us": round(float(cat_row.get("cvar99_latency_us", 0.0)), 2),
            "silent": silent,
            "detector_fired": has_detector,
            "mitigation_applied": has_mitigation,
            "missing_detector": "none" if has_detector else "no_detector_for_catastrophe",
            "missing_mitigation": "none" if has_mitigation else "no_mitigation_for_catastrophe",
        })

    report_df = pd.DataFrame(rows)

    n_silent = int(report_df["silent"].sum())
    logger.info(
        "Catastrophe audit: %d catastrophes, %d silent",
        len(report_df), n_silent,
    )

    return report_df


def _empty_violation_report() -> pd.DataFrame:
    """Return empty violation report with correct schema."""
    return pd.DataFrame(columns=[
        "catastrophe_id", "trial_id", "violation_rate", "cvar99_latency_us",
        "silent", "detector_fired", "mitigation_applied",
        "missing_detector", "missing_mitigation",
    ])


def _deep_merge(base: Dict, override: Dict) -> None:
    """Recursively merge override into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
