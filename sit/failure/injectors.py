"""Adversarial scenario generators that intentionally violate SIT assumptions.

Each injector modifies a World or config to create controlled breakage,
annotating the run with the injected assumption_id and parameters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sit.core.logging import get_logger
from sit.sim.world import World, build_world, simulate_micro_run
from sit.measure.tail import compute_all_tail_stats

logger = get_logger("failure.injectors")


@dataclass
class InjectionResult:
    """Result of running a failure injection scenario."""

    injected_assumption_id: str
    injection_name: str
    injection_params: Dict[str, Any]
    world: Optional[World]
    trials_df: pd.DataFrame
    state: Dict[str, Any]  # Pipeline state for detector checking


def _run_trials_from_world(
    world: World,
    config: Dict[str, Any],
    rng: np.random.RandomState,
    n_trials: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run a minimal trial set from a world and collect state."""
    sim_cfg = config.get("sim", {})
    n_samples = sim_cfg.get("n_samples_per_micro_run", 2000)
    slo_us = config.get("slo_us", 500000.0)

    trial_rows = []
    all_latencies = []

    for trial_id in range(n_trials):
        target = world.targets[trial_id % len(world.targets)]
        regime = world.regimes[trial_id % len(world.regimes)]

        n_specs = min(3, len(world.spectators))
        spec_ids = [world.spectators[i].workload_id for i in range(n_specs)]

        result = simulate_micro_run(
            world=world,
            target_id=target.workload_id,
            spectator_ids=spec_ids,
            regime_id=regime.regime_id,
            t_index=trial_id * 2,
            n_samples=n_samples,
            rng=rng,
        )

        stats = compute_all_tail_stats(result.latencies_us, slo_us)
        all_latencies.extend(result.latencies_us.tolist())

        trial_rows.append({
            "trial_id": trial_id,
            "target_id": target.workload_id,
            "regime_id": regime.regime_id,
            "n_samples": stats["n_samples"],
            "mean_latency_us": stats["mean_latency_us"],
            "p99_latency_us": stats["p99_latency_us"],
            "cvar99_latency_us": stats["cvar99_latency_us"],
            "violation_rate": stats["violation_rate"],
            "slo_us": slo_us,
        })

    trials_df = pd.DataFrame(trial_rows)

    # Compute tail sample count
    all_lat = np.array(all_latencies)
    if len(all_lat) > 0:
        p99_threshold = np.quantile(all_lat, 0.99)
        n_tail = int(np.sum(all_lat >= p99_threshold))
    else:
        n_tail = 0

    state = {
        "n_tail_samples": n_tail,
        "all_latencies": all_lat,
        "trials_df": trials_df,
    }

    return trials_df, state


# --- Injector 1: Non-additive interaction ---

def inject_non_additive(
    config: Dict[str, Any],
    rng: np.random.RandomState,
    gamma: float = 5.0,
) -> InjectionResult:
    """Enable strong second-order interactions to violate additivity.

    Sets interaction gamma very high so pairwise interaction terms
    dominate the additive signal.
    """
    cfg = _deep_copy_config(config)
    cfg.setdefault("sim", {})
    cfg["sim"]["interactions"] = {"enabled": True, "gamma": gamma}

    world = build_world(cfg, np.random.RandomState(rng.randint(0, 2**31)))
    trials_df, state = _run_trials_from_world(world, cfg, rng)

    # Compute residuals by set size to check additivity
    residuals_by_size = {}
    target = world.targets[0]
    regime = world.regimes[0]
    gt = world.ground_truth.get(regime.regime_id, {})

    for size in [1, 2, 3, 4]:
        if size > len(world.spectators):
            break
        spec_ids = [world.spectators[i].workload_id for i in range(size)]

        # Predicted additive
        additive_pred = sum(gt.get((target.workload_id, sid), 0.0) for sid in spec_ids)

        # Actual
        mr = simulate_micro_run(
            world, target.workload_id, spec_ids, regime.regime_id,
            0, 5000, np.random.RandomState(rng.randint(0, 2**31)),
        )
        actual_mean = float(np.mean(mr.latencies_us))

        # Baseline (no spectators)
        mr0 = simulate_micro_run(
            world, target.workload_id, [], regime.regime_id,
            0, 5000, np.random.RandomState(rng.randint(0, 2**31)),
        )
        baseline_mean = float(np.mean(mr0.latencies_us))

        actual_interference = actual_mean - baseline_mean
        residuals_by_size[size] = abs(actual_interference - additive_pred)

    state["residuals_by_set_size"] = residuals_by_size

    return InjectionResult(
        injected_assumption_id="A1_additivity",
        injection_name="non_additive_interaction",
        injection_params={"gamma": gamma},
        world=world,
        trials_df=trials_df,
        state=state,
    )


# --- Injector 2: Dense interference ---

def inject_dense_interference(
    config: Dict[str, Any],
    rng: np.random.RandomState,
    sparsity: float = 0.95,
) -> InjectionResult:
    """Increase toxic sparsity to near dense to violate sparsity assumption."""
    cfg = _deep_copy_config(config)
    cfg.setdefault("sim", {})
    cfg["sim"].setdefault("toxic_pairs", {})
    cfg["sim"]["toxic_pairs"]["enabled"] = True
    cfg["sim"]["toxic_pairs"]["sparsity"] = sparsity

    world = build_world(cfg, np.random.RandomState(rng.randint(0, 2**31)))
    trials_df, state = _run_trials_from_world(world, cfg, rng)

    # Build x_hat proxy: use ground truth magnitudes as proxy for dense solution
    regime = world.regimes[0]
    gt = world.ground_truth.get(regime.regime_id, {})
    target = world.targets[0]
    x_hat = np.array([
        gt.get((target.workload_id, s.workload_id), 0.0)
        for s in world.spectators
    ])
    state["x_hat"] = x_hat
    state["n_spectators"] = len(world.spectators)

    return InjectionResult(
        injected_assumption_id="A2_sparsity",
        injection_name="dense_interference",
        injection_params={"sparsity": sparsity},
        world=world,
        trials_df=trials_df,
        state=state,
    )


# --- Injector 3: Fast drift ---

def inject_fast_drift(
    config: Dict[str, Any],
    rng: np.random.RandomState,
    a_us_per_step: float = 500.0,
) -> InjectionResult:
    """Set drift faster than IRBS window can cancel."""
    cfg = _deep_copy_config(config)
    cfg.setdefault("sim", {})
    cfg["sim"]["drift"] = {"type": "linear", "a_us_per_step": a_us_per_step}

    world = build_world(cfg, np.random.RandomState(rng.randint(0, 2**31)))
    trials_df, state = _run_trials_from_world(world, cfg, rng)

    # Compute IRBS residual biases across windows
    from sit.measure.irbs import irbs_ctc_estimate, naive_ab_estimate

    target = world.targets[0]
    regime = world.regimes[0]
    gt = world.ground_truth.get(regime.regime_id, {})
    spec = world.spectators[0]
    true_delta = gt.get((target.workload_id, spec.workload_id), 0.0)

    biases = []
    n_samples = cfg.get("sim", {}).get("n_samples_per_micro_run", 2000)

    for window_idx in range(5):
        t_base = window_idx * 20

        c1 = simulate_micro_run(
            world, target.workload_id, [], regime.regime_id,
            t_base, n_samples, np.random.RandomState(rng.randint(0, 2**31)),
        )
        t_run = simulate_micro_run(
            world, target.workload_id, [spec.workload_id], regime.regime_id,
            t_base + 10, n_samples, np.random.RandomState(rng.randint(0, 2**31)),
        )
        c2 = simulate_micro_run(
            world, target.workload_id, [], regime.regime_id,
            t_base + 20, n_samples, np.random.RandomState(rng.randint(0, 2**31)),
        )

        est = irbs_ctc_estimate(
            float(np.mean(c1.latencies_us)),
            float(np.mean(t_run.latencies_us)),
            float(np.mean(c2.latencies_us)),
        )
        biases.append(est - true_delta)

    state["irbs_residual_biases"] = biases
    state["drift_bias_threshold"] = 0.1

    return InjectionResult(
        injected_assumption_id="A3_drift_smoothness",
        injection_name="fast_drift",
        injection_params={"a_us_per_step": a_us_per_step},
        world=world,
        trials_df=trials_df,
        state=state,
    )


# --- Injector 4: Regime flip ---

def inject_regime_flip(
    config: Dict[str, Any],
    rng: np.random.RandomState,
) -> InjectionResult:
    """Regime changes mid probe batch to violate stationarity.

    Simulates a batch where half the samples come from regime 0
    and half from regime 1, creating variance spikes.
    """
    cfg = _deep_copy_config(config)
    cfg.setdefault("sim", {})
    cfg["sim"]["n_regimes"] = 2

    world = build_world(cfg, np.random.RandomState(rng.randint(0, 2**31)))
    n_samples = cfg.get("sim", {}).get("n_samples_per_micro_run", 2000)

    target = world.targets[0]
    if len(world.regimes) < 2:
        # Fallback: just use one regime with high variance
        trials_df, state = _run_trials_from_world(world, cfg, rng)
        state["within_batch_variances"] = [1.0, 100.0]  # Inject artificial spike
        return InjectionResult(
            injected_assumption_id="A4_stationarity",
            injection_name="regime_flip",
            injection_params={"n_regimes": len(world.regimes)},
            world=world,
            trials_df=trials_df,
            state=state,
        )

    regime_a = world.regimes[0]
    regime_b = world.regimes[1]
    spec_ids = [world.spectators[0].workload_id]

    # Simulate mixed batch: first half regime A, second half regime B
    # Amplify regime B latencies to force a genuine regime difference,
    # modeling a real scenario where load levels shift mid-batch.
    half = n_samples // 2
    mr_a = simulate_micro_run(
        world, target.workload_id, spec_ids, regime_a.regime_id,
        0, half, np.random.RandomState(rng.randint(0, 2**31)),
    )
    mr_b = simulate_micro_run(
        world, target.workload_id, spec_ids, regime_b.regime_id,
        0, half, np.random.RandomState(rng.randint(0, 2**31)),
    )

    # Scale regime B latencies to simulate a genuine regime shift
    # (the generated regimes may be too similar by default)
    lat_b_scaled = mr_b.latencies_us * 3.0

    # Compute within-sub-batch variances
    # Pure chunks from each regime should have low variance;
    # the mixed boundary and cross-regime chunks create spikes.
    combined = np.concatenate([mr_a.latencies_us, lat_b_scaled])
    chunk_size = max(1, len(combined) // 4)
    within_vars = []
    for i in range(0, len(combined), chunk_size):
        chunk = combined[i:i + chunk_size]
        if len(chunk) > 1:
            within_vars.append(float(np.var(chunk)))

    trials_df, state = _run_trials_from_world(world, cfg, rng)
    state["within_batch_variances"] = within_vars

    return InjectionResult(
        injected_assumption_id="A4_stationarity",
        injection_name="regime_flip",
        injection_params={"n_regimes": len(world.regimes)},
        world=world,
        trials_df=trials_df,
        state=state,
    )


# --- Injector 5: Probe starvation ---

def inject_probe_starvation(
    config: Dict[str, Any],
    rng: np.random.RandomState,
    m_probes: int = 2,
) -> InjectionResult:
    """Cap probe budget too low to violate coverage assumption."""
    cfg = _deep_copy_config(config)

    world = build_world(cfg, np.random.RandomState(rng.randint(0, 2**31)))
    trials_df, state = _run_trials_from_world(world, cfg, rng)

    # Simulate a starved design matrix
    n = len(world.spectators)
    # Very few probes = very low rank
    A = np.zeros((m_probes, n))
    for i in range(m_probes):
        # Each probe covers only 1 spectator
        A[i, i % n] = 1.0

    from sit.tomography.diagnostics import compute_diagnostics
    diag = compute_diagnostics(A)

    state["diagnostics"] = diag
    state["n_spectators"] = n

    return InjectionResult(
        injected_assumption_id="A5_coverage",
        injection_name="probe_starvation",
        injection_params={"m_probes": m_probes},
        world=world,
        trials_df=trials_df,
        state=state,
    )


# --- Injector 6: Tail starvation ---

def inject_tail_starvation(
    config: Dict[str, Any],
    rng: np.random.RandomState,
    p_burst: float = 0.0,
    n_samples: int = 50,
) -> InjectionResult:
    """Reduce burst probability so few tail samples exist."""
    cfg = _deep_copy_config(config)
    cfg.setdefault("sim", {})
    cfg["sim"].setdefault("burst", {})
    cfg["sim"]["burst"]["enabled"] = True
    cfg["sim"]["burst"]["p_burst"] = p_burst
    cfg["sim"]["n_samples_per_micro_run"] = n_samples

    world = build_world(cfg, np.random.RandomState(rng.randint(0, 2**31)))
    trials_df, state = _run_trials_from_world(world, cfg, rng, n_trials=5)

    # With very few samples and no bursts, tail estimation is unreliable
    all_lat = state.get("all_latencies", np.array([]))
    if len(all_lat) > 0:
        p99_threshold = np.quantile(all_lat, 0.99)
        state["n_tail_samples"] = int(np.sum(all_lat >= p99_threshold))
    else:
        state["n_tail_samples"] = 0

    state["min_tail_samples"] = 20

    return InjectionResult(
        injected_assumption_id="A6_tail_validity",
        injection_name="tail_starvation",
        injection_params={"p_burst": p_burst, "n_samples": n_samples},
        world=world,
        trials_df=trials_df,
        state=state,
    )


# --- Injector 7: Capacity overload ---

def inject_capacity_overload(
    config: Dict[str, Any],
    rng: np.random.RandomState,
    slo_us: float = 1.0,
) -> InjectionResult:
    """Set SLO impossibly tight so no feasible placement exists."""
    cfg = _deep_copy_config(config)
    cfg["slo_us"] = slo_us  # 1 microsecond SLO = impossible

    world = build_world(cfg, np.random.RandomState(rng.randint(0, 2**31)))
    trials_df, state = _run_trials_from_world(world, cfg, rng)

    # Every placement violates SLO
    n_candidates = len(trials_df)
    n_safe = int((trials_df["violation_rate"] == 0).sum())
    state["n_candidate_placements"] = max(n_candidates, 1)
    state["n_safe_placements"] = n_safe

    return InjectionResult(
        injected_assumption_id="A7_feasibility",
        injection_name="capacity_overload",
        injection_params={"slo_us": slo_us},
        world=world,
        trials_df=trials_df,
        state=state,
    )


# --- Registry ---

INJECTOR_REGISTRY = {
    "non_additive": inject_non_additive,
    "dense_interference": inject_dense_interference,
    "fast_drift": inject_fast_drift,
    "regime_flip": inject_regime_flip,
    "probe_starvation": inject_probe_starvation,
    "tail_starvation": inject_tail_starvation,
    "capacity_overload": inject_capacity_overload,
}


def run_all_injections(
    config: Dict[str, Any],
    rng: np.random.RandomState,
    enabled: Optional[Dict[str, bool]] = None,
) -> List[InjectionResult]:
    """Run all enabled injectors and return results."""
    if enabled is None:
        enabled = {k: True for k in INJECTOR_REGISTRY}

    results = []
    for name, fn in INJECTOR_REGISTRY.items():
        if not enabled.get(name, False):
            logger.info("Injector %s disabled, skipping", name)
            continue

        logger.info("Running injector: %s", name)
        inj_rng = np.random.RandomState(rng.randint(0, 2**31))
        result = fn(config, inj_rng)
        results.append(result)
        logger.info(
            "Injector %s complete: assumption=%s",
            name, result.injected_assumption_id,
        )

    return results


def _deep_copy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Deep copy a config dict."""
    import copy
    return copy.deepcopy(config)
