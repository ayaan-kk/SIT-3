"""Identifiability diagnostics and mitigation for tomography.

Analyzes the design matrix A for ill-conditioning and provides
mitigation strategies (adding extra probes, regularization tuning).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sit.core.logging import get_logger
from sit.probe.budget import ProbeBudget
from sit.probe.selection import coverage_aware_probes, compute_coverage

logger = get_logger("tomography.diagnostics")


def compute_diagnostics(A: np.ndarray) -> Dict[str, float]:
    """Compute identifiability diagnostics for design matrix A.

    Args:
        A: Design matrix of shape (m, n).

    Returns:
        Dict with diagnostic metrics:
            - rank: Matrix rank
            - rank_eff: Effective rank (from singular values)
            - cond_est: Condition number estimate
            - coherence: Maximum absolute correlation between columns
            - coverage_min: Minimum column sum (worst-covered spectator)
            - coverage_max: Maximum column sum
            - coverage_mean: Mean column sum
            - density: Fraction of nonzero entries
    """
    m, n = A.shape

    if m == 0 or n == 0:
        return {
            "rank": 0,
            "rank_eff": 0.0,
            "cond_est": float("inf"),
            "coherence": 0.0,
            "coverage_min": 0.0,
            "coverage_max": 0.0,
            "coverage_mean": 0.0,
            "density": 0.0,
        }

    # SVD for rank and condition
    svd_vals = np.linalg.svd(A, compute_uv=False)
    rank = int(np.sum(svd_vals > 1e-10))

    # Effective rank: exp(entropy of normalized singular values)
    if svd_vals.max() > 0:
        p = svd_vals / svd_vals.sum()
        p = p[p > 0]
        rank_eff = float(np.exp(-np.sum(p * np.log(p))))
    else:
        rank_eff = 0.0

    # Condition number
    nonzero_sv = svd_vals[svd_vals > 1e-10]
    if len(nonzero_sv) > 0:
        cond_est = float(nonzero_sv[0] / nonzero_sv[-1])
    else:
        cond_est = float("inf")

    # Coherence: max absolute off-diagonal correlation between columns
    coherence = _column_coherence(A)

    # Coverage: column sums
    col_sums = A.sum(axis=0)
    coverage_min = float(col_sums.min())
    coverage_max = float(col_sums.max())
    coverage_mean = float(col_sums.mean())

    # Density
    density = float(A.sum() / A.size)

    diagnostics = {
        "rank": rank,
        "rank_eff": rank_eff,
        "cond_est": cond_est,
        "coherence": coherence,
        "coverage_min": coverage_min,
        "coverage_max": coverage_max,
        "coverage_mean": coverage_mean,
        "density": density,
    }

    logger.info(
        "Diagnostics: rank=%d/%d, rank_eff=%.1f, cond=%.1f, "
        "coherence=%.3f, coverage=[%.0f, %.0f]",
        rank, min(m, n), rank_eff, cond_est,
        coherence, coverage_min, coverage_max,
    )

    return diagnostics


def _column_coherence(A: np.ndarray) -> float:
    """Compute maximum absolute off-diagonal correlation between columns."""
    n = A.shape[1]
    if n <= 1:
        return 0.0

    # Normalize columns
    norms = np.linalg.norm(A, axis=0)
    # Avoid division by zero
    norms = np.where(norms > 0, norms, 1.0)
    A_norm = A / norms

    # Gram matrix
    G = A_norm.T @ A_norm

    # Zero diagonal
    np.fill_diagonal(G, 0.0)

    return float(np.max(np.abs(G)))


def is_well_conditioned(
    diagnostics: Dict[str, float],
    max_cond: float = 100.0,
    min_coverage: float = 1.0,
    max_coherence: float = 0.99,
) -> bool:
    """Check if the design matrix is well-conditioned for recovery.

    Args:
        diagnostics: Output from compute_diagnostics().
        max_cond: Maximum acceptable condition number.
        min_coverage: Minimum acceptable coverage count.
        max_coherence: Maximum acceptable coherence.

    Returns:
        True if all conditions are met.
    """
    ok_cond = diagnostics["cond_est"] <= max_cond
    ok_cov = diagnostics["coverage_min"] >= min_coverage
    ok_coh = diagnostics["coherence"] <= max_coherence

    is_ok = ok_cond and ok_cov and ok_coh

    if not is_ok:
        logger.warning(
            "Design matrix is ill-conditioned: cond=%.1f (max=%.1f), "
            "coverage_min=%.0f (min=%.0f), coherence=%.3f (max=%.3f)",
            diagnostics["cond_est"], max_cond,
            diagnostics["coverage_min"], min_coverage,
            diagnostics["coherence"], max_coherence,
        )

    return is_ok


def mitigate_design(
    A: np.ndarray,
    spectator_ids: List[str],
    rng: np.random.RandomState,
    extra_probes: int = 20,
    target_coverage_min: float = 2.0,
) -> np.ndarray:
    """Mitigate ill-conditioned design by adding coverage-aware probes.

    Generates extra probe sets targeting under-covered spectators
    and appends them to the design matrix.

    Args:
        A: Original design matrix (m, n).
        spectator_ids: Ordered spectator IDs.
        rng: Random state.
        extra_probes: Number of extra probes to add.
        target_coverage_min: Target minimum coverage per spectator.

    Returns:
        Augmented design matrix (m + extra_probes, n).
    """
    from sit.tomography.design import build_design_matrix

    n = len(spectator_ids)

    # Determine set size from original A
    if A.shape[0] > 0:
        avg_row_sum = A.sum(axis=1).mean()
        set_size = max(1, int(round(avg_row_sum)))
    else:
        set_size = min(3, n)

    budget = ProbeBudget(
        m_probes=extra_probes,
        set_size=set_size,
        max_repeats=extra_probes,  # Relaxed for mitigation
    )

    new_probes = coverage_aware_probes(spectator_ids, budget, rng)
    A_extra = build_design_matrix(new_probes, spectator_ids)

    A_augmented = np.vstack([A, A_extra])

    new_diag = compute_diagnostics(A_augmented)
    logger.info(
        "Mitigation: added %d probes, new coverage_min=%.0f (target=%.0f), "
        "new cond=%.1f",
        extra_probes, new_diag["coverage_min"], target_coverage_min,
        new_diag["cond_est"],
    )

    return A_augmented
