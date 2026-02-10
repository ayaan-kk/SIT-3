"""Information gain scoring for active probe selection.

Provides uncertainty-based and hybrid scoring functions that combine
reconstruction uncertainty with diversity to guide adaptive probing.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from sit.core.logging import get_logger

logger = get_logger("probe.info_gain")


def uncertainty_scores(
    sigma: np.ndarray,
    x_hat: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> np.ndarray:
    """Score spectators by reconstruction uncertainty.

    score(s) = alpha * sigma_s + beta * |x_hat_s|

    Args:
        sigma: Per-spectator uncertainty vector of shape (n,).
        x_hat: Current estimate vector of shape (n,), optional.
        alpha: Weight on uncertainty.
        beta: Weight on estimated magnitude.

    Returns:
        Score vector of shape (n,).
    """
    scores = alpha * sigma
    if x_hat is not None and beta > 0:
        scores = scores + beta * np.abs(x_hat)
    return scores


def hybrid_score(
    candidate_idx: int,
    current_set: List[int],
    sigma: np.ndarray,
    K: np.ndarray,
    hybrid_lambda: float = 0.5,
    epsilon: float = 1e-6,
) -> Tuple[float, Dict[str, float]]:
    """Compute hybrid score combining uncertainty and diversity.

    J(S + {s}) = sigma_s + lambda * logdet_increment(S, s)

    Args:
        candidate_idx: Index of candidate spectator.
        current_set: List of already-selected indices.
        sigma: Uncertainty vector of shape (n,).
        K: Kernel matrix of shape (n, n).
        hybrid_lambda: Trade-off parameter (higher = more diversity).
        epsilon: Regularization for logdet.

    Returns:
        Tuple of (total_score, breakdown_dict).
    """
    from sit.probe.diversity import logdet_increment

    unc_val = float(sigma[candidate_idx])
    div_val = logdet_increment(current_set, candidate_idx, K, epsilon)

    total = unc_val + hybrid_lambda * div_val

    breakdown = {
        "uncertainty": unc_val,
        "logdet_increment": div_val,
        "hybrid_lambda": hybrid_lambda,
        "total": total,
    }

    return total, breakdown
