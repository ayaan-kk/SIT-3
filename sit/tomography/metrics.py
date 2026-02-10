"""Recovery metrics for tomography evaluation.

Computes L1, L2, relative L2, top-k recall, precision@k, NDCG@k,
and sign consistency between estimated and true interference vectors.
"""

from typing import Dict, Optional

import numpy as np

from sit.core.logging import get_logger

logger = get_logger("tomography.metrics")


def l1_error(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """L1 error: sum |x_hat - x_true|."""
    return float(np.sum(np.abs(x_hat - x_true)))


def l2_error(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """L2 error: ||x_hat - x_true||_2."""
    return float(np.linalg.norm(x_hat - x_true))


def relative_l2_error(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """Relative L2 error: ||x_hat - x_true||_2 / ||x_true||_2.

    Returns inf if x_true is zero.
    """
    norm_true = np.linalg.norm(x_true)
    if norm_true < 1e-15:
        if np.linalg.norm(x_hat - x_true) < 1e-15:
            return 0.0
        return float("inf")
    return float(np.linalg.norm(x_hat - x_true) / norm_true)


def topk_recall(
    x_hat: np.ndarray,
    x_true: np.ndarray,
    k: Optional[int] = None,
) -> float:
    """Top-k recall: fraction of true top-k recovered in estimated top-k.

    If k is None, uses the number of nonzero entries in x_true.

    Args:
        x_hat: Estimated vector.
        x_true: True vector.
        k: Number of top components to consider.

    Returns:
        Recall in [0, 1].
    """
    if k is None:
        k = max(1, int(np.sum(x_true > 1e-10)))
    k = min(k, len(x_true))

    if k == 0:
        return 1.0

    true_topk = set(np.argsort(-x_true)[:k])
    hat_topk = set(np.argsort(-x_hat)[:k])

    overlap = len(true_topk & hat_topk)
    return float(overlap / k)


def precision_at_k(
    x_hat: np.ndarray,
    x_true: np.ndarray,
    k: Optional[int] = None,
) -> float:
    """Precision@k: of the top-k in x_hat, how many are in true top-k.

    Same as recall when using same k for both.
    """
    return topk_recall(x_hat, x_true, k)


def ndcg_at_k(
    x_hat: np.ndarray,
    x_true: np.ndarray,
    k: Optional[int] = None,
) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Uses x_true values as relevance scores. Measures how well
    x_hat ranks the spectators by interference magnitude.

    Args:
        x_hat: Estimated interference vector.
        x_true: True interference vector (relevance).
        k: Number of positions to consider.

    Returns:
        NDCG in [0, 1].
    """
    n = len(x_true)
    if k is None:
        k = max(1, int(np.sum(x_true > 1e-10)))
    k = min(k, n)

    if k == 0 or np.max(x_true) < 1e-15:
        return 1.0

    # Ranking by x_hat (descending)
    hat_order = np.argsort(-x_hat)

    # DCG: sum_{i=1}^{k} x_true[hat_order[i-1]] / log2(i+1)
    dcg = 0.0
    for i in range(k):
        dcg += x_true[hat_order[i]] / np.log2(i + 2)

    # Ideal DCG: sort x_true descending
    true_order = np.argsort(-x_true)
    idcg = 0.0
    for i in range(k):
        idcg += x_true[true_order[i]] / np.log2(i + 2)

    if idcg < 1e-15:
        return 1.0

    return float(dcg / idcg)


def sign_consistency(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """Fraction of components with correct sign (nonzero vs zero).

    A component is "consistent" if both are zero or both are nonzero.
    """
    n = len(x_true)
    if n == 0:
        return 1.0

    true_nz = x_true > 1e-10
    hat_nz = x_hat > 1e-10
    consistent = (true_nz == hat_nz)
    return float(np.mean(consistent))


def compute_all_recovery_metrics(
    x_hat: np.ndarray,
    x_true: np.ndarray,
    k: Optional[int] = None,
) -> Dict[str, float]:
    """Compute all recovery metrics.

    Args:
        x_hat: Estimated interference vector.
        x_true: True interference vector.
        k: Top-k parameter (None = auto from x_true sparsity).

    Returns:
        Dict with all metric values.
    """
    metrics = {
        "l1_error": l1_error(x_hat, x_true),
        "l2_error": l2_error(x_hat, x_true),
        "relative_l2": relative_l2_error(x_hat, x_true),
        "topk_recall": topk_recall(x_hat, x_true, k),
        "precision_at_k": precision_at_k(x_hat, x_true, k),
        "ndcg_at_k": ndcg_at_k(x_hat, x_true, k),
        "sign_consistency": sign_consistency(x_hat, x_true),
    }

    logger.info(
        "Recovery: L1=%.4f, L2=%.4f, rel_L2=%.4f, "
        "recall@k=%.3f, NDCG@k=%.3f, sign=%.3f",
        metrics["l1_error"], metrics["l2_error"], metrics["relative_l2"],
        metrics["topk_recall"], metrics["ndcg_at_k"], metrics["sign_consistency"],
    )

    return metrics
