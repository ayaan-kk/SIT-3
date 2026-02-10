"""Diversity metrics and kernel computation for probe selection.

Provides RBF and cosine similarity kernels, plus diversity metrics
(mean pairwise similarity, log-determinant, coverage entropy) used
to evaluate and guide probe set diversity.
"""

from typing import List, Optional

import numpy as np

from sit.core.logging import get_logger

logger = get_logger("probe.diversity")


def rbf_kernel(
    F: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Compute RBF (Gaussian) kernel matrix.

    K_{ij} = exp(-||f_i - f_j||^2 / (2 * sigma^2))

    Args:
        F: Feature matrix of shape (n, d).
        sigma: Kernel bandwidth parameter.

    Returns:
        Kernel matrix K of shape (n, n).
    """
    n = F.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    # Pairwise squared distances
    sq_norms = np.sum(F ** 2, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * F @ F.T
    dist_sq = np.maximum(dist_sq, 0.0)

    K = np.exp(-dist_sq / (2.0 * sigma ** 2))
    return K


def cosine_kernel(F: np.ndarray) -> np.ndarray:
    """Compute cosine similarity kernel matrix.

    K_{ij} = (f_i . f_j) / (||f_i|| * ||f_j||)

    Args:
        F: Feature matrix of shape (n, d).

    Returns:
        Kernel matrix K of shape (n, n) with values in [-1, 1].
    """
    n = F.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    norms = np.linalg.norm(F, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    F_norm = F / norms
    K = F_norm @ F_norm.T
    K = np.clip(K, -1.0, 1.0)
    return K


def build_kernel(
    F: np.ndarray,
    kernel_type: str = "rbf",
    sigma: float = 1.0,
) -> np.ndarray:
    """Build a similarity kernel matrix.

    Args:
        F: Feature matrix of shape (n, d).
        kernel_type: "rbf" or "cosine".
        sigma: Bandwidth for RBF kernel.

    Returns:
        Kernel matrix K of shape (n, n).
    """
    if kernel_type == "rbf":
        return rbf_kernel(F, sigma)
    elif kernel_type == "cosine":
        return cosine_kernel(F)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def mean_pairwise_similarity(
    S_indices: List[int],
    K: np.ndarray,
) -> float:
    """Compute mean pairwise similarity among selected spectators.

    Args:
        S_indices: List of selected spectator indices.
        K: Full kernel matrix of shape (n, n).

    Returns:
        Mean pairwise similarity (0 if |S| < 2).
    """
    s = len(S_indices)
    if s < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(s):
        for j in range(i + 1, s):
            total += K[S_indices[i], S_indices[j]]
            count += 1

    return total / count if count > 0 else 0.0


def logdet_diversity(
    S_indices: List[int],
    K: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """Compute log-determinant diversity of a set.

    logdet(K_S + epsilon * I) measures the volume spanned by
    the selected spectators in feature space.

    Args:
        S_indices: List of selected spectator indices.
        K: Full kernel matrix of shape (n, n).
        epsilon: Regularization for numerical stability.

    Returns:
        Log-determinant value (higher = more diverse).
    """
    s = len(S_indices)
    if s == 0:
        return 0.0

    K_S = K[np.ix_(S_indices, S_indices)] + epsilon * np.eye(s)
    sign, logdet = np.linalg.slogdet(K_S)

    if sign <= 0:
        return -float("inf")
    return float(logdet)


def logdet_increment(
    S_indices: List[int],
    new_idx: int,
    K: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """Compute the incremental log-determinant gain from adding new_idx.

    Uses the Schur complement formula:
    logdet(K_{S+new}) = logdet(K_S) + log(K[new,new] + eps - k_new^T K_S^{-1} k_new)

    Args:
        S_indices: Current selected indices.
        new_idx: Index to add.
        K: Full kernel matrix.
        epsilon: Regularization.

    Returns:
        Incremental logdet gain.
    """
    if len(S_indices) == 0:
        return float(np.log(K[new_idx, new_idx] + epsilon))

    S = list(S_indices)
    K_S = K[np.ix_(S, S)] + epsilon * np.eye(len(S))
    k_new = K[np.array(S), new_idx]

    try:
        L = np.linalg.cholesky(K_S)
        v = np.linalg.solve(L, k_new)
        schur = K[new_idx, new_idx] + epsilon - np.dot(v, v)
    except np.linalg.LinAlgError:
        # Fallback: compute full logdet difference
        old_val = logdet_diversity(S, K, epsilon)
        new_val = logdet_diversity(S + [new_idx], K, epsilon)
        return new_val - old_val

    if schur <= 0:
        return -float("inf")
    return float(np.log(schur))


def coverage_entropy(coverage_counts: dict) -> float:
    """Compute entropy of coverage distribution.

    Higher entropy means more uniform coverage across spectators.

    Args:
        coverage_counts: Dict mapping spectator_id to count.

    Returns:
        Shannon entropy of the normalized coverage distribution.
    """
    counts = np.array(list(coverage_counts.values()), dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0

    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))
