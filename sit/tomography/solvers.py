"""Nonnegative elastic net solver for tomography inverse problem.

Solves:  min_{x >= 0}  (1/2m)||Ax - y||^2  +  lambda_1 ||x||_1  +  (lambda_2/2)||x||^2

Uses coordinate descent with nonnegativity projection.
"""

from typing import Optional, Tuple

import numpy as np

from sit.core.logging import get_logger

logger = get_logger("tomography.solvers")


def solve_nonneg_elastic_net(
    A: np.ndarray,
    y: np.ndarray,
    lambda_1: float = 0.01,
    lambda_2: float = 0.01,
    max_iter: int = 5000,
    tol: float = 1e-8,
    warm_start: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve nonnegative elastic net via coordinate descent.

    min_{x >= 0}  (1/2m)||Ax - y||^2  +  lambda_1 ||x||_1  +  (lambda_2/2)||x||^2

    Args:
        A: Design matrix of shape (m, n).
        y: Measurement vector of shape (m,).
        lambda_1: L1 regularization strength.
        lambda_2: L2 regularization strength.
        max_iter: Maximum coordinate descent iterations.
        tol: Convergence tolerance on x updates.
        warm_start: Optional initial x vector.

    Returns:
        Solution vector x_hat of shape (n,), all entries >= 0.
    """
    m, n = A.shape
    assert y.shape == (m,), f"y shape {y.shape} doesn't match A rows {m}"

    # Precompute column norms
    col_norms_sq = np.sum(A ** 2, axis=0)  # shape (n,)

    # Initialize
    if warm_start is not None:
        x = warm_start.copy()
    else:
        x = np.zeros(n, dtype=np.float64)

    # Residual: r = y - A @ x
    r = y - A @ x

    for iteration in range(max_iter):
        x_old = x.copy()

        for j in range(n):
            # Compute partial residual (add back contribution of x_j)
            r += A[:, j] * x[j]

            # Compute unconstrained update
            rho_j = A[:, j] @ r / m

            # Denominator: column norm / m + lambda_2
            denom = col_norms_sq[j] / m + lambda_2

            if denom <= 0:
                x[j] = 0.0
            else:
                # Soft threshold with nonnegativity
                # For nonneg: x_j = max(0, (rho_j - lambda_1) / denom)
                x[j] = max(0.0, (rho_j - lambda_1) / denom)

            # Update residual
            r -= A[:, j] * x[j]

        # Check convergence
        dx = np.max(np.abs(x - x_old))
        if dx < tol:
            logger.debug(
                "Converged after %d iterations (max_dx=%.2e)", iteration + 1, dx,
            )
            break

    # Final objective value
    obj = _objective(A, y, x, lambda_1, lambda_2)
    nnz = int(np.sum(x > 1e-12))
    logger.info(
        "Solver: %d iters, obj=%.4f, nnz=%d/%d, ||x||_1=%.2f, max(x)=%.2f",
        min(iteration + 1, max_iter), obj, nnz, n, np.sum(x), np.max(x) if n > 0 else 0.0,
    )

    return x


def _objective(
    A: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    lambda_1: float,
    lambda_2: float,
) -> float:
    """Compute the elastic net objective value."""
    m = A.shape[0]
    residual = y - A @ x
    data_fit = np.sum(residual ** 2) / (2 * m)
    l1_term = lambda_1 * np.sum(np.abs(x))
    l2_term = (lambda_2 / 2) * np.sum(x ** 2)
    return data_fit + l1_term + l2_term


def solve_nonneg_lasso(
    A: np.ndarray,
    y: np.ndarray,
    lambda_1: float = 0.01,
    max_iter: int = 5000,
    tol: float = 1e-8,
) -> np.ndarray:
    """Convenience: nonneg Lasso (elastic net with lambda_2 = 0)."""
    return solve_nonneg_elastic_net(A, y, lambda_1=lambda_1, lambda_2=0.0,
                                     max_iter=max_iter, tol=tol)


def cross_validate_lambda(
    A: np.ndarray,
    y: np.ndarray,
    lambda_candidates: Optional[np.ndarray] = None,
    n_folds: int = 5,
    lambda_2: float = 0.01,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, np.ndarray]:
    """Select lambda_1 via k-fold cross-validation.

    Args:
        A: Design matrix (m, n).
        y: Measurement vector (m,).
        lambda_candidates: Array of lambda_1 values to try.
        n_folds: Number of CV folds.
        lambda_2: Fixed L2 regularization.
        rng: Random state for fold assignment.

    Returns:
        Tuple of (best_lambda_1, cv_errors array of shape (n_candidates,)).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    m = A.shape[0]

    if lambda_candidates is None:
        # Default: log-spaced from 1e-4 to 1.0
        lambda_candidates = np.logspace(-4, 0, 20)

    # Create fold assignments
    fold_ids = np.arange(m) % n_folds
    rng.shuffle(fold_ids)

    cv_errors = np.zeros(len(lambda_candidates))

    for li, lam in enumerate(lambda_candidates):
        fold_errors = []
        for fold in range(n_folds):
            train_mask = fold_ids != fold
            test_mask = fold_ids == fold

            A_train = A[train_mask]
            y_train = y[train_mask]
            A_test = A[test_mask]
            y_test = y[test_mask]

            if len(y_train) == 0 or len(y_test) == 0:
                continue

            x_hat = solve_nonneg_elastic_net(
                A_train, y_train, lambda_1=lam, lambda_2=lambda_2,
                max_iter=2000, tol=1e-6,
            )

            pred = A_test @ x_hat
            mse = np.mean((y_test - pred) ** 2)
            fold_errors.append(mse)

        cv_errors[li] = np.mean(fold_errors) if fold_errors else float("inf")

    best_idx = int(np.argmin(cv_errors))
    best_lambda = float(lambda_candidates[best_idx])

    logger.info(
        "CV selected lambda_1=%.6f (fold MSE=%.4f)",
        best_lambda, cv_errors[best_idx],
    )

    return best_lambda, cv_errors
