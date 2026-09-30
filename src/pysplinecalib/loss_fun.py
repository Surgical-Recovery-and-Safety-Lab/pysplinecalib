"""Penalized logistic loss and gradient (pure NumPy/SciPy)."""

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit


def pen_ll_fun(
    beta: NDArray[np.float64],
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    lam: float = 0.0,
    weight_vec: NDArray[np.float64] | None = None,
    max_exp: int = 50,
) -> float:
    """Compute the penalized log loss.

    Parameters
    ----------
    beta : ndarray of shape (n_features,)
        Coefficient vector.
    X : ndarray of shape (n_samples, n_features)
        Design matrix.
    y : ndarray of shape (n_samples,)
        Targets in [0, 1]. Soft labels are allowed.
    lam : float, default=0.0
        Ridge penalty strength.
    weight_vec : ndarray of shape (n_samples,), optional
        Per-sample weights.
    max_exp : int, default=50
        Unused. Kept for API compatibility.

    Returns
    -------
    loss : float
        Weighted mean log loss plus the ridge penalty.
    """
    s = X @ beta
    per_obs = np.logaddexp(0.0, s) - y * s
    if weight_vec is not None:
        per_obs = per_obs * weight_vec
    return float(per_obs.sum() / X.shape[0] + lam * (beta @ beta) / X.shape[1])


def pen_ll_fun_grad(
    beta: NDArray[np.float64],
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    lam: float = 0.0,
    weight_vec: NDArray[np.float64] | None = None,
    max_exp: int = 50,
) -> tuple[float, NDArray[np.float64]]:
    """Compute the penalized log loss and its gradient.

    Parameters
    ----------
    beta : ndarray of shape (n_features,)
        Coefficient vector.
    X : ndarray of shape (n_samples, n_features)
        Design matrix.
    y : ndarray of shape (n_samples,)
        Targets in [0, 1]. Soft labels are allowed.
    lam : float, default=0.0
        Ridge penalty strength.
    weight_vec : ndarray of shape (n_samples,), optional
        Per-sample weights.
    max_exp : int, default=50
        Unused. Kept for API compatibility.

    Returns
    -------
    loss : float
        Weighted mean log loss plus the ridge penalty.
    grad : ndarray of shape (n_features,)
        Gradient of the loss with respect to ``beta``.
    """
    n_samples, n_features = X.shape
    s = X @ beta
    per_obs = np.logaddexp(0.0, s) - y * s
    resid = expit(s) - y
    if weight_vec is not None:
        per_obs *= weight_vec
        resid *= weight_vec
    loss = float(per_obs.sum() / n_samples + lam * (beta @ beta) / n_features)
    grad = X.T @ resid / n_samples + 2.0 * lam * beta / n_features
    return loss, grad
