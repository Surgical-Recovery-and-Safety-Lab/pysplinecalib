"""Calibration of predicted probabilities."""
from __future__ import division

import numpy as np
import scipy as sp
import scipy.optimize  # noqa: F401
import random
import warnings
from scipy.special import expit
from scipy.stats import binom

from .loss_fun import pen_ll_fun_grad

def _natural_cubic_spline_basis_expansion(xpts, knots):
    """Compute the natural cubic spline basis for points and knots.

    Parameters
    ----------
    xpts : array-like of shape (n_points,)
        Points at which to evaluate the basis.
    knots : array-like of shape (n_knots,)
        Sorted knot locations.

    Returns
    -------
    outmat : ndarray of shape (n_points, n_knots)
        Basis matrix: intercept, linear term, then n_knots - 2 nonlinear terms.
    """
    xpts = np.asarray(xpts, dtype=np.float64)
    knots = np.asarray(knots, dtype=np.float64)
    last = knots[-1]
    tail = np.maximum(xpts - last, 0.0) ** 3
    d = (np.maximum(xpts[:, None] - knots[None, :-1], 0.0) ** 3 - tail[:, None]) / (
        last - knots[:-1]
    )
    outmat = np.empty((len(xpts), len(knots)))
    outmat[:, 0] = 1.0
    outmat[:, 1] = xpts
    outmat[:, 2:] = d[:, :-1] - d[:, -1:]
    return outmat


def logreg_cv(X, y, num_folds, reg_param_vec, method, max_iter,
              tol, weightvec=None, random_state=42, reg_prec=4, ps_mode='fast'):
    """Routine to find the best fitting penalized Logistic Regression.

    User must provide, the X, y, number of folds, range of `lambda` parameter
    and other specs for the optimization.
    """
    fn_vec = get_stratified_foldnums(y, num_folds, random_state=random_state)
    preds = np.zeros(len(y))
    ll_vec = np.zeros(len(reg_param_vec))
    start_coef_vec = np.zeros(X.shape[1])
    num_folds_to_search = 1 if ps_mode == 'fast' else num_folds

    # Build the fold splits once, rather than for every lambda
    splits = []
    for fn in range(num_folds_to_search):
        tr = fn_vec != fn
        te = fn_vec == fn
        splits.append((te, X[tr, :], y[tr], X[te, :],
                       None if weightvec is None else weightvec[tr]))

    for i, lam_val in enumerate(reg_param_vec):
        for te, X_tr, y_tr, X_te, weightvec_tr in splits:
            opt_res = sp.optimize.minimize(pen_ll_fun_grad,
                                           start_coef_vec,
                                           (X_tr, y_tr, float(lam_val),
                                            weightvec_tr),
                                           method=method,
                                           jac=True,
                                           options={"gtol": tol,
                                                    "maxiter": max_iter})
            if not opt_res.success:
                warnings.warn("Optimization did not converge for lambda={}".format(lam_val))
            preds[te] = expit(X_te.dot(opt_res.x))
        if ps_mode == 'fast':
            ll_vec[i] = my_log_loss(y[fn_vec == 0], preds[fn_vec == 0])
        else:
            ll_vec[i] = my_log_loss(y, preds)
    best_index = np.argmin(np.round(ll_vec, decimals=reg_prec))
    best_lam_val = reg_param_vec[best_index]

    # Final fit on all of the data
    opt_res = sp.optimize.minimize(pen_ll_fun_grad,
                                   start_coef_vec,
                                   (X, y, float(best_lam_val), weightvec),
                                   method=method,
                                   jac=True,
                                   options={"gtol": tol,
                                            "maxiter": max_iter})
    if not opt_res.success:
        warn_str = """Optimization did not converge for final fit.
                    This is usually due to numerical issues.
                    Consider increasing `max_iter` or `tol`"""
        warnings.warn(warn_str)

    return(best_lam_val, ll_vec, opt_res)


def my_logit(vec, base=np.exp(1), eps=1e-16):
    vec = np.clip(vec, eps, 1-eps)
    return (1/np.log(base)) * np.log(vec/(1-vec))


def my_log_loss(truth_vec, pred_vec, eps=1e-16):
    pred_vec = np.clip(pred_vec, eps, 1-eps)
    val = np.mean(truth_vec*np.log(pred_vec)+(1-truth_vec)*np.log(1-pred_vec))
    return(-val)


def get_stratified_foldnums(y, num_folds, random_state=42):
    """Given an outcome vector y, assigns each data point to a fold in a stratified manner.
    
    Assumes that y contains only integers between 0 and num_classes-1
    """
    fn_vec = -1 * np.ones(len(y))
    for y_val in np.unique(y):
        curr_yval_indices = np.where(y==y_val)[0]
        np.random.seed(random_state)
        np.random.shuffle(curr_yval_indices)
        index_indices = np.round((len(curr_yval_indices)/num_folds)*
                                 np.arange(num_folds+1)).astype(int)
        for i in range(num_folds):
            fold_to_assign = i if ((y_val%2)==0) else (num_folds-i-1)
            fn_vec[curr_yval_indices[index_indices[i]:index_indices[i+1]]] = fold_to_assign
    return(fn_vec)

