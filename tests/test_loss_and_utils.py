import numpy as np
import pytest
from scipy.optimize import check_grad

from pysplinecalib.calib_utils import (
    _natural_cubic_spline_basis_expansion,
    get_stratified_foldnums,
    logreg_cv,
    my_log_loss,
    my_logit,
)
from pysplinecalib.loss_fun import pen_ll_fun, pen_ll_fun_grad


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = rng.uniform(size=60)  # soft labels
    w = rng.uniform(0.5, 2.0, size=60)
    beta = rng.normal(size=4)
    return X, y, w, beta


def _reference_loss(beta, X, y, lam, w=None):
    p = 1 / (1 + np.exp(-(X @ beta)))
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    if w is not None:
        ll = ll * w
    return ll.sum() / len(y) + lam * np.sum(beta**2) / len(beta)


@pytest.mark.parametrize("weighted", [False, True])
def test_loss_matches_reference(problem, weighted):
    X, y, w, beta = problem
    w = w if weighted else None
    expected = _reference_loss(beta, X, y, 0.3, w)
    assert pen_ll_fun(beta, X, y, 0.3, w) == pytest.approx(expected)
    loss, _ = pen_ll_fun_grad(beta, X, y, 0.3, w)
    assert loss == pytest.approx(expected)


@pytest.mark.parametrize("weighted", [False, True])
def test_gradient_matches_finite_differences(problem, weighted):
    X, y, w, beta = problem
    w = w if weighted else None
    err = check_grad(
        lambda b: pen_ll_fun_grad(b, X, y, 0.3, w)[0],
        lambda b: pen_ll_fun_grad(b, X, y, 0.3, w)[1],
        beta,
    )
    assert err < 1e-5


def test_loss_is_stable_for_extreme_scores():
    X = np.array([[1.0], [1.0]])
    y = np.array([0.0, 1.0])
    loss, grad = pen_ll_fun_grad(np.array([1000.0]), X, y)
    assert np.isfinite(loss) and np.all(np.isfinite(grad))
    assert loss == pytest.approx(500.0)


def test_basis_matches_loop_reference():
    rng = np.random.default_rng(1)
    x = rng.normal(size=25)
    knots = np.sort(rng.normal(size=6))

    def d(k, xv):
        return (
            np.maximum(xv - knots[k - 1], 0) ** 3 - np.maximum(xv - knots[-1], 0) ** 3
        ) / (knots[-1] - knots[k - 1])

    expected = np.zeros((len(x), len(knots)))
    expected[:, 0] = 1
    expected[:, 1] = x
    for i in range(1, len(knots) - 1):
        expected[:, i + 1] = d(i, x) - d(len(knots) - 1, x)
    np.testing.assert_allclose(
        _natural_cubic_spline_basis_expansion(x, knots), expected
    )


def test_my_logit_and_log_loss():
    np.testing.assert_allclose(my_logit(np.array([0.5])), [0.0], atol=1e-12)
    np.testing.assert_allclose(my_logit(np.array([0.9]), base=10), [np.log10(9)])
    assert np.isfinite(my_logit(np.array([0.0, 1.0]))).all()
    assert my_log_loss(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(
        np.log(2)
    )
    assert np.isfinite(my_log_loss(np.array([1.0]), np.array([0.0])))


def test_stratified_foldnums_balanced():
    y = np.array([0] * 20 + [1] * 10 + [2] * 10)
    fn = get_stratified_foldnums(y, 5)
    assert set(fn) == set(range(5))
    for c in range(3):
        counts = np.bincount(fn[y == c].astype(int), minlength=5)
        assert counts.max() - counts.min() <= 1


def _cv_data():
    rng = np.random.default_rng(2)
    x = rng.normal(size=200)
    X = np.column_stack([np.ones(200), x])
    y = (rng.uniform(size=200) < 1 / (1 + np.exp(-2 * x))).astype(float)
    return X, y


@pytest.mark.parametrize("mode", ["fast", "full"])
@pytest.mark.parametrize("weighted", [False, True])
def test_logreg_cv_uses_all_data_in_final_fit(mode, weighted):
    X, y = _cv_data()
    w = np.ones(len(y)) if weighted else None
    lams = np.array([1e-3, 1e-1, 10.0])
    best, ll_vec, res = logreg_cv(
        X, y, 4, lams, "L-BFGS-B", 500, 1e-6, weightvec=w, ps_mode=mode
    )
    assert best in lams and ll_vec.shape == (3,)
    # Stationarity on the full data at the chosen lambda
    _, grad = pen_ll_fun_grad(res.x, X, y, float(best), w)
    assert np.linalg.norm(grad) < 1e-4
    assert res.x[1] > 0


def test_logreg_cv_warns_when_not_converged():
    X, y = _cv_data()
    with pytest.warns(UserWarning, match="did not converge"):
        logreg_cv(X, y, 3, np.array([1e-3]), "L-BFGS-B", 1, 1e-12)


def test_stratified_foldnums_reproducible_and_leaves_global_rng_alone():
    y = np.array([0] * 20 + [1] * 20)
    np.random.seed(123)
    state = np.random.get_state()[1].copy()
    a = get_stratified_foldnums(y, 4, random_state=7)
    b = get_stratified_foldnums(y, 4, random_state=7)
    c = get_stratified_foldnums(y, 4, random_state=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    np.testing.assert_array_equal(np.random.get_state()[1], state)
