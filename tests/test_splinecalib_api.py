import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from pysplinecalib import SplineCalib  # noqa: E402

FAST = dict(reg_param_vec=np.logspace(-2, 2, 3), cv_spline=3, max_iter=200)


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.02, 0.98, size=300)
    y = (rng.uniform(size=300) < p**1.5).astype(int)
    return p, y


@pytest.fixture
def multiclass_data():
    rng = np.random.default_rng(4)
    raw = rng.dirichlet(np.ones(3), size=240)
    y = np.array([rng.choice(3, p=r) for r in raw])
    return raw, y


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_init_options():
    sc = SplineCalib(reg_param_vec=[0.1, 1.0], logodds_eps=1e-3)
    assert isinstance(sc.reg_param_vec, np.ndarray)
    assert sc.logodds_eps == 1e-3 and not sc.logodds_eps_auto
    assert SplineCalib().logodds_eps_auto
    with pytest.warns(UserWarning, match="param_search_mode"):
        sc = SplineCalib(param_search_mode="bogus")
    assert sc.param_search_mode == "full"


def test_fit_calibrate_binary_and_aliases(binary_data):
    p, y = binary_data
    sc = SplineCalib(**FAST)
    sc.fit(p, y)
    out = sc.calibrate(p)
    assert out.shape == p.shape and np.all((out >= 0) & (out <= 1))
    for alias in (sc.transform, sc.predict, sc.predict_proba):
        np.testing.assert_array_equal(alias(p), out)


def test_fit_does_not_mutate_input(binary_data):
    p, y = binary_data
    before = p.copy()
    SplineCalib(**FAST).fit(p, y)
    np.testing.assert_array_equal(p, before)


def test_two_column_input_roundtrip(binary_data):
    p, y = binary_data
    two = np.column_stack([1 - p, p])
    sc = SplineCalib(**FAST)
    sc.fit(two, y)
    out = sc.calibrate(two)
    assert out.shape == two.shape
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    np.testing.assert_allclose(out[:, 1], sc.calibrate(p))


def test_list_truth_values_accepted(binary_data):
    p, y = binary_data
    SplineCalib(**FAST).fit(p, list(y))


def test_no_unity_prior_and_no_logodds(binary_data):
    p, y = binary_data
    sc = SplineCalib(unity_prior=False, logodds_scale=False, **FAST)
    sc.fit(p, y)
    assert not sc.use_weights
    assert np.all(np.isfinite(sc.calibrate(p)))


def test_custom_unity_prior_gridpts_weight_not_squared(binary_data):
    p, y = binary_data
    grid = np.linspace(0.05, 0.95, 10)
    sc = SplineCalib(unity_prior_gridpts=grid, unity_prior_weight=20, **FAST)
    sc.fit(p, y)
    np.testing.assert_allclose(sc.final_weightvec[len(p) :], 20 / 10)
    assert sc.final_weightvec[len(p) :].sum() == pytest.approx(20)


def test_full_search_mode_and_numeric_logodds_eps(binary_data):
    p, y = binary_data
    sc = SplineCalib(param_search_mode="full", logodds_eps=1e-4, **FAST)
    sc.fit(p, y)
    assert sc.reg_param_scores.shape == (3,)


def test_knot_selection_branches(binary_data):
    p, y = binary_data
    with pytest.raises(Exception, match="3 unique"):
        SplineCalib(**FAST).fit(np.array([0.2, 0.2, 0.8, 0.8]), np.array([0, 0, 1, 1]))
    with pytest.raises(Exception, match="knot_sample_size"):
        SplineCalib(knot_sample_size=0, add_knots=None, **FAST).fit(p, y)
    sc = SplineCalib(knot_sample_size=0, add_knots=[0.2, 0.5, 0.8], **FAST)
    np.testing.assert_array_equal(sc._get_knot_vec(p), [0.2, 0.5, 0.8])
    with pytest.warns(UserWarning, match="force_knot_endpts"):
        sc = SplineCalib(knot_sample_size=1, **FAST)
        sc._get_knot_vec(p)
    assert not sc.force_knot_endpts
    sc = SplineCalib(
        force_knot_endpts=False, knot_sample_size=10, add_knots=None, **FAST
    )
    assert len(sc._get_knot_vec(p)) == 10
    sc = SplineCalib(knot_sample_size=10, add_knots=None, **FAST)
    knots = sc._get_knot_vec(p)
    assert p.min() in knots and p.max() in knots and len(knots) == 10
    few = np.array([0.1, 0.5, 0.9] * 10)
    sc = SplineCalib(add_knots=None, **FAST)
    np.testing.assert_array_equal(sc._get_knot_vec(few), [0.1, 0.5, 0.9])


def test_calibrate_before_fit_warns():
    sc = SplineCalib()
    sc.n_classes = 1
    with pytest.warns(UserWarning, match="not fit"):
        assert sc.calibrate(np.array([0.5])) is None


def test_multiclass_fit_calibrate_and_options_forwarded(multiclass_data):
    raw, y = multiclass_data
    sc = SplineCalib(
        reg_prec=2,
        param_search_mode="full",
        force_knot_endpts=False,
        logodds_eps=1e-3,
        **FAST,
    )
    sc.fit(raw, y, verbose=True)
    assert len(sc.binary_splinecalibs) == 3
    for b in sc.binary_splinecalibs:
        assert b.reg_prec == 2
        assert b.param_search_mode == "full"
        assert b.force_knot_endpts is False
        assert b.logodds_eps == 1e-3
    out = sc.calibrate(raw)
    assert out.shape == raw.shape
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


def test_plots(binary_data, multiclass_data):
    p, y = binary_data
    sc = SplineCalib(**FAST)
    sc.fit(p, y, verbose=True)
    sc.show_spline_reg_plot()
    sc.show_calibration_curve()
    sc.show_calibration_curve(scaling="logit", show_baseline=True)

    raw, yy = multiclass_data
    mc = SplineCalib(**FAST)
    mc.fit(raw, yy)
    with pytest.warns(UserWarning, match="class number"):
        mc.show_spline_reg_plot()
    with pytest.warns(UserWarning, match="class number"):
        mc.show_calibration_curve()
    mc.show_spline_reg_plot(class_num=1)
    mc.show_calibration_curve(class_num=1)


def test_single_column_input(binary_data):
    p, y = binary_data
    sc = SplineCalib(**FAST)
    sc.fit(p, y)
    out = sc.calibrate(p[:, None])
    assert out.shape == p.shape
    np.testing.assert_allclose(out, sc.calibrate(p))


def test_unsupported_input_shape_raises(binary_data):
    p, y = binary_data
    sc = SplineCalib(**FAST)
    sc.fit(p, y)
    with pytest.raises(ValueError, match="got"):
        sc.calibrate(np.ones((4, 3)))
    with pytest.raises(ValueError, match="got"):
        sc.calibrate(np.ones((2, 2, 2)))


def test_knots_reproducible_and_random_state_respected(binary_data):
    p, _ = binary_data
    np.random.seed(5)
    state = np.random.get_state()[1].copy()
    a = SplineCalib(random_state=1, add_knots=None, **{**FAST})._get_knot_vec(p)
    b = SplineCalib(random_state=1, add_knots=None, **{**FAST})._get_knot_vec(p)
    c = SplineCalib(random_state=2, add_knots=None, **{**FAST})._get_knot_vec(p)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    np.testing.assert_array_equal(np.random.get_state()[1], state)


def test_random_state_reaches_cv_folds(binary_data, monkeypatch):
    import pysplinecalib.pysplinecalib as mod

    seen = {}
    real = mod.logreg_cv

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "logreg_cv", spy)
    p, y = binary_data
    SplineCalib(random_state=11, **FAST).fit(p, y)
    assert seen["random_state"] == 11
