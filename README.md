# pysplinecalib

Spline-based probability calibration for binary and multiclass classifiers, in pure
Python.

`pysplinecalib` is a fork of [`splinecalib`](https://github.com/numeristical/splinecalib)
by Brian Lucena. It implements the method described in
[*Spline-Based Probability Calibration*](https://arxiv.org/abs/1809.07751). The
Cython extension has been replaced with NumPy/SciPy, so the package installs with no
compiler.

## Installation

From PyPI:

```bash
pip install pysplinecalib
```

From Github:

```bash
pip install git+https://github.com/Surgical-Recovery-and-Safety-Lab/pysplinecalib.git
```

For development:

```bash
git clone https://github.com/Surgical-Recovery-and-Safety-Lab/pysplinecalib.git
cd pysplinecalib
pip install -e ".[dev]"
```

Requires Python 3.10 or later.

## Usage

Fit the calibrator on model outputs from a held-out calibration set, then use it to
calibrate new predictions.

### Binary classification

```python
from pysplinecalib import SplineCalib

# p_calib: predicted probability of class 1, shape (n_samples,)
# y_calib: true labels in {0, 1}, shape (n_samples,)
calibrator = SplineCalib()
calibrator.fit(p_calib, y_calib)

p_test_calibrated = calibrator.calibrate(p_test)
```

A 2-column array of shape `(n_samples, 2)` is also accepted, and `calibrate` returns
the same shape it was given.

### Multiclass classification

```python
# probs_calib: shape (n_samples, n_classes); y_calib: integers in 0..n_classes-1
calibrator = SplineCalib()
calibrator.fit(probs_calib, y_calib)

probs_test_calibrated = calibrator.calibrate(probs_test)  # rows sum to 1
```

One binary calibrator is fit per class, and the calibrated outputs are renormalised.
The individual calibrators are available as `calibrator.binary_splinecalibs`.

### Diagnostics

```python
calibrator.show_calibration_curve()   # calibrated vs. uncalibrated probability
calibrator.show_spline_reg_plot()     # cross-validated loss vs. regularization
```

For multiclass calibrators, pass `class_num=` to select a class.

### Main parameters

| Parameter | Default | Purpose |
|---|---|---|
| `knot_sample_size` | `30` | Number of knots sampled from the training values |
| `reg_param_vec` | 17 values in `[1e-4, 1e4]` | Regularization strengths searched by cross-validation |
| `cv_spline` | `5` | Number of cross-validation folds |
| `param_search_mode` | `'fast'` | `'fast'` scores lambda on one fold, `'full'` on all folds |
| `unity_prior` | `True` | Add synthetic points on `y = x` to favour the identity |
| `unity_prior_weight` | `20` | Total weight of the synthetic points |
| `logodds_scale` | `True` | Fit the spline on the log-odds scale |

See `help(SplineCalib)` for the full list.

## Differences from `splinecalib`

- Pure Python: the Cython loss and gradient are now NumPy/SciPy code, and the spline
  basis is vectorised.
- The final fit in `logreg_cv` now uses all the data and the requested optimisation
  `method`. Upstream refit on the last cross-validation training split, so
  calibrated values can differ slightly.
- With custom `unity_prior_gridpts`, the prior weight is no longer applied twice.
- Multiclass calibrators now inherit `reg_prec`, `param_search_mode`,
  `force_knot_endpts` and `logodds_eps` from the parent.
- `fit` no longer modifies the array passed in.

See [CHANGELOG.md](CHANGELOG.md) for the full list.

## Development

```bash
ruff check src
ruff format --check src
pytest --cov=pysplinecalib
```

## License

MIT. Copyright (c) 2022 numeristical and (c) 2026 Surgical Recovery and Safety
Laboratory. See [LICENSE](LICENSE).

## References

Lucena, B. *Spline-Based Probability Calibration.* <https://arxiv.org/abs/1809.07751>
