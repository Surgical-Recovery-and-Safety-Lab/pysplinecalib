# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to Semantic Versioning ([SemVer](https://semver.org/spec/v2.0.0.html)).


## [Unreleased]

## [0.0.1] - 2026-10-01

First release of `pysplinecalib`, a pure Python fork of `splinecalib` 0.0.13 by
Brian Lucena. Results can differ slightly from upstream (see *Changed* and *Fixed*).

### Added
* New structure in the package so `tests` have their own folder and source code is
in the `src/pysplinecalib` folder.
* The `CHANGELOG.md` file.
* `SplineCalib` is exported from the package root.
* Unit tests for the loss, gradient, spline basis, cross-validation and the
`SplineCalib` API (coverage from 74% to 98%).
* Type hints throughout the package, a `py.typed` marker and a `mypy` check.
* GitHub Actions workflows for tests (Python 3.10-3.12 and the minimum supported
dependency versions), and for lint, type checking and a build check.
* A README with installation, usage and a list of differences from `splinecalib`.

### Changed
* Fold assignment (`get_stratified_foldnums`) and knot selection (`_get_knot_vec`) use a
local `numpy.random.Generator` seeded from `random_state` instead of reseeding the
global `np.random` and `random` modules. The global random state is no longer
modified, but the folds and sampled knots differ from earlier versions, so calibrated
outputs can change slightly.
* Updated the `pyproject.toml` file to include formatting and author information.
* Raised the dependency minimums to `numpy>=2.1`, `scipy>=1.14` and `matplotlib>=3.9`;
`pandas` and `joblib` were not used by the package and are no longer runtime
dependencies (`pandas` and `scikit-learn` are dev dependencies for the tests).
* `SplineCalib.logodds_eps` now has a placeholder value before `fit` when set to
`'auto'`, and a non-numeric value other than `'auto'` raises at construction.
* Updated the version number in the `__init__.py` file from 0.0.13 to 0.0.1.
* Pure Python/NumPy implementation: the Cython loss was replaced by `loss_fun.py`
(stable `logaddexp` loss, `X.T @ resid` gradient) and the spline basis is vectorised.
* `logreg_cv` builds the fold splits once, and the final fit now uses all the data
and the chosen `method` (it previously reused the last fold's training split).

### Fixed
* Binary calibration accepts `(n_samples, 1)` input, and any unsupported shape raises a
clear `ValueError` (it previously crashed with `UnboundLocalError`).
* `random_state` is now passed to the cross-validation fold assignment in `fit`; it was
previously ignored and the folds always used seed 42.
* `logodds_eps` string check used `type(x == str)` and is now `isinstance`.
* `fit` no longer clips `y_model` in place.
* The unity prior weight was squared when custom `unity_prior_gridpts` were given.
* Multiclass calibrators now receive `reg_prec`, `param_search_mode`,
`force_knot_endpts` and `logodds_eps`.
* Removed a stray `build-backend` key from the `[project]` table.

### Removed
* Cython sources (`loss_fun_c.pyx`/`.c`) and the Cython build requirement.
* Setup and documentation that was not needed from the fork.

