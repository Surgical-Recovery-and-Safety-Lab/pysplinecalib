# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to Semantic Versioning ([SemVer](https://semver.org/spec/v2.0.0.html)).


## [Unreleased]

### Added
* New structure in the package so `tests` have their own folder and source code is
in the `src/pysplinecalib` folder.
* The `CHANGELOG.md` file.
* `SplineCalib` is exported from the package root.

### Changed
* Updated the `pyproject.toml` file to include formatting and author information.
* Updated the version number in the `__init__.py` file from 0.0.13 to 0.0.1.

* Pure Python/NumPy implementation: the Cython loss was replaced by `loss_fun.py`
(stable `logaddexp` loss, `X.T @ resid` gradient) and the spline basis is vectorised.
* `logreg_cv` builds the fold splits once, and the final fit now uses all the data
and the chosen `method` (it previously reused the last fold's training split).

### Fixed
* `logodds_eps` string check used `type(x == str)` and is now `isinstance`.
* `fit` no longer clips `y_model` in place.
* The unity prior weight was squared when custom `unity_prior_gridpts` were given.
* Multiclass calibrators now receive `reg_prec`, `param_search_mode`,
`force_knot_endpts` and `logodds_eps`.
* Removed a stray `build-backend` key from the `[project]` table.

### Removed
* Cython sources (`loss_fun_c.pyx`/`.c`) and the Cython build requirement.
* Setup and documentation that was not needed from the fork.

