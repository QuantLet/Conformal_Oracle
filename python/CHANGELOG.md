# Changelog

## [0.3.1] - 2026-07-15

### Fixed
- **Split-conformal quantile now finite-sample valid (coverage bugfix).** The
  conformal correction used the plain empirical quantile
  `np.quantile(scores, 1 - alpha)`, which loses the finite-sample coverage
  guarantee: the correct threshold is the `ceil((n + 1)(1 - alpha))`-th order
  statistic (Vovk et al. 2005; Lei et al. 2018), one step more conservative.
  The gap is `O(1/n)` — negligible on large calibration sets but material at
  short windows. On the **250-day rolling correction** it systematically
  **under-covered**: mean realised violation 0.016 against a 0.010 target,
  versus 0.010 once corrected (Basel Green-zone share 72% → 95% on the 216
  model–asset panel of the companion study). Users who ran
  `audit(..., mode='rolling')` or `compute_qv_roll_from_scores` on releases up
  to and including 0.3.0 have rolling results that under-cover and should
  re-run on 0.3.1. Static correction (`compute_qv_stat`, `ConformalShift`,
  static `audit`) is affected only negligibly (large calibration window) but was
  corrected for consistency. `AdaptiveConformalInference` / `ACICalibrator` are
  unchanged. New shared helper `conformal.quantile.conformal_quantile`.

### Added
- `recalibration.diagnose_scale()` / `recalibration.ScaleDiagnostic`: a
  location-scale diagnostic reporting the multiplicative (scale) share of the
  one-parameter conformal shift. Diagnostic only.
- `recalibration.ACICalibrator`: Adaptive Conformal Inference (Gibbs & Candès
  2021) wrapped as a calibrator with `gamma` selected by first-half validation.

## [0.3.0] - 2026-05-12

### Migration

See [`docs/migration_v0.3.md`](docs/migration_v0.3.md) for a complete
upgrade guide including import path changes and the new `[benchmarks]` extra.

### Added
- `forecast=` parameter on `audit()` for dependency-agnostic audits
  using pre-computed quantile paths.
- `classify_regime()` top-level entry point returning a `RegimeVerdict`
  dataclass.
- `compare_forecasters()` top-level entry point returning a
  `ComparisonResult` with pairwise Diebold-Mariano tests.
- `conformal_oracle.contrib.benchmarks` subpackage: canonical home for
  GJR-GARCH, GARCH-Normal, and Historical Simulation forecasters.
- `conformal_oracle.contrib.tsfm` subpackage: canonical home for TSFM
  wrappers (Chronos, Lag-Llama, Moirai, TimesFM).
- `[benchmarks]` optional-dependency extra for `arch>=6.0`.
- `[all]` meta-extra installing benchmarks + all TSFMs.

### Changed
- `arch>=6.0` removed from core dependencies (moved to `[benchmarks]`).
- `audit()` signature: `forecaster` is now optional (keyword-only when
  `forecast=` is used).
- `conformal_oracle.forecasters` is now a compatibility shim that
  re-exports from `contrib.*` with `DeprecationWarning`.

### Deprecated
- `audit_static()`, `audit_rolling()`, `audit_with_benchmarks()` as
  top-level imports. Use `audit(mode=...)` or `compare_forecasters()`.
- Importing forecasters from `conformal_oracle.forecasters` (use
  `conformal_oracle.contrib.benchmarks` or `.contrib.tsfm`).

## [0.2.2] - 2026-05-11

PyPI metadata fix (project description rendering).

## [0.2.1] - 2026-05-11

Worked example notebooks, CITATION.cff, Trusted Publisher workflow.

## [0.2.0] - 2026-05-10

Initial public release. Static and rolling conformal audit pipelines,
9 recalibration baselines, 4 TSFM wrappers, panel-level inference,
full backtesting diagnostics.

[0.3.0]: https://github.com/QuantLet/Conformal_Oracle/compare/v0.2.2-python...v0.3.0-python
[0.2.2]: https://github.com/QuantLet/Conformal_Oracle/compare/v0.2.1-python...v0.2.2-python
[0.2.1]: https://github.com/QuantLet/Conformal_Oracle/compare/v0.2.0-python...v0.2.1-python
[0.2.0]: https://github.com/QuantLet/Conformal_Oracle/releases/tag/v0.2.0-python
