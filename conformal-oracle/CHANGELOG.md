# Changelog

All notable changes to `conformal-oracle` are documented here. The format is
loosely based on [Keep a Changelog](https://keepachangelog.com/); versions
follow the project's own numbering.

## [0.3.1] — 2026-07-15

Supporting diagnostics for *Recalibrating Tail Event Forecasts under Temporal
Dependence*: a coverage bugfix in the conformal quantile, plus a scale
diagnostic and an ACI baseline.

### Fixed
- **Split-conformal quantile now finite-sample valid (coverage bugfix).** The
  conformal correction previously used the plain empirical quantile
  `np.quantile(scores, 1 - alpha)`, which loses the finite-sample coverage
  guarantee: the correct threshold is the `ceil((n + 1)(1 - alpha))`-th order
  statistic (Vovk et al. 2005; Lei et al. 2018), one step more conservative.
  The gap is `O(1/n)` — negligible on large calibration sets but material at
  short windows. On the **250-day rolling correction** it systematically
  **under-covered**: mean realised violation 0.016 against a 0.010 target
  across the 216 model–asset pairs, versus 0.0098 once corrected (Basel
  Green-zone share 72% → 95%). Users who ran `audit(..., mode='rolling')` or
  `compute_qv_roll_from_scores` on releases up to and including 0.3.0 have
  rolling results that under-cover and should re-run on 0.3.1. The static
  correction (`compute_qv_stat`, `ConformalShift`, static `audit`) is affected
  only negligibly because it calibrates on a large window, but was corrected
  for consistency. `AdaptiveConformalInference` / `ACICalibrator` are
  unchanged: they calibrate on the full (large) calibration set where the
  correction is immaterial, and their adaptive miscoverage level is a distinct
  design choice.
- New shared helper `conformal.quantile.conformal_quantile(scores, alpha)` is
  the single source of the correct order statistic, used by the rolling,
  static, and `ConformalShift` correction paths.

### Added
- `recalibration.diagnose_scale()` and `recalibration.ScaleDiagnostic`: a
  location-scale diagnostic that fits `VaR_cp = a_hat + b_hat * VaR_raw` by
  alpha-level linear quantile regression (reusing `LinearQuantileRegression`)
  and reports the share of the one-parameter conformal shift attributable to
  the multiplicative term. Diagnostic only — not a competing corrector.
- `recalibration.ACICalibrator`: Adaptive Conformal Inference (Gibbs & Candès,
  2021) wrapped as a calibrator that is API-consistent with the rolling
  `RecalibrationMethod` (`fit` / `apply` / `apply_online`), with step-size
  `gamma` selected from the grid `{0.001, 0.005, 0.01, 0.05}` by first-half
  validation. Wraps the existing `AdaptiveConformalInference` (kept unchanged
  for backward compatibility).

### Tests
- `test_recalibration/test_t11_scale_diagnostic.py` (7 tests).
- `test_recalibration/test_t12_aci_calibrator.py` (9 tests).

### Changed
- Version bumped to `0.3.1`. The latest release on PyPI is `0.3.0`, so `0.3.1`
  is the correct next version; the committed tree had merely gone stale
  (`pyproject.toml = 0.1.0`, `__version__ = 0.2.0-beta`) and both are now
  aligned to `0.3.1`.
