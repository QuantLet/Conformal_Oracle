# Changelog

## [Unreleased]

### Changed

- **The links return to `danpele/Conformal_Oracle`, which is now canonical in
  every sense.** Control of the repository authorised to publish is control of
  the package: with trusted publishing, anyone who can modify or trigger the
  release workflow in that repository can publish a version of it. On a personal
  account that is the author alone; in an organisation it is everyone with write
  access there, and the administrators who can change environments, workflows
  and protection rules. The publisher stays on the personal account, so the
  links point at the same place. Package metadata, README links, Colab badges,
  clone instructions and `CITATION.cff` all move back.
- **`QuantLet/Conformal_Oracle` becomes an automatic mirror**, updated by a
  workflow here on every push to `main` and every `v*-python` tag, through a
  deploy key with write access to that repository alone. Its release workflow
  stays disabled, so only this repository can publish, and its README says it is
  a mirror.
- `PUBLISH_LOG.md` and the historical files under `R8/history/` keep their links
  as records of what happened at the time.

## [0.5.1] - 2026-09-21

### Changed

- **Documentation only; the package code is identical to 0.5.0.** The
  description that reaches PyPI cannot be edited after a release, so these
  corrections ship as a patch.
- The opening paragraph announces 0.5.0's intensity rather than 0.4.0's
  separated protocol.
- The earlier manuscript is cited by one deposit tag, `R8-2026-09-13-repair1`,
  the deposit its R7 and R8 APIs shipped from, with its later extensions named;
  the current manuscript's deposit, `R9-2026-09-21`, is identified as separate,
  so the `R9` prefix no longer stands for two manuscripts without qualification.
- The result about estimating the intensity is attributed to the estimator the
  manuscript actually tests, a plug-in built from a Bartlett long-run variance
  and a kernel density at the fitted shift.
  `diagnostics.optimism.first_order_shrinkage` is named as a different
  estimator of the same quantity, keeping `validated=False` for its own reason.

## [0.5.0] - 2026-09-21

### Added

- **`ConformalShift(intensity=...)`**: the deployed correction is
  `intensity * c_hat`, validated to lie in `[0, 1]`. The default stays `1.0`,
  the whole conformal shift, so no result from 0.4.0 or earlier moves unless the
  argument is passed. `ConformalShift.shift` exposes the correction actually
  applied.

  Intensity `0.5` is the average of the raw and the fully corrected threshold.
  At that intensity the leading coefficient of the local-bias corollary is
  `(f/8)(sigma^2 - 3 delta^2)`, so the correction pays over a region three times
  wider in squared bias than the whole shift, costs a quarter as much when no
  correction was needed, and is the optimal intensity at the boundary where the
  whole shift stops paying. On the evaluated panels it lowered quantile loss
  against the whole shift at every calibration length in every universe, and
  against the raw forecast once the shift was fitted on 1000 calibration pairs.
  The recommended static setting is `intensity=0.5` with at least 1000
  calibration pairs; that evidence is retrospective.

- **A short-window warning.** `ConformalShift.fit` raises a `UserWarning` when
  `ceil((n + 1)(1 - alpha)) >= n`, at `alpha = 0.01` any window of 198 pairs or
  fewer, stating that the fitted shift is the largest calibration score.

- **Tests** for the intensity, the endpoints, the rejection of values outside
  `[0, 1]`, the warning, and the agreement of the two version declarations.

### Changed

- **Canonical repository.** Every live link in the package metadata, the README
  and the notebooks now points to `QuantLet/Conformal_Oracle`, which holds the
  source of every published release. The links previously pointed to
  `danpele/Conformal_Oracle`, which is three releases behind, so the PyPI page
  for 0.4.0 sends readers to 0.3.0-era code. `PUBLISH_LOG.md` keeps its links as
  the record of where the 0.3.0 release happened. This was reversed before the
  next release; see the unreleased entry at the top.
- **Release workflow** moved into this repository, tag-triggered on `v*-python`
  through trusted publishing. It stays inert until the PyPI trusted publisher is
  transferred, so that only one repository can publish.
- README: the companion-paper citation now names the current manuscript; a new
  correction-intensity section; a note on how the intensity relates to the R7
  pre-deployment indication rule; a sentence in the regime-classification
  documentation stating that the diagnosis uses the whole fitted shift.

### Note

The intensity is not estimated from data anywhere in the public API. The
estimator of the same quantity, `diagnostics.optimism.first_order_shrinkage`,
keeps `validated=False`: on the evaluated panels it loses to the fixed `0.5` in
every supported comparison between them, and at short windows it degenerates to
the whole shift.

## [0.4.0] - Unreleased

### Added (R8, 13 September 2026)

- `conformal_oracle.recalibration.one_coefficient`: `shift_cp`, `shift_erm`,
  `vol_cp`, `vol_erm`, `weighted_quantile`, `fit_one_coefficient_corrections`
  and `OneCoefficientCorrections.apply`.
- `conformal_oracle.diagnostics.optimism`: `blocked_cv_optimism` (factor
  `2(K-1)/(2K-1)`), `block_bootstrap_optimism`, `training_loss_change`,
  `first_order_shrinkage` (always `validated=False`).
- `conformal_oracle.diagnostics.paired_bootstrap`: `paired_calendar_bootstrap`
  with pointwise and simultaneous bands, `calendar_seed`.
- `past_loss_selection` and `PastLossSelection` in `conformal_oracle.deployment`.
- `tests/test_r8_extensions.py`, including archive-backed reproductions of the
  stored optimism estimates and of the published static-minus-raw bands.
- README and citation updated to manuscript R8 and its replication tag.

### Added (R7 workflows)

- `SeparatedSplitConformalVaR`: explicit-gap single-split protocol with
  calibration, unused-gap and evaluation indices. The shift uses the same
  calibration order statistic as the contiguous estimator.
- `proxy_separation_gap`: R7 absolute lag-one persistence proxy, with the
  numerical-zero fallback applied only at or below `1e-12`. Returned metadata
  explicitly states `proxy_based=True` and `certified=False`.
- `recalibration_indication`: calibration-only Basel-or-Kupiec deployment
  decision, with diagnostics, reasons and calibration provenance.
- `selectively_recalibrate`: static or causal rolling correction using a
  fixed pre-deployment decision; the skip path preserves raw forecasts.
- Focused correctness, validation, indexing, leakage and compatibility tests;
  an executable R7 API example and an optional read-only artifact integration
  script. Examples, scripts and test fixtures are included in the sdist.

### Compatibility and scope

- Existing contiguous/rolling computations, signatures and stored manuscript
  outputs are unchanged. The pre-existing 0.3.4 bootstrap correction is retained.
- Theorem 4.5 applies to the separated construction under its maintained
  assumptions, not automatically to any chosen gap. The proxy is not a
  validated mixing-rate estimator or a certificate of those assumptions.
- The indication rule is an operational policy, not a promise of better
  Quantile Score. Rolling updates do not change the initial decision.
- Prepared as a local candidate; this entry does not imply PyPI publication.

## [0.3.4] - 2026-08-24

Prepared locally on 2026-08-31; documentation refreshed for manuscript R7 on
2026-09-06. These preparation dates are not PyPI publication dates.

### R7 documentation and packaging

- Update the companion manuscript title and credit Gneiting and Resin (2023)
  for the calibration/score-decomposition framework, including constant
  translation; distinguish the displacement from its associated score gain.
- Distinguish contiguous static and rolling APIs from the separated estimator
  covered by R7 Theorem 4.5. The separated protocol and pre-deployment
  indication rule are not implemented as package APIs.
- Document the finite conformal rank, maximum-score fallback and its lack of
  the usual coverage guarantee, and the fact that lower Quantile Score is
  better. Correction-magnitude labels do not validate a forecaster.
- Update active API documentation and identify older replication notebooks as
  legacy protocols, not reproductions of the current R7 tables.
- Include methodology, conventions, known limitations and release notes in
  the source distribution. Retain version 0.3.4 because the existing candidate
  was not published; preserve previously built artifacts separately.
- No new numerical or API behaviour changes relative to the already prepared
  0.3.4 source. The bootstrap fix below remains a behaviour change relative
  to the published 0.3.2 release.

### Fixed

- **`bootstrap_qv_ci` computed its replicates with a different estimator from the
  point estimate it was reported around.** Each bootstrap replicate used
  `np.quantile(sample, 1 - alpha)`, the plain empirical quantile, while
  `conformal_shift` returns `conformal_quantile()`, the
  `ceil((n + 1)(1 - alpha))`-th order statistic. The two differ by one order
  statistic — a median 5% of the shift on the study panel, and far more on short
  windows. The interval was therefore centred on a quantity the caller was not
  estimating. Replicates now call `conformal_quantile`.

### Note on short calibration samples

`conformal_quantile` returns the sample maximum whenever
`ceil((n + 1)(1 - alpha)) >= n`, which holds for `n < 2/alpha - 1` — at
`alpha = 0.01`, for every calibration sample of 198 observations or fewer. In
that regime the shift is an extreme-value statistic with no stable variance, not
a noisy interior order statistic. This has always been the behaviour; it is now
documented, and `cfp_config.conformal_index` reports it.

## [0.3.3] - 2026-08-22

### Fixed (documentation of a released estimator)

- **`conformal_shift` and `compute_qv_roll` documented an estimator they do not
  implement.** Both docstrings described the correction as the plain empirical
  `(1 - alpha)` quantile of the nonconformity scores. Both call
  `conformal_quantile()`, which returns the finite-sample split-conformal
  threshold, the `ceil((n + 1)(1 - alpha))`-th order statistic. The
  implementation has been correct since 0.3.1; the description has been wrong
  since the function was introduced, in every released version.

  **No behaviour changes in this release.** `conformal/quantile.py` is untouched
  and is byte-identical to the file shipped in 0.3.1 and 0.3.2. Numerical output
  of 0.3.3 is identical to 0.3.2 on all inputs.

  **Why a release rather than a note.** A user reading the docstring and
  reimplementing from it obtains `np.quantile(scores, 1 - alpha)`, which differs
  by one order statistic. The gap is `O(1/n)`: negligible on a long calibration
  set, material on a 250-day window, and large enough to change the *sign* of the
  correction when the calibration scores straddle zero near the `(1 - alpha)`
  level. On the panel used in the accompanying paper the two conventions differ
  in sign on at least one asset (-0.000176 against +0.000708). A user who
  compared their reimplementation against this package and found a discrepancy
  would have had the docstring on their side and the code against them.

  This is the second defect this convention has produced, in the opposite
  direction to the first: 0.3.1 fixed an implementation that had drifted from the
  documented estimator, and 0.3.3 fixes documentation that had drifted from the
  implemented one.

### Known issue

- `conformal-oracle/` in the accompanying research repository is a stale
  duplicate of this package at version 0.3.1 and carries the same docstring
  defect. It is not the distribution source and should be reconciled against
  `python/` or removed before any JOSS submission.

## [0.3.2] - 2026-07-16

### Documentation
- README now describes the 0.3.1 rolling-mode coverage fix and the new
  `diagnose_scale` / `ScaleDiagnostic` and `ACICalibrator` APIs, and corrects
  the rolling-mode description (trailing 250-day window, not an expanding one).
  No code changes; identical to 0.3.1 in behaviour.

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

[0.3.0]: https://github.com/danpele/Conformal_Oracle/compare/v0.2.2-python...v0.3.0-python
[0.2.2]: https://github.com/danpele/Conformal_Oracle/compare/v0.2.1-python...v0.2.2-python
[0.2.1]: https://github.com/danpele/Conformal_Oracle/compare/v0.2.0-python...v0.2.1-python
[0.2.0]: https://github.com/danpele/Conformal_Oracle/releases/tag/v0.2.0-python
