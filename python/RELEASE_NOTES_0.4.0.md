# conformal-oracle 0.4.0 — R7 workflows and R8 analysis tools

Prepared locally on 2026-09-07; extended on 2026-09-13 with the R8 tools
(one-coefficient corrections, nuisance-free optimism estimators, paired
calendar bootstrap, past-loss selection). Publication is a separate step.

This minor release adds the separated single-split estimator, the
non-certified R7 proxy-gap utility, the calibration-only Basel-or-Kupiec
indication rule, and selective static/causal rolling application. Explicit
result metadata distinguishes the estimator, information window and correction.

The existing contiguous/rolling algorithms, parameter names and defaults are
unchanged; the quality pass below corrects type annotations. The prior 0.3.4 bootstrap fix remains included; it changes bootstrap
intervals relative to public 0.3.2, not point estimates. No stored forecasts,
manuscript results, empirical tables or generated numbers are modified.

The separated construction belongs to Theorem 4.5 only under its maintained
assumptions. A user-specified gap does not verify those assumptions, and the
absolute lag-one score autocorrelation is an operational persistence proxy,
not a validated beta-mixing-rate estimator. Gap metadata is always
`proxy_based=True, certified=False`. Contiguous static and rolling correction
do not acquire a dependence guarantee through this release.

The indication uses the complete calibration window, unrounded annualized
Basel counts, and strict `Kupiec p < kupiec_level`. The existing trailing-window
Basel diagnostic is unchanged. Decisions contain their evidence, reasons and
calibration fingerprint. Rolling replay updates the correction using past
outcomes, never the initial decision. Skip preserves raw quantile bytes.

The optional `scripts/reproduce_r7_deployment.py` reads full replication
artifacts and reports calibration decisions separately from ex-post outcomes.
No target manuscript counts are hard-coded into algorithm outputs. The sdist
contains its integration script, unit tests, fixtures, examples and docs;
external manuscript datasets are not bundled.

Remaining limits: caller-supplied chronology cannot be independently verified;
legacy `RegimeVerdict.R_bootstrap_ci` semantics remain documented but unchanged;
the finite maximum-score fallback does not retain the usual conformal
finite-sample coverage guarantee. See `KNOWN_ISSUES.md` and the release
validation report for test skips and any outstanding lint/type-check failures.

## Local quality pass (2026-09-07)

Ruff findings in examples, notebooks and tests have been corrected. Audit
and recalibration annotations now distinguish the two result types and use
the existing recalibration protocol. Numerical pandas boundaries explicitly
request NumPy arrays. Lazy optional models have concrete type annotations,
and isotonic `apply()` before `fit()` now raises a clear `RuntimeError`.

Mode-dependent keyword forwarding remains dynamically typed (`Any`); direct
worker parameters retain their types. This does not statically validate every
forwarded option. No checker rules, exclusions or ignore directives were added.
The whole-package mypy run still fails on unavailable or untyped dependencies;
see `docs/quality_checks.md` for the environment and remaining scope.
