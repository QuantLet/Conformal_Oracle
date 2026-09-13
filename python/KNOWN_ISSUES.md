# Known issues

## R7 scope and current limitations

- `audit` still exposes contiguous static and rolling modes. Use the separate
  `SeparatedSplitConformalVaR` API for the R7 single-split construction.
  `proxy_separation_gap` does not estimate a certified beta-mixing rate or
  establish the theorem's assumptions.
- `recalibration_indication` implements the calibration-only rule. Legacy
  `signal-preserving`/`replacement` labels describe correction magnitude,
  not forecast information content or a decision to deploy recalibration.
- When the conformal rank exceeds the calibration-sample size, the helper
  returns the sample maximum instead of `+inf`. The finite fallback lacks
  the usual finite-sample coverage guarantee. Empty scores return `0.0`.
- `RegimeVerdict.R_bootstrap_ci` is misnamed: static mode passes through the
  shift CI rather than a ratio CI; rolling mode supplies a shift
  mean-plus/minus-1.96-standard-deviation band, not a bootstrap CI for `R`.
  Version 0.4.0 does not change this legacy API behaviour.
- The decision API cannot determine whether arrays a caller labels
  "calibration" were genuinely available before deployment. Its restricted
  signature, provenance checks and causal rolling replay prevent internal
  leakage, not misuse of caller-supplied data or a noncausal base forecaster.

## Historical test issue (superseded)

## `test_panel/test_t1_diagnostic_regression.py::test_clustered_se_differs_from_ols`

**Current status:** the named test no longer exists. It was replaced by
`test_cluster_se_matches_independent_reference` (a statsmodels reference)
and `test_cluster_se_collapses_to_ols_without_clustering` (a negative control).
The historical account below is retained for context, not as a current failing
test or a reason to weaken the replacement checks.

**Historical status:** pre-existing on the un-merged 0.3.0 tree and unrelated
to the 0.3.1 conformal-quantile fix. Deterministic, not flaky.

**Symptom:** the test asserts clustered SEs differ from OLS SEs by more than a
threshold; for `pi_raw` they differ by only **4.6%**, failing the check.

**Do not "fix" by moving the threshold yet.** There is a docstring/assertion
mismatch to resolve first: the docstring says the SEs should differ by **>50%**
while the assertion checks **>10%**, and the observed value is 4.6%. The likely
root cause is upstream of the threshold: the synthetic panel **fixture is not
generating the intended within-cluster dependence**, so clustered and OLS SEs
come out close by construction. Investigate the fixture (`tests/fixtures/`)
before changing any threshold — otherwise a loosened threshold would mask a
fixture that is not testing what the name claims.

**Scope:** panel diagnostic-regression statistics only. The released conformal
recalibration functionality (rolling/static correction, ACI, scale diagnostic)
was not affected by that historical test-design issue. Current validation
results must be read from the latest release validation report.
