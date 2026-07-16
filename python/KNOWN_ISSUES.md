# Known issues

## `test_panel/test_t1_diagnostic_regression.py::test_clustered_se_differs_from_ols`

**Status:** pre-existing (fails identically on the un-merged 0.3.0 tree),
unrelated to the 0.3.1 conformal-quantile fix. Deterministic, not flaky.

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
is unaffected and fully green.
