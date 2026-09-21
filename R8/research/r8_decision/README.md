# Stronger comparators for the R8 research plan

This is an executed development experiment, separate from the canonical R8
article. Read `PROTOCOL.md` before interpreting any output. Inputs are the
current August 2026 forecasts and return series, including the repaired
Lag-Llama calendar outputs. No foundation-model inference is performed.

From the project root, in the locked R8 statistical environment:

```sh
/private/tmp/irfa-r8-conda-clean/bin/python -m pytest research/r8_decision/test_methods.py -q
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_decision/run.py --workers 3
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_decision/aggregate.py
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_decision/validate.py
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_decision/plot.py
```

The environment is the R8 Conda lock documented in the base release. An
equivalent installation can use a different interpreter path; reproducing
package versions alone need not reproduce different binary builds.

`run.py` verifies completed outputs before resuming. Each of the 216 pair
directories stores forecasts, fitted parameters, every tuning trial, LP
certificates, all DtACI experts and pre-outcome probabilities, seeded
backtests, input hashes and a completion receipt. `aggregate.py` verifies
all bindings again and computes equal-pair means and common-calendar block
intervals. Its scale and cryptocurrency exclusions are declared sensitivities,
not filters defining a preferred result.

`validate.py` independently reruns four pairs spanning both TSFM interfaces
and an econometric benchmark, and perturbs future outcomes to check the
complete selection boundary. It records exact replay and current manuscript
hashes in `artifacts/r8_decision/validation.json`.

Primary outputs:

- `artifacts/r8_decision/results/summary.csv`
- `artifacts/r8_decision/results/intervals.csv`
- `artifacts/r8_decision/results/sensitivity_intervals.csv`
- `artifacts/r8_decision/results/decisions.csv`
- `artifacts/r8_decision/results/diagnostics.csv`
- `artifacts/r8_decision/comparison_frontier.png` and `.svg`
- `docs/IRFA_STRONG_COMPARATORS_RESULTS.md`

The first implementation run is retained under `implementation_v0` only for
provenance. Before aggregate outcome inspection, a tied-score inverse-CDF
edge case was identified, fixed and covered by a new independent test. All
216 pairs were rerun. That superseded run is not an alternative result set.

The finite projected DtACI comparator is explicitly an operational adaptation.
The unprojected expert mechanism's nonfinite outputs are reported separately;
none are discarded from scoring to make that method appear competitive.
The POT comparator is a plug-in tail fit, not an extreme-conformal confidence
bound. The loss gate is an empirical policy, not a certified no-harm rule.

The inference conditions on estimated forecasts and selected policies.
Neither this retrospective experiment nor its resampling intervals establish
external validation or account for all fitting and researcher-selection
uncertainty. No canonical R8 table is overwritten by these scripts.
