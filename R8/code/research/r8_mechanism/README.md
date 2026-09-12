# Controlled recalibration mechanisms

This development experiment implements [PROTOCOL.md](PROTOCOL.md). The
canonical R8 paper is not changed by these scripts. The mathematical
development is in [RISK_THEORY.md](RISK_THEORY.md).

Use the recorded statistical runtime:

```sh
/private/tmp/irfa-r8-conda-clean/bin/python -m pytest -q research/r8_mechanism/test_engine.py
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_mechanism/run.py --init
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_mechanism/run.py --workers 3
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_mechanism/aggregate.py
```

The initialisation saves 1,500 independent seeds and all generated input
arrays. GARCH inputs exactly reuse the older control's seeds and warm-up.
Shared paths produce 144 configurations with 500 repetitions each. Neither
72,000 configuration repetitions nor 552,000 method evaluations is a count
of independent histories. Completed blocks are checksum-verified and skipped.
Changes in estimator code, source inputs or protocol cause a binding failure.

For fresh-process verification, execute the following four blocks:

```sh
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_mechanism/run.py --single ar normal 0.8 125 0 25 --replay
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_mechanism/run.py --single ar t5 0.8 1000 475 500 --replay
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_mechanism/run.py --single garch normal 0 1000 0 25 --replay
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_mechanism/run.py --single garch t5 0 1000 475 500 --replay
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_mechanism/validate.py
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_mechanism/plot.py
```

`artifacts/r8_mechanism/blocks/` stores all 640 block receipts, fit parameters,
hyperparameter trials, optimiser diagnostics, per-history losses, and
prediction-moment sums. All coefficients and transformations needed to
reconstruct test predictions are retained. `results/` holds the full
replication table, paired summaries, prediction bias/variance decomposition,
count-variance comparisons and the old-reference replay comparison.

The validation script regenerates every input history, checks exact equality
of tail counts across the two AR marginal transformations, checks all
location-equivariance and n=125 POT90 fallback identities, evaluates 32 iid
order-statistic risks by independent beta-density quadrature, and compares
the four fresh-process blocks exactly. The old-reference comparison covers
all 112,000 saved GARCH method/replication cells from the previous experiment.

Original-return pinball loss is the estimand. Multiplying by 10,000 is a
display convention only. Lower is better. Paired Monte Carlo standard errors
are calculated across the 500 independent histories within each law/configuration.
Pointwise bands are descriptive; they are not simultaneous discovery bands.
No finite-moment theorem is asserted for all fitted polynomial/selection
estimators. Median, upper quantiles, maximum prediction and concentration
of regret in the five largest replicates make their extreme fits visible.

The AR experiment evaluates an independent marginal return, not tomorrow's
conditional return. The GARCH experiment knows the true conditional sigma
and evaluates exact loss on a fixed independent grid of volatility states.
Its reported simulation uncertainty conditions on that grid; it does not
measure integration error for the stationary volatility distribution.
POT is an approximation at a fixed threshold, and the 90% fit at n=125
falls back to CP by the predeclared minimum tail count. The L1 selection
method and its very short inner validation window are part of the tested
algorithm, not a claim about all possible regularisation procedures.

The theoretical note's count and quantile arguments are separate from the
computational checks. Its local result requires stronger moment assumptions
and a deterministic separation. It does not validate the contiguous or
rolling estimator, and is not a rare-event alpha_n limit or an estimated
deployment rule. Plot exports use transparent PNG/SVG, vivid colours and
legends outside the panels at the bottom.

An executed-source snapshot, including the in-project imported dependencies,
is retained under `artifacts/r8_mechanism/code_snapshot/`, with original
relative paths and hashes in `final_audit.json`. This preserves the source
used for this development run if the working manuscript or shared helpers
are revised later. The dependency snapshot complements the stored paths,
block receipts and runtime versions; it is not a claim that an arbitrary
Python installation will give bitwise-identical optimiser output.
