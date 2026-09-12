# Transfer of the existing stronger comparators to ten native-tail models

10 September 2026, authorised by the instruction to continue after the
ten-model native and policy validations. Earlier model losses are known;
this is a retrospective extension of an existing specification. No new
comparison, hyperparameter or exclusion is chosen from its forthcoming
stronger-comparator results.

Use exactly the ten models, 24 assets, 512-return warm-up, native 1% target
and 70/30 split in `r8_model_extension/PROTOCOL.md`. Retain every pair.
The completed common-date and policy paths are cross-checks, not substitutes
for re-estimating competitors whose original calibration support differed.
Keep all existing native forecasts and completed archives unchanged.

Transfer the full definitions in `r8_decision/PROTOCOL.md`: State-L1 and its
separately labelled clipping diagnostic, POT-Shift, POT-Vol, projected DtACI
with exact randomisation-expected loss and eight seeded paths, the
unprojected nonfinite-output audit, the past-loss gate and past minimum.
Reuse its numerical functions unchanged, including inner split, fitting,
selection, seeds, projection, fallbacks and diagnostic conventions.

Recompute the full-calibration controlled family using the unchanged
`r8_commodity_etp/controlled_comparisons.py` functions: Raw, Shift-CP,
Shift-ERM, Vol-CP, Vol-ERM, State2-ERM and State4-ERM. Recompute the existing
125/250/500 rolling-window selection and its calibration-coverage gate on
the common support, without changing the last-30%-of-calibration selection.
The existing ACI comparator keeps its three step sizes (.001,.005,.01),
past-only calibration coverage criterion, clipping and test-start reset.
This is an ACI transfer, not a new online algorithm.

All twenty existing evaluation rows are retained. Store all fitted
coefficients, hyperparameter trials, optimiser certificates, thresholds,
daily paths, adaptive experts/probabilities and seed diagnostics. Compare
Raw/Static/Rolling250 and the past-loss choices against the already
independently validated ten-model archives on every date. Forecasts or
infinite-output probabilities are never removed to improve a comparison.

Compute 999 paired circular calendar-block resamples at 20 and 60 calendar
days using the original `r8_decision/aggregate.py` bootstrap algorithm and
seed rule. Its six simultaneous contrasts against Shift-CP are State-L1,
POT-Shift, POT-Vol, projected-DtACI expected loss, Loss-gate and Past-minimum.
Keep the original pair-equal weighting, calibration-scale normalisation,
non-crypto sensitivity and descriptive class/model exclusions. No new
statistical interpretation of marginal coverage or theorem applicability.

Repeat every pair in a fresh process, compare all numerical outputs, and
independently reconstruct scores/events and mixture losses. Test LP/GPD/
adaptive mechanics with the existing tests and perturb future outcomes for
each of the three new forecasters. Independently reconstruct bootstrap
counts, asset/pair weighting and reported intervals from daily paths. Bind
protocol, code, inputs and outputs. Do not integrate numerical claims until
these checks pass.

This stage does not claim to rerun the broad GAMLSS/boosting/multiplicative
baseline table, all four-alpha experiments or external validation. Those
retain their explicitly identified scope until separately transferred.
The external protocol still requires August 2026 for all twelve industry
portfolios; the official page checked today continues to list July 2026.
