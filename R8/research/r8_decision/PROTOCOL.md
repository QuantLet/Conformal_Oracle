# Stronger correction comparators and a past-loss gate

Declared 9 September 2026, after the user's approval of the research plan
and before computing this extension. This is retrospective development on
the previously examined R8 panel, not an untouched confirmatory experiment.

## Inputs and estimand

All 216 R8 model–asset pairs, corrected Lag-Llama calendar forecasts included.
No new data, base-model inference, or GAMLSS fits. Primary alpha is 0.01.
Use the existing 70/30 chronological split and identical per-pair test dates.
Report original-return pinball loss times 10,000, with each pair equally
weighted. Also report calibration-scale-normalised loss, class/model
exclusions, violation frequency, threshold magnitude and fitting diagnostics.
No outcome-based pair removal. Reference results are the saved matched-loss
and policy paths, whose returns, dates and raw predictions must match.

## Common inner split

For every pair let n_cal = floor(.7 n), and v = max(1000, floor(.7 n_cal)).
Require v < n_cal and at least 250 validation observations; otherwise fail
the run rather than silently modify the protocol. Fit on [0,v), select on
[v,n_cal), test on [n_cal,n). Static competitors refit on the full calibration
block after selection. All hyperparameter trials, including unsuccessful
ones, are stored. The gate below deliberately does not refit its static
candidate after validation. Selection ties prefer the simpler state model,
stronger penalty, or lower POT threshold in the order listed below.

## Regularised state correction

Use the existing lagged 20-return volatility, with 1e-8 floor; standardise
log volatility using fitting data only. Model the return residual y-q with
intercept plus linear or cubic powers of this predictor, in units of the
fitting sample's median volatility. Minimise mean original pinball loss in
those units plus lambda times the L1 norm of non-intercept coefficients.
Candidate p = 2,4 and lambda = 0, 1e-4, 1e-3, 1e-2, .1, 1. Select the
smallest validation loss, using fewer coefficients then larger lambda for
exact ties. Refit that specification on all calibration observations.
Solve the convex linear program with primal/dual certificates. Preserve all
finite extrapolations. A separately named diagnostic clips the standardised
log-volatility input to its fitting range, with the SAME fitted coefficients
and selected hyperparameters; it is not the primary estimator.

## POT correction of scores

Two distinct candidates: constant correction of S=q-y, and volatility-scaled
correction of S/sigma. Fit a GPD to strict exceedances over empirical 90th or
95th percentile (linear interpolation); select the threshold on the inner
validation block by return pinball loss, then refit on full calibration.
The exceedance fraction is the actual fraction, not the nominal 10% or 5%.
Use deterministic GPD maximum likelihood with location fixed to zero, after
normalising excesses by their mean. Archive shape, scale, threshold, tail
count, likelihood, support and optimiser status. Use the exponential limit
when shape is near zero. Fewer than 20 excesses, numerical failure, or an
invalid fitted support triggers the corresponding empirical correction
(Shift-CP or Vol-CP), with an explicit failure/fallback record. No tail or
forecast winsorisation. Negative-shape estimates are retained and flagged
when outside the usual regular likelihood range.

This is a plug-in POT comparator. It is not Pasche–Lam–Engelke's confidence-
inflated extreme conformal interval, and carries no asserted finite-sample
conformal coverage guarantee. The distinction is scientifically relevant:
https://arxiv.org/html/2505.08578v3, equations 10–13.

## Finite-threshold DtACI comparator

Use the expert-weight and fixed-share updates of Gibbs–Candès Algorithm 1:
https://jmlr.org/papers/v25/22-1218.html
Author reference code: https://github.com/isgibbs/DtACI/blob/main/DtACI.R

Window = 500 previous scores. Seven step sizes:
.001, .002, .004, .008, .016, .032, .064.
Initial expert levels alpha; equal weights. Fixed share sigma=1/1000 and
eta=sqrt(3*(log(2*7*500)+1)/(500*(alpha*(1-alpha))**2)), following the paper's
fixed heuristic with the actual target alpha. Start at observation 500,
warm through calibration, and update online after each observed outcome.

The operational candidate is explicitly **DtACI-projected**: after each
update project each expert's level to [1/501,500/501]; interpolate the past
empirical score quantile linearly at 1-level. Compute each expert's error
from its actual return threshold. Beta is the inverse of this interpolated
quantile curve, clipped to [0,1] for scores beyond sample support; tied scores
are counted and actual strict violations govern the state update. Projection
and finite quantiles modify the original algorithm; its original coverage
or regret theorem is not claimed for this candidate. Log projection events.

All experts update irrespective of the sampled output. Store every expert
threshold and its pre-outcome probability. Thus expected daily pinball loss
and expected violation rate over expert randomisation are computed exactly,
not as the loss of an averaged threshold. Archive a primary seeded path and
eight independently seeded selection paths for seed-sensitivity and ordinary
backtests. Seeds are derived from 20260909, pair identity and replicate index.
The expected-loss row has no Kupiec/independence count: fractional expected
hits are not a realised Bernoulli path. Primary seed path remains separately
labelled. No choice of seed by performance.

In addition run unprojected states with the extended empirical quantile
(+infinite score threshold for level<=0, -infinite for level>=1), actual
boundary errors, and the same weights/step sizes. Report nonfinite support
probability and seeded nonfinite forecasts. Do not silently drop infinite
forecasts, replace them with finite numbers, or claim an infinite-loss row
is a competitive finite-threshold forecast. This audit distinguishes the
operational adaptation from the unmodified expert mechanism.

## Exploratory past-loss gate

Candidates are Raw, Shift-CP fitted on [0,v), Vol-ERM fitted on [0,v), and
Rolling500 (past-only daily update). Score on [v,n_cal). For each of the
three non-raw candidates compute paired loss differences to raw. Circular
block bootstrap with 499 draws, native-observation blocks 20 and 60, and a
fixed pair-derived seed. For each block length form a simultaneous one-sided
95% upper band using the maximum centred standardised deviation across the
three candidates. Take the larger upper bound over the two block lengths.
Choose the candidate with the smallest upper bound if that bound is strictly
negative, otherwise Raw. Freeze that choice for the test window; rolling
still updates from past scores. Keep static fitting coefficients unchanged
after validation. Archive both bands, losses, selected candidate, and the
same candidate family selected by minimum past loss without a band.

This is a conservative empirical decision heuristic, not a proven no-harm
policy or conformal coverage certificate. Report avoided and forgone gains,
selection frequency and full-window loss. The primary policy comparator is
the existing past-selected rolling window, with Rolling500 and Vol-ERM also
reported. Underpowered selection that always chooses Raw is not sufficient
evidence of usefulness.

## Evaluation and verification

Use 999 common-calendar circular bootstrap draws, blocks 20 and 60 days,
preserving contemporaneous shocks and availability. The simultaneous family
contains all finite new primary competitors and the proposed gate versus
Shift-CP; the clipping diagnostic and primary randomisation seed are clearly
separate sensitivities. Also report paired policy differences to selected
rolling, Rolling500 and Raw. These intervals condition on fitted forecasts
and decisions; they do not include fitting/selection uncertainty.

Tests must include LP primal–dual agreement and the unpenalised reference,
analytic GPD tail inversion including shape zero and negative shape,
DtACI state/weight recursion checked independently, exact mixture losses,
strict boundary violations, and prefix invariance after changing future
outcomes. Verify all 216 input bindings, saved reference paths and current
manuscript hashes; fresh replay of representative pairs must reproduce all
numeric outputs. Store all fitted parameters, daily forecasts, code hashes,
package versions and aggregation receipts. No canonical manuscript result
is overwritten by this development run.

The extension answers whether stronger comparators and a loss-aware gate
warrant the next theory/controlled-experiment phase. Results are reported
even if they eliminate the scalar's advantage or show no gate benefit.
