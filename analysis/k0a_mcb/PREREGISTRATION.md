# K0a — is q̂_V a rewriting of uMCB? Pre-registration

Written before `analysis/umcb/MEMO.md` was opened and before any decomposition was
run in this session. The two interpretations below are fixed now, so that reading
the numbers cannot choose between them afterwards.

## The object

Gneiting & Resin (2023) decompose the mean score of a quantile forecast as

    S̄ = MCB − DSC + UNC,

with MCB the miscalibration component, DSC discrimination, UNC the uncertainty of
the unconditional reference. The **unconditional** miscalibration component uMCB
is the part of MCB that a single constant shift of the forecast can remove: the
gap between the mean score of the forecast as issued and the mean score of the
best constant-shifted version of it.

q̂_V is the conformal shift: one number per pair, the order statistic of the
calibration nonconformity scores at level 1−α.

The manuscript already asserts (Section 3.2) that q̂_V "quantifies miscalibration
on a continuous, signed scale", and already says elsewhere that it estimates the
unconditional miscalibration component. R̄ was retracted as an independent
statistic when it turned out to correlate with raw π̂ at Spearman 0.9912. The
partial evidence therefore runs against q̂_V being anything new, and this check is
designed to be able to say so.

## Unit of analysis

One row = one pair (forecaster × asset) at α = 0.01, on the sequence panel:
13 × 24 = **312 rows**.
What varies from row to row: the forecaster and the asset, and therefore the
calibration and test windows. Range: q̂_V from about −0.007 to +0.04 in log-return
units; uMCB in units of the quantile score, of order 1e-5 to 1e-3.

Both quantities are computed on the **test window**, from the same stored series,
with the calibration window used only to fix q̂_V — which is the asymmetry that
matters: q̂_V is estimated out of sample and uMCB in sample. Both an in-sample uMCB
(computed on the test window, using the test-window optimal shift) and an
out-of-sample uMCB (computed on the test window, using the calibration-window
optimal shift) are reported. Conflating them would manufacture the answer.

## The question, in the form it will be answered

Is q̂_V a monotone rewriting of the shift that minimises the test-window quantile
score — that is, does the ordering of the 312 pairs by q̂_V reproduce the ordering
by the score-optimal shift — and does the residual difference matter at α = 0.01?

**Decision rule, fixed now.** "Rewriting" means Spearman correlation between q̂_V
and the score-optimal shift δ* above 0.95 **and** median |q̂_V − δ*| below one
quarter of the interquartile range of q̂_V. Anything else is "differs".

## The two interpretations, written in advance

**If it is a rewriting.** Section 6 reports q̂_V as a *measurement* — the size of
the correction a forecaster needs, in the units of the forecast — and not as a
statistic with independent content. The sentence to write is that the conformal
shift is the unconditional miscalibration component of the Gneiting–Resin
decomposition, estimated out of sample by a single order statistic, and that its
value to a reader is that it is on the scale of the quantity being corrected
rather than on the scale of a score. Nothing in the abstract changes; the abstract
does not claim q̂_V is a new statistic. The ranking table by q̂_V loses its status
as evidence about model quality and keeps its status as a description of how much
correction each pair required. This is the outcome the existing evidence points
to, and it costs the paper nothing it has not already conceded for R̄.

**If it differs at α = 0.01.** The difference is the finding, and it has to be
characterised before it is used: an order statistic at the 1% level of n ≈ 3,000
calibration scores is a different estimator from a score-minimising shift, with
different variance and different sensitivity to the shape of the score
distribution near the tail. If the two orderings separate, the pairs on which they
separate are named and the mechanism identified, and only then does §6 report q̂_V
as carrying information the decomposition does not. A high correlation with a
handful of separating pairs is the *first* interpretation, not the second.

## AE-3

The recalibration improves uMCB by construction and leaves DSC and UNC untouched,
so in-sample scores should improve by approximately the uMCB term. The pre-registered
check: the mean quantile-score improvement from static recalibration, computed on
the calibration window, against the in-sample uMCB of the raw series on that same
window. These should agree to within Monte Carlo error if the intervention does
what the decomposition says it does. Reported as a scatter and a ratio, per pair.

## Negative control

The decomposition is run on a forecast constructed to have zero unconditional
miscalibration (the empirical α-quantile of the test window itself, held constant)
and on one constructed with a large known shift. The first must return uMCB near
zero; the second must return uMCB near the known value. If either fails, the
decomposition code is wrong and no result from it is reported.
