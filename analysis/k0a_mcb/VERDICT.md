# K0a — verdict: q̂_V is the argmin, uMCB is the value, and neither orders the panel

Run against `PREREGISTRATION.md` in this directory. 312 pairs, 13 forecasters ×
24 assets, α = 0.01. `k0a_result.json` carries every figure.

## Correction to the task premise

Phase 2 was not "never begun". It was run on 15 August, broke on the corrected
panel, and was re-run on 20 August; `analysis/umcb/MEMO.md` and
`analysis/umcb/RERUN_NOTE.md` are its output and the macro `\nuMCBShareWellSpec`
= 0.270 in `numbers.tex` comes from it. What that run did **not** do is the
comparison the pre-registration asks for, so this is new work, not a repeat.

## The decomposition separates into two objects, and they are not the same one

For pinball loss at level α, the shift minimising the mean score is
$\delta^\star = Q_{1-\alpha}(\{\hat q^{lo}_t - r_t\})$ — the plain empirical
quantile of the nonconformity scores. Checked numerically against a 4001-point
grid on five random pairs: closed form and grid argmin agree to between 5e-6 and
7e-4, within the grid spacing.

- $\delta^\star$ is a **shift**, in return units. $\hat q_V$ estimates it.
- uMCB is the **score reduction achieved at that shift**. It is not a shift.

The relation is $\mathrm{uMCB} \approx \tfrac12 f \hat q_V^2$ with $f$ the residual
density at the α-quantile, and the implied $f$ on this panel spans a factor of 481
(`analysis/umcb/MEMO.md`). So the answer to "is q̂_V a rewriting of uMCB?" is:
**no, it is a rewriting of the argmin, and the map from the argmin to uMCB is not
one-to-one across pairs.**

## The measurement

| comparison | all 312 | well-specified 264 | truncated 48 |
|---|---|---|---|
| Spearman(q̂_V, δ* on the **calibration** window) | **0.9929** | 0.9885 | 0.9993 |
| Spearman(q̂_V, δ* on the **test** window) | 0.6972 | **0.5171** | 0.9460 |
| Spearman(\|q̂_V\|, uMCB on the test window) | 0.7042 | 0.5266 | 0.9240 |
| median \|q̂_V − δ*_test\| ÷ IQR(q̂_V) | 0.361 | 0.620 | 0.197 |

On the window it is estimated on, q̂_V *is* δ* — ρ = 0.993, and the residual is one
order statistic plus an interpolation convention. On the evaluation window the
rank correlation falls to **0.52 among well-specified series**. Under the
pre-registered rule (ρ > 0.95 and median gap below a quarter of the IQR) the
verdict on that second comparison is "differs", on every subset.

**But the mechanism is estimation error, not independent content.** The two
quantities differ because a 1%-level order statistic estimated on one window does
not reproduce the 1%-level optimum of the next window, not because q̂_V reads
something uMCB cannot. The pre-registration anticipated this reading and fixed the
consequence in advance: *"A high correlation with a handful of separating pairs is
the first interpretation, not the second."* Here there is not even a high
correlation, and what that argues against is treating **either** ordering as
evidence about a forecaster.

## AE-3, verified

Recalibration improves the unconditional miscalibration term and leaves the
others alone, so the in-sample score improvement should equal uMCB. It does:

- mean quantile-score improvement from q̂_V on the calibration window: **5.493e-04**
- in-sample uMCB on that same window: **5.494e-04**
- per-pair ratio: median **0.99987**, 5th percentile 0.9704, 95th 1.0020

The 37 of 312 pairs falling more than 1% short are the O(1/n) gap that
Section 3.2.1 already flags: q̂_V is the $\lceil (n+1)(1-\alpha)\rceil$-th order
statistic, not the interpolated empirical quantile, so it is *near* the argmin and
not at it. One pair has ratio −0.365 — the conformal convention makes the
calibration-window score worse there. That is the sign-straddling case §3.2.1
describes, and it now has a measured instance.

## Out of sample, which is the number that matters

The correction captures a median **88.6%** of the achievable in-sample uMCB on the
test window, and the 5th percentile is **−12.0** — a pair where it costs twelve
times what the best shift would have gained. **66 of 312 pairs are made worse on
the test window by correcting them.** That reproduces `\nDegradedStatic` = 66
exactly, from an independent recomputation, and it is the §7 result arriving from
the decomposition rather than from the deterioration count.

## Negative controls

- A forecast already at the optimal shift returns uMCB 0.000e+00 and δ* −7.6e-17.
- A forecast displaced by a known 0.01 returns δ* +0.010000 and uMCB 1.09e-04.

Both fire. The decomposition code can tell a calibrated forecast from a shifted one.

## What Section 6 does with this

**q̂_V is written as a measurement, not as a statistic.** The sentence to write:
the conformal shift is an out-of-sample estimate of the constant displacement that
minimises the quantile score — the argmin of the Gneiting–Resin unconditional
miscalibration term — reported in the units of the forecast rather than of a
score. Its value to a reader is the scale it is on. It does not rank forecasters:
its rank correlation with the evaluation-window optimum is 0.52 on well-specified
series, and the ordering by q̂_V is therefore a description of how much correction
each pair required on its own calibration window, nothing more.

**The abstract does not change.** It does not claim q̂_V is a new statistic.

Table `tab:master`'s ordering by $\bar R$ inherits this: it keeps its status as a
description and loses any residual reading as evidence of model quality. That is
the same disposition R̄ already received when it was retracted as an independent
statistic, so it costs the paper nothing further.
