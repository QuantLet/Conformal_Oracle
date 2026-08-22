# What each level of information constrains

Phase 2, step 4, rewritten. The earlier version of this document said that
level 2 "identifies" sign, monotonicity and scale. That is wrong, and the
feasibility result is what shows it: **scale constrains, it does not identify.**
A pair matched on coverage, variance, mean absolute deviation and the scale band
still exists, so adding the band shrinks the set of indistinguishable
alternatives without reducing it to a point.

The correct statement uses one quantity throughout. Let `delta` be the
truncation depth — the probability mass discarded from each tail of the
predictive law before the alpha-quantile is read. For a given information level
and a given restriction on the return law, `delta*` is the largest truncation
that admits an alternative model no diagnostic at that level can distinguish
from a correctly specified one. Every level reduces `delta*`; none sends it to
zero.

## The ladder

All figures are for `alpha = 0.01` against a standardised Student-t(5) honest
model with a true 1% quantile of `-2.6065`. `delta*` converges **from above** as
the discretisation is refined, so each row is an upper bound and the bias
overstates the blind spot.

| information available | restriction on the return law | `delta*` | reported quantile | VaR understated by |
|---|---|---|---|---|
| exceedance path only | none | **0.218** | -0.638 | **75.5%** |
| exceedance path only | unimodal | **0.066** | -1.320 | **49.4%** |
| exceedance path only | unimodal, 4th moment <= honest model's | 0.046 | -1.504 | 42.3% |
| exceedance path only | unimodal, Pareto tail index 5 beyond 3 sigma | 0.024 | -1.814 | 30.4% |
| exceedance path only | GARCH class, standardised Student-t innovations | — | -2.330 | 10.6% |
| + series and returns, scale band at -1.8 | unimodal | **0.024** | -1.800 | **30.9%** |
| + series and returns, scale band at -2.05 | unimodal | 0.013 | -2.050 | 21.3% |
| + the predictive object | unimodal | — | — | not yet computed |

The GARCH row carries no `delta*`: that class fixes the innovation family to a
one-parameter standardised Student-t, so truncation depth does not index it. The
entry is the lightest alpha-quantile the family can reach, the Gaussian limit.

**The real exhibit sits outside every row.** The Chronos default implies
`delta` of roughly **0.388**, beyond the unrestricted 0.218. That is why Kupiec
rejects it on 24 of 24 assets: it is not in the blind region at all. Detection
and diagnosis separate cleanly — the same Kupiec rule flags 11 of 13
forecasters, so the rejection carries no information about which is defective.

## What each level adds

**Level 1, the exceedance path `{V_t}`.** Identifies the law of
`u_t = F_t(q_t)` and nothing else (Proposition 1). Kupiec, both Christoffersen
components and the Basel traffic light are functions of the path. The class
extends further than that: both models in the pair satisfy
`E[V_t - alpha | G_{t-1}] = 0` exactly, so **every test of a predictable
conditional-moment restriction on the hit sequence** has power equal to its size,
which is where the Engle–Manganelli dynamic quantile test lives. Verified: DQ
returns `p = 0.943` on the honest model and `p = 0.831` on the alternative.
`sigma(V, q)` is not blind in full — the q-marginals differ — but no test in use
within that class exploits the difference, because doing so needs a scale
reference that `sigma(V, q)` does not contain.

**Level 2, the reported series and the returns `{q_t, r_t}`.** Adds the
threshold's level and the magnitude of returns. Sign, monotonicity across alpha
and alignment are **invariants**: a series violating them is not a Value-at-Risk
series, so these do reduce the admissible set by exclusion rather than by
constraint. The remaining checks are bands, and the scale band is the one that
binds here: it cuts `delta*` from 0.066 to 0.024 and the residual understatement
from 49.4% to 30.9%. Also at this level, and outside Proposition 1, sit the
magnitude-reading statistics — Acerbi–Székely `Z2` and the Fissler–Ziegel joint
score.

**Level 3, the predictive object `{F_t^M}`.** Adds support cardinality and tail
reach. It does **not** add dispersion discrimination for this pair: both models
have predictive standard deviation equal to realised, to machine precision,
because the construction matches variance. Only the scale band fires. Dispersion
catches a predictive law of the wrong width; scale catches a quantile at the
wrong place in a law of the right width. The real Chronos default fails both; the
constructed pair fails only the second, and the two bands should not be
presented as interchangeable.

## The scale band, calibrated

The band edge was chosen by hand at `-1.8`. It is now the only free parameter in
the gate for which a trade-off curve exists, so it should be set by that curve
rather than by judgement.

| band edge | residual `delta*` | residual understatement | series blocked on the real panel |
|---|---|---|---|
| -1.40 | 0.056 | 46.3% | 2 |
| -1.80 (current) | 0.024 | **30.9%** | 2 |
| -2.00 | 0.015 | 23.3% | 2 |
| **-2.05** | **0.013** | **21.3%** | **2** |
| -2.085 | 0.012 | 20.0% | 3 (Lag-Llama) |
| -2.15 | 0.010 | 17.5% | 5 (+ EWMA, GJR-GARCH) |
| -2.30 | 0.006 | 11.8% | 7 |

Every row is LP-feasible under the unimodal class, so each residual is attained
by an actual pair rather than merely bounded.

The panel is cleanly separated: the eleven well-specified series occupy
`[-2.528, -2.085]` and the two truncated Chronos defaults sit at `-0.208` and
`-0.127`. **Tightening the edge from -1.8 to -2.05 cuts the residual
understatement from 30.9% to 21.3% at no false-positive cost** — the same two
series block either way. The hand-chosen value gives away ten percentage points
of residual for nothing. Beyond `-2.085` the gate starts blocking legitimate
forecasters, Lag-Llama first, and the trade-off becomes real.

This also disposes of the in-sample objection to the band. The edge is no longer
defended by "a practitioner would endorse it"; it is set where the residual curve
meets the panel's own separation, and both quantities are reported.
