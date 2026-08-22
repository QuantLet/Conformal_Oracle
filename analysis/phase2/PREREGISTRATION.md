# Phase 2 pre-registration

Written before any computation. Standing rule 3.

## The characterisation I expect to prove

Let `q_t` be the reported lower alpha-quantile, predictable w.r.t. `G_{t-1}`, and
`F_t` the conditional law of `r_t` given `G_{t-1}`. Define

    u_t := F_t(q_t) = P(r_t < q_t | G_{t-1}),

the *realised* conditional tail probability the reported threshold cuts off. I
expect to show that the law of the exceedance path factorises as

    P(V_{1:T} = v_{1:T}) = E[ prod_t u_t^{v_t} (1 - u_t)^{1 - v_t} ],

so the law of V depends on the joint law of (forecast, return) **only** through
the law of the process (u_t). Everything else about the pair -- predictive
dispersion, tail shape, support cardinality, the entire predictive object away
from the reported quantile -- is unidentified.

## What I expect the construction to require

To make two models observationally equivalent I must equalise the u-law. Under a
**fixed** return DGP a truncated forecaster has u_t > alpha and IS detectable --
which is why Kupiec rejects the truncated Chronos series on 24 of 24 assets. So
the equivalence must vary the DGP as well as the forecaster. I expect this and
will state it as the substantive content, not as a caveat: the exceedance
sequence cannot separate "the forecaster is broken" from "the return regime is
fatter-tailed than the one the forecaster was validated in".

## Feasibility risk I have identified in advance

Matching the pair on variance, mean absolute deviation AND the alpha-quantile
requires the defective model's return law H to be **platykurtic** -- lighter
tailed than Gaussian. Scale mixtures of normals cannot do this (kurtosis >= 3),
so a Student-t or normal-mixture family will fail. I will therefore construct H
as a discrete symmetric law on a grid, which is also faithful to the exhibit
(Chronos's predictive law is discrete over 4093 bins). All three constraints are
linear in the grid probabilities, so this is a linear feasibility problem.

**If the linear program is infeasible, Phase 2 stops and I report that**, per the
brief. I will not weaken the proposition into a sentence that sounds like one.

## Numbers I expect

  N1  Under standardised t_5, the true 1% quantile is about -2.61.
  N2  Discarding 5% from each tail moves the reported 1% quantile to about
      -1.43, i.e. roughly 45% of the way to zero.
  N3  A unit-variance law with its 1% quantile at -1.43 is far lighter-tailed
      than Gaussian (-2.33), so the LP will need most of its mass near +-1.
  N4  The two models will have identical exceedance-path likelihoods to machine
      precision, and Kupiec / Christoffersen / Basel will return identical
      verdicts on both by construction, not by simulation luck.

## Boundary check, pre-registered

I expect magnitude-based tests (Acerbi-Szekely Z2, Fissler-Ziegel) to separate
the truncated Chronos series from the analytic one on the real panel, because
they read exceedance *size*, which is outside the test class. From the artefact
I have already read this session, Z2 rejects the truncated series on 24/24
assets raw. I expect FZ to order them in the same direction. If either fails to
separate, that is the more informative outcome and extends the proposition's
reach; I will report it either way.
