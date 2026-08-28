# Phase 2 — what the indication rule costs, measured three ways

Pre-registration in `PREREG_BENEFIT.md`, written before any of this was read.
Unit: one pair, one forecaster on one asset at alpha = 0.01, 312 of them. The
pre-registered counts hold: 204 zone upgrades, 174 score deteriorations, 71 pairs
that are both. Every pair has a defined dQS and a defined distance to nominal, so
the falsification condition does not fire.

## The outcome was the third one

The pre-registration allowed for the two alternative measures disagreeing with
each other, and that is what happened. The 20 zone upgrades the deployable rule
gives up read:

| measure | cost |
|---|---|
| Basel zone upgrades foregone | **20** of 20, by construction |
| distance to nominal, \|pi_hat - alpha\| | **20** of 20 move closer to nominal |
| quantile score | **9** of 20; the other 11 would have been *worse* on the score |

So the 20 survives one alternative and not the other, and the split is total:
zero pairs where both alternatives agree there is no loss. The eleven are pairs
where the correction buys coverage and pays for it in score --- the
calibration-sharpness trade-off, arriving at the level of individual pairs.

## The part that was not anticipated, and it is the result

The two coverage-side measures are not independent, and the panel shows exactly
how they fail to be:

| relation | cells |
|---|---|
| zone upgrade **and** closer to nominal | 204 |
| zone upgrade but **not** closer to nominal | **0** |
| closer to nominal but **no** zone upgrade | **70** |

Not one pair is upgraded by the traffic light without moving closer to nominal,
and seventy move closer without the light noticing. The zone is not a second
opinion on the benefit; it is a strictly coarser reading of the same movement,
and the coarsening discards 70 of the 274 pairs that improve --- a quarter of
them. That is Proposition 5.1 measured on the panel rather than proved in the
limit: an instrument that resolves a sign cannot count a magnitude, and here it
undercounts by construction, never in the other direction.

Rank correlations over the 312 cells put the same thing numerically. Zone against
distance to nominal, 0.51; zone against score improvement, 0.58; distance against
score, 0.29. The zone agrees with each of the other two about as much as they
disagree with one another, which is what a coarsening of one of them looks like.

## What Section 7 must therefore say

The cost of the indication rule is **20 pairs that would have moved closer to
nominal, of which 11 would have paid for that with a worse quantile score, so 9
by the only measure here that is not a function of the violation count**. The
zone figure is kept because it is what a supervisor reads and what the rule is
keyed on, and it is reported as an administrative count, not as the benefit.

The interpretation written in advance for this branch stands: the indication rule
inherits the traffic light's resolution floor. That does not weaken the rule --
it locates it. A rule triggered by a sign can be audited by a sign; it cannot
have its cost priced in one, and Section 7 prices it in the score instead.
