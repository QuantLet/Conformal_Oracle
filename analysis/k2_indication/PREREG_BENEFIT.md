# Phase 2 — pre-registration, written before the numbers are read

## The question

Proposition 5.1 says the Basel traffic light identifies sign(pi - tau) in the
limit and nothing else. Section 7's indication rule is *triggered* on the Basel
zone and its cost is *counted* in Basel zone upgrades given up: 20 of 204. Both
sides of that ledger are therefore read through an instrument this paper has just
shown to resolve only a sign.

So: does the cost survive being measured by something the traffic light does not
determine? Two alternatives, both already computed per pair in
`analysis/ae_point4/pairs_long.csv`:

- **dQS** -- the change in quantile score, a strictly consistent scoring rule for
  the quantile, which orders forecasts without reference to any zone;
- **distance to nominal**, |pi_hat - alpha| before against after, which is the
  quantity the guarantee actually targets.

## The unit, declared before running

One row is a **pair**: one forecaster on one asset at alpha = 0.01, restricted to
the 174 pairs the rolling estimator degrades. Expected row counts: 204 zone
upgrades, 174 deteriorations, 20 upgrades given up by the deployable rule, 11 of
those also worse on the score. These come from `zone_tradeoff.csv` and
`gate_ledger_overlap.json` and are checked before anything else is read.

## The two interpretations, written in advance

**If the 20 survives** -- if the upgrades the rule gives up are also losses on
dQS and on distance to nominal, at a similar count -- then the zone is standing
in for a benefit that exists independently of it. The indication rule's cost is
real, the traffic light happens to track it here, and Section 7 reports the cost
in all three measures with the zone as the operational one because that is what a
supervisor reads. Proposition 5.1 then constrains what the zone can *diagnose*
without undermining what the rule *costs*.

**If the 20 does not survive** -- if most of those pairs are unchanged or improved
on dQS and on distance to nominal -- then "zone upgrade given up" is not a benefit
foregone but a reclassification foregone. The cost was an artefact of the
measuring instrument, and the rule inherits exactly the limitation Proposition 5.1
establishes: a rule keyed on a sign cannot have its cost counted in that sign.
This is the stronger result, not a problem. It says the indication rule and the
traffic light share a resolution floor, and Section 7 must report the cost in
dQS and distance to nominal, keeping the zone count only as what changes
administratively.

**A third outcome is possible and is not a fudge**: the two alternative measures
may disagree with each other. dQS rewards sharpness as well as calibration;
distance to nominal does not. If they split, that is reported as a split, and the
rule's cost is stated as measure-dependent -- which is itself a finding about
what "benefit" means for a recalibration step.

## What would falsify the exercise itself

If fewer than half the 174 degraded pairs have a defined dQS and a defined
distance-to-nominal change, the panel is too thin to answer the question and the
result is BLOCKED rather than reported.
