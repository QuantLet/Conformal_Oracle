# Is a Basel zone upgrade a benefit? Result

Run against `PREREGISTRATION.md`. 312 pairs, α = 0.01, from
`analysis/ae_point4/pairs_long.csv` (static columns reproduce `all_results.csv`
to 1e-16). Negative control separates the two constructed cases.

## The proxy holds on coverage and fails on score

| | static | rolling |
|---|---|---|
| zone upgrades **U** | 182 | 205 |
| of those, coverage also improved (**C**) | **182 (100.0%)** | **205 (100.0%)** |
| of those, score also improved (**S**) | 176 (96.7%) | **133 (64.9%)** |
| upgraded while coverage moved *away* from nominal | **0** | **0** |
| upgraded while the score got worse | 6 | **72** |
| not upgraded, yet coverage improved | 73 of 130 | 69 of 107 |

**The asymmetric case the pre-registration told me to look for does not occur.**
No pair on this panel is upgraded while $|\hat\pi - \alpha|$ grows. Proposition 5.1
warns that the zone reads a threshold crossing rather than a distance, and on
these data the crossing happens to coincide with a genuine move toward nominal in
every one of 387 upgrades. That is a fact about the panel, not a general
guarantee, and it is the honest reading: the pre-registered 95% bar is met for
**U ⟹ C**.

**It is not met for U ⟹ S.** Under the rolling estimator, 72 of 205 upgrades are
pairs whose quantile score got *worse*. A zone upgrade is a reliable indicator
that coverage moved toward nominal and an unreliable indicator that the forecast
improved by a proper scoring rule. The two disagree because the zone reads only
the exceedance count, and the score reads the magnitudes as well.

## What that does to the gated rule's ledger

The rule applies the correction when the raw series fails on the gating window
(zone $\neq$ Green or Kupiec $p \le 0.05$).

| estimator | signal | upgrades | kept | **lost** | of those, score was also worse |
|---|---|---|---|---|---|
| static | calibration | 182 | 172 | 10 | **0** |
| static | evaluation (oracle) | 182 | 182 | 0 | 0 |
| rolling | calibration | 205 | 185 | **20** | **11** |
| rolling | evaluation (oracle) | 205 | 205 | 0 | 0 |

**Of the 20 rolling zone upgrades the deployable rule gives up, 11 are pairs the
correction degrades on the quantile score.** Skipping those is not a cost; it is
the rule working. The net cost is **9**, not 20.

And the benefit and cost columns overlap: **72 of the 205 upgrades are
simultaneously among the 174 score deteriorations.** The two counts cannot be
traded off against each other as printed, because a third of one is inside the
other.

## Consequence for Section 7

1. The 205 and the 20 stay printed, as counts of zone changes.
2. They stop being the benefit ledger. The benefit is reported in two measures
   that are not functions of the zone: the change in $|\hat\pi - \alpha|$ and the
   change in quantile score.
3. The cost of the gated rule is reported as 20 upgrades forgone **of which 11
   were deteriorations**, net 9.
4. The reason is stated once and attributed to Proposition 5.1 rather than
   presented as a separate caveat: a rule keyed on the Basel zone inherits the
   zone's resolution, which is a sign.
