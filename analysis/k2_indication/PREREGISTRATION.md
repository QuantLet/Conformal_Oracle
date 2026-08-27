# Is a Basel zone upgrade a benefit? Pre-registration

Written before the counts were computed.

## Why the question arises now

Proposition 5.1 says the traffic-light classification identifies
$\operatorname{sign}(\pi-\tau)$, $\tau = 4/250$, and nothing else in the limit.
Section 7 counts **zone upgrades** as the benefit of applying the correction and
reports that gating on the calibration window costs 20 of 205 of them. If the
classification carries only a sign, then a zone upgrade records that $\hat\pi$
crossed $\tau$ --- not that the forecast got closer to the level it reports.
Those are different events and the paper has been treating them as one.

## Unit of analysis

One row = one pair (forecaster x asset) at $\alpha = 0.01$ on the sequence panel.
**Row count: 312.** Two estimators, static and rolling, reported separately.
What varies row to row: forecaster, asset, test-window length (450-1,880 dates).
Range: $\hat\pi$ in [0, 0.45]; zones in {Green, Yellow, Red}.

Source: `analysis/ae_point4/pairs_long.csv`, whose static columns reproduce
`all_results.csv` to 1e-16.

## The three events, named before counting

For each pair, with $\alpha = 0.01$ and $\tau = 4/250$:

- **U** (zone upgrade): the Basel zone improves, raw to corrected.
- **C** (coverage improves): $|\hat\pi_{\text{cor}} - \alpha| < |\hat\pi_{\text{raw}} - \alpha|$.
- **S** (score improves): $\Delta\mathrm{QS} > 0$.

The paper's benefit measure is U. The question is how far U, C and S disagree.

## What each outcome means, fixed in advance

**If U implies C and S on almost every pair** (say at least 95%), the zone upgrade
is a serviceable proxy, Proposition 5.1's warning does not bite on this panel, and
Section 7 keeps its counts with one sentence recording that the proxy was checked.

**If U and C disagree materially**, then the count of upgrades is measuring
threshold crossings and not calibration, and Section 7 must report the benefit in
a measure that is not a function of the zone: the change in $|\hat\pi - \alpha|$
and the change in quantile score, per pair. The 205 and the 20 stay printed, as
counts of what they are, and lose their role as the benefit ledger.

**The asymmetric case is the one to look for.** A pair can be upgraded while
$\hat\pi$ moves *away* from $\alpha$: raw $\hat\pi$ just above $\tau$, corrected
$\hat\pi$ far below $\alpha$. That is an over-correction rewarded by the zone. Its
count is the number that decides the section.

## Negative control

Two constructed pairs, scored through the same code path:
one with $\hat\pi$ moving 0.017 -> 0.0155 (a genuine crossing, coverage improved),
one with 0.017 -> 0.001 (crossing, coverage worsened in absolute distance from
$\alpha$). The first must register U and C; the second must register U and not C.
If the classifier cannot separate them it is not measuring what this asks.
