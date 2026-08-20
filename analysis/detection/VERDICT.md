# Phase 0e verdict: branch C3 — SUPERSEDES the C1 reading below

## Correction, 2026-08-18

**This document originally concluded C1. That conclusion was wrong and is
withdrawn.** It is kept in full below because the error is instructive and the
detection numbers themselves are correct; only their interpretation was not.

The test that settles it: R is a monotone re-encoding of the raw violation rate.
Spearman(R, pihat_raw) = +0.9912 across the 14 well-specified forecasters
(p = 6.5e-12) and +0.9941 across all 16. Detection using pihat alone reproduces
every figure attributed to the conformal statistic:

    pihat > 0.10                     sens 4/5  spec 8/8  p = 0.00699
    pihat < 0.005                    sens 1/5  spec 8/8  p = 0.38462
    either                           sens 5/5  spec 8/8  p = 0.00078
    q_V union (for comparison)       sens 5/5  spec 8/8  p = 0.00078

q_V therefore adds nothing over the violation rate as a defect detector. The
finding is real but it is about a THRESHOLD, not a statistic: a severe cut on
coverage separates defects from ordinary miscalibration, where Kupiec does not
because its threshold is calibrated to sample size rather than to severity.

The error was comparing q_V against Kupiec's p-value rather than against the
quantity Kupiec is computed from. The pre-registration anticipated circularity
and in-sample fitting; it did not anticipate that the detector and its baseline
were the same measurement expressed differently.

This is the AE's B1 objection, confirmed in its sharper form: q_V is not an
informative quantity for acceptably well-specified forecasters, because among
them it is a rank-preserving transform of a quantity already reported.

**Consequences.** The audit-statistic framing goes. Conformal recalibration is
the recalibration theorem plus the gate rule, and the structural gate is the
contribution. The retrospective finding (Task 4) survives intact but changes
meaning: the submitted Table 1 partitioned the panel correctly on R > 1, and it
would have partitioned it equally well on the raw violation rate printed in the
adjacent column.

---

# Superseded C1 reading, retained for the record


Rules and branch readings were fixed in `PREREGISTRATION.md` before any of this
was computed.

## The detection result

| detector | sensitivity | specificity | Fisher exact p | foundation | classical |
|---|---|---|---|---|---|
| q̂_V magnitude, R̄ > 1 | 4/5 | 8/8 | 0.0070 | 4/4 | 0/1 |
| q̂_V sign, majority negative | 1/5 | 8/8 | 0.3846 | 0/4 | 1/1 |
| **union of the two (post-hoc)** | **5/5** | **8/8** | **0.00078** | **4/4** | **1/1** |
| Kupiec, majority fail | 5/5 | 2/8 | 0.4872 | 4/4 | 1/1 |
| Christoffersen, majority fail | 0/5 | 8/8 | 1.0000 | 0/4 | 0/1 |

**C1 holds.** q̂_V dominates both backtests, and the dominance is not marginal.

Kupiec attains 5/5 sensitivity by flagging 11 of 13 forecasters; its specificity
is 2/8 and its Fisher p is 0.49. A rule that fires on 85% of everything has no
discriminating power, and this is the sense in which the paper's title is
correct: the backtest cannot detect a defect, because it does not distinguish
one from ordinary under-coverage.

Christoffersen detects nothing — 0/5 — and is *undefined on 22 of 24 assets*
for GJR-GARCH, the one forecaster whose defect made exceedances too rare to
populate a transition table. The defect suppressed the test that would have
caught it.

### The threshold is not tuned

Sensitivity and specificity are identical for every R̄ threshold from 0.5 to
3.0 (4/5 and 8/8 throughout). The nearest unlabelled forecaster is Lag-Llama at
0.357 and the nearest labelled one is TimesFM-2.5 at 3.183 — a ninefold void.
Only above 5.0 does sensitivity fall, as the two sign-flipped series drop out.

### Qualification: the union is post-hoc

The two q̂_V rules were pre-registered separately and are exactly
complementary — magnitude catches the four foundation defects and misses the
classical one; sign catches the classical one and misses all four foundation
ones. Their union achieves 5/5 at 8/8, but "use both" is a rule formed after
seeing which each caught, and it is labelled as such wherever it appears.

Neither rule generalises across families alone. That is a real limitation and
not a presentational one: a reader should not take away that R̄ > 1 is a
general-purpose defect screen. It is a screen for defects that inflate the
required correction, which is a specific and mechanically explicable class.

## The refutation that matters: this was not circular

The pre-registration flagged the obvious threat — the defects were found partly
by looking at q̂_V, so high sensitivity could be an artefact of the search
process. Task 4 refutes this directly, from the submitted artefacts rather than
from the current tree.

**Table 1 of the IJF submission partitions the panel on exactly this statistic:**

    Panel A: Signal-preserving recalibration   (|q̂_V|/|VaR_raw| < 1)
        Moirai 1.1 0.11 · Lag-Llama 0.36 · GJR-GARCH 0.16
        GARCH-N 0.17 · Hist. Sim. 0.11 · EWMA 0.18

    Panel B: Effective replacement             (|q̂_V|/|VaR_raw| > 1)
        Chronos-Small 17.3 · Chronos-Mini 23.5
        TimesFM 2.5 3.2 · Moirai 2.0 3.2

Panel B is exactly the four defective foundation-model series. The line was
drawn at R = 1, the four defects fell on one side of it, and the partition was
named after a property of forecaster behaviour.

The fifth defect appears too. In the submitted cross-sectional table,
GJR-GARCH's correlation between q̂_V and annualised volatility is **−0.786**,
the only negative value against +0.67 to +0.97 for the nine other models. The
manuscript addresses it at line 423:

> "...and inverts for GJR-GARCH, whose conditional variance dynamics absorb
> volatility into raw VaR."

So all three defect families were flagged by q̂_V **in the submitted paper**,
in printed tables, before anyone knew a defect existed. Every flag was read as a
property of the model.

This is not a claim that the statistic was validated in advance. It is the
narrower and more uncomfortable claim that it fired, visibly, and that the
failure was one of interpretation rather than of instrumentation. It also
disposes of the circularity objection in the direction that matters: the flags
predate the discovery.

## Flagged but unlabelled — six cases, each requiring a verdict

Kupiec flags Chronos-Small-A, Chronos-Mini-A, EWMA, GARCH-N, Hist-Sim and
Lag-Llama. Under the pre-registered protocol each is traced, not counted.

Preliminary reading, to be completed: all six are cases of ordinary
under-coverage at α = 0.01 (π̂ from 0.0158 to 0.0294), which is what the panel
does — no forecaster of sixteen passes Kupiec on more than 15 of 24 assets.
None is flagged by either q̂_V rule. This is consistent with Kupiec measuring
miscalibration rather than defect, which is what it was designed to measure.

**q̂_V produces no unlabelled flags at all**, so its specificity of 8/8 rests on
no traced cases. This is a small sample and the correct statement is that no
false positive has yet been found, not that none exists.

## What this does to the paper, per the pre-registration

C1 was pre-registered as: title becomes an answer rather than a complaint;
abstract leads with the detection result; conformal recalibration becomes the
method with a validated second use; the truncation becomes the worked example.

Two amendments the result itself forces:

1. The detection claim must be stated as **the union of two complementary
   rules, one of which was formed post-hoc**, not as "R̄ > 1 detects defects".
2. The retrospective finding is stronger than the detection rate and should lead
   over it. "The statistic flagged all three defect families in the submitted
   paper and every flag was read as a model property" is a more useful sentence
   than any sensitivity figure, and it is the one a referee cannot dismiss as
   post-hoc fitting.
