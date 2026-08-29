# Material for the coauthor message — facts only, not a draft

Assembled 2026-08-29 for the message to Lessmann and Härdle. Every figure below
was recomputed in this session from the artefact named beside it. This file is
source material; the message itself is not drafted here.

## 1. What was actually circulated under five names

Checked across all five signed files --- `submission_IJF/main_R1.tex`, its
anonymised twin, both `Recalibrating_Tail_Risk_Forecasts` variants, and the
2026-05-17 fixed version:

| figure | in the signed versions? | status |
|---|---|---|
| $\hat\rho \in [0.18, 0.67]$ | **yes, all five** | wrong twice over |
| coverage floor "approximately 88\%" | **yes, all five** | wrong |
| alternative threshold $1.32\sigma$ | **no, none of them** | never circulated |

**The distinction matters for what the message is.** The first two are
corrections to numbers that carry the coauthors' names. The third is an internal
defect in an unpublished draft, caught before it left the building, and it is not
a retraction --- it is worth mentioning only because it is the reason the
verification was tightened.

## 2. The three figures, with what replaced them

**Score persistence, $\hat\rho \in [0.18, 0.67]$.** Two independent defects in
one clause. The upper end is a pre-correction vintage: TimesFM's persistence
moved with the 2026-08-17 sign correction and the six-pair range is now
**[0.183, 0.618]**. And the range was computed over the six pairs of the
bound-validation table while the sentence attaches it to the four-pair gap
ablation, whose full-period range is **[+0.05, +0.46]**. Artefacts:
`Quantlets/CO_bound_validation/tab_bound_validation.csv`,
`Quantlets/CO_robustness/gap_ablation.csv`.

**Coverage floor "approximately 88\%".** The guaranteed floor is the
\emph{minimum} over the pairs, **85.2\%**. 88.1\% is the maximum. The published
sentence reports the best case as if it were the guarantee. $\hat\Delta_n \in
[0.109, 0.138]$ and empirical coverage averaging 98.9\% are unchanged. Artefact:
same file.

**Constructed pair, $1.32\sigma$.** The linear programme returns **1.46σ**; the
alternative holds 56\% of the honest threshold, not "barely half". 1.32 is the
value at which the sentence's own word comes out exact, and it is produced by no
script in the repository. Artefact: `analysis/phase2/pair.npz`.

## 3. Two tables in the submitted version that were computed from superseded series

Both are printed in `submission_IJF/main_R1.tex`, both last regenerated
2026-05-31, neither re-run after the 17 August sign correction. The models that
move are exactly the three whose series were corrected.

- `tab_regime_sensitivity`: TimesFM 2.5 and Moirai 2.0 show 24 of 24 at every
  threshold, the fingerprint of the ~99\% raw violation rate the sign defect
  produced; recomputed they are 11--1 and 12--2. The headline moves from
  **138/240 to 121/240**.
- `tab_bound_validation`: $\hat\rho$ for TimesFM 0.64 → 0.62, Moirai 0.49 → 0.41.

These are the erratum, and they are the reason the SSRN replacement cannot wait
for the journal decision.

## 4. What the paper claims now, in three sentences

1. The Basel traffic light identifies the sign of $\pi - \tau$ in the limit and
   nothing else, the interval it cannot resolve does not shrink with evidence,
   and on the panel a zone upgrade is a strict sub-event of a move toward
   nominal --- 0 pairs upgraded without moving closer, 70 moving closer with no
   zone change.
2. A one-sided split-conformal shift restores marginal coverage exactly under
   exchangeability and asymptotically under geometric $\beta$-mixing with an
   explicit remainder, and it does so whatever forecaster it is applied to, which
   is why the corrected column cannot rank the series that produced it.
3. Recalibration is therefore an intervention with an indication rather than a
   default step, and its cost cannot be priced in the instrument that triggers
   it: the deployable gate costs 9 pairs on the quantile score against the 20
   zone upgrades the traffic light records.

## 5. What changed in what the paper claims, against the IJF abstract

| IJF claim | now |
|---|---|
| tail sparsity makes flexible post-processing fragile | **kept and strengthened** --- it binds on the forecaster as well as the correction, with a measured instance |
| a one-parameter conformal shift as the low-dimensional alternative | **kept**, and the coverage theory moved into the body with the remainder explicit and the separation condition the split does not meet reported |
| the shift is "a signed audit statistic" measuring whether the base retains tail information | **withdrawn as a statistic.** It is a measurement in the units of the forecast --- the argmin of the unconditional miscalibration term --- and its rank correlation with the evaluation-window optimum is 0.52 on well-specified series, so it does not order forecasters out of sample |
| "a sharp regime separation", signal-preserving versus replacement | **narrowed to a description.** $\bar R$ ordering records how much correction each pair required; the separation was measured on a panel holding only gross failures and correct specifications |
| rolling recalibration improves coverage | **kept**, with the cost now stated: it degrades 89 of 94 pairs whose raw forecast already passes Kupiec |
| fitted per-asset GARCH retains a Quantile Score advantage | **kept, qualified**: the two GJR variants are not distinguishable, and the sign of the comparison depends on the weighting |
| scalar shifts do not restore conditional independence | **kept** |
| Expected Shortfall as supplementary diagnostic | **kept** |
| --- | **new**: Propositions 5.1--5.3, and the indication result |

## 6. What follows

Replace the SSRN preprint; submit to IRFA. The replacement carries the two
corrected tables and the corrected figures of section 2 above.
