*Draft for Daniel. Two artefacts are public and carry numbers this paper now
identifies as artefacts of our own code: SSRN preprint 6757685 and the Quantinar
course. Below: a version note for SSRN, a shorter one for Quantinar, and the
reasoning for posting a note rather than replacing the PDF.*

---

## Why a note and not a silent replacement

A replaced PDF leaves every existing citation, download and course viewing
pointing at figures that were never withdrawn. Anyone who read the earlier
version has no way to learn that 98.8% and 99.0% were a sign inversion in our
generation code. A version note is also the only form that survives being
mirrored: SSRN's revision history, RePEc and the scrapers that index preprints
all keep the abstract page, and a note there travels with the record.

The note should be posted **before** the corrected manuscript is submitted
anywhere, so the sequence in the public record is discovery, disclosure,
resubmission — not the reverse.

---

## Version note for SSRN 6757685

**CORRECTION NOTICE — VERSION 2, [date]**

**This paper's principal empirical results were incorrect. The corrections
change them by factors between four and seventy. Readers should not rely on
Version 1.**

An audit of our own forecast pipeline traced five defects to specific lines of
code. Four affect reported numbers:

| Series | Defect | Version 1 | Corrected |
|---|---|---|---|
| Moirai 2.0 | the α-quantile was stored as −F⁻¹(α) rather than F⁻¹(α), so the threshold pointed at the wrong tail | violation rate 0.988 at α = 0.01 | 0.0178 |
| TimesFM 2.5 | the same sign inversion | 0.990 | 0.0143 |
| Chronos-Small / Mini | sampled at the checkpoint default `top_k = 50`, which truncates a 4093-bin predictive distribution to 50 bins before any quantile is computed | 0.388 / 0.419 | 0.0175 / 0.0178 |
| GJR-GARCH | an unstandardised Student-t quantile with degrees of freedom hard-coded at five, inflating every threshold by about 45% | 0.004 | 0.0200 |

A fifth defect, a ticker alias and an undated forecast vintage, affected inputs
and has been quarantined.

**What this does to the paper's claims.** The central finding of Version 1 — that
the predictive interface, sample-based against quantile-grid, governs extreme-tail
calibration — does not survive. On the corrected panel the best raw
foundation-model series at the 1% level is a quantile-grid model, and the
within-family Moirai gap is 0.24 percentage points rather than 97. That
comparison was also not a controlled one: the two Moirai releases differ in
architecture, pretraining corpus and output parameterisation together.

The statistic q̂_V, presented in Version 1 as an audit instrument, is withdrawn as
such: among well-specified forecasters it is a rank-preserving transform of the
raw violation rate (Spearman 0.99), so it adds nothing to a quantity already
printed beside it.

**Why the errors survived our checks.** Every backtest in Version 1 was computed
after conformal recalibration. That correction restores marginal coverage
whatever forecaster it is applied to, so on the sign-inverted series it produced
19 of 24 assets in the Basel green zone and every backtest passed. The method
worked; that is precisely why the defects were invisible.

**What replaces it.** A corrected manuscript is in preparation. Its subject is
the diagnostic gap this episode exposes: a coverage backtest reports that a
series miscovers, never why, and cannot distinguish a poor model from a truncated
sampler, an inverted sign or a misaligned series. It proposes structural checks
applied before calibration, and it reproduces the Chronos result from a public
checkpoint without our data.

**Replication.** Corrected series, the diagnostic scripts that establish each
defect, and the validation gate are in the replication package at
[repository URL]. The scripts that produce every number are versioned; the
figures in the corrected manuscript are generated from the artefacts rather than
transcribed.

We are grateful to have found this ourselves rather than in review, and we would
rather correct it in public than defend it.

---

## Shorter note for the Quantinar course

The course slides report violation rates of 98.8% and 99.0% for Moirai 2.0 and
TimesFM 2.5 at the 1% level, and 38.8% / 41.9% for Chronos. **All four were
defects in our own code, not properties of those models.** Corrected, the four
series run between 1.4% and 1.8%.

The affected slides should be withdrawn or annotated. The teaching point that
survives is a better one than the original: the errors passed every backtest in
the paper, because they were all computed after a conformal correction that
restores coverage regardless of what it is applied to.

---

## Sequence

1. Post the SSRN version note. Keep Version 1 accessible, marked.
2. Correct or withdraw the affected Quantinar material.
3. Send the co-author message (`drafts/coauthor_message.md`) — this should
   precede both, not follow them.
4. Only then submit the corrected manuscript.
