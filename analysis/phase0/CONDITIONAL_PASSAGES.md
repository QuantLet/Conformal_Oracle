# Passages conditional on results not yet read

**Status 2026-08-28: the first table is resolved, the second is applied.**
Reading (v) itself remains BLOCKED — the 24-asset panel has never run — but the
4-asset dose-response settles what it can, and the results are in
`analysis/ml/RESULTS_READING_V.md` against
`analysis/ml/PREREG_READING_V.md`. One row of seven was false as written:

| row | verdict |
|---|---|
| "blocks 0 of 312 cells" | stands; no ML cell reaches the lower edge, most negative −2.871, margin 0.629 |
| "the scale check is one-sided" | stands, and sharper: the upper edge fires 15 times on the ML family |
| "whether a lower edge is needed cannot be determined" | stands, and the ML family cannot determine it either — 4 assets can show the edge binding and cannot show that it does not |
| "nine checks exercised and one that was not" | stands |
| the margin paragraph, "nothing in this panel occupies the intermediate range" | **FALSE as written.** 15 of 20 LightGBM cells sit in the gap; 0 of 20 quantile-forest cells do. Rewritten per estimator, and the edge's false-positive region is now bounded in closed form |
| Table 2 row 4, tightened edge −1.940 | **recomputed, unchanged.** 0 of 40 ML cells lie in the strip the tightening adds |
| S6 δ\* figures | unaffected, as declared |

Two further findings from the same pass, neither anticipated here: the worked
example of a forecaster at −1.9σ̂ was called a false positive on a threshold-unit
reading, and exceeds at 2.3–2.9× nominal, so blocking it is correct; and this
family's coverage statistic lives on a grid of 0.5× nominal at 200 dates, so the
"0.6x to 1.0x" description below is off the grid it is measured on.

The second table's rows are applied: Remark 3.1 is restated across both layers,
§3.2.4 cites it on the correction layer explicitly, and the exhibit is written up
as Supplement S.13.

---


Marked before the results exist, so that new data cannot silently contradict
standing text. Nothing here is rewritten until the reading returns.

## Conditional on reading (v) — the quantile forest and the lower band edge

The quantile forest sits at 0.6x to 1.0x nominal on four assets. If the 24-asset
panel puts any cell below the gate's lower scale edge of -3.500, the following
passages become false as written and must be revised together, not one at a time.

| location | current text | what breaks |
|---|---|---|
| S7.2, "One of our own checks cannot fail" | "blocks 0 of 312 cells" | the count and the claim; the edge would have bound |
| S7.2, same paragraph | "on this panel the scale check is one-sided" | false: the check becomes two-sided |
| S7.2, same paragraph | "whether a lower edge is needed at all cannot be determined from these data" | false: it would be determined |
| S7.2, closing | "the block count is nine checks exercised and one that was not" | becomes ten exercised |
| S7.2, margin paragraph | "the separation here is clean because this panel contains no series that is *moderately* misspecified" | a forecaster at 0.6x nominal occupies exactly the intermediate range this sentence declares empty |
| S7.1, Table 2 row 4 | band tightened to -1.940, residual 25.6% | the tightened edge was calibrated on a panel with nothing in the intermediate range; the false-positive count at -1.940 must be recomputed with the ML cells included |
| S6, Proposition 6.2 discussion | delta* figures | unaffected: they are analytic, not panel-derived |

The last row matters most and is the one a referee would find. Section 7.1's
residual-versus-margin trade-off was computed on 312 cells drawn from two clearly
separated populations. Adding a family that sits between them changes the
false-positive count at every candidate edge, and therefore the whole of
Table 2's fourth row.

## Conditional on the ML exhibit entering beside Remark 3.1

If the LightGBM result is placed beside the tail-sparsity remark rather than as a
second exhibit in Section 4, then:

| location | current text | what changes |
|---|---|---|
| Remark 3.1 title and first sentence | scoped to the recalibration layer | becomes a statement about both layers: the constraint binds an estimator fitted to the tail as well as a correction estimated from it |
| S3.2.4 | "The comparison tests Remark 3.1: at alpha = 0.01, additional parameters increase variance faster than they reduce bias" | cites the remark in its old, single-layer form |
| B6 recount list | — | both of the above join it |

## Readings (ii) and (iv): resolved, both falsified

Reading (ii) fails on all three control knobs (factors 3.31, 2.53, 1.60 against a
1.5 threshold). Reading (iv) is contradicted (`num_leaves` 31 -> 127 leaves
pi-hat at 0.0300, ratio 1.00). See `analysis/ml/DOSE_RESPONSE_REPORT.md`.

**Consequence for placement, now forced rather than chosen.** The result is not a
configuration trap and must not be placed beside Section 4, whose mechanism is a
silent default surviving a sensitivity sweep. It is an instance of the
tail-sparsity constraint and goes beside Remark 3.1. The two rows already listed
under "conditional on the ML exhibit entering beside Remark 3.1" are therefore
no longer conditional -- they are required.
