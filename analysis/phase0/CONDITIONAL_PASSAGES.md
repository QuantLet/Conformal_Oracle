# Passages conditional on results not yet read

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

## Conditional on readings (ii) and (iv)

Nothing in the manuscript is conditional on these yet, because no ML text has
been written. They govern what may be claimed, not what must be revised.
