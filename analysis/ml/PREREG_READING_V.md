# Pre-registration — reading (v) on the ML cells, and what it drags with it

Written 2026-08-28, before the counts below are computed.

## Disclosure, because the order matters

The scale ratios of the ML cells were computed first, to answer reading (v) of
`analysis/phase0/CONDITIONAL_PASSAGES.md`. That answer is on the page before this
file was written, and it is reported in the first section below with no
pre-registration claimed for it, because none was written in time. What is
pre-registered here is the **second** question, which arose from seeing those
ratios: whether the ML cells occupy the intermediate range that Section 7.2
declares empty, and what that does to Table 2's fourth row.

The quantitative claims below — how many cells, on which side of which edge, and
the false-positive count at the tightened edge — have not been computed at the
time of writing. Stating which part is pre-registered and which is not is the
only honest way to report a question that arrived out of order.

## Reading (v) itself: BLOCKED, with one direction settled

The pre-registered object is **the 24-asset panel**. It does not exist:
`analysis/ml/series/lgbm_default/` is empty and no panel run has completed. What
exists is `dose_response_raw.csv` — **4 assets x 5 leaf settings x 2 estimators
x 200 dates**, so 40 cells against the 312 the sequence panel carries.

A subset settles reading (v) in one direction only. If any of the 40 cells fell
below the lower scale edge of −3.500, the edge would have bound and the passages
in `CONDITIONAL_PASSAGES.md` would be false as written — an existence claim a
subset can establish. That the 40 do **not** fall below it says nothing about the
other 20 assets, so band 3 is not reached and reading (v) stays **BLOCKED** until
the panel runs.

Most negative ratio over the 40 cells: **−2.871**, quantile forest on SP500 at
`min_data_in_leaf = 500`. Margin from the edge: 0.629.

The `blind_gate.py` band logic is unchanged and is not re-run on a partial panel;
it is written against the 24-asset object and reporting its verdict on 4 assets
would be the R3 failure — a correct calculation on the wrong unit.

## The second question, pre-registered

Section 7.2's margin paragraph says the separation between well-specified and
truncated cells "is clean because this panel contains no series that is
*moderately* misspecified". `CONDITIONAL_PASSAGES.md` anticipated that a
forecaster at 0.6x to 1.0x nominal "occupies exactly the intermediate range this
sentence declares empty".

The relevant band edge for that is the **upper** one, −1.800, not the lower one
reading (v) asks about, and the relevant figures are already in the registry:
the worst well-specified cell at **−1.947**, the best truncated cell at
**−0.283**, an upper margin of **0.147**, and the tightened edge of Table 2 row 4
at **−1.940** with a margin of **0.007** above the worst well-specified cell.

**The unit.** One row is a **cell**: one estimator, one asset, one leaf setting,
at alpha = 0.01. Expected rows: 40 = 2 x 4 x 5. What varies between rows is the
estimator class, the asset, and `min_data_in_leaf` over {1, 5, 20, 100, 500}.
This is not the 312-cell sequence panel and no count from it is comparable to a
count from that panel; the two are reported side by side and never pooled.

### If the ML cells land in the gap — between −1.947 and −0.283

Then the margin paragraph is false as written and must be revised, and the whole
of Table 2 row 4 with it. A tightened edge at −1.940 was calibrated on a panel
containing nothing between the two populations; a family sitting in that gap
changes the false-positive count at every candidate edge, and the 0.007 margin
that made −1.940 look safe is a property of the panel it was fitted on. This is
the row `CONDITIONAL_PASSAGES.md` calls the one a referee would find, and the
correct response is to report the recomputed count, not to move the edge.

### If they do not — if every ML cell sits inside [−3.5, −1.947]

Then the margin paragraph stands, the ML family is simply another well-specified
population, and Table 2 row 4 is unchanged. The ML exhibit would then support
Remark 3.1 on the tail-sparsity argument alone and carry no consequence for the
gate, which is a weaker but perfectly reportable result.

### The third outcome, and it is the one to watch

The two estimators may **split**: LightGBM in the gap and the quantile forest
outside it, or the reverse. They are different estimators with different
tail behaviour, and `DOSE_RESPONSE_REPORT.md` already records that they respond
differently to `num_leaves`. A split is reported as a split, per estimator, and
the margin paragraph is then false for the family that lands in the gap
regardless of what the other does — one counterexample is enough to empty a
claim that a range contains nothing.

## What would falsify the exercise itself

If fewer than 200 dates carry a finite ratio in any cell, that cell is dropped
and the count is reported over what remains, with the drop stated. If more than
a quarter of the 40 cells drop, the exercise is BLOCKED rather than reported.

## What no outcome licenses

None of this resolves reading (v). The lower edge and the upper edge are
different questions about different tails of the same band, and an answer about
the upper edge on 4 assets must not be written up as though the 24-asset panel
had run.
