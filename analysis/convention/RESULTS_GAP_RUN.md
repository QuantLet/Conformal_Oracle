# The separation gap on the full panel — results

Run 2026-08-30 against `PREREG_GAP_RUN.md`. **312 of 312 cells, 0 blocked**, 2.1
seconds of compute.

## The first pre-registered branch

| quantity | value |
|---|---|
| gap $g_n$ | 5 to 30 observations, median **8** |
| as a share of the test window | **0.46\%** |
| median $\lvert\Delta\hat\pi\rvert$ | **0.000051** |
| max $\lvert\Delta\hat\pi\rvert$ | **0.000521** |
| Basel zone changes | **0** of 312 |
| Kupiec verdict flips at 5\% | **1** of 312 |

The panel-wide maximum, 0.00052, reproduces the four-pair ablation's 0.0005
almost exactly, so the ablation was representative rather than lucky.

**Theorem 4.5 now covers the estimator as run.** The price is 0.46\% of each test
window and one Kupiec pass.

## The third branch fired too, and it is the one worth naming

The pre-registration allowed for the effect being negligible on $\hat\pi$ while
still moving a count, because Kupiec passes and Basel zones are threshold
functions of the rate. That is exactly what happened: one cell crosses the 5\%
boundary on a rate that moved by less than a thousandth. It is reported as the
sensitivity of a thresholded statistic and not as instability in the estimator,
which is what the branch was written to distinguish in advance.

## A correction made before the run, not after

The pre-registration first stated the gap as coming off the **end of the
calibration block**. `gap_ablation.py`'s `conformal_with_gap` sets
`test_start = n_cal + gap`, so it comes off the **front of the test block**: the
calibration sample is untouched and the shift is bit-identical to the ungapped
one. The two differ in what can move --- under the implementation, only the
evaluation window --- and the prediction sharpens accordingly, to a change of
order $g_n/n_{\mathrm{test}}$. The observed median, 0.000051 against a window of
about 1{,}500 and a gap of 8, sits where that predicts.

## What these numbers are not

`run_gap_panel.py` recomputes the split from the stored scores and reports its
own levels: 297 green and 230 Kupiec passes before the gap. These are **not** the
manuscript's headline counts, which come from the full pipeline and differ in
window construction. Only the **differences** enter the manuscript, and both arms
are computed by the same code on the same cells, which is what the claim needs.

## Not in scope

The rolling estimator. Theorem 4.5 covers the single split, and a gap inside a
trailing window is a different object. Section 4.5's statement is unchanged.
