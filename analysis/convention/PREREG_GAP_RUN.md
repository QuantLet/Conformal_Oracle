# Pre-registration — running the panel with the theorem's gap inserted

Written 2026-08-30, before the run.

## Why

Theorem 4.5 requires a separation gap $g_n$ between calibration and test blocks.
The protocol runs $g_n = 0$, so the theorem does not cover the estimator as run,
and Section 4.4 reports that as a gap rather than closing it. With the title
reduced to the conformal claim, this is the one hole the title exposes directly.

The ablation already run on four pairs shows $g_n = \lceil c\log n\rceil$ costs
5--11 observations and moves $\hat\pi$ by at most 0.0005 over the full window.
This extends it from 4 pairs to all 312.

## The unit

One row is a **cell**: one forecaster, one asset, $\alpha = 0.01$. Expected rows
**312**, the sequence panel. What varies is the gap, and the direction matters: `gap_ablation.py`'s
`conformal_with_gap` sets `test_start = n_cal + gap`, so the **calibration block
is untouched and the test block loses its first $g_n$ observations**. This
paragraph first said the opposite --- that the gap comes off the end of the
calibration block --- and it is corrected here before the run rather than after,
because the two differ in what changes: under the implementation the shift
itself is identical to the ungapped one and only the evaluation window moves.
$c = 1/|\log\hat\rho|$ with $\hat\rho$ the AR(1) coefficient of the
calibration scores.

Expected gap size: 9--10 observations on a mean calibration block of 3,626, so
about 0.28\%.

## Predictions, both directions

That correction sharpens the prediction. Since the shift does not change, any
movement in $\hat\pi$ comes only from dropping $g_n$ test observations, so the
expected change is of order $g_n/n_{\mathrm{test}}$ --- about ten in fifteen
hundred, under one percent of the window --- and any larger movement means the
dropped observations were not typical of the window.

**If the panel-wide effect matches the four-pair ablation** --- median absolute
change in $\hat\pi$ below 0.001, no Basel zone changes, no Kupiec flips beyond a
handful --- then the gapped estimator is the one reported, Theorem 4.5 covers the
protocol as run, and Section 4.4 becomes a remark that the gap costs 0.28\% of
the calibration sample rather than a concession that the theorem does not apply.

**If it does not** --- if zone changes or Kupiec flips appear at a rate that
would move a headline count --- then the gap is not free at panel scale, the
four-pair ablation was unrepresentative, and Section 4.4 keeps its concession
with the panel-wide magnitude now stated. That is a worse outcome for the paper
and a better one for the reader, and it is reported either way.

**The third possibility.** The effect may be negligible on $\hat\pi$ and visible
on the *counts* that are functions of it, because a count near a threshold moves
on an arbitrarily small change in the rate. Kupiec passes and Basel zones are
both threshold functions. If $\hat\pi$ moves by less than 0.001 and a headline
count still moves, that is not a contradiction and is reported as what it is:
the sensitivity of a thresholded statistic, not instability in the estimator.

## What would falsify the exercise

If any cell has $n_{\mathrm{cal}} \le g_n$ the gap cannot be applied there and
the run is BLOCKED for that cell, reported rather than silently skipped. With
$n_{\mathrm{cal}}$ in the thousands and $g_n$ around ten, none is expected.

## What is not in scope

The rolling estimator. Theorem 4.5 covers the single split only, and inserting a
gap into a trailing window is a different object. The rolling results stand as
they are and Section 4.4's statement about them is unchanged.
