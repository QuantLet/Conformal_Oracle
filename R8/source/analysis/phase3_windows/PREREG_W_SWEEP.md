# Pre-registration — the rolling calibration window, w in {125, 250, 500}

Written 2026-08-29, before the sweep runs.

## The question and why it is not the one R2-1 asked

R2-1 asked for an ablation over the rolling calibration window. Section 3.2.1
already answers part of it analytically and exactly: equation~(8) sets
$k = \lceil (w+1)(1-\alpha)\rceil$, and $k \ge w$ whenever $w < 2/\alpha - 1$, so
at $\alpha = 0.01$ the shift is the **window maximum** for every window of 198
observations or fewer. The three windows a reader would compare are therefore

| $w$ | $k$ | which order statistic | |
|---|---|---|---|
| 125 | 125 | the **maximum** of the window | an extreme-value statistic |
| 250 | 249 | second largest | interior |
| 500 | 496 | fifth largest | interior |

These are arithmetic and need no data. What needs data is the consequence: does
the change of estimator at $w = 125$ show up in the panel, or is it a distinction
without a difference?

## The unit

One row is a **cell**: one forecaster, one asset, one window, at $\alpha = 0.01$.
Expected rows: $13 \times 24 \times 3 = 936$. What varies is the trailing window
over $\{125, 250, 500\}$; everything else is held at the manuscript's
configuration. The 13 forecasters are the sequence panel, because the three
dynamic-quantile models store no series.

## Predictions, in both directions

The extreme-value reading says the maximum of a window has a variance that does
not shrink at the $1/\sqrt{w}$ rate the interior order statistics do, and that it
sits further into the tail, so the shift is larger and the corrected forecast
more conservative.

**If $w = 125$ separates from the other two**, the prediction holds: its shift
should be larger on most cells, its corrected violation rate further below
nominal, and its dispersion across time larger by more than the $\sqrt{2}$ that
halving the window alone would give. Section 7 then reports that
$w \in \{125, 250, 500\}$ is not three points on a variance curve, and the
requested ablation is answered by saying one of its points is a different
estimator.

**If $w = 125$ sits on a smooth curve with the other two**, the analytic point is
correct and immaterial: the estimator changes kind without changing behaviour at
this sample size, and Section 7 says exactly that. This is the outcome that costs
the paper its sharpest answer to R2-1, and it is reported as readily.

**A third outcome, and it is the one to watch.** The separation may appear in
dispersion but not in coverage, or the reverse. Coverage at $\alpha = 0.01$ is a
count of rare events and is the less sensitive of the two; dispersion of the
shift over time is continuous and measured on the same dates. If they split, the
dispersion result is the one with the resolution to carry a claim, and the
coverage result is reported with the grid it lives on, as guard 6 requires.

## What would falsify the exercise itself

If fewer than 90% of the expected 936 cells produce a defined shift on every one
of the three windows, the comparison is not like-for-like and the result is
BLOCKED rather than reported. Cells are compared on the intersection of dates
available at all three windows, so the longest window sets the common sample and
no cell is compared against a different span of history than another.
