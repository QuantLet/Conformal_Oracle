# What the gap does to the pipeline's own numbers

Run 2026-08-31 by `measure_gap_on_pipeline.py`, which imports the shipped
evaluation driver and re-runs its backtest with the separation inserted. Writes
nothing. This supersedes the standalone comparison in `RESULTS_GAP_RUN.md` as
evidence for a decision, because it is computed on the object the tables read.

## At alpha = 0.01, the 312-cell panel

| quantity | contiguous split | with $g_n$ |
|---|---|---|
| Basel green | **278** | **278** |
| Kupiec passes | **230** | **229** |
| Basel zone changes | --- | **0** of 312 |
| Christoffersen verdict changes | --- | **1** |
| max $\lvert\Delta\hat\pi\rvert$ | --- | **0.000521** |
| gap $g_n$ | --- | 5 to 30, median 8 |

## Across all four levels, 1,248 cells

max $\lvert\Delta\hat\pi\rvert$ **0.002789**, **0** zone changes, **11** Kupiec
verdict flips.

## Why this matters more than the earlier run

`run_gap_panel.py` re-split the stored scores itself and reported 297 green and
230 Kupiec passes. The 230 matches; the green does not, because that script
scales the traffic light differently from the driver. The paired differences it
reported were right, but its levels were its own. This run has neither problem:
both arms come from `conformal_backtest` in `run_full_evaluation.py`, the
function that produced `all_results.csv`.

## The decision this supports, and the constraint it collides with

Switching the reported panel to the gapped estimator would make
Theorem 4.5 cover the estimator the tables report, which is what the title
promises, at a cost of **one Kupiec pass at alpha = 0.01** and no change to any
Basel classification. The compute is seconds: `run_full_evaluation.py --verify`
reproduces the committed artefact in 5 s.

It is not done here, because it changes a figure placed under an explicit
do-not-touch instruction: Table S.25's 82 of 312 rejections, equivalently 230
passes, becomes 83 and 229. That instruction was given on the understanding that
the figure was correct, which it is; the switch would make it correct about a
different estimator. Lifting it is the author's call, not this file's.
