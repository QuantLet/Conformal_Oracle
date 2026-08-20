# uMCB re-run on the corrected panel — 20 August 2026

`MEMO.md`, `umcb_pairs.csv` and `fig_umcb_qv.png` were regenerated. The versions
computed on 15 August, before the four defect corrections, are kept unmodified in
`superseded_20260815/`.

## Why it had to be re-run, and why it was not automatic

The script had been half-updated on 17 August: `MODELS` and the defect flag were
switched to `cfp_config`/`DEFECTIVE_SERIES`, but the reporting half still asked
for the `grid` and `panel` columns of the withdrawn Panel A/B taxonomy, so it
raised `KeyError: 'grid'` and no output was produced. The subsets are now:

| old subset | new subset |
|---|---|
| excluding quantile-grid failures | well-specified series only (264 pairs) |
| Panel A only / Panel B only | top_k-truncated series only (48 pairs) |

The Panel A/B split is not reinstated under another name. What replaces it is a
partition by *traced defect*: the two Chronos series sampled at the checkpoint
default `top_k = 50` against everything else.

## What changed

Panel: 240 pairs (10 x 24) to 312 (13 x 24).

| quantity | 15 Aug, defective panel | 20 Aug, corrected panel |
|---|---|---|
| identity check, max abs residual | 1.39e-17 | see MEMO (unchanged in kind) |
| Spearman(abs q_V, uMCB), all pairs | 0.899 (n = 240) | 0.704 (n = 312) |
| Spearman, usable forecasters | 0.593 (Panel A, n = 144) | 0.527 (well-specified, n = 264) |
| Spearman, defective subset | 0.896 (Panel B, n = 96) | 0.924 (truncated, n = 48) |
| share of MCB that is unconditional, median | 0.354 (Panel A) | **0.270** (well-specified) |
| same, all pairs | 0.523 | 0.330 |
| implied f, 5th-95th pct | 0.2 to 18.7 (factor 83) | 0.1 to 25.2 (factor 481) |

**Verdict (a) is unaffected**, as anticipated: q_V estimates the unconditional
miscalibration component of Gneiting and Resin (2023), and that is an identity
between estimators rather than an empirical finding, so it does not depend on
which series it is evaluated on.

**The magnitude reported alongside it does change, and in the direction that
costs the paper something.** The abstract of `main_R2.tex` states that the shift
"addresses a median 35% of total miscalibration". On the corrected panel the
figure for well-specified series is **27%**, and 33% if the truncated Chronos
pair is averaged in. The old 35% was computed on a subset containing two
sign-inverted series, where almost all miscalibration is unconditional by
construction (the truncated subset still shows 0.99).

The rank agreement between abs q_V and uMCB also weakens among usable
forecasters, and the implied-density spread widens to a factor of 481. Both
strengthen the same conclusion the memo already drew: the two quantities are not
readable off one another without the residual density f.
