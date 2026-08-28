# L1 — do the shipped tables reproduce from the shipped code?

Diagnosis only. Nothing repaired. 27 August 2026.

**Method.** `make.sh tables` was run twice in the same environment: once with the
pre-R14 analytic panels and once with the corrected ones. The first run is the
control — its inputs are byte-identical to what produced the committed files, so
anything that moves in it moves for a reason other than R14. Ten files moved.
Each is then classified on two axes the summary must not merge: does an emitter
exist, and does it reproduce the **form** (columns, structure, prose) as well as
the **values**.

A table that comes out with a different *shape* is not numerical drift. It is
evidence that the committed file was written by code other than the code in the
repository — the Table S.10 pattern.

## The ten, with verdicts

| # | artefact | emitter | form | values | verdict |
|---|---|---|---|---|---|
| 1 | `CO_multi_quantile_panel/tab_panel_pooled.csv` | `run_panel_pooled.py` | yes | 1e-16 | **REPRODUCES** |
| 2 | `CO_quantile_scores/tab_dm_pvalues.csv` | `run_dm_pvalues.py` | yes | 1.1e-14 | **REPRODUCES** |
| 3 | `CO_asset_overview/tab_assets.tex` | `run_asset_overview.py` | **no** | 186/186 identical | **EDITED AFTER GENERATION** |
| 4 | `CO_robustness_inner7/tab_tail_closure_extended.tex` | `run_inner7_tail_closure.py` | **no** | 102/102 identical | **EDITED AFTER GENERATION** |
| 5 | `CO_full_evaluation/tab_master_results.tex` | two, and `make.sh` calls the wrong one | **no** — 12 columns vs 10 | 169 vs 125 tokens | **WRONG EMITTER WIRED** |
| 6 | `CO_full_evaluation/tab_master_results.csv` | as above | **no** | `raw_kup` 0 vs 13 | **WRONG EMITTER WIRED** |
| 7 | `CO_regime_sensitivity/tab_regime_sensitivity.tex` | `run_regime_sensitivity.py` | yes | **35/150 differ** | **PRE-CORRECTION VINTAGE** |
| 8 | `CO_bound_validation/tab_bound_validation.csv` | `run_bound_validation.py` | yes | **9/39 differ** | **PRE-CORRECTION VINTAGE** |
| 9 | `CO_bound_validation/tab_bound_validation.tex` | as above | **no** (model label) | as above | **PRE-CORRECTION VINTAGE** |
| 10 | `CO_gbm_qr/gbm_qr_results.csv` | `baseline_gbm_qr.py` | yes | **58/216 rows** | **UNSEEDED STOCHASTIC ESTIMATOR** |

## What each class means

### REPRODUCES (1, 2) — not divergence

`tab_panel_pooled.csv` differs only in the last digit of a float repr
(`0.0016808411753621007` against `0.001680841175362102`); `tab_dm_pvalues.csv` to
1.1e-14. Neither `.tex` companion appears in the drift list at all, because both
round to printed precision. These two are noise from the arithmetic library, not
a reproduction failure, and they should be removed from the count before anyone
reads it as nine or ten broken tables.

### EDITED AFTER GENERATION (3, 4) — values sound, form is not the emitter's

`tab_assets.tex` ships `\hline\hline` where the emitter writes `\toprule`;
`tab_tail_closure_extended.tex` ships a table note ending "see the discussion"
where the emitter writes "it cannot fail." Every number in both is identical.

`git log -S` shows **neither emitter has ever contained the shipped string**, so
this is not an older version of the generator: the `.tex` was hand-edited after
being generated. Harmless today, and a live trap — a rebuild silently reverts an
intentional editorial change, and nothing announces it.

### WRONG EMITTER WIRED (5, 6) — already known, still wired wrong

The published `tab_master_results.tex` has twelve columns over ten models, with a
CC-pass column and an $\bar R$ column. `run_master_table.py` — the script
`make.sh` calls at target T4 — emits ten columns over nine models and places
TimesFM 2.5 and Moirai 2.0 in the wrong panel. It cannot produce this table and
`MANIFEST.md` already records it as `DIFFERS` and marks that script **SUPERSEDED**.

`rebuild_master_table.py` was written to close exactly this gap and reproduces
109 of 110 cells. So the repair here is a one-line change in `make.sh`, not an
investigation. The defect is that the manifest recorded the finding and the build
was never repointed, so `make.sh tables` still regresses the file every time it
runs.

Neither file is `\input` by `main_R2.tex`, which uses `tab_master_results_r2`
from `build_table1_r2.py`, nor read by `paper_numbers.py`.

### PRE-CORRECTION VINTAGE (7, 8, 9) — the serious ones

All three were last committed **2026-05-31** and never regenerated after the sign
defect was corrected on **2026-08-17**. The signature is unambiguous: the models
that move are exactly the three whose series were corrected — GJR-GARCH,
Moirai 2.0, TimesFM 2.5 — and no others.

`tab_regime_sensitivity.tex`:

| row | shipped | recomputed |
|---|---|---|
| TimesFM 2.5, all nine thresholds | 24, 24, 24, 24, 24, 24, 24, 24, 24 | 11, 7, 7, 7, 5, 4, 4, 2, 1 |
| Moirai 2.0 | 24 × 9 | 12, 12, 10, 11, 9, 9, 7, 4, 2 |
| GJR-GARCH | 0 × 9 | 8, 5, 2, 4, 3, 2, 2, 1, 0 |
| $R > 1.2$ headline | 138/240 | **121/240** |

A row of 24/24 across every threshold is the fingerprint of the ~99% raw
violation rate the sign defect produced. `tab_bound_validation` carries the same
defect more quietly: $\hat\rho$ for TimesFM 0.64 → 0.62 and for Moirai 0.49 →
0.41, with every other row unchanged.

**Neither table is in `main_R2.tex`. Both are in the submitted `main_R1.tex`, six
references between them, and the frozen copies under `submission_IJF/` are
byte-identical to the committed ones.** The version under review therefore prints
two tables computed from series this project has since established were wrong.
That is an erratum, and it is the reason L1 blocks the SSRN replacement rather
than merely the next rebuild.

`MANIFEST.md` grades `tab_regime_sensitivity.tex` **OK — reproduced by
run_regime_sensitivity.py**. That verdict was true when it was recorded and is
false now. See the note below.

### UNSEEDED STOCHASTIC ESTIMATOR (10)

`baseline_gbm_qr.py` trains LightGBM with `feature_fraction=0.9`,
`bagging_fraction=0.8`, `bagging_freq=5` and **sets no seed** — not `seed`, not
`bagging_seed`, not `feature_fraction_seed`. It relies on the library's default
seeding, which is stable within a version and is not a contract across versions.
`requirements.txt` pins lightgbm 4.6.0; 4.7.0 is installed.

Evidence that separates the candidate causes:

- the two rebuilds are **byte-identical to each other**, so it is deterministic
  within this environment — not run-to-run noise;
- `baseline_gbm_qr.py` reads none of the analytic panels (its own nine-model dict
  excludes them), so R14 cannot be the cause;
- **58 of 216 rows** change their violation count, up to 45% relative — far too
  large for arithmetic;
- rows for `Chronos-Mini`, whose input series was never corrected, are among
  those that move, which rules out the data-vintage explanation for this file.

What remains is the training seed. This is the only one of the ten whose cause is
inferred rather than demonstrated: proving it requires installing lightgbm 4.6.0
and re-running, which has not been done.

It matters beyond its own file: `gbm_qr_results.csv` feeds
`compile_tab_baselines.py`, which emits `tab_baselines` — Table S.25 of the
supplement, already carried in the ledger as `NOT_EMITTED`.

## The instrument that should have caught this

`analysis/provenance/build_manifest.py` is the project's own answer to this
question and it did not raise these. Two structural reasons, both worth fixing
before it is trusted again:

1. **It compares against the frozen submission, not the working tree.** Its
   inventory comes from `submission_IJF/main_R1.tex` and each regenerated
   artefact is compared to the copy under `submission_IJF/`. Drift introduced
   *after* the submission is outside its field of view by construction.
2. **Its verdicts are snapshots with no expiry.** `tab_regime_sensitivity.tex` is
   recorded `OK`. It was OK on the day it ran. The 17 August correction changed
   the inputs underneath it and nothing re-ran the manifest, so a stale `OK` now
   reads as a live guarantee.

This is the same failure mode as the three already in `PROTOCOL.md`, in a new
place: not a check that cannot fail, but a check whose **passing verdict outlives
the state it described**.
