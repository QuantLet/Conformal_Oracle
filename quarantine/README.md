# Quarantine — undated and stale forecast series

Removed from `cfp_ijf_data/` on 2026-08-17 because the promotion gate read them
as part of the live population. Retained, not deleted: they are part of the
provenance record, and an undated vintage that nobody can place should be
unpromotable rather than invisible.

## `2026-08-17_undated_six_column_vintage/` — 121 series

A second copy of every asset for Chronos-Small, Chronos-Mini, TimesFM 2.5,
Moirai 2.0 and Lag-Llama, stored beside the live files under the macOS
duplicate-name convention `<ASSET> 2.parquet`.

**What they are, as far as the files allow.** They carry a six-column schema —
`VaR_0.01, VaR_0.025, VaR_0.05, VaR_0.1, mean, std` — with no `df_student` and
no `ES_student_0.025`. The live grid-model files carry eight columns including
both. Six columns is the signature of the *sample-based* writer in
`CFP_Moirai_Forecasts.ipynb`, not the grid writer.

Their values behave accordingly: for the grid models they are correctly signed
(SP500 median VaR₀.₀₁ = −0.0246 for Moirai 2.0, −0.0219 for TimesFM) and
monotone across α, where the live files are positive and reversed; and their
`std` sits at the order of realised volatility rather than at the fitted
Student-t scale.

**Most defensible reading:** an earlier vintage, predating the Student-t tail
closure, when the grid models were still read as sample quantiles. The
eight-column files replaced them when the closure was introduced, and brought
the sign error with it.

**Limitation, stated rather than buried.** They cannot be dated. iCloud mtimes
are not evidence, `cfp_ijf_data/` has no git history, and no cell in the current
pipeline writes a six-column file for these two models. The characterisation
rests on schema and value ranges alone.

They are **not** a usable correction: their `mean` and `std` differ
substantially from the live files, so they are a different inference run, not
the same numbers with the sign fixed. The corrected series in
`analysis/recompute/corrected/` were rebuilt from the live files' own fitted
parameters and are the ones to promote.

For Chronos the `" 2"` files differ from the live ones only in the fifth decimal
— the same `top_k = 50` run with a different seed. No corrected Chronos vintage
exists anywhere.

## `2026-08-17_stale_CACT_ticker/` — 5 series

`CACT.parquet` in five model directories. `CACT` was the CAC 40 ticker before
the data rebuild renamed it `FCHI`. Six pipeline scripts still referenced the
old name and silently skipped the asset, producing 23-asset tables that looked
complete; that was corrected on 2026-08-15. These forecast files are the other
half of the same defect — the scripts stopped looking for `CACT`, but its
forecasts were never removed.

Moirai 1.1 is the only model directory that contained neither artefact,
consistent with its series having been regenerated after the rename.

## Also removed

Eight `.ipynb_checkpoints/` directories inside `cfp_ijf_data/`.

## State after quarantine

Every model directory holds exactly 24 series; `benchmarks/` holds 100
(4 estimators × 25, the 25th being the `CACT` file for each, retained here
pending the same treatment). The promotion gate reads only what the manifest
names and fails on anything it finds that is not listed.
