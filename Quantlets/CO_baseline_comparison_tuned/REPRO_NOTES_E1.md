# REPRO_NOTES — E1: Tuned GBM-QR Ablation

**Quantlet:** `CO_baseline_comparison_tuned`
**Script:** `run_tuned_gbm_qr.py`
**Date:** 2026-05-08

## Reproduction

```bash
cd "/Users/danielpele/Documents/2026 CFP LLM VaR"
python3 Quantlets/CO_baseline_comparison_tuned/run_tuned_gbm_qr.py
```

## Dependencies

- Python 3.10+, LightGBM, NumPy, Pandas, SciPy
- Input data: `cfp_ijf_data/returns/SP500.csv`, `cfp_ijf_data/{model_dir}/SP500.parquet`

## Grid specification

| Parameter | Values |
|-----------|--------|
| `n_estimators` | 100, 500 |
| `max_depth` | 3, 5 |
| `learning_rate` | 0.01, 0.05 |

8 configs × 13 base models = 104 individual fits, all on SP500.

**Corrected 2026-08-28.** This note was written on 2026-05-08 against a nine-model
run and said "8 configs × 9 base models = 72 individual fits". The shipped
`tuned_gbm_qr_grid.csv` has 104 rows over thirteen models and
`tuned_gbm_qr_summary.csv` records `n_pairs = 13` on every row; the emitted
`tab_gbm_tuned.tex` prints `0/13` and `8/13`. The artefact was re-run onto
thirteen series and this note was not, so Supplement S.5 carried 5/9 and 88.9%
beside a table reading 8/13 and 84.6%. 88.9% is 8/9 and corresponds to no count
in the current archive. All figures below are recomputed from the shipped grid.

## Key result

Best QS config (n=100, d=3, lr=0.05): QS=4.38×10⁻⁴, π̂=.0155, 8/13 Kupiec rejections, 84.6% Green.
Conservative config (n=100, d=3, lr=0.01): QS=4.61×10⁻⁴, π̂=.0110, 0/13 rejections, 100% Green.

The QS-optimal tuning overshoots the 1% target (π̂=.0155 > .010) and loses coverage validity on
8 of 13 base models, confirming Remark 3.2's prediction that at the 1% tail with αT ≈ 15
effective observations, additional GBM parameters add variance faster than they reduce bias.

These figures are emitted as macros by `scripts/paper_numbers.py` under the `Sup`
prefix, so the supplement no longer carries them as literals and this note is no
longer a source the manuscript reads from.

## Outputs

- `tuned_gbm_qr_grid.csv` — 104-row full results (per config × per model)
- `tuned_gbm_qr_summary.csv` — 8-row config-level summary
- `tab_baselines_tuned_row.tex` — LaTeX row for best config, compatible with tab_baselines.tex
