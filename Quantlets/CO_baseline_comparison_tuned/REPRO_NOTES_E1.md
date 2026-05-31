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

8 configs × 9 base models = 72 individual fits, all on SP500.

## Key result

Best QS config (n=100, d=3, lr=0.05): QS=4.40×10⁻⁴, π̂=.015, 5/9 Kupiec rejections, 88.9% Green.
Conservative config (n=100, d=3, lr=0.01): QS=4.63×10⁻⁴, π̂=.011, 0/9 rejections, 100% Green.

The QS-optimal tuning overshoots the 1% target (π̂=.015 > .010) and loses coverage validity on
5 of 9 base models, confirming Remark 3.2's prediction that at the 1% tail with αT ≈ 15
effective observations, additional GBM parameters add variance faster than they reduce bias.

## Outputs

- `tuned_gbm_qr_grid.csv` — 72-row full results (per config × per model)
- `tuned_gbm_qr_summary.csv` — 8-row config-level summary
- `tab_baselines_tuned_row.tex` — LaTeX row for best config, compatible with tab_baselines.tex
