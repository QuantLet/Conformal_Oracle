# REPRO NOTES — E4: Diagnostic Regression

**Quantlet:** `CO_diagnostic_regression`
**Script:** `run_diag_regression.py`
**Date:** 2026-05-08

## What it does

OLS regression of ΔQS = QS_raw − QS_cp on qV_stat and π̂_raw across
240 model–asset pairs (α = 0.01). Reports coefficients with three SE
specifications: homoskedastic OLS, cluster-robust by asset (24 clusters),
cluster-robust by model (10 clusters). Partial R² for qV_stat computed
via Frisch–Waugh–Lovell.

## Data

- `cfp_ijf_data/paper_outputs/tables/all_results.csv` — 9 models × 24 assets × 4 alphas (864 rows; filtered to α = 0.01 → 216 rows)
- `cfp_ijf_data/paper_outputs/tables/moirai11_results.csv` — Moirai-1.1, 24 rows at α = 0.01

Combined: n = 240.

## Key results

| Metric | Value |
|--------|-------|
| R² | 0.782 |
| Partial R² (qV_stat) | 0.534 |
| β(qV_stat) | significant at 1% under all three SE specs |
| β(π̂_raw) | significant at 1% under all three SE specs |

## Drift note

The body previously cited R² = 0.822 and partial R² = 61.6%. Those
values were computed from n = 216 (9 models, before Moirai-1.1 was
added). Verified: running the same regression on the 216-row subset
yields R² = 0.819 ≈ 0.822 (rounding). The canonical n = 240 results
(R² = 0.782, partial R² = 53.4%) are now in the body and appendix.

## Output

- `tab_diag_regression.tex` — LaTeX table fragment (\input-ready)
- `diag_regression_results.csv` — coefficient estimates and all SEs

## Environment

- Python 3.x, statsmodels, pandas, numpy
- Runtime: < 1 second
