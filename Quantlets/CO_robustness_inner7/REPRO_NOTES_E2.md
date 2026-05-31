# REPRO_NOTES — E2: Inner-7-Deciles Tail-Completion Robustness

**Quantlet:** `CO_robustness_inner7`
**Script:** `run_inner7_tail_closure.py`
**Date:** 2026-05-08

## Reproduction

```bash
cd "/Users/danielpele/Documents/2026 CFP LLM VaR"
python3 Quantlets/CO_robustness_inner7/run_inner7_tail_closure.py
```

## Dependencies

- Python 3.10+, NumPy, Pandas, SciPy
- Input data: `cfp_ijf_data/{timesfm25,moirai2}/{SP500,BTC,NATGAS}.parquet`
- Input returns: `cfp_ijf_data/returns/{SP500,BTC,NATGAS}.csv`

## Method

The original `scripts/tail_completion_robustness.py` fits parametric distributions
to the full 9-decile quantile grid (u = 0.1, …, 0.9) reconstructed from stored
Student-t parameters (df_student, mean, std).

The inner-7-deciles variant drops the outermost quantiles (u = 0.1 and u = 0.9)
and refits the Student-t using only u ∈ {0.2, 0.3, …, 0.8}. This tests whether
tail-extrapolation to the 1% level is driven by the extreme ends of the predictive grid.

Nelder-Mead fitting with maxiter=2000, xatol=1e-8. Conformal pipeline uses
F_C = 0.70, α = 0.01 (same as main analysis).

## Outputs

- `inner7_tail_closure.csv` — full results (4 closures × 2 models × 3 assets = 24 rows)
- `tab_tail_closure_extended.tex` — LaTeX table extending tab_tail_closure.tex with inner-7 rows
