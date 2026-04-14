# CO_baseline_comparison

Conformal recalibration vs 4 alternatives (Table 11).

## Baselines

1. **Scale correction** — multiplicative rescaling of VaR by observed/expected violation ratio
2. **Historical quantile** — replace model quantile with empirical quantile of past returns
3. **Quantile regression** — fit QR on model forecast to correct conditional quantile
4. **Isotonic regression** — monotone recalibration of predicted quantiles

## Output

- `results/tab_baseline_comparison.tex`
- `results/fig_baseline_qs.pdf`
