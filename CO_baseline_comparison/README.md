# CO_baseline_comparison

Conformal recalibration vs 4 alternatives (Table 11)

**Paper:** Distribution-Free Recalibration of Tail Quantile Forecasts under Temporal Dependence

**Repository:** [QuantLet/Conformal_Oracle](https://github.com/QuantLet/Conformal_Oracle)

## Baselines

| # | Method | Parameters | Description |
|---|--------|-----------|-------------|
| 1 | Scale correction | 1 (scalar) | Multiplicative rescaling by observed/expected violation ratio |
| 2 | Historical quantile | 0 | Replace model quantile with empirical quantile of past returns |
| 3 | Quantile regression | ~3 | Fit QR on model forecast to correct conditional quantile |
| 4 | Isotonic regression | O(n) | Monotone recalibration of predicted quantiles |
| 5 | **Conformal (ours)** | **1 (q_V)** | **One-sided conformal shift** |

## Key Finding

Isotonic regression achieves only 53.7% Basel Green despite richer parameterisation — binning fragments the sparse 1% tail sample. At extreme quantiles, the bias-variance trade-off favours the one-parameter conformal shift.

## Files

- `CO_baseline_comparison.ipynb` — main notebook
- Results in `results/`

## Usage

See the main repository README for full reproduction instructions.
