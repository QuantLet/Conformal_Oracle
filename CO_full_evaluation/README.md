# CO_full_evaluation

Main evaluation pipeline: 9 models x 24 assets (Tables 1-8, Figures 1-5)

**Paper:** Distribution-Free Recalibration of Tail Quantile Forecasts under Temporal Dependence

**Repository:** [QuantLet/Conformal_Oracle](https://github.com/QuantLet/Conformal_Oracle)

## Models

| # | Forecaster | Type |
|---|-----------|------|
| 1 | Chronos-Small | TSFM |
| 2 | Chronos-Mini | TSFM |
| 3 | TimesFM 2.5 | TSFM |
| 4 | Moirai 1.1-R | TSFM |
| 5 | Lag-Llama | TSFM |
| 6 | GJR-GARCH | Parametric |
| 7 | GARCH-N | Parametric |
| 8 | Historical Simulation | Parametric |
| 9 | EWMA | Parametric |

## Pipeline

1. Load forecast parquets (5 TSFMs + 4 benchmarks, 24 assets)
2. Compute nonconformity scores: s_t = q_t^lo - r_t
3. Calibrate conformal threshold q_V (70/30 split)
4. Correct VaR: VaR_t = -(q_t^lo - q_V)
5. Run backtests: Kupiec, Christoffersen, Basel TL, Acerbi Z2
6. Generate tables and figures

## Files

- `CO_full_evaluation.ipynb` — main notebook
- Results in `results/`

## Usage

See the main repository README for full reproduction instructions.
