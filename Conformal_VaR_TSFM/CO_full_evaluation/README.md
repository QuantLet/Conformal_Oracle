# CO_full_evaluation

Main pipeline: 9 forecasters x 24 assets, Tables 1-8, Figures 1-5.

## Notebooks (run in order)

| # | Notebook | Description |
|---|----------|-------------|
| 1 | CFP_Data_Download | 24 asset returns from Yahoo Finance (2000-2026) |
| 2 | CFP_Chronos_Forecasts | Chronos-Small and Chronos-Mini VaR forecasts |
| 3 | CFP_TimesFM_Forecasts | TimesFM 2.5 quantile-to-VaR via Student-t fit |
| 4 | CFP_Moirai_Forecasts | Moirai 1.1-R VaR forecasts |
| 5 | CFP_LagLlama_Forecasts | Lag-Llama VaR forecasts |
| 6 | CFP_Parametric_Benchmarks | GJR-GARCH, GARCH-N, HS, EWMA |
| 7 | CFP_Conformal_Calibration | One-sided conformal calibration + Kupiec/Basel/Acerbi |
| 8 | CFP_qV_Diagnostic | Conformal threshold ranking and rolling overlay |
| 9 | CFP_Multi_Quantile | Evaluation at alpha = 0.01, 0.025, 0.05, 0.10 |
| 10 | CFP_Panel_Pooled | Cross-asset pooled backtest with HAC errors |
| 11 | CFP_Scoring_Rules | Quantile Score + Diebold-Mariano tests |
| 12 | CFP_Traffic_Light | Basel traffic light heatmaps |
| 13 | CFP_Paper_Tables_Figures | All publication-ready tables and figures |

## Output

- `results/` — parquet forecasts, backtest CSVs, LaTeX tables, PDF figures
