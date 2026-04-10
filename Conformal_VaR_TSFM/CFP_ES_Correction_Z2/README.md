# CFP_ES_Correction_Z2

**Heuristic ES Correction and Z2 Backtest (Table C.1)**

Computes the heuristic Expected Shortfall correction from Appendix C of
"Distribution-Free Recalibration of Tail Quantile Forecasts under Temporal Dependence"
(Pele, Lessmann, Hardle, 2026) and evaluates the Acerbi-Szekely Z2 backtest statistic
before and after correction across 9 forecasters and 24 financial assets.

ES is derived from each model's mean/std output using the Gaussian ES formula.
The conformal ES correction shifts ES by q_hat_E computed on calibration-set VaR
violation days (Definition C.2).

## Dependencies

```
numpy, pandas, scipy
```

## Usage

```bash
cd CFP_ES_Correction_Z2
python CFP_ES_Correction_Z2.py
```

## Output

- `table_c1_es_correction.csv` — per-model-asset Z2 results
- LaTeX table block printed to stdout (Table C.1 of the paper)

## Links

- Paper: [Conformal_Oracle](https://github.com/QuantLet/Conformal_Oracle/)
- Main pipeline: [CO_full_evaluation](../CO_full_evaluation/)
