# CFP_Baselines_EVT_FHS

**EVT-POT and Filtered Historical Simulation Baselines (Table 12)**

Computes two standalone VaR models as baselines for comparison with conformal
recalibration methods in "Distribution-Free Recalibration of Tail Quantile
Forecasts under Temporal Dependence" (Pele, Lessmann, Hardle, 2026).

- **EVT-POT** (McNeil and Frey, 2000): GARCH(1,1) + GPD tail fit
- **Filtered Historical Simulation** (Barone-Adesi et al., 1999): GARCH(1,1) + empirical quantile of standardised residuals

Both use daily re-estimation on 250-day rolling windows across 24 assets.

## Dependencies

```
numpy, pandas, scipy, arch
```

## Usage

```bash
python CFP_Baselines_EVT_FHS.py
```

## Output

- `baselines_evt_fhs.csv` — per-asset results
- `baselines_evt_fhs_summary.csv` — means across 24 assets
- LaTeX rows for Table 12 printed to stdout

## Links

- Paper: [Conformal_Oracle](https://github.com/QuantLet/Conformal_Oracle/)
- Main pipeline: [CO_full_evaluation](../CO_full_evaluation/)
