# CFP_Capital_Charge

**Cumulative Capital Charge Comparison (Section 6)**

Computes cumulative daily capital charges under Basel zone multipliers for
S&P 500 under three configurations:
1. Lag-Llama at Yellow-zone multiplier (k=3.65)
2. Conformally corrected Lag-Llama at Green-zone multiplier (k=3.00)
3. GJR-GARCH with conformal correction at Green-zone multiplier (k=3.00)

From "Distribution-Free Recalibration of Tail Quantile Forecasts under
Temporal Dependence" (Pele, Lessmann, Hardle, 2026).

## Dependencies

```
numpy, pandas, matplotlib, scipy
```

## Usage

```bash
cd CFP_Capital_Charge
python CFP_Capital_Charge.py
```

## Output

- `capital_charge_cumulative.pdf` — publication-quality figure (300 DPI)
- `capital_charge_cumulative.png` — preview figure (150 DPI)
- Summary statistics printed to stdout

## Links

- Paper: [Conformal_Oracle](https://github.com/QuantLet/Conformal_Oracle/)
- Main pipeline: [CO_full_evaluation](../CO_full_evaluation/)
