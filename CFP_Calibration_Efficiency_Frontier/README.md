# CFP_Calibration_Efficiency_Frontier

**Two-Panel Calibration-Efficiency Frontier (Figure 3)**

Generates the calibration-efficiency frontier showing the trade-off between
empirical coverage (1 - violation rate) and VaR width for four representative
forecasters before and after conformal correction.

- Panel (a): Zoomed view near the 99% target
- Panel (b): Full scale including Chronos-Small (raw coverage ~61%)

From "Distribution-Free Recalibration of Tail Quantile Forecasts under
Temporal Dependence" (Pele, Lessmann, Hardle, 2026).

## Dependencies

```
numpy, pandas, matplotlib
```

## Usage

```bash
cd CFP_Calibration_Efficiency_Frontier
python CFP_Calibration_Efficiency_Frontier.py
```

## Output

- `calibration_efficiency_frontier_v2.pdf` — publication-quality figure (300 DPI)
- `calibration_efficiency_frontier_v2.png` — preview figure (150 DPI)

## Links

- Paper: [Conformal_Oracle](https://github.com/QuantLet/Conformal_Oracle/)
- Main pipeline: [CO_full_evaluation](../CO_full_evaluation/)
