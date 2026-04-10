# Conformal_VaR_TSFM

**Distribution-Free Recalibration of Tail Quantile Forecasts under Temporal Dependence**

Pele, D.T., Lessmann, S., Hardle, W.K. (2026)

## Quantlets

| Quantlet | Description |
|----------|-------------|
| CO_full_evaluation | Main pipeline: 9 models x 24 assets, Tables 1-8, Figures 1-5 |
| CO_simulation_study | Monte Carlo: 5 DGPs x 500 reps |
| CO_rolling_conformal | Rolling vs static correction (Table 10) |
| CO_baseline_comparison | Conformal vs 4 alternatives (Table 11) |
| CFP_ES_Correction_Z2 | Heuristic ES correction and Z2 backtest (Table C.1) |
| CFP_Capital_Charge | Cumulative capital charge comparison (Section 6) |
| CFP_Calibration_Efficiency_Frontier | Two-panel calibration-efficiency frontier (Figure 3) |

## Data

All return series sourced from Yahoo Finance.
Pinned TSFM checkpoints listed in Table 2 of the paper.

## Requirements

Python 3.10+, torch, chronos, timesfm, uni2ts,
gluonts, arch, statsmodels, scipy, pandas, numpy

## Citation

```bibtex
@article{pele2026conformal,
  title={Distribution-Free Recalibration of Tail Quantile
         Forecasts under Temporal Dependence},
  author={Pele, Daniel Traian and Lessmann, Stefan
          and H{\"a}rdle, Wolfgang Karl},
  journal={Working Paper},
  year={2026}
}
```

## License

MIT
