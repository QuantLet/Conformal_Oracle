# Conformal_VaR_TSFM

**Distribution-Free Recalibration of Tail Quantile Forecasts under Temporal Dependence**

Pele, D.T., Bolovăneanu, V., Ginavar, A.T., Lessmann, S., Hardle, W.K. (2026)

## Quantlets

| Quantlet | Description |
|----------|-------------|
| CO_full_evaluation | Main pipeline: 9 models x 24 assets, Tables 1-8, Figures 1-5 |
| CO_baseline_comparison | Conformal vs 4 recalibration alternatives (Table 11) |
| CO_gbm_qr | Gradient-boosted quantile regression baseline (LightGBM), Table 12 |
| CO_simulation_study | Monte Carlo: 5 DGPs x 500 reps (Section 5.7) |
| CO_coverage | Coverage recovery comparison (Figure 3) |
| CO_cross_sectional | Cross-sectional q_V vs asset characteristics |
| CO_cross_model | Cross-model threshold comparison (Figure 2) |
| CO_frontier | Coverage-efficiency frontier (Figure 5) |
| CO_garch_conformal | Parametric benchmark conformal correction |
| CO_heatmap | q_V heatmap across models and assets (Figure 6) |
| CO_multi_quantile_panel | Multi-quantile panel evaluation (Table 5) |
| CO_pipeline | Forecasting pipeline diagram |
| CO_quantile_scores | Quantile Score evaluation and DM tests (Table 7) |
| CO_raw_traffic_light | Basel traffic light matrix (Table 3) |
| CO_rolling_qV | Rolling q_V stability analysis (Figure 7) |
| CO_score_comparison | One-sided vs two-sided score comparison |
| CO_sharpness_penalty | Calibration-efficiency trade-off |
| CO_sign_diagnostic | q_V sign diagnostic heatmap |
| CFP_ES_Correction_Z2 | Heuristic ES correction and Z2 backtest (Table C.1) |
| CFP_Capital_Charge | Cumulative capital charge comparison (Section 6) |
| CFP_Calibration_Efficiency_Frontier | Two-panel calibration-efficiency frontier (Figure 3) |

## Data

All return series sourced from Yahoo Finance (24 assets, 2000-2026).
Pinned TSFM checkpoints listed in Table 2 of the paper.

## Requirements

Python 3.10+, torch, chronos, timesfm, uni2ts,
gluonts, arch, statsmodels, scipy, pandas, numpy, lightgbm

## Citation

```bibtex
@article{pele2026conformal,
  title={Distribution-Free Recalibration of Tail Quantile
         Forecasts under Temporal Dependence},
  author={Pele, Daniel Traian and Bolovăneanu, Vlad and Ginavar, Andrei Theodor and Lessmann, Stefan
          and H{\"a}rdle, Wolfgang Karl},
  journal={Working Paper},
  year={2026}
}
```

## License

MIT
