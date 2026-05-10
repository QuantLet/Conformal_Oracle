# Conformal_Oracle

**Recalibrating Tail Event Forecasts under Temporal Dependence**

Pele, D.T., Bolovăneanu, V., Ginavar, A.T., Lessmann, S., Härdle, W.K. (2026)

This repository contains the reproducible code units (Quantlets) for all tables
and figures in the paper, plus a standalone Python package (`conformal-oracle`)
implementing the conformal recalibration audit framework for user-supplied
probabilistic forecasters.

The [`python/`](python/) directory contains the package with install instructions
and a quickstart guide. The Quantlets below are self-contained directories with
a `Metainfo.txt` (QuantNet standard), a Python script (`.py`), a Jupyter
notebook (`.ipynb`), and one or more outputs (`.tex`, `.csv`, `.pdf`, `.png`).

## Quantlets

| Quantlet | Output | Description |
|----------|--------|-------------|
| CO_data_returns | data | Download 24 asset return series from Yahoo Finance |
| CO_asset_overview | Table 1 | Asset universe (24 assets, 5 classes) |
| CO_model_overview | Table 2 | Model overview (6 TSFMs + 4 benchmarks) |
| CO_cross_sectional | Table 3 | Cross-sectional correlations of q_V |
| CO_full_evaluation | Table 4 | Master results: violation rates, Kupiec, Basel, QS |
| CO_qV_ranking | Figure | q_V ranking bar chart |
| CO_multi_quantile_panel | Tables 5-7 | Multi-quantile evaluation, panel-pooled, panel by class |
| CO_quantile_scores | Table 8 | Diebold-Mariano p-values (HLN correction) |
| CO_garch_conformal | Table 9 | Rolling vs static conformal correction |
| CO_bound_validation | Table 11 | Coverage bound evaluation (Theorem 3.5) |
| CO_gbm_qr | prerequisite | GBM-QR baseline (LightGBM) |
| CO_gamlss | prerequisite | GAMLSS-SST baseline |
| CO_baselines_evt_fhs | prerequisite | EVT-POT + FHS baselines |
| CO_baseline_comparison | Table 12 | Composite recalibration method comparison |
| CO_baseline_comparison_tuned | Table E.1 | Tuned GBM-QR comparison |
| CO_fz_scores | Table 13 | Fissler-Ziegel joint VaR-ES scores |
| CFP_ES_Correction_Z2 | Table C.14 | ES correction + Acerbi-Szekely Z2 backtest |
| CO_robustness | Tables D.15-D.18 | WCP, calibration fraction, MC robustness |
| CO_robustness_inner7 | Table E.2 | Extended tail closure (inner 7 assets) |
| CO_regime_sensitivity | Table E.5 | Regime-conditional sensitivity analysis |
| CO_panel_wildcluster | Table E.3 | Wild cluster bootstrap panel inference |
| CO_diagnostic_regression | Table E.4 | Diagnostic regression analysis |
| CO_rolling_qV | Figure 1 | Rolling q_V on S&P 500 + realised volatility |
| CO_heatmap | Figure 2 | Basel Traffic Light heatmap (10 x 24) |
| CFP_Calibration_Efficiency_Frontier | Figure 3 | Calibration-efficiency frontier |
| CO_violation_rates | Figure 4 | Raw vs corrected violation rates |
| CO_simulation_study | Table 10 + Figure 5 | Monte Carlo q_V distribution (5 DGPs, 500 reps) |
| CO_covid_response_lag | Figure 6 | COVID-19 response lag |
| CO_drift_diagnostic | Figure 7 | Distributional drift diagnostic (TV distance) |
| CFP_Capital_Charge | Figure 8 | Cumulative capital charge comparison |

## Data

All return series sourced from Yahoo Finance (24 assets, 2000-2026).
TSFM quantile forecasts produced by the upstream pipeline in the
[main repository](https://github.com/danpele/Conformal_Oracle).
Pinned TSFM checkpoints listed in Table 2 of the paper.

## Running individual Quantlets

Each Quantlet can be run standalone from the repository root:

```bash
# Run the Python script directly
python CO_full_evaluation/run_master_table.py

# Or open the Jupyter notebook for interactive exploration
jupyter notebook CO_full_evaluation/CO_full_evaluation.ipynb
```

## Quantlet structure

Each directory follows the QuantNet standard:

```
CO_full_evaluation/
├── Metainfo.txt                    # QuantNet metadata (name, keywords, description)
├── run_master_table.py             # Standalone Python script
├── CO_full_evaluation.ipynb        # Jupyter notebook (same logic, interactive)
└── tab_master_results.tex          # Output (table or figure)
```

## Requirements

Python 3.10+, torch, chronos, timesfm, uni2ts,
gluonts, arch, statsmodels, scipy, pandas, numpy, lightgbm

## Citation

```bibtex
@article{pele2026conformal,
  title={Recalibrating Tail Event Forecasts
         under Temporal Dependence},
  author={Pele, Daniel Traian and Bolov{\u{a}}neanu, Vlad and Ginavar, Andrei Theodor
          and Lessmann, Stefan and H{\"a}rdle, Wolfgang Karl},
  journal={International Journal of Forecasting},
  year={2026}
}
```

## License

MIT
