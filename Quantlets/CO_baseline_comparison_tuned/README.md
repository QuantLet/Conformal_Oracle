<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: CO_baseline_comparison_tuned

Published in: Recalibrating Tail Risk Forecasts under Temporal Dependence

Description: Tuned GBM-QR ablation (TODO E1). Runs 8-config hyperparameter grid for LightGBM quantile regression on S&P 500, testing n_estimators in {100,500}, max_depth in {3,5}, learning_rate in {0.01,0.05}. Confirms bias-variance trade-off at the 1% tail (Remark 3.2).

Keywords: GBM-QR, quantile regression, hyperparameter tuning, LightGBM, bias-variance, VaR

See also: CO_gbm_qr, CO_baseline_comparison

Author: Daniel Traian Pele

Submitted: 2026-05-08

Datafiles: cfp_ijf_data/returns/SP500.csv, cfp_ijf_data/{model_subdir}/SP500.parquet

Output: tab_baselines_tuned_row.tex, tuned_gbm_qr_grid.csv, tuned_gbm_qr_summary.csv

```
