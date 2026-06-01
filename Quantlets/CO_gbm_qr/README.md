<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: CO_gbm_qr

Published in: Recalibrating Tail Risk Forecasts under Temporal Dependence

Description: Gradient-boosted quantile regression baseline (LightGBM) for 1% VaR recalibration. Features: base-model lower quantile, lagged 5-day and 20-day realised volatility. Produces the GBM-QR row of Table 12.

Keywords: quantile regression, gradient boosting, LightGBM, conformal prediction, Value-at-Risk, tail sparsity, bias-variance

See also: CO_baseline_comparison, CO_gamlss

Author: Daniel Traian Pele

Submitted: 2026-04-25

Datafile: cfp_ijf_data/{model}/*.parquet, cfp_ijf_data/benchmarks/*.parquet, cfp_ijf_data/returns/*.csv

Output: gbm_qr_results.csv, gbm_qr_detail.csv

```
