<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: CO_gamlss

Published in: Recalibrating Tail Risk Forecasts under Temporal Dependence

Description: GAMLSS baseline with Fernandez-Steel skewed-t innovations for 1% VaR recalibration. Location linear in base-model quantile forecast, log-scale linear in lagged realised volatility, shape and skewness intercept-only. Produces the GAMLSS-SST row of Table 12.

Keywords: GAMLSS, skewed-t, conditional density, conformal prediction, Value-at-Risk, tail sparsity, location-scale model

See also: CO_baseline_comparison, CO_gbm_qr

Author: Daniel Traian Pele

Submitted: 2026-04-25

Datafile: cfp_ijf_data/{model}/*.parquet, cfp_ijf_data/benchmarks/*.parquet, cfp_ijf_data/returns/*.csv

Output: gamlss_results.csv

```
