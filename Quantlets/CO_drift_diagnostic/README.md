<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: CO_drift_diagnostic

Published in: Recalibrating Tail Risk Forecasts under Temporal Dependence

Description: Distributional drift diagnostic via empirical total variation distance on rolling conformal scores for Lag-Llama on S&P 500 with w=250 (Figure 7). Splits each window into two halves, estimates TV distance via histogram binning, and applies the KS two-sample test as complementary significance assessment.

Keywords: conformal prediction, VaR, distributional drift, total variation, rolling window, non-exchangeability, KS test, beta-mixing

See also: CO_bound_validation, CO_garch_conformal

Author: Daniel Traian Pele

Submitted: 2026-04-25

Datafile: cfp_ijf_data/lagllama/SP500.parquet, cfp_ijf_data/returns/SP500.csv

Output: fig_drift_diagnostic.pdf, fig_drift_diagnostic.png

```
<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/Conformal_Oracle/main/Quantlets/CO_drift_diagnostic/fig_drift_diagnostic.png" alt="Image" />
</div>

