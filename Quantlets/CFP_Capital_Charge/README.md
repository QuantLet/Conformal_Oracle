<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: CFP_Capital_Charge

Published in: Recalibrating Tail Risk Forecasts under Temporal Dependence

Description: Cumulative daily capital charges under Basel zone multipliers for raw versus conformally corrected VaR forecasts on S&P 500 (Figure 8). Compares Lag-Llama (Yellow zone, k=3.65) against corrected Lag-Llama and GJR-GARCH (Green zone, k=3.00).

Keywords: capital charge, Basel traffic light, VaR, conformal prediction, zone reclassification, regulatory capital

See also: CO_full_evaluation, CFP_ES_Correction_Z2

Author: Daniel Traian Pele

Submitted: 2026-04-25

Datafile: cfp_ijf_data/returns/SP500.csv, cfp_ijf_data/lagllama/SP500.parquet, cfp_ijf_data/benchmarks/SP500_gjr_garch.parquet

Output: capital_charge_cumulative.pdf, capital_charge_cumulative.png, capital_charge_results.csv

```
<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/Conformal_Oracle/main/Quantlets/CFP_Capital_Charge/capital_charge_cumulative.png" alt="Image" />
</div>

