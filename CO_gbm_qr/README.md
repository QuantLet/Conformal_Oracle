<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: CO_gbm_qr

Published in: Distribution-Free Recalibration of Tail Quantile Forecasts under Temporal Dependence

Description: Gradient-boosted quantile regression baseline (LightGBM) for 1% VaR recalibration across 216 model-asset pairs; empirical validation of the tail sparsity principle (Proposition 3.5).

Keywords: quantile regression, gradient boosting, LightGBM, conformal prediction, Value-at-Risk, tail sparsity, post-processing

Author: Daniel Traian Pele

Datafile: baseline_gbm_qr.py

Output: gbm_qr_results.csv, gbm_qr_baseline.tex
```

## Summary

Tests whether nonlinear feature interactions improve upon the scalar conformal
correction in the sparse tail regime. Features: base-model lower quantile,
lagged 5-day and 20-day realised volatilities. Training uses LightGBM with
early stopping on a 20% validation fold.

## Headline result

Across 216 model-asset pairs, GBM-QR achieves:

- pi_hat: 0.018 (target 0.01)
- Kupiec rejections: 123/216 (57%)
- Quantile Score: 5.47e-4
- Mean VaR width: 0.033
- Basel Green: 105/216 (48.6%)

Far below the 86.6% Green rate of the static conformal shift, confirming
Proposition 3.5: multi-parameter recalibration overfits the sparse tail.
