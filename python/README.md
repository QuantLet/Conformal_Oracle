# conformal-oracle

[![PyPI version](https://img.shields.io/pypi/v/conformal-oracle)](https://pypi.org/project/conformal-oracle/)
[![Python](https://img.shields.io/pypi/pyversions/conformal-oracle)](https://pypi.org/project/conformal-oracle/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/conformal-oracle)](https://pypi.org/project/conformal-oracle/)

Conformal recalibration and backtesting for extreme financial quantiles.

Given any return series and either a forecaster object or a
pre-computed quantile path, `conformal-oracle` computes a
one-parameter conformal correction and reports coverage, Quantile Score and
correction-magnitude diagnostics. Version 0.4.0 adds an explicit separated
single-split protocol and a calibration-only selective-deployment policy.
The legacy regime labels are descriptive, not a validation of the forecaster.

The core install is **dependency-agnostic**: it needs only NumPy,
pandas, SciPy, statsmodels, and matplotlib. No forecaster library
is required unless you use the built-in benchmark wrappers.

Companion software for:

> Pele, D.T., Bolovaneanu, V., Ginavar, A.T., Lessmann, S., Hardle, W.K.
> "When Does Recalibration Improve Value-at-Risk Forecasts? Estimation Cost,
> Dependence and How Much to Correct" (2026).

The R7 and R8 APIs of version 0.4.0 accompany the earlier manuscript, "Conformal
Recalibration of Extreme Tail Quantiles under Temporal Dependence", replication
tag `R9-2026-09-17-v2`, and are unchanged in this release.

## Scope and interpretation

The companion study examines validity under temporal dependence, the
comparison of scalar and richer recalibration under tail sparsity, and the
decision to recalibrate before deployment. `audit(mode="static")` remains the
contiguous operational estimator; `audit(mode="rolling")` remains a rolling
heuristic. Separate public APIs implement the separated estimator and the
pre-deployment indication rule without changing either existing audit mode.

The manuscript's separated coverage theorem concerns the separated single-split
estimator under its maintained assumptions, not the contiguous static or rolling
API. The gap experiment
uses a proxy-based separation rule, not a certified estimate of the mixing rate.
Marginal coverage after correction does not validate the full predictive
distribution, and a shift can worsen Quantile Score (lower is better).

[Gneiting and Resin (2023)](https://doi.org/10.1214/23-EJS2180) provide the
calibration and score-decomposition framework, including constant-translation
recalibration. We use it to interpret what this restricted correction can
remove, not to introduce a new miscalibration measure. The displacement and
its associated score reduction are different quantities; the finite-sample
conformal order statistic need not exactly minimise the calibration score.

## Version 0.5.0

`ConformalShift` takes an `intensity` argument; the deployed correction is
`intensity * c_hat`. The default stays `1.0`, so nothing moves for existing
callers. The recommended static setting is `intensity=0.5` on at least 1000
calibration pairs. `fit` now warns when the conformal rank reaches the
calibration sample size. See the correction-intensity section below. The R7 and
R8 APIs are unchanged.

## Version 0.4.0

This release adds the R8 analysis tools listed in the next section and the
two R7 workflows below (separated estimator and pre-deployment indication). It also
retains the earlier bootstrap fix: each
`bootstrap_qv_ci` replicate uses the same conformal order statistic as the
point estimate. The published 0.3.2 release still uses the plain empirical
quantile in those replicates. Existing static/rolling algorithm outputs and
API signatures are preserved. Check the
[PyPI release history](https://pypi.org/project/conformal-oracle/#history)
for publication status; a local build does not establish that a release
has been uploaded.

All three corrections use `ceil((n+1)(1-alpha))`, not an interpolated empirical
quantile. When that rank exceeds the calibration-sample size, the existing
implementation returns the sample maximum as a finite proxy; this case does
not retain the usual finite-sample conformal coverage guarantee. See the
[methodology](https://github.com/QuantLet/Conformal_Oracle/blob/main/python/docs/methodology.md)
and [changelog](https://github.com/QuantLet/Conformal_Oracle/blob/main/python/CHANGELOG.md).

## R8: estimation cost, correction form and selection

Four additions implement the R8 analysis of when a correction pays.

```python
import numpy as np
from conformal_oracle import (
    fit_one_coefficient_corrections, blocked_cv_optimism, block_bootstrap_optimism,
    first_order_shrinkage, paired_calendar_bootstrap, past_loss_selection,
)

# One-coefficient corrections of a lower quantile q (calibration block).
# sigma is a past-only volatility proxy, e.g. the SD of the previous 20 returns.
fit = fit_one_coefficient_corrections(q_cal, r_cal, sigma_cal, alpha=0.01)
corrected = fit.apply(q_test, sigma_test)   # 'Shift-CP', 'Shift-ERM', 'Vol-CP', 'Vol-ERM'

# Optimism of the fitted conformal shift: how much in-sample loss flatters it.
scores = q_cal - r_cal
cv = blocked_cv_optimism(scores, alpha=0.01)          # factor 2(K-1)/(2K-1) = 8/9
boot = block_bootstrap_optimism(scores, alpha=0.01)   # circular blocks, ceil(n^(1/3))
print(cv.penalty, boot.penalty, cv.estimated_out_of_sample_loss_change)
first_order_shrinkage(cv).validated                  # False: diagnostic only

# Paired loss comparison across pairs with a common-calendar block bootstrap.
bands = paired_calendar_bootstrap(pair_losses, [("Shift-CP", "Raw")], scale=1e4)

# Selection on an inner validation block: past-loss minimum and cautious gate.
choice = past_loss_selection({"Raw": l_raw, "Shift-CP": l_cp, "Vol-ERM": l_vol}, key="asset")
```

`shift_erm` and `vol_erm` minimise calibration pinball loss over constant and
volatility-proportional shifts; `vol_cp` is the conformal order statistic of
the standardised scores. The optimism estimators recover the manuscript's
leading penalty within its accuracy criterion on the synthetic study; the
shrinkage factor built from them failed the study's value criterion and is
exposed only with `validated=False`. `paired_calendar_bootstrap` reproduces
the manuscript's static-minus-raw bands from the stored 240-pair losses, and
`past_loss_selection` implements the two selection policies of Section 7.
Tests in `tests/test_r8_extensions.py` check these against the research
archive when it is present.

## Install

```bash
pip install conformal-oracle                 # core (no arch dep)
pip install conformal-oracle[benchmarks]     # + GJR-GARCH, GARCH-Normal
pip install conformal-oracle[chronos]        # + Chronos TSFM
pip install conformal-oracle[all]            # everything
```

For development:

```bash
git clone https://github.com/QuantLet/Conformal_Oracle.git
cd Conformal_Oracle/python
pip install -e ".[dev,benchmarks]"
```

## Quickstart -- agnostic audit (no forecaster dependency)

```python
import pandas as pd
from conformal_oracle import audit

returns = pd.read_csv("returns.csv", index_col=0, parse_dates=True).squeeze()
# q_lo: your model's predicted 1% quantile, same index as returns
q_lo = pd.read_csv("my_var_forecast.csv", index_col=0, parse_dates=True).squeeze()

result = audit(returns, forecast=q_lo, alpha=0.01, mode="static")
print(result.summary())

# Rolling mode: re-estimates the conformal correction from a
# trailing 250-day window (an operational heuristic under dependence)
result_roll = audit(returns, forecast=q_lo, alpha=0.01, mode="rolling")
print(result_roll.summary())
```

No `arch`, no `torch`, no heavyweight dependency -- just your
quantile series.

## Correction intensity

`ConformalShift(intensity=...)` applies a fraction of the fitted shift; the
deployed correction is `intensity * c_hat`. The default, `1.0`, is the whole
shift, so results from 0.4.0 and earlier are unchanged unless the argument is
passed.

```python
from conformal_oracle import audit
from conformal_oracle.contrib.benchmarks import GJRGARCHForecaster
from conformal_oracle.recalibration import ConformalShift

# Recommended static setting: the average of the raw and the fully corrected
# threshold, fitted on at least 1000 calibration pairs.
result = audit(
    returns, GJRGARCHForecaster(), alpha=0.01, mode="static",
    recalibration=ConformalShift(intensity=0.5),
)
```

`recalibration=` belongs to the forecaster path. On the agnostic `forecast=`
path, apply the intensity to the quantile series yourself:

```python
shift = ConformalShift(intensity=0.5)
shift.fit(-q_cal.to_numpy(), r_cal.to_numpy(), alpha=0.01)
q_corrected = q_lo - shift.shift          # q_lo is the lower-tail quantile
```

**Why 0.5.** The whole shift lowers expected loss only when the correction the
forecaster needs is larger than the standard error of the fitted quantile. At
intensity 0.5 the leading coefficient of the local-bias corollary becomes
`(f/8)(sigma^2 - 3 delta^2)`: the correction pays over a region three times
wider in squared bias, costs a quarter as much when no correction was needed,
and is the optimal intensity at that boundary. On the panels of the current
manuscript, intensity 0.5 lowered quantile loss against the whole shift at
every calibration length in every universe, and against the raw forecast once
the window held 1000 pairs. The evidence is retrospective.

**The intensity is not estimated.** `diagnostics.optimism.first_order_shrinkage`
estimates the same quantity and keeps `validated=False`: on those panels the
estimated intensity loses to the fixed 0.5 in every supported comparison between
them, and at short windows it degenerates to the whole shift.

**Short windows.** When the conformal rank reaches the calibration sample size,
at `alpha = 0.01` any window of 198 pairs or fewer, the fitted shift is the
largest calibration score and `ConformalShift.fit` raises a `UserWarning`.

## R7: separated single-split estimator

```python
from conformal_oracle import SeparatedSplitConformalVaR, proxy_separation_gap

n_cal = int(0.70 * len(returns))
gap_info = proxy_separation_gap(
    (q_lo.iloc[:n_cal] - returns.iloc[:n_cal]).to_numpy(),
    context_length=512,
)
separated = SeparatedSplitConformalVaR(
    alpha=0.01, calibration_fraction=0.70,
    gap=gap_info.gap, minimum_evaluation_size=100,
).split(returns, q_lo)
print(separated.q_v_stat, separated.evaluation_indices[0])
print(gap_info.certified)  # False: this is an operational proxy
```

You can instead supply an explicit integer gap directly. The shift uses only
the original calibration block; the gap removes evaluation observations, not
calibration observations. Indices are zero-based: evaluation begins at
`n_cal + gap`, one position after the last calibration index plus the gap.
Returned forecasts are **lower return quantiles**, not positive-loss VaR.
An empty or undersized evaluation window raises `ValueError`.

The lag-one absolute autocorrelation is not a validated beta-mixing-rate
estimator. Neither a user-chosen gap nor `certified=False` proxy metadata
establishes the maintained assumptions of Theorem 4.5.

## R7: decide before deployment

```python
from conformal_oracle import recalibration_indication, selectively_recalibrate

cal_returns, cal_quantiles = returns.iloc[:n_cal], q_lo.iloc[:n_cal]
decision = recalibration_indication(
    calibration_returns=cal_returns,
    calibration_quantiles=cal_quantiles,
    alpha=0.01, kupiec_level=0.05,
)
selected = selectively_recalibrate(
    q_lo.iloc[n_cal:],
    calibration_returns=cal_returns,
    calibration_quantiles=cal_quantiles,
    decision=decision, method="static",
)
print(decision.apply, decision.reasons)
# selected.raw_quantiles and selected.final_quantiles have the same horizon.
```

Apply if the calibration Basel zone is not Green **or** the calibration
Kupiec p-value is below `kupiec_level`; otherwise preserve raw forecasts.

This rule and the correction intensity answer different questions and compose.
The R7 rule decides **whether** to correct, from calibration coverage alone, and
accompanies the earlier manuscript. The intensity decides **how much** to apply
once the answer is yes, and the current manuscript recommends `intensity=0.5`
with at least 1000 calibration pairs. `selectively_recalibrate` applies the
whole shift; to deploy the recommended setting after a positive decision, build
the correction with `ConformalShift(intensity=0.5)`.
The decision function accepts no evaluation outcomes. Callers must supply a
genuine calibration block and already causal forecasts; array values alone
cannot establish their provenance. The helper verifies the decision's
calibration fingerprint before applying it.

For `method="rolling", window=250`, additionally provide
`evaluation_returns=returns.iloc[n_cal:]` for chronological replay. Each
time-t correction uses only outcomes before t; the initial decision remains
fixed. The skip path needs no evaluation outcomes. An ex-post score comparison
is separate from the policy and cannot be used to choose the initial decision.

See the [API reference](https://github.com/QuantLet/Conformal_Oracle/blob/main/python/docs/api.md)
for the precise R7 Basel convention and a read-only artifact reproduction command.

## Quickstart -- with a forecaster object

```python
from conformal_oracle import audit
from conformal_oracle.contrib.benchmarks import GJRGARCHForecaster

result = audit(returns, GJRGARCHForecaster(), alpha=0.01, mode="rolling")
print(result.summary())
```

## Regime classification

These are legacy correction-magnitude labels. Neither label establishes
forecasting quality, conditional calibration, or whether recalibration should
be deployed. `classify_regime()` is not the paper's indication rule.

```python
from conformal_oracle import classify_regime

verdict = classify_regime(returns, forecast=q_lo, mode="rolling")
print(verdict.regime)       # "signal-preserving" or "replacement"
print(verdict.R)            # replacement ratio
print(verdict.basel_zone)   # "green", "yellow", or "red"
```

## Compare multiple forecasters

```python
from conformal_oracle import compare_forecasters

comp = compare_forecasters(
    returns,
    {"model_A": q_lo_A, "model_B": q_lo_B},
    mode="rolling",
)
print(comp.comparison_table())
print(comp.dm_matrix())
```

## Custom forecaster

Any object implementing `fit(returns)` and `forecast(returns, t)` works:

```python
from conformal_oracle._types import SampleDistribution

class MyForecaster:
    def fit(self, returns): pass
    def forecast(self, returns, t):
        hist = returns.iloc[max(0, t-250):t]
        return SampleDistribution(samples=hist.values)

result = audit(returns, MyForecaster(), alpha=0.01)
```

## Worked examples

- [Separated and selective R7 workflows](examples/05_r7_workflows.py) --
  deterministic API smoke example, no fitted model or external dataset.
- [Optional R7 artifact integration](scripts/reproduce_r7_deployment.py) --
  recalculates decisions from stored calibration evidence and reports ex-post
  outcomes separately; requires the complete replication artifacts.
- [Quickstart (S&P 500)](examples/notebooks/quickstart_sp500.ipynb) --
  Static and rolling conformal audits with GJR-GARCH and Lag-Llama.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuantLet/Conformal_Oracle/blob/main/python/examples/notebooks/quickstart_sp500.ipynb)
- [Legacy Table 4 replication](examples/notebooks/reproduce_table4_full.ipynb) --
  9 forecasters x 24 assets under an earlier protocol, with checkpointing.
  This notebook does not reproduce the current R7 tables.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuantLet/Conformal_Oracle/blob/main/python/examples/notebooks/reproduce_table4_full.ipynb)

## Documentation

- [API Reference](https://github.com/QuantLet/Conformal_Oracle/blob/main/python/docs/api.md)
- [Methodology](https://github.com/QuantLet/Conformal_Oracle/blob/main/python/docs/methodology.md)
- [Conventions](https://github.com/QuantLet/Conformal_Oracle/blob/main/python/docs/conventions.md) (return units, VaR sign, alpha)
- [Migration Guide (v0.3)](https://github.com/QuantLet/Conformal_Oracle/blob/main/python/docs/migration_v0.3.md)

## Requirements

Python 3.10+, numpy, pandas, scipy, statsmodels, matplotlib.

GARCH benchmarks require `arch>=6.0` (install with `[benchmarks]`).
TSFM wrappers require PyTorch and model-specific packages (see extras).

## License

MIT
