# conformal-oracle

[![PyPI version](https://img.shields.io/pypi/v/conformal-oracle)](https://pypi.org/project/conformal-oracle/)
[![Python](https://img.shields.io/pypi/pyversions/conformal-oracle)](https://pypi.org/project/conformal-oracle/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/conformal-oracle)](https://pypi.org/project/conformal-oracle/)

Conformal recalibration audit for tail quantile forecasters.

Given any return series and either a forecaster object or a
pre-computed quantile path, `conformal-oracle` computes a
one-parameter conformal correction (static or rolling), classifies
the forecast as signal-preserving or replacement, and reports a
full backtest panel.

The core install is **dependency-agnostic**: it needs only NumPy,
pandas, SciPy, statsmodels, and matplotlib. No forecaster library
is required unless you use the built-in benchmark wrappers.

Implements the methodology from:

> Pele, D.T., Bolovaneanu, V., Ginavar, A.T., Lessmann, S., Hardle, W.K.
> "Recalibrating Tail Event Forecasts under Temporal Dependence" (2026).

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

# Rolling mode: re-estimates the conformal correction in an
# expanding window (more realistic for live deployment)
result_roll = audit(returns, forecast=q_lo, alpha=0.01, mode="rolling")
print(result_roll.summary())
```

No `arch`, no `torch`, no heavyweight dependency -- just your
quantile series.

## Quickstart -- with a forecaster object

```python
from conformal_oracle import audit
from conformal_oracle.contrib.benchmarks import GJRGARCHForecaster

result = audit(returns, GJRGARCHForecaster(), alpha=0.01, mode="rolling")
print(result.summary())
```

## Regime classification

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

- [Quickstart (S&P 500)](examples/notebooks/quickstart_sp500.ipynb) --
  Static and rolling conformal audits with GJR-GARCH and Lag-Llama.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danpele/Conformal_Oracle/blob/main/python/examples/notebooks/quickstart_sp500.ipynb)
- [Reproduce Table 4 (Full replication)](examples/notebooks/reproduce_table4_full.ipynb) --
  9 forecasters x 24 assets, full master evaluation table with checkpointing.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danpele/Conformal_Oracle/blob/main/python/examples/notebooks/reproduce_table4_full.ipynb)

## Documentation

- [API Reference](docs/api.md)
- [Methodology](docs/methodology.md)
- [Conventions](docs/conventions.md) (return units, VaR sign, alpha)
- [Migration Guide (v0.3)](docs/migration_v0.3.md)

## Requirements

Python 3.10+, numpy, pandas, scipy, statsmodels, matplotlib.

GARCH benchmarks require `arch>=6.0` (install with `[benchmarks]`).
TSFM wrappers require PyTorch and model-specific packages (see extras).

## License

MIT
