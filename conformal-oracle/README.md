# conformal-oracle

Conformal recalibration audit for tail quantile forecasters.

Given any black-box probabilistic forecaster and a return series,
`conformal-oracle` computes a one-parameter conformal correction
(static or rolling), classifies the forecaster as signal-preserving
or replacement, and reports a full backtest panel.

Implements the methodology from:

> Pele, D.T., Bolovăneanu, V., Ginavar, A.T., Lessmann, S., Härdle, W.K.
> "Recalibrating Tail Event Forecasts under Temporal Dependence" (2026).

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart — static audit

```python
import pandas as pd
from conformal_oracle import audit_static
from conformal_oracle.forecasters import GJRGARCHForecaster

returns = pd.read_csv("returns.csv", index_col=0, parse_dates=True).squeeze()
result = audit_static(returns, GJRGARCHForecaster(), alpha=0.01)
print(result.summary())
```

## Quickstart — rolling audit

```python
from conformal_oracle import audit_rolling
from conformal_oracle.forecasters import GJRGARCHForecaster

result = audit_rolling(returns, GJRGARCHForecaster(), alpha=0.01, window=250)
print(result.summary())
```

## Quickstart — benchmark comparison

```python
from conformal_oracle import audit_with_benchmarks

comp = audit_with_benchmarks(returns, my_forecaster, benchmarks=["gjr_garch", "hist_sim"])
print(comp.comparison_table())
print(comp.diebold_mariano(baseline="gjr_garch"))
print(comp.comparison_table_latex())
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

result = audit_static(returns, MyForecaster(), alpha=0.01)
```

See `examples/04_custom_forecaster.py` for a full example.

## Documentation

- [API Reference](docs/api.md)
- [Methodology](docs/methodology.md)
- [Conventions](docs/conventions.md) (return units, VaR sign, alpha)

## Requirements

Python 3.10+, numpy, pandas, scipy, statsmodels, arch, matplotlib.

## License

MIT
