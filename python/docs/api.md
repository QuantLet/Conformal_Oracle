# API Reference

## Top-level functions

### `audit(returns, forecaster, alpha=0.01, mode="static", **kwargs)`

Convenience dispatcher. Routes to `audit_static()` or `audit_rolling()`.

```python
from conformal_oracle import audit
result = audit(returns, forecaster, mode="static")
```

### `audit_static(returns, forecaster, alpha=0.01, calibration_split=0.70, warmup=50, seed=2026)`

Static conformal audit. Returns `StaticAuditResult`.

```python
from conformal_oracle import audit_static
from conformal_oracle.forecasters import GJRGARCHForecaster

result = audit_static(returns, GJRGARCHForecaster(), alpha=0.01)
print(result.summary())
```

### `audit_rolling(returns, forecaster, alpha=0.01, window=250, warmup=250, persistence=20, seed=2026)`

Rolling conformal audit. Returns `RollingAuditResult`.

```python
from conformal_oracle import audit_rolling
result = audit_rolling(returns, GJRGARCHForecaster(), alpha=0.01)
```

### `audit_with_benchmarks(returns, forecaster, benchmarks=["gjr_garch", "hist_sim"], alpha=0.01, mode="rolling", seed=2026, **kwargs)`

Audit user's forecaster alongside reference benchmarks.
Returns `BenchmarkComparison`.

```python
from conformal_oracle import audit_with_benchmarks
comp = audit_with_benchmarks(returns, my_forecaster, mode="static")
print(comp.comparison_table())
print(comp.diebold_mariano(baseline="gjr_garch"))
```

## Distribution types

### `SampleDistribution(samples: np.ndarray)`

Monte Carlo samples. Methods: `quantile(alpha)`, `expected_shortfall(alpha)`, `cdf(x)`.

### `QuantileGridDistribution(levels: np.ndarray, quantiles: np.ndarray)`

Finite quantile grid with parametric tail completion.
Methods: `quantile(alpha, completion="student_t")`, `expected_shortfall(alpha, completion="student_t")`, `cdf(x, completion="student_t")`.

### `ParametricDistribution(location, scale, family, df=None, skew=None)`

Closed-form parametric family ("normal", "student_t", "skewed_t").
Methods: `quantile(alpha)`, `expected_shortfall(alpha)`, `cdf(x)`.

## Forecaster protocol

```python
class Forecaster(Protocol):
    def fit(self, returns: pd.Series) -> None: ...
    def forecast(self, returns: pd.Series, t: int) -> PredictiveDistribution: ...
```

The forecaster sees only `returns.iloc[:t]` (history up to t-1).

## Built-in forecasters

- `GJRGARCHForecaster(window=250, distribution="skewt")`
- `GARCHNormalForecaster(window=250)`
- `HistoricalSimulationForecaster(window=250)`

## Diagnostics

- `kupiec_pof_pvalue(violations, alpha)` → float
- `christoffersen_pvalue(violations, alpha)` → dict with "unconditional", "independence", "joint"
- `basel_traffic_light(violations, window=250)` → "green" | "yellow" | "red"
- `z2_statistic(violations, realised, es_forecasts, alpha, stabilised=True)` → float
- `quantile_score(realised, forecasts, alpha)` → float
- `fissler_ziegel_fz0(realised, var_forecasts, es_forecasts, alpha)` → float
- `diebold_mariano_pvalue(losses_a, losses_b, horizon=1, hln_correction=True)` → float

## Reporting

- `audit_result_to_latex_row(result, name)` → str
- `comparison_to_latex(results_dict, caption, label)` → str (full table)
- `plot_rolling_diagnostic(result, figsize, save_path)` → Figure
