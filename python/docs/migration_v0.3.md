# Migration Guide: v0.1.x to v0.2.0

## Summary of changes

v0.2.0 makes the core `conformal-oracle` install **dependency-agnostic**.
The `arch` library is no longer a mandatory dependency. GARCH-based
benchmark forecasters have moved to `conformal_oracle.contrib.benchmarks`
and require the `[benchmarks]` extra.

A new `forecast=` parameter on `audit()` lets you pass a pre-computed
quantile path instead of a forecaster object, eliminating all forecaster
dependencies from the audit pipeline.

## Breaking changes

### `arch` is no longer a core dependency

If you use `GJRGARCHForecaster` or `GARCHNormalForecaster`, install
with the benchmarks extra:

```bash
pip install conformal-oracle[benchmarks]
```

### Forecaster imports have moved

| Old path | New path |
|----------|----------|
| `conformal_oracle.forecasters.GJRGARCHForecaster` | `conformal_oracle.contrib.benchmarks.GJRGARCHForecaster` |
| `conformal_oracle.forecasters.GARCHNormalForecaster` | `conformal_oracle.contrib.benchmarks.GARCHNormalForecaster` |
| `conformal_oracle.forecasters.HistoricalSimulationForecaster` | `conformal_oracle.contrib.benchmarks.HistoricalSimulationForecaster` |
| `conformal_oracle.forecasters.tsfm.ChronosForecaster` | `conformal_oracle.contrib.tsfm.ChronosForecaster` |
| `conformal_oracle.forecasters.tsfm.LagLlamaForecaster` | `conformal_oracle.contrib.tsfm.LagLlamaForecaster` |
| `conformal_oracle.forecasters.tsfm.TimesFM25Forecaster` | `conformal_oracle.contrib.tsfm.TimesFM25Forecaster` |
| `conformal_oracle.forecasters.tsfm.MoiraiForecaster` | `conformal_oracle.contrib.tsfm.MoiraiForecaster` |

The old import paths still work but emit `DeprecationWarning` and
will be removed in v0.3.0.

### Deprecated functions

| Deprecated | Replacement |
|------------|-------------|
| `audit_static(returns, fc, ...)` | `audit(returns, fc, mode="static", ...)` |
| `audit_rolling(returns, fc, ...)` | `audit(returns, fc, mode="rolling", ...)` |
| `audit_with_benchmarks(returns, fc, ...)` | `compare_forecasters(returns, {...}, ...)` |

These functions still work but emit `DeprecationWarning`.

## New features

### `forecast=` parameter (agnostic audit)

```python
from conformal_oracle import audit

result = audit(returns, forecast=q_lo_series, alpha=0.01, mode="static")
```

When using `forecast=`, ES-related diagnostics (Z2 statistic, FZ score)
are set to `NaN` because the expected shortfall cannot be computed from
a single quantile path.

### `classify_regime()`

```python
from conformal_oracle import classify_regime

verdict = classify_regime(returns, forecast=q_lo, mode="rolling")
print(verdict.regime, verdict.R, verdict.basel_zone)
```

### `compare_forecasters()`

```python
from conformal_oracle import compare_forecasters

comp = compare_forecasters(
    returns,
    {"model_a": q_lo_a, "model_b": q_lo_b},
    mode="rolling",
)
print(comp.comparison_table())
print(comp.dm_matrix())
```

## Upgrade checklist

1. Change `pip install conformal-oracle` to
   `pip install conformal-oracle[benchmarks]` if you use GARCH forecasters.
2. Update imports from `conformal_oracle.forecasters` to
   `conformal_oracle.contrib.benchmarks` (or `.contrib.tsfm`).
3. Replace `audit_static()` / `audit_rolling()` calls with
   `audit(mode="static")` / `audit(mode="rolling")`.
4. Consider switching to `forecast=` if you have pre-computed quantiles.
