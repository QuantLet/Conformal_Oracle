# API Reference

## Top-level functions

### `audit(returns, forecaster=None, *, forecast=None, alpha=0.01, mode="static", recalibration=None, **kwargs)`

Supply exactly one of `forecaster` or the pre-computed lower-quantile series
`forecast`. The dispatcher supports only `"static"` (contiguous split) and
`"rolling"`. For the separated construction and the pre-deployment policy,
use the new APIs below; neither is silently enabled by an existing audit mode.
A `recalibration` baseline is supported only
with a forecaster object, not with the `forecast=` path.

```python
from conformal_oracle import audit
result = audit(returns, forecast=q_lo, mode="static")
```

The pre-computed quantile path is aligned to returns by index. It does not
supply Expected Shortfall, so ES diagnostics are unavailable. See
[Methodology](methodology.md) for the order-statistic convention and validity
boundaries.

### `classify_regime(returns, *, forecast=None, forecaster=None, alpha=0.01, mode="rolling", **kwargs)`

Returns `RegimeVerdict` with legacy magnitude labels `"signal-preserving"`
and `"replacement"`. These labels do not measure forecasting ability and
are not the R7 indication rule. The field `R_bootstrap_ci` is a legacy naming
mismatch: static mode carries the shift CI, while rolling mode carries a
mean-plus/minus-1.96-standard-deviation band for the shift. Neither is a
bootstrap confidence interval for the dimensionless ratio `R`.

### `compare_forecasters(returns, forecasts=None, *, alpha=0.01, mode="rolling", test="dm_hac", **kwargs)`

Compares named forecasters or pre-computed quantile series through the audit
API and returns `ComparisonResult`. Quantile Score is a loss; lower is better.

## R7 APIs (added in 0.4.0)

All names and result types below are importable from `conformal_oracle`.
Inputs are chronological, aligned one-dimensional arrays or paired Series
with identical, unique, increasing indexes. No function fits a base forecaster;
its quantile forecasts must already be causal.

### `SeparatedSplitConformalVaR(*, gap, alpha=0.01, calibration_fraction=0.70, minimum_evaluation_size=1)`

Call `.split(returns, quantiles)` to return `SeparatedSplitResult`.
`gap` must be an explicit nonnegative integer, not a Boolean. Define
`n_cal = int(calibration_fraction * len(returns))`. Returned integer indices
are zero-based positions:

- `calibration_indices`: `[0, n_cal)`;
- `gap_indices`: `[n_cal, n_cal + gap)`;
- `evaluation_indices`: `[n_cal + gap, len(returns))`.

`q_v_stat` uses exactly `conformal_quantile(quantiles[:n_cal] -
returns[:n_cal], alpha)`. The gap and evaluation returns do not affect it.
`raw_quantiles` and `corrected_quantiles` contain evaluation observations
only; correction subtracts the shift from the lower return quantile.
`evaluation_index` retains input Series labels, when supplied. Also returned:
`alpha`, `calibration_fraction`, `gap`, `conformal_rank`,
`maximum_score_fallback`, `certified=False`, and sample-size properties
`n_calibration` and `n_evaluation`.

Too few calibration observations, or fewer than `minimum_evaluation_size`
observations after the gap, raise `ValueError`. The default minimum of one
is a computational bound, not a recommendation for statistical adequacy.
`gap=0` provides a contiguous comparator. No explicit gap is itself a
certificate that the theorem's maintained assumptions hold.

### `proxy_separation_gap(scores, *, context_length, safety_factor=1.1, minimum_log_gap=5, numerical_zero_threshold=1e-12)`

Supply **calibration scores only**. The function calculates Pearson
correlation of adjacent score slices, takes its absolute value, and returns
`context_length + ceil(safety_factor * log(n_cal) / abs(log(rho_tilde)))`.
Only if `rho_tilde <= numerical_zero_threshold` is the logarithmic component
replaced by `minimum_log_gap`. Five is not a universal floor. Context length
and the fallback are nonnegative integers; the safety factor must exceed one.

At least three finite scores and a defined lag-one correlation with absolute
value below one are required. Constant/undefined or unit-magnitude proxies
raise `ValueError`. Near-unit persistence may produce a very large gap; the
split estimator subsequently checks evaluation feasibility.

`GapResult` contains `gap`, `context_length`, `log_gap`, `persistence_proxy`
(also available as `rho_tilde`), `signed_autocorrelation`, `safety_factor`,
`n_calibration`, `minimum_log_gap`, `numerical_zero_threshold`, `near_zero`,
`proxy_based=True`, and `certified=False`. This is an operational persistence
proxy, not a validated estimator of the beta-mixing rate or a check of the
theorem's assumptions.

### `recalibration_indication(*, calibration_returns, calibration_quantiles, alpha=0.01, kupiec_level=0.05, rule="basel_or_kupiec")`

No evaluation-window arguments are accepted. The supported rule applies
recalibration iff the calibration Basel zone is not Green **or** the
calibration Kupiec p-value is strictly below `kupiec_level`.

For the R7 policy, count strict violations `return < quantile` over the
**entire calibration window**, annualize as `violations * 250 / n_cal`
without rounding, and classify Green `<=4`, Yellow `<=9`, otherwise Red.
This is distinct from the existing `basel_traffic_light` helper, which uses
a trailing window and rounding; its behavior is unchanged. The policy's
fixed Basel thresholds refer to the 1% convention and are not rescaled for
other `alpha` values. Kupiec uses the configured `alpha`.

`RecalibrationDecision` is frozen and records `apply`, lowercase
`basel_zone`, `kupiec_statistic`, `kupiec_pvalue`, `kupiec_level` (also
`level`), `reasons`, `information_window="calibration"`, `alpha`,
`n_calibration`, `n_violations`, `scaled_violations_250`, `rule`, and
`calibration_fingerprint`. Reasons are `basel_not_green` and/or
`kupiec_rejection`; an empty tuple means skip.

### `selectively_recalibrate(raw_quantiles, *, calibration_returns, calibration_quantiles, decision, method="static", window=250, evaluation_returns=None)`

Supply raw **evaluation** lower quantiles and the calibration arrays used
for the required decision. The helper validates that decision against the
same calibration evidence and settings, never evaluation outcomes.
Indexed evaluation must start strictly after calibration.

- On skip, `final_quantiles` preserves raw values, dtype and bytes exactly;
  correction metadata indicates no correction. No evaluation outcomes are
  required or read.
- Static application subtracts the one calibration shift from every raw
  evaluation quantile. Evaluation outcomes are not read.
- Rolling application requires at least `window` calibration scores and
  aligned `evaluation_returns`. It initializes from the final `window`
  calibration scores; forecast t uses only scores through t-1. Raw, not
  already-corrected, quantiles produce new scores. The initial decision is
  never changed by observed evaluation returns.

`SelectiveRecalibrationResult` preserves independent read-only arrays
`raw_quantiles`, `final_quantiles`, and `corrections`, plus `decision`,
`method`, `window`, `n_calibration`, `calibration_fingerprint`,
`evaluation_index`, `finite_sample_rank`, and `maximum_score_proxy`.
`alpha` and `applied` are properties. Rank/fallback metadata are `None` on
skip. Static and skipped rolling use `window=None`; applied rolling records
its actual window.

Neither fingerprints nor argument names can establish that caller-supplied
data were truly available before deployment. They detect inconsistent reuse;
users remain responsible for truthful chronology and causal base forecasts.

### Optional R7 artifact integration

From the package directory, with the package installed:

```sh
python scripts/reproduce_r7_deployment.py \
    --data-root /path/to/cfp_ijf_data \
    --artifact-root /path/to/replication/root
```

The artifact root contains `Quantlets/cfp_config.py` and
`analysis/ae_point4/pairs_long.csv`. The script checks complete configured
support, generates decisions from calibration evidence, and separately
replays static/rolling corrections for ex-post score and zone summaries.
It compares per-pair values with stored artifacts and writes JSON to stdout,
not the manuscript or data files. Missing inputs or mismatches produce a
nonzero exit. Parquet loading requires a pandas Parquet engine such as
`pyarrow`, which is an integration-only prerequisite, not a core dependency.

The manuscript artifacts use `p <= 0.05`; this API follows the requested
strict `p < kupiec_level` convention. The integration report explicitly
counts exact-cutoff cases instead of silently ignoring the distinction.

## Compatibility functions

The following top-level imports remain available with deprecation warnings.
Prefer `audit()` and `compare_forecasters()` for new code.

### `audit_static(returns, forecaster, alpha=0.01, calibration_split=0.70, warmup=50, seed=2026)`

Static conformal audit. Returns `StaticAuditResult`.

```python
from conformal_oracle import audit_static
from conformal_oracle.contrib.benchmarks import GJRGARCHForecaster

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

The forecaster must use only `returns.iloc[:t]` (history up to t-1) when
producing the time-t forecast; the API passes the series and the position,
so this no-lookahead obligation belongs to the forecaster implementation.

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
