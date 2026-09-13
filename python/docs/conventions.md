# Conventions

These conventions must be understood before using the package.
They document the package API. See [Methodology](methodology.md) for the
distinction between the separated, contiguous and rolling estimators in R7.

## Returns

- `pandas.Series` of **log returns** in **decimal form** (not percent).
- Negative values = losses.
- The index should be a `DatetimeIndex` (business days) but the
  package operates positionally, not by date.

**Example:** a daily return of -2.3% is stored as `-0.023`.

## VaR sign convention

VaR is reported as a **positive number** representing the loss threshold.

A **violation** occurs when `r_t < -VaR_t`.

**Example:** if `VaR_t = 0.025` and `r_t = -0.030`, then `-0.030 < -0.025`
is `True`, so this is a violation (the loss exceeded the VaR forecast).

## Quantile level alpha

`alpha` is the **lower-tail probability**: small values (e.g., 0.01)
correspond to extreme left-tail risk.

The forecast is the `alpha`-quantile of the predictive distribution.
For `alpha = 0.01`, the forecast is the 1st percentile.

## Calibration split (static mode)

`calibration_split` is a float in `(0, 1)`. Default `0.70`.

The first 70% of the return series is the calibration set (used to
compute `qV_stat`); the remaining 30% is the test set (used for
backtesting).

The split is **chronological and contiguous** — no shuffling and no separation
gap. It is not the separated estimator covered by R7 Theorem 4.5.

`SeparatedSplitConformalVaR` instead accepts `calibration_fraction` and an
explicit `gap`. It preserves the same first calibration block and starts
evaluation at zero-based position `n_cal + gap`. It returns lower quantiles
(`raw - shift`), not the positive-loss VaR returned by the audit reporting API.
The gap does not certify the theorem's assumptions.

## Rolling window

`window` is an integer (default 250, i.e., one trading year).

`qV_roll(t)` is computed from the most recent `window` nonconformity
scores ending at `t-1`.

All modes use the `ceil((n+1)(1-alpha))`-th order statistic on their available
calibration scores, with `n=window` in the rolling helper. The helper clips
ranks above `n` to the sample maximum, a finite fallback without the usual
conformal coverage guarantee. An empty score array returns `0.0`.

## Warmup

`warmup` is an integer (default 250 for rolling, 50 for static).

Minimum number of observations before the first valid forecast.
Forecasts at `t < warmup` are skipped. Users should set this to
match their forecaster's minimum context requirement (e.g., 512 for
a TSFM with `context_length=512`).

## Sample size

`SampleDistribution` uses `N=1000` samples by default, matching the
paper's Monte Carlo convention.

## Block bootstrap

Stationary block bootstrap with geometric block lengths.
Mean block length = 20, B = 999 replications.

## Regime classification

- **Signal-preserving:** legacy label for a correction that does not exceed
  the raw VaR scale in magnitude; not evidence that the forecast is informative.
- **Replacement:** legacy label for a correction exceeding the raw VaR scale;
  not a test of whether the forecast is uninformative.

Threshold: `R = |qV| / mean(|VaR_raw|) > 1.0` → replacement.

In rolling mode, a **persistence rule** requires `R_t > 1.0` for
at least 20 consecutive trading days to trigger "replacement",
avoiding transient classification flips from short volatility spikes.

Neither label implements the paper's pre-deployment indication rule. Quantile
Score is a loss (lower is better); improved marginal coverage can coexist
with a worse score and does not validate the base model.

Use `recalibration_indication` for the calibration-only R7 rule. Its Basel
zone uses the complete calibration count annualized to 250 observations,
without rounding (Green `<=4`, Yellow `<=9`, otherwise Red); its Kupiec
trigger is strict `p < kupiec_level`. Existing trailing-window diagnostic
behavior is unchanged. Evaluation outcomes may update future rolling shifts,
but cannot change the fixed initial decision.
