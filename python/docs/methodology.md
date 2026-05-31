# Methodology

This package implements the conformal recalibration framework from:

> Pele, D.T., Bolovăneanu, V., Ginavar, A.T., Lessmann, S., Härdle, W.K.
> "Recalibrating Tail Event Forecasts under Temporal Dependence" (2026).

## The conformal correction

Given a black-box forecaster that produces predictive distributions
F_t at each time step, the nonconformity score is:

    S_t = F_t^{-1}(alpha) - r_t

where `F_t^{-1}(alpha)` is the alpha-quantile of the predictive
distribution and `r_t` is the realised return.

### Static mode

The **static conformal correction** `qV_stat` is the empirical
`(1-alpha)`-quantile of the calibration scores `{S_1, ..., S_n_cal}`.

The corrected VaR forecast is:

    VaR_corrected(t) = -(F_t^{-1}(alpha) - qV_stat)

Under exchangeability, this guarantees finite-sample coverage at
level `(1-alpha)`. Under beta-mixing temporal dependence, coverage
holds approximately with a bound that depends on the mixing rate
(Theorem 3.5 of the paper).

### Rolling mode

The **rolling conformal correction** `qV_roll(t)` is the empirical
`(1-alpha)`-quantile of the most recent `w` nonconformity scores:

    qV_roll(t) = Quantile_{1-alpha}({S_{t-w}, ..., S_{t-1}})

This adapts to non-stationarity. In the paper, rolling correction
lifts Basel Green-zone compliance from 85% to 98%.

## Regime classification

The **replacement ratio** measures the correction's magnitude
relative to the raw forecast:

    R = |qV| / mean(|VaR_raw|)

- `R < 1`: **signal-preserving** — the forecaster provides meaningful
  risk information; conformal calibration fine-tunes it.
- `R > 1`: **replacement** — the correction dominates; the forecaster's
  signal is uninformative at this tail level.

In rolling mode, a persistence rule requires `R_t > 1` for at least
`K=20` consecutive days to trigger the replacement classification,
avoiding transient flips from short volatility spikes.

## Drift diagnostic

The **distributional drift diagnostic** `delta_hat_w(t)` measures
non-stationarity in the score distribution via total variation (TV)
distance between the first and second halves of each rolling window:

    delta_hat_w(t) = 0.5 * sum_b |p_{1,b}(t) - p_{2,b}(t)|

High drift values indicate that the rolling correction may be
tracking a moving target, suggesting the forecaster's distributional
assumptions may be structurally misspecified.

## Coverage validity

The conformal correction provides valid finite-sample coverage under
the exchangeability assumption. Under temporal dependence (beta-mixing),
coverage holds approximately. The paper proves an explicit bound on
coverage error as a function of the mixing rate (Theorem 3.5), and
validates it empirically on 24 assets across 10 forecasting models.

## Diagnostics

The package computes standard backtesting diagnostics:

- **Kupiec POF test**: unconditional coverage test (LR ~ chi2(1))
- **Christoffersen test**: conditional coverage (independence + coverage)
- **Basel traffic light**: Green (<=4/250), Yellow (5-9/250), Red (>=10/250)
- **Acerbi-Szekely Z2**: ES backtest with stabilised denominator.
  Z2 near zero is expected under correct specification; the sign
  indicates direction of ES bias (negative = mild over-prediction).
- **Quantile score**: pinball loss (proper scoring rule for quantiles)
- **Fissler-Ziegel FZ_0**: joint VaR-ES consistent scoring function
- **Diebold-Mariano test**: pairwise comparison of predictive accuracy
  with Newey-West HAC variance and HLN small-sample correction
