# Methodology

This package supplies separated, contiguous and rolling recalibration tools
and a calibration-only indication rule accompanying:

> Pele, D.T., Bolovăneanu, V., Ginavar, A.T., Lessmann, S., Härdle, W.K.
> "Conformal Recalibration of Extreme Tail Quantiles under Temporal Dependence"
> (2026, manuscript R7).

## The conformal correction

Given a black-box forecaster that produces predictive distributions
F_t at each time step, the nonconformity score is:

    S_t = F_t^{-1}(alpha) - r_t

where `F_t^{-1}(alpha)` is the alpha-quantile of the predictive
distribution and `r_t` is the realised return.

### Static mode

For `n` calibration scores, define `k = ceil((n+1)(1-alpha))`.
The **static conformal correction** `qV_stat` is the `k`-th smallest score,
not the interpolated empirical quantile `np.quantile(scores, 1-alpha)`.
The existing static audit uses one chronological, **contiguous** calibration/test
split. For the same supplied calibration score block,
`SeparatedSplitConformalVaR` uses the same conformal order statistic but starts
evaluation after an explicit unused gap. The shift itself is unchanged;
no gap or evaluation outcomes enter its estimation. Supply already-aligned
forecasts after any model warmup when comparing with a forecaster-object audit.

The corrected VaR forecast is:

    VaR_corrected(t) = -(F_t^{-1}(alpha) - qV_stat)

For exchangeable calibration and evaluation scores, the conformal rank gives
finite-sample marginal coverage at least `1-alpha` when `k <= n`.
When `k > n`, the formal conformal threshold is `+inf`, but this package
returns the largest observed score as a finite proxy. That fallback does not
retain the usual finite-sample coverage guarantee. Empty score arrays return
`0.0` for compatibility; this is not a calibrated estimate.

### Rolling mode

The **rolling conformal correction** `qV_roll(t)` uses the same conformal
order statistic on the most recent `w` nonconformity scores:

    k = ceil((w+1)(1-alpha))
    qV_roll(t) = k-th smallest of {S_{t-w}, ..., S_{t-1}}

The same maximum-score fallback applies when `k > w`. Rolling recalibration
is an operational heuristic under temporal dependence, not an estimator
covered by the paper's separated single-split theorem. Better coverage can
come with worse Quantile Score; empirical outcomes depend on the evaluation
sample and do not supply a general deployment guarantee.

## Regime classification

The **replacement ratio** measures the correction's magnitude
relative to the raw forecast:

    R = |qV| / mean(|VaR_raw|)

- `R <= 1`: legacy label **signal-preserving**; the correction does not
  exceed the mean absolute raw VaR in magnitude.
- `R > 1`: legacy label **replacement**; the correction exceeds that scale.

These labels describe correction magnitude only. They do not establish
information content, conditional calibration or the benefit of deployment.
The ratio carries no sign; inspect the signed shift separately.

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

R7 Theorem 4.5 covers the **separated single-split estimator**, asymptotically
under the maintained dependence assumptions. It does not cover the contiguous
static or rolling audits. `SeparatedSplitConformalVaR` implements the separated
construction; it does not verify the maintained assumptions or certify that
an arbitrary explicit gap is sufficient.

The paper's operational gap experiment uses a proxy-based implementation of
the separation rule. Its absolute lag-one score autocorrelation is not a
validated estimator of the mixing rate, and the proxy does not certify the
finite-sample guarantee. `proxy_separation_gap` implements this operational rule:

    rho_tilde = abs(Corr(scores[:-1], scores[1:]))
    log_gap = ceil(safety_factor * log(n_cal) / abs(log(rho_tilde)))
    gap = context_length + log_gap

The default safety factor is `1.1`. Only at `rho_tilde <= 1e-12` does the
utility replace the logarithmic term with `minimum_log_gap=5`; five is not a
floor for all finite persistence estimates. Negative autocorrelations use
their absolute values. Undefined/constant-series correlations and absolute
correlations of one are rejected. Metadata always marks the result as
`proxy_based=True, certified=False`.

## Interpretation and selective deployment

[Gneiting and Resin (2023)](https://doi.org/10.1214/23-EJS2180) provide the
general calibration hierarchy and score-decomposition framework, including
the unconditional component obtained through optimal constant translation.
The displacement minimises the canonical Quantile Score over translations;
the associated score reduction is the unconditional miscalibration component,
not the displacement itself. The conformal ceiling rule estimates the target
displacement but need not be the exact empirical-score minimiser.

The companion study specialises dependent-data coverage results to financial
score processes, compares scalar and richer recalibration under tail sparsity,
and evaluates a pre-deployment indication rule. These are not claims of a new
score decomposition. A scalar correction does not in general repair the full conditional
predictive distribution, and marginal coverage is not model validation.

`recalibration_indication` implements the paper's pre-deployment policy:
apply when the calibration Basel zone is not Green or the calibration Kupiec
p-value is below the configured significance level. It accepts only
explicitly named calibration arrays, records their fingerprint and returns
a fixed decision with its diagnostics and reasons. `selectively_recalibrate`
checks that calibration provenance, applies static or causal rolling shifts
when indicated and preserves raw forecasts otherwise. Test-window outcomes
are not inputs to the decision. In a rolling replay they enter only the
history for subsequent corrections, never the initial policy.

`classify_regime()` remains a descriptive magnitude diagnostic, not this rule.
The indication rule does not promise a score improvement on every series.
Avoided deteriorations and lost upgrades are ex-post evaluation summaries,
not inputs or outputs hard-coded into the decision algorithm.

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
