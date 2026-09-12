# Controlled short-window experiment

Declared 9 September 2026 before computing this experiment, following the
author's approval of the research plan and the next controlled experiment.
This is a development study. The existing panel and earlier simulations
have already been examined; this is not an untouched empirical confirmation.

## Question and fixed design

Separate dependence of tail counts, persistence of conditional volatility,
and the shape of a missing correction. Test whether regularisation and a
structured tail model change the short-window comparisons. Include an
already-correct forecast; loss reduction is not presumed.

Use 500 independent calibration histories per generator, windows
125, 250, 500 and 1,000, and alpha 0.01 and 0.05. Reuse each history across
windows, levels and distortions. All configurations and fitting failures
are reported. No replication, threshold, state or method is removed because
of its result. Monte Carlo standard errors use paired independent histories,
not the number of configurations, methods or test states.

### A. Tail-count dependence at a fixed marginal distribution

Generate stationary Gaussian AR(1) paths of length 1,000 with phi in
{0, 0.5, 0.8}: the initial state is N(0,1); subsequent innovations are
independent N(0,1), with multiplier sqrt(1-phi^2). Use the same underlying
innovation path across phi and across two monotone marginal transformations:
normal, and variance-standardised Student t(5). Return scale is
V0=sqrt(1e-5/(1-.10-.85)). Marginal distributions are continuous and paths
have no constructed ties. Record any numerical CDF endpoint clipping.

The raw threshold is the true marginal alpha quantile plus either zero or
0.25 V0. At phi>0 this is NOT the conditional quantile. Evaluate against an
independent draw from the known marginal distribution, using exact expected
pinball loss and marginal violation probability. This isolates the cost of
dependent calibration, not the performance of a contiguous forecasting rule.

Methods: Raw, Shift-CP, Shift-ERM, POT80-Shift and POT90-Shift. The last is a
threshold sensitivity. Fit upper-tail scores; translation equivariance may
be used to share one fit across raw offsets and, for POT, target levels.
There are 96 configurations but only 500 independent latent histories.

Calculate tail-indicator covariances by deterministic one-dimensional
Gaussian integration. Report both the exact finite-n count variance and
the long-run variance, with an explicit bound on the truncated covariance
sum. The normal and t(5) transformations share the same indicators.

### B. Conditional volatility and correction shape

Reuse the exactly regenerable calibration histories of the existing GARCH
control: omega=1e-5, a=.10, b=.85, 2,000 warm-up observations, normal or
variance-standardised t(5) innovations. Reuse its 500 seeds per innovation
law and fixed independent 1,024 test volatility states. The warm-up
approximates stationarity; it does not initialise the stationary GARCH law
exactly. The oracle conditional quantile remains known at every state.

Supply true conditional sigma to every correction requiring volatility.
This is an ideal-proxy mechanism experiment, not a claim about estimated
empirical volatility. Raw = oracle plus one of:

- zero;
- constant 0.25 V0;
- 0.25 V0 + 0.75 V0 log(sigma/V0).

The oracle hit indicators are iid under both innovation laws despite
volatility persistence. Compute conditional expected pinball loss and
violation probability analytically at the fixed test states. No test return
sampling or hyperparameter choice from test states is used.

Methods: Raw, Shift-CP, Shift-ERM, Vol-CP, Vol-ERM, State2-ERM,
State4-ERM, State-L1, State-L1-clipped, POT80-Shift, POT80-Vol,
POT90-Shift and POT90-Vol. All have the same calibration information.
The clipped state prediction is a separately named sensitivity using the
same selected coefficients. There are 48 configurations and 1,000
independent calibration histories. Together A and B use 1,500 independent
histories, not 72,000 independent experiments.

## Estimation

CP uses rank ceil((n+1)(1-alpha)); ERM uses the empirical inverse-CDF
quantile. Vol-ERM minimises return-unit loss through sigma weights.
State corrections use intercept plus linear or cubic powers of standardised
log sigma. For L1, use the previous experiment's p in {2,4}, lambda in
{0, 1e-4, 1e-3, 1e-2, .1, 1}, with an unpenalised intercept and response
scaled by fitting median sigma. Fit on the first floor(.7n) observations;
select by mean pinball loss on the remainder; refit on all n. Exact ties
prefer smaller p then larger lambda. Every trial and LP certificate is
saved. At n=125 the validation tail count is deliberately small: selection
cost is part of the experiment. This proportional split differs from the
long-panel split used in the earlier empirical experiment.

POT80 is fixed in advance at the empirical 80th percentile, so n=125 gives
25 strict exceedances. POT90 is a sensitivity, not a selected alternative.
Use the existing deterministic three-start GPD likelihood implementation,
actual exceedance fraction, and minimum 20 exceedances. Hence POT90 at
n=125 falls back to the corresponding CP estimator; label and count this.
Numerical failures also fall back, with their reasons saved. Retain and
flag shapes <= -0.5 and all finite extrapolations. No clipping of GPD shapes
or outcome-driven threshold choice. This is a plug-in tail model, without
an asserted conformal guarantee or exact GPD specification for these laws.

Exact location equivariance may reduce repeated fits. In B, constant and
zero distortions give identical corrected paths for unrestricted scalar
and intercept-containing state fits; the original raw paths still differ.
Unpenalised state families also contain the stated log-sigma distortion.
Penalised state fits under the state distortion must be estimated separately.
All transformations used to reuse a fit must be saved and tested.

## Outputs and checks

Save source/protocol hashes, package versions, seeds, generated calibration
paths, all fit parameters/trials, per-history expected losses and violation
rates, and prediction moments. Use exact loss, not a quadratic approximation,
for conclusions. Report paired loss differences, Monte Carlo standard errors,
fitting/selection frequencies, tail counts and sensitivity results. Pointwise
Monte Carlo intervals are descriptive, not a search-adjusted discovery claim.

Check analytic loss against numerical integration, alpha propagation,
stationary AR construction, covariance/count formulas, translation
equivariance, LP certificates, oracle loss optimality, complete configuration
coverage, and representative fresh-process replay. Compare old GARCH
reference methods with their existing saved outputs, allowing only recorded
LP nonuniqueness or floating-point tolerance, never silent numerical drift.
Check that all tracked canonical R8 sources and PDFs remain unchanged.

Only after these checks will results be interpreted for the paper. A theory
development note may state separately proved results under explicit stronger
assumptions; neither simulation nor this independent-test design transfers
Theorem 4.5 to contiguous or rolling estimators.
