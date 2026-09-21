# Regime changes and the cost of adaptation

Declared 10 September 2026, before generating this study's paths or results.
The user authorised continuation of the outstanding regime-change experiment.
This is a controlled development experiment, not external confirmation.

## Design fixed before results

Use 500 independent innovation histories under each of two laws: standard
normal and variance-standardised Student t(5). There are 1,000 independent
histories in total, shared within each law across scenarios, methods and
alpha in {0.01, 0.05}. Each history contains 1,000 warm-up innovations and
2,500 retained observations. Seeds are SHA-256-derived from
`20260910/regime/{law}/{replication}`. Do not increase replication counts,
discard histories or tune the design in response to estimated effects.

Baseline return standard deviation is V0=sqrt(0.00001/(1-.10-.85)). Returns
are sigma_t times the innovation; the volatility path is deterministic.
The first evaluated forecast is at retained index 1250 (zero-based), the
break is at index 1500, and the last forecast is at index 2499. Report the
250 pre-break days and four fixed post-break blocks: days 1–20, 21–125,
126–500 and 501–1000, as well as the complete 1,000-day post-break horizon.

Eight scenarios, all retained in the report:

| Scenario | Return scale | Raw quantile |
|---|---|---|
| correct | V0 throughout | True conditional quantile |
| biased | V0 throughout | Truth + 0.5 V0 throughout |
| bias appears | V0 throughout | Truth; then truth + 0.5 V0 |
| bias disappears | V0 throughout | Truth + 0.5 V0; then truth |
| bias reverses | V0 throughout | Truth + 0.5 V0; then truth − 0.5 V0 |
| scale jump, oracle | V0; then 2 V0 | True conditional quantile throughout |
| steady scale, EWMA | V0 throughout | Law's unit-variance quantile times EWMA sigma |
| scale jump, EWMA | V0; then 2 V0 | Same past-only EWMA forecaster |

The first five manipulate forecast error, not the return law. The last
three separate an actual market-scale break from lag in the base model.
EWMA uses lambda=.94, initial variance V0 squared, and every warm-up return.
The forecast for t uses only returns strictly before t. An oracle raw
forecast is a deliberate control, not a deployable volatility estimator.
Corrections requiring scale receive the raw forecaster's own scale:
true sigma for oracle controls and estimated sigma for EWMA controls.

## Fixed correction policies

Compare Raw, Static-CP, Rolling125/250/500/1000, VolRolling250/500,
projected DtACI500, PastSelectedRolling and InitialKupiecGate.
CP always uses ceil((w+1)(1-alpha)); thus window and finite-sample rank
effects are both present, explicitly as in the paper. Do not interpret the
window contrast as a pure sample-size effect.

Static-CP uses the 1,000 scores at indices 250:1250. Rolling corrections
use raw forecast-minus-return scores strictly preceding each origin.
VolRolling uses the same CP convention on scores divided by the past
forecaster scale, multiplied by the scale known at the forecast origin.

Projected DtACI follows the already implemented seven-expert, finite-range
version in research/r8_decision/methods.py, with a 500-score window.
Initialise experts at alpha with uniform weights at index 1000; update
through the 250 validation observations before deployment and then the
test observations. Keep its gamma grid, sharing parameter, learning-rate
formula and [1/501,500/501] level projection unchanged. Propagate alpha
explicitly for the 5% sensitivity. Evaluate the distribution over experts
using its exact conditional expected pinball loss and expected violation
probability, never the loss of an averaged quantile. No ordinary Kupiec
test is applied to its fractional expected hits.

PastSelectedRolling chooses once among 125/250/500/1000 by realised mean
pinball loss at indices 1000:1250, with exact ties favouring the larger
window. InitialKupiecGate uses that frozen choice if the raw forecast's
Kupiec LR test on these same 250 observations rejects at 5%; otherwise it
retains Raw throughout. This is an initial-deployment Kupiec-only control,
not the existing empirical gate's complete specification and not a new
adaptive decision rule. Selection and gating precede both evaluation and
the unannounced break; neither is reopened after observing the break.
The experiment tests obsolescence of a past decision, not break detection.

## Evaluation and interpretation

The current innovation is independent of past forecasts. Integrate its
known law analytically to obtain conditional expected loss and violation
probability at each saved forecast. These are losses of contiguous online
policies in a nonstationary design, not the previous independent-test
stationary experiment. No beta-mixing theorem is invoked across a break.

Report QS and excess QS over the known conditional oracle, paired changes
versus Raw, Static-CP, Rolling500 and PastSelectedRolling, expected
violation rates and absolute quantile error. Normalise excess loss by
V0 and also save return-unit values. Monte Carlo uncertainty uses the 500
independent histories within each law. Pointwise 95% MC intervals are
descriptive; there is no multiple-comparison discovery claim or unique
winner criterion. Do not use a favourable scenario to suppress the rest.
Save realised losses and hits as diagnostics, but use integrated losses
for headline comparisons. Save complete predictions, DtACI experts and
pre-outcome weights, input paths, selection decisions and per-history
period metrics. A DtACI mean prediction is for plotting only.

## Verification before interpretation

Bind protocol, sources and environment before the full run. Preserve
canonical sources/PDFs and existing empirical matrices during this phase.
Test analytic integrals independently, finite-sample ranks, exact update
timing, selected-policy invariance to future perturbations, sequential
EWMA replay, scalar-versus-vector DtACI equivalence including ties, and
oracle optimality. Regenerate every innovation path and exactly replay
all production blocks in fresh processes. Check completeness, finite
outputs, mixture probabilities and all file hashes. Fail closed on changed
bindings or missing blocks. Protocol amendments for defects must be
archived and disclosed before aggregate comparisons are interpreted.

This study does not replace the external 12-industry evaluation, whose
required August 2026 endpoint is not yet available at the official source.
