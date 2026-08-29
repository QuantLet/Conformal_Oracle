# Pre-registration — does the residual density explain the rank-correlation collapse?

Written 2026-08-29, before anything below is computed.

## The proposal under test

Section 6 reports two facts side by side: the implied residual density $f$ spans
a factor of 489 across the panel, and the rank correlation between $\hat q_V$ and
its target falls from 0.99 within the estimation window to 0.52 across the window
boundary. The proposal is that the first explains the second, which would make
them one mechanism rather than two observations.

## The objection to the proposal, stated before measuring

K0a reports **two** correlations and they are different objects:

| correlation | what it compares | is $f$ in the map? |
|---|---|---|
| $\rho(\hat q_V,\ \delta^\star_{\text{test}})$ = 0.52 | a shift against a shift | **no** |
| $\rho(\lvert\hat q_V\rvert,\ \mathrm{uMCB}_{\text{test}})$ = 0.53 | a shift against a score reduction | **yes** |

$f$ is the conversion factor between a displacement and the score reduction it
buys, through $\mathrm{uMCB} \approx \tfrac12 f \hat q_V^2$. It therefore cannot
enter the first correlation at all: both sides are displacements in return units
and no density appears in the relation between them. The first collapse must be
estimation error across the window boundary, and calling it a density effect
would be a unit confusion of exactly the kind Section 7 was corrected for on the
$-1.9\hat\sigma$ example.

So the proposal is tested only on the second correlation, and the first is
reported as what it is.

## The unit

One row is a **pair**: one forecaster on one asset at $\alpha = 0.01$. Expected
rows: 312, of which 264 well-specified and 48 truncated. The density is defined
only where the quadratic expansion is meaningful, so the truncated series are
excluded --- their $\hat q_V$ is not a small displacement --- and pairs with
numerically negative in-sample uMCB are dropped rather than clipped. Expected
rows for the density test: 259, the count Section 6 already reports.

## Predictions, written in both directions

**Test 1 --- does $f$ enter the shift-to-shift collapse?** Prediction: **no**.
$\rho(\hat q_V, \delta^\star_{\text{test}})$ should show no systematic relation
to the pair's implied $f$. Operationally: split the 259 pairs at the median
implied $f$ and compute the rank correlation within each half.

- **If both halves return a correlation near the pooled 0.52**, the density is
  irrelevant to this collapse, as predicted, and the text says the collapse is
  estimation error and nothing else.
- **If the halves differ materially** --- say by more than 0.15 in $\rho$ --- then
  something links them that the algebra does not, and that is a finding to
  report and explain, not to fold into the density story. It would most likely
  mean $f$ is proxying for something else, tail heaviness or sample length, and
  the text would have to say so rather than claim a mechanism.

**Test 2 --- does $f$ explain the shift-to-score collapse?** Prediction: **yes**,
and this is the one the proposal is really about. If the density is the missing
conversion factor, then replacing $\lvert\hat q_V\rvert$ with the pair's own
$\tfrac12 f \hat q_V^2$ should recover a much higher correlation with uMCB.

- **If $\rho(\tfrac12 f \hat q_V^2,\ \mathrm{uMCB})$ is materially above
  $\rho(\lvert\hat q_V\rvert,\ \mathrm{uMCB}) = 0.53$** --- say above 0.85 --- the
  density is the mechanism, the two observations become one, and Section 6 says
  so.
- **If it is not materially higher**, the density is not the explanation, the two
  facts stay two facts, and Section 6 keeps them adjacent without a causal claim
  between them. This is the outcome that costs the paper a sentence it would
  like to have, and it is reported as readily as the other.

**A third outcome.** The recovered correlation may be high by construction rather
than by mechanism: $f$ is *solved for* from uMCB and $\hat q_V$, so
$\tfrac12 f \hat q_V^2$ reproduces uMCB identically and any correlation it shows
is an identity, not evidence. If the recovered $\rho$ is 1.000 to numerical
precision, that is the tautology and not the mechanism, and the test has to be
run the other way: with $f$ estimated **independently** of uMCB, from the
empirical residual density at the $\alpha$-quantile, and only then is a high
correlation informative.

This third branch is the likely one and it is written down first so that a
correlation of 1.000 is read as a circularity rather than a confirmation.

## What would falsify the exercise itself

If fewer than 200 of the 312 pairs admit a defined implied density, the panel is
too thin and the result is BLOCKED rather than reported.
