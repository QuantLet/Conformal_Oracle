# The residual density and the rank-correlation collapse — results

Run 2026-08-29 against `PREREG_DENSITY_MECHANISM.md`. Unit: one pair, one
forecaster on one asset at alpha = 0.01. 261 of 312 admit a defined implied
density: the 48 truncated series are excluded before looking, and 3 more have a
non-positive calibration-window uMCB. Above the pre-registered floor of 200, so
the exercise is not blocked.

**Both pre-registered predictions are wrong, in opposite directions.** The
proposal is right in substance and wrong in mechanism, and the mechanism it is
right by is not the one either the proposal or the objection named.

## 0. First, the density figure itself was the wrong object

Section 6 said the implied residual density "spans a factor of 489 across this
panel", taken from `analysis/umcb/MEMO.md`. That number divides a **test-window**
uMCB by a **calibration-window** $\hat q_V$. The two come from different samples,
so the estimation noise in the denominator does not cancel against the numerator
and the ratio inherits both. It is a cross-window quantity, not a property of the
density.

Computed within one window --- calibration-window uMCB against the same window's
$\hat q_V$ --- the implied density spans **14.8** between its 5th and 95th
percentiles, 0.32 to 4.76, median 1.74.

| object | 5th--95th percentile of implied $f$ | span |
|---|---|---|
| test uMCB over calibration $\hat q_V$ (as printed) | 0.05 -- 21.8 | **417** |
| `umcb_pairs.csv`, which is the same cross-window quantity | 0.05 -- 21.6 | 422 |
| calibration uMCB over calibration $\hat q_V$ | **0.32 -- 4.76** | **14.8** |

489 was the seventh figure in this project to mean something other than what its
sentence said. The correction reduces the claim by an order of magnitude and it
still supports the sentence it is in.

## 1. Test 1 --- prediction: the density does not enter the shift-to-shift collapse. FALSIFIED

The pre-registration argued from the algebra: $\rho(\hat q_V,
\delta^\star_{\text{test}})$ compares a displacement with a displacement, no
density appears in the relation between them, and attributing the collapse to $f$
would be the unit confusion Section 7 was just corrected for.

Split at the median implied $f$, the pre-registered threshold being a difference
of 0.15:

| half | $n$ | $\rho(\hat q_V, \delta^\star_{\text{test}})$ |
|---|---|---|
| low $f$ | 131 | **0.361** |
| high $f$ | 130 | **0.572** |
| pooled | 261 | 0.517 |

The halves differ by **0.212**, past the threshold. The prediction fails.

**Why it fails, and the objection's error.** The algebra is right about the
*definition* and silent about the *sampling distribution*. The asymptotic
standard deviation of a sample $\alpha$-quantile is
$\sqrt{\alpha(1-\alpha)/n}\,/\,f(q_\alpha)$: the density enters through the
variance of the estimator, not through the score conversion. A dense residual
tail pins the quantile down; a sparse one leaves it loose. So $f$ governs how
noisy $\hat q_V$ is as an estimate of $\delta^\star$, and a panel spanning a
factor of 15 in $f$ spans the same factor in estimation noise.

That is a testable functional form, not an interpretation. Regressing
$\log|\hat q_V - \delta^\star_{\text{test}}|$ on $\log f$ over the 261 pairs:

- slope **−0.940** (s.e. 0.103), against the classical prediction of **−1**;
  **0.6 standard errors** from it, $r = -0.494$, $p = 1.8 \times 10^{-17}$;
- the magnitude matches as well as the exponent: the median observed gap is
  **1.11** times the median $\sqrt{\alpha(1-\alpha)(1/n_{\text{cal}} +
  1/n_{\text{test}})}\,/\,f$, an 11\% discrepancy with no fitted constant.

So the collapse from 0.99 to 0.52 is estimation error, and its size is predicted
in closed form by the residual density. That is a better sentence than
"estimation error" and a better one than the density story as proposed.

## 2. Test 2 --- prediction: the density explains the shift-to-score collapse. FALSIFIED

This was the proposal as stated, and the threshold was $\rho$ above 0.85.

| comparison | $\rho$ |
|---|---|
| $\lvert\hat q_V\rvert$ against test-window uMCB | 0.531 |
| $\tfrac12 f \hat q_V^2$ against test-window uMCB | **0.529** |
| $\tfrac12 f \hat q_V^2$ against calibration-window uMCB | **1.0000** |

Supplying each pair's own density recovers nothing: 0.529 against 0.531, a
change of 0.002 in the wrong direction. The third row is the pre-registered
circularity branch and it fires exactly as written --- $f$ is *solved for* from
the calibration-window uMCB, so $\tfrac12 f \hat q_V^2$ reproduces it as an
identity to four decimal places, and a reader shown only that row would take a
tautology for a confirmation. Writing that branch down first is what makes the
1.0000 readable.

The density is the conversion factor **within** a window and carries no
information **across** one. Both correlations sit at 0.53 for the same reason
Test 1 gives: what limits them is the sampling noise in $\hat q_V$, which the
conversion factor cannot undo.

## 3. What Section 6 says

One mechanism, not two observations, and not the mechanism proposed:

> The residual density at the $\alpha$-quantile varies by a factor of 15 across
> the panel, and the standard deviation of a sample quantile is inversely
> proportional to it. The collapse in rank correlation from 0.99 within the
> estimation window to 0.52 across it is therefore estimation error of a size the
> density predicts: regressed on the log density, the log gap has slope −0.94
> against a theoretical −1, and its median is 1.11 times the predicted standard
> deviation.

What Section 6 must **not** say is that the density explains why $\hat q_V$ and
uMCB order the panel differently. It does not: supplying it changes 0.531 to
0.529.
