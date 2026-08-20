# The six flagged-but-unlabelled cases, traced

The pre-registration required that every detector flag on an unlabelled
forecaster be **traced to a verdict, not counted as a false positive**. An
untraceable flag is a false positive; a traceable one is the next defect. This
completes that obligation. All six flags come from Kupiec; neither q̂_V rule
fired on any unlabelled forecaster.

Recall that `none` never meant "clean" — it meant not known to be defective.
These six are the cases where that distinction had to be paid for.

| forecaster | π̂(0.01) | verdict |
|---|---|---|
| GARCH-N | 0.0193 | expected — Gaussian innovation |
| EWMA | 0.0208 | expected — Gaussian innovation |
| Hist-Sim | 0.0158 | expected — order-statistic variance |
| Chronos-Small-A | 0.0175 | characterised — residual tail deficiency |
| Chronos-Mini-A | 0.0178 | characterised — residual tail deficiency |
| **Lag-Llama** | **0.0294** | **no defect found — under-dispersion** |

---

## GARCH-N and EWMA — Gaussian innovations on fat-tailed returns

Both apply `norm.ppf(α)` to a conditional scale. A Gaussian innovation
under-covers the 1% tail of daily financial returns by roughly a factor of two;
this is textbook and is why the Student-*t* variant exists. GJR-GARCH-*t*, on
the same panel and the same windows, runs at 0.0145 against these two at 0.0193
and 0.0208.

Neither is a defect. Both are reproducible from committed code — Hist-Sim and
EWMA verified exactly by `scripts/verify_producers.py`, GARCH-N verified to
`ML_REFIT` (same estimator, same quantile map, optimiser differences across
library versions of ~4% of |VaR|, which is a reproducibility limit rather than a
coverage cause and cannot produce a systematic 2× deficit).

The two known irregularities in this pair were checked and neither explains the
flag. EWMA's documentation describes a truncated 250-day weighted sum while its
data came from the RiskMetrics recursion — a real documentation defect worth
1.2e-10 in VaR, seven orders below the discrepancy at issue. GARCH-N's
cross-version irreproducibility is unsigned noise.

## Hist-Sim — the estimator's own variance

The 1% quantile of a 250-observation window is approximately its 2.5th order
statistic. That estimator has high variance and a known small-sample downward
bias in |VaR| at extreme levels, and it holds constant for long stretches as the
window slides. Its coverage at α = 0.10 is far better (11/24 passes at 0.01
against a much higher rate at 0.10), which is the signature of order-statistic
sparsity rather than of a systematic error.

Not a defect, and reproduced bit-identically on 21 of 24 assets by the committed
producer; the other three differ because their input returns were revised
upstream.

## Chronos-Small-A and Chronos-Mini-A — the characterised residual

These are the analytic series, after the `top_k` truncation is removed. Their
residual is already characterised in the paper: π̂/α falls monotonically from
1.75 at α = 0.01 to 1.04 at α = 0.10, so the deficiency is specifically in the
tail and largely gone by the 10% level, where they pass Kupiec on 20 and 22 of
24 assets.

The flag is real and is reported in the paper as a limitation. It is a property
of the model's quantisation and pretraining, not a pipeline defect: the series
pass every structural check except coverage on CBU0, and the estimator that
produces them has no free parameters to misconfigure.

## Lag-Llama — no defect found, and the reason is on the record

This is the flag that had no verdict, and it was carried as an open anomaly for
most of the audit before being resolved here. π̂ = 0.0294 is 2.9× nominal, the
worst of any correctly-implemented forecaster in the panel.

Evidence gathered:

| quantity | value | reading |
|---|---|---|
| predictive std / realised σ | 0.799 | 20% under-dispersed |
| median VaR_0.01 / σ | −2.085 | shallower than Gaussian's −2.326 |
| distinct VaR values | 100% of observations | no truncation, no atoms |
| π̂(0.10)/π̂(0.01) | 6.15 | responds to α normally |
| π̂/α across levels | 2.94, 2.64, 2.26, 1.81 | proportional, not collapsing |
| sampling call | `predict(ds, num_samples=1000)` | no `top_k`, no temperature |

The truncation signature is absent on every axis on which Chronos showed it. The
Chronos defect produced 50 distinct values out of 1000 draws, dispersion 0.117,
and a violation rate almost flat in α (ratio 1.08). Lag-Llama produces a distinct
value on every date, dispersion 0.799, and a ratio of 6.15. It is under-dispersed
uniformly rather than collapsed.

The sampling call carries no truncation parameter, so there is no configuration
analogue to disable. 1000 samples does place the 1% quantile at the 10th order
statistic, which carries Monte Carlo error — but that adds noise symmetrically
and cannot produce a systematic 20% narrowing.

**Verdict: no defect traced.** Lag-Llama's predictive distribution is too narrow
by about a fifth, uniformly across quantile levels, with no configuration
parameter implicated and no truncation signature. Under the pre-registered
protocol this is a false positive for Kupiec, and it is recorded as such.

The honest limit of this verdict: "no defect found" is not "no defect exists".
Establishing that would require the same depth of audit the labelled forecasters
received — reading the model's predictive distribution directly, as was done for
Chronos — and that has not been done for Lag-Llama. What can be said is that the
three mechanisms already identified in this project do not account for it.

---

## Consequence for the reported specificity

Kupiec's specificity was recorded as 2/8. All six flags are now traced and none
is a defect, so the figure stands as measured: Kupiec fires on ordinary
under-coverage, which is what it was designed to do, and cannot separate that
from a defect.

q̂_V produced no unlabelled flags, so its 8/8 specificity rests on no traced
cases. With eight unlabelled forecasters the correct statement remains that no
false positive has been found, not that none exists — and this is now moot for
the paper, since B1 established that q̂_V is a rank-preserving transform of the
violation rate and therefore adds nothing over it.
