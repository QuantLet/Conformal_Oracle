# Cover letter — International Review of Financial Analysis

Dear Editors,

We submit **"What Backtests Cannot Diagnose: Structural Validation of Tail-Risk
Forecasting Pipelines."**

## This is not the paper we submitted to the International Journal of Financial Economics

We state this at the top because the referee pools of the two journals overlap,
and a reviewer who saw the earlier manuscript should be able to establish in one
paragraph whether this is a resubmission or a different paper. It is a different
paper, and we would rather say so than have it inferred.

The earlier manuscript was rejected for pursuing three claims at once: that
foundation models are miscalibrated in the extreme tail, that a conformal shift
repairs them, and that the shift's magnitude is itself a diagnostic. The referees
were right. We have not repackaged those claims; we have dropped two of them and
replaced the third.

**What the earlier paper claimed, and what became of it.**

| earlier claim | disposition |
|---|---|
| Foundation models fail catastrophically at the 1% level (violation rates near 99% for two of them) | **Withdrawn.** Those rates were a sign-convention error in our own forecast-reduction code. Corrected, the same series run at 0.014 and 0.018, alongside the GARCH benchmarks. |
| The conformal correction's magnitude, `R`, is an audit statistic carrying information the standard backtests do not | **Withdrawn.** Among well-specified forecasters `R` is a rank-preserving transform of the violation rate (Spearman 0.99); a rule built on the rate alone reproduces every figure we attributed to `R`. |
| The predictive interface (quantile grid versus sampled paths) drives tail failure | **Withdrawn.** The corrected panel does not separate the two interfaces, and the within-family comparison we used as a control varies architecture, patching and pretraining corpus simultaneously. |
| Recalibration restores coverage | **Retained, and narrowed to a corollary.** It restores marginal coverage and nothing else, and it is an intervention with an indication rather than a default step. |

**What the paper now claims** is a single identification result, and it is new
work rather than a rewriting. The exceedance sequence a Value-at-Risk backtest
consumes identifies exactly one functional of the joint law of forecast and
return — the tail probability the reported threshold actually cuts off. Every
test measurable with respect to that sequence therefore has power equal to its
size against alternatives agreeing on it, and this covers Kupiec, both
Christoffersen components, the Basel traffic light, and the Engle–Manganelli
dynamic quantile test with any predictable instrument.

We quantify the resulting blind spot with one number, the truncation depth an
indistinguishable alternative can reach, and give an explicit pair attaining it:
under a unimodal return law a Value-at-Risk understated by 49% is exactly
unidentifiable while coverage, exceedance clustering, point-forecast accuracy and
predictive dispersion are all held fixed. Structural checks on the forecast
series reduce the residual to 31% and **do not close it** — a result we report
against our own proposed instrument.

## Why IRFA

The paper is methodological but its object is operational: what a risk function
can and cannot learn from the evidence a backtest produces, and what a capital
requirement computed from a threshold and a violation count does not determine.
The empirical work spans 24 assets and 16 forecasters, and the foundation-model
case study is reproducible from a public checkpoint and published code without
our data. We believe this fits IRFA's interest in risk measurement and in the
validation of forecasting practice more closely than a journal focused on
forecasting methods alone.

## Reproducibility, stated plainly

Every numeric claim in the manuscript is emitted as a macro by a script that
recomputes it from the artefacts and fails the build when a value drifts. Two
build guards run with negative controls: each is required to fail on a case
constructed to make it fail before it is allowed to report a pass.

We should disclose what that discipline found in our own manuscript. Converting
the prose to generated macros — forced conversion, not inspection — surfaced
assertions that no artefact produces, including a cluster-robust p-value whose
correction inverts the conclusion of the paragraph containing it. Seventy-nine
numeric claims sat outside the reach of the checking script, because a value that
is never emitted cannot be checked by a tool that checks emitted values. That is
the same structure the paper attributes to a degenerate test statistic, and we
report it in Section 7 rather than in a footnote. We also report that one of our
own ten structural checks blocks nothing on any series in the panel and therefore
cannot fail informatively.

We would rather submit a paper that says this than one that does not.

## Declarations

The manuscript is not under consideration elsewhere. All data are public: prices
from Yahoo Finance, all model checkpoints open-weight on HuggingFace. Code,
result matrices for all 384 model–asset pairs, and the provenance record are in
the Quantlet repository accompanying the paper. We used large language models for
language editing and code review; all scientific content and conclusions are our
own.

We suggest reviewers with expertise in backtesting and regulatory risk
measurement, conformal prediction under dependence, and time-series foundation
models. We have no objection to reviewers who saw the earlier submission — the
table above is written for them.

Yours sincerely,

Daniel Traian Pele, on behalf of the authors
Bucharest University of Economic Studies
danpele@ase.ro
