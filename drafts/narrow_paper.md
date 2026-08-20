# A Sampling Default Silently Truncates Tail-Risk Forecasts

*Draft, §1–§6. Every number here comes from a public checkpoint or from series
this paper regenerates; nothing depends on the parent project's panel design,
its recalibration method, or its conformal statistic.*

---

## Abstract

A time-series foundation model that reports a predictive distribution is
increasingly used to produce the quantiles from which risk measures are read.
We show that the packaged inference routine of a widely used checkpoint family
truncates that distribution before any quantile is computed, that the truncation
is invisible under standard forecast evaluation, and that it changes a
Value-at-Risk conclusion by a factor of twenty-two.

Chronos-T5 emits a categorical distribution over 4093 quantised value bins. Its
`predict` routine defaults to `top_k = 50`, a value inherited from text
generation, discarding 98.8% of the support and discarding it from the tails
inward. The median is unaffected: a point-forecast evaluation on this
configuration is entirely normal. Across 24 assets the empirical violation rate
of a nominal 1% Value-at-Risk reaches 0.3884 for Chronos-Small and 0.4188 for
Chronos-Mini — a thirty-nine-fold exceedance placing every asset in the Basel
red zone.

We establish the mechanism by controlled experiment rather than inference. On
identical contexts and seeds, predictive dispersion recovers monotonically as
the truncation is relaxed — 0.12, 0.33, 0.95, 1.06 of realised volatility at
`top_k` of 50, 200, 1000 and 4094, on four unlike assets — and one thousand
draws under the default contain exactly 50 distinct values. A practitioner
running a sensitivity analysis over the parameters that are documented and
discussed would find nothing: temperature moves dispersion by 0.001 across a
fourfold range, and nucleus sampling by 0.006. The one parameter that matters is
the one nobody varies.

Sampling is also unnecessary. At a horizon of one step the predictive law is
categorical and its quantiles are available in closed form. We give an estimator
that reads them directly from the model's logits — softmax, map token
identifiers to bin centres, scale, invert the cumulative distribution — which
agrees with full-vocabulary sampling to within Monte Carlo error and has no
`num_samples`, no `top_k`, no seed, and costs one forward pass. Under it the
violation rate falls to 0.0175 and 0.0178.

The residual is stated rather than left to be found. At 1.75× nominal the
corrected forecasts still fail unconditional coverage on 16 of 24 assets. That
is unremarkable in context: on the same panel a GARCH benchmark with a Gaussian
innovation fails on the same 16, and no forecaster we examine passes on more
than 15 of 24 at the 1% level. Correcting the default does not make Chronos well
calibrated for tail risk. It makes Chronos indistinguishable from a standard
volatility model, having been worse than one by a factor of twenty.

---

## 1. Introduction

A risk manager needs one number: the threshold a return will breach with
probability α. A modern forecaster does not produce that number. It produces a
predictive object — a set of sampled paths, a grid of quantiles, a fitted
parametric law — from which the number must be extracted. This paper is about
that extraction step, and about a defect that lives in it, is invisible to the
evaluation ordinarily applied, and is large.

The defect is a default argument. Chronos-T5 (Ansari et al., 2024) tokenises a
real-valued series into a fixed vocabulary and forecasts autoregressively over
it. At a horizon of one step the model emits a single categorical distribution
over 4093 quantised value bins, which is precisely the object a tail-risk
forecast requires. The packaged `predict` routine samples that distribution
under a default `top_k = 50`: attention is restricted to the fifty
highest-probability tokens before sampling begins. In text generation, where the
parameter originates, this is a sensible fluency heuristic — it suppresses
implausible continuations. Applied to a distribution over numerical magnitudes
it discards 98.8% of the support, and because the discarded tokens are the
low-probability ones, it discards the tails.

The consequence is precisely localised, which is why it survives. The mode is
untouched. The median is untouched. Mean absolute error, root mean squared
error, and every point-forecast diagnostic behave normally, because the fifty
retained bins are exactly the ones that carry the central mass. Only the region
a risk measure reads is destroyed, and only the evaluation a risk measure uses
would reveal it.

### 1.1 Why a default rather than a bug

Nothing in the checkpoint is broken and nothing in the library is incorrectly
implemented. `top_k = 50` is a reasonable default for the task the parameter was
designed for, and the routine does exactly what it documents. The defect is
entirely in the transfer: a parameter whose meaning is calibrated to one problem
retains its value when the object being sampled changes from a distribution over
tokens to a distribution over magnitudes.

This matters for how the finding should be read. It is not a criticism of the
checkpoint's authors, and we make no claim that any published result is affected
— we have not audited any. It is an argument that the conversion from a
predictive object to a risk number is a step at which assumptions travel
silently, and that it deserves the explicit validation that estimation and
backtesting already receive.

### 1.2 What we do

Section 2 establishes the mechanism by controlled experiment. Contexts, seeds
and weights are held fixed and only the sampling configuration varies. The
readings were fixed in advance.

Section 3 removes the sampler entirely. At horizon one no sampling is needed:
the predictive law is categorical and its quantiles are available in closed
form. The analytic estimator is validated against full-vocabulary sampling
before any conclusion is drawn from it, on the principle that a reconstruction
of a model's support must be checked against the model.

Section 4 gives the consequences for tail-risk evaluation across 24 assets and
four quantile levels, including the residual miscalibration that survives the
correction.

Section 5 asks why standard diagnostics did not reveal this, and finds that at
α = 0.01 they largely cannot: the Christoffersen independence test is undefined
on 53.5% of our model–asset pairs, falling monotonically to 0.0% at α = 0.10,
because a 1% tail generates too few exceedances to populate a transition table.
Unconditional coverage is better defined and rejects 70% of pairs at α = 0.01 —
a test that rejects the majority of defensible models carries little information
about any one of them. Both instruments are weakest precisely at the level at
which the Basel traffic-light system operates.

Section 6 proposes validating a predictive distribution structurally, before it
is calibrated or evaluated, and gives ten such checks with the scoping each
requires. We state plainly that these were written after the defect was known
and that their performance on it is therefore in-sample.

---

## 2. The mechanism

### 2.1 Design

The experiment varies one thing. For each of two checkpoints
(`chronos-t5-small`, `chronos-t5-mini`), four assets (SP500, GOLD, BTC, EURUSD)
and 200 evenly spaced dates with a full 512-observation context, we draw 1000
samples at horizon one under eight configurations: `top_k` ∈ {50, 200, 1000,
4094}, `top_p` ∈ {0.9, 0.99} at `top_k = 50`, and temperature ∈ {0.5, 2.0} at
`top_k = 50`. Context, seed and weights are identical across configurations.

Readings fixed in advance: if the truncation is the mechanism, dispersion
recovers monotonically in `top_k` and is unmoved by the other two parameters,
and the number of distinct sampled values under the default equals `top_k`
exactly.

### 2.2 Result

| configuration | dispersion, Small | dispersion, Mini | distinct values |
|---|---|---|---|
| `top_k = 50` (default) | 0.121 | 0.115 | **50** |
| `top_k = 200` | 0.331 | 0.328 | 198 |
| `top_k = 1000` | 0.958 | 0.945 | 500 |
| `top_k = 4094` (full) | 1.087 | 1.059 | 514 |
| temperature 0.5 @ k=50 | 0.120 | 0.114 | 50 |
| temperature 2.0 @ k=50 | 0.121 | 0.115 | 50 |
| `top_p = 0.9` @ k=50 | 0.115 | 0.109 | 45 |

Dispersion is the ratio of predictive standard deviation to the 250-day realised
volatility of the asset; 1.0 is the target.

All three readings hold. Dispersion recovers monotonically and reaches its
target only at full vocabulary. One thousand draws under the default contain
exactly 50 distinct values, on every asset and both checkpoints — the signature
of the mechanism rather than evidence about the model. And the parameters a
practitioner would vary do nothing: a fourfold change in temperature moves
dispersion by 0.001.

That last row is the reason the defect is durable. A sensitivity analysis over
temperature and nucleus sampling — the parameters that are documented, discussed
in the literature, and exposed in every tutorial — returns a flat response and
correctly reports the model as robust.

---

## 3. Reading the distribution instead of sampling it

### 3.1 The estimator

At prediction length one, a Chronos-T5 checkpoint emits logits over its
vocabulary for a single position. The predictive law is therefore categorical
and exactly available:

1. softmax the logits to probabilities over token identifiers;
2. map each identifier to its bin centre, offset by the special-token count;
3. multiply by the tokenizer's own scale, the mean absolute value of the context;
4. sort the support, cumulative-sum the probabilities;
5. the α-quantile is the smallest value whose cumulative probability reaches α.

This has no `num_samples`, no `top_k`, no `top_p`, no temperature and no seed.
It costs one forward pass and is exact given the model's output.

### 3.2 Validation before use

A reconstruction of a model's support is a claim about the model and is checked
against it. Over 40 dates we compare the analytic quantiles against sampling at
full vocabulary with 4000 draws: predictive standard deviations agree to within
0.3%, and the violation rates at all four levels are identical to four decimal
places. The two routes disagree only where Monte Carlo error predicts they
should.

### 3.3 Cost

Across 24 assets and both checkpoints the analytic route required one forward
pass per date — 121,923 dates per checkpoint — and completed in roughly ten
minutes per checkpoint on a laptop GPU, against 1000 sampled paths per date for
the sampling route. Reading the distribution is not only more accurate than
sampling it; at horizon one it is cheaper.

---

## 4. Consequences for tail-risk evaluation

24 assets, four quantile levels, a 70/30 chronological split, and no
recalibration of any kind — these are the raw forecasts.

| | π̂(0.01) | π̂(0.025) | π̂(0.05) | π̂(0.10) | Kupiec passes, out of 24 |
|---|---|---|---|---|---|
| Chronos-Small, default | 0.3884 | 0.3962 | 0.4044 | 0.4177 | 0 / 0 / 0 / 0 |
| Chronos-Small, analytic | 0.0175 | 0.0338 | 0.0577 | 0.1036 | 8 / 10 / 14 / 20 |
| Chronos-Mini, default | 0.4188 | 0.4261 | 0.4340 | 0.4477 | 0 / 0 / 0 / 0 |
| Chronos-Mini, analytic | 0.0178 | 0.0333 | 0.0558 | 0.0989 | 8 / 12 / 17 / 22 |

### 4.1 The default produces a forecast that is not a quantile

The first and third rows are almost flat in α. Between the 1% and the 10% level
the nominal exceedance probability changes tenfold; the realised rate changes by
a factor of 1.08. This is the mechanism showing through: a distribution
truncated to its fifty most probable bins has so little spread that its α
quantiles are nearly coincident, and the resulting series responds to the
quantile level it was asked for almost not at all.

That property is diagnostic in its own right and requires no benchmark, no
backtest and no comparison — a forecast whose exceedance rate is invariant to
its own nominal level is not reporting a quantile. It is the seventh of the
checks in Section 6.

### 4.2 The residual is a tail phenomenon, and shrinks

Read through its own predictive distribution, Chronos is not well calibrated at
the 1% level: 0.0175 against 0.0100 is 1.75× nominal, and unconditional coverage
rejects on 16 of 24 assets. We state this rather than let it be found.

But the residual is not a level shift. It is concentrated in the tail and
declines monotonically as the tail probability rises:

| α | π̂ / α, Small | π̂ / α, Mini |
|---|---|---|
| 0.01 | 1.75 | 1.78 |
| 0.025 | 1.35 | 1.33 |
| 0.05 | 1.15 | 1.12 |
| 0.10 | 1.04 | 0.99 |

At α = 0.10 the analytic forecasts are essentially calibrated — 0.1036 and
0.0989 against a nominal 0.10, passing Kupiec on 20 and 22 of 24 assets. What
remains after the truncation is removed is a specifically *tail* deficiency,
which is what one would expect of a model whose quantisation grid is trained on
predominantly non-financial series, and which is a claim about the model rather
than about its interface.

### 4.3 Context for the residual

At α = 0.01, on the same panel and the same split:

| forecaster | π̂ | Kupiec passes |
|---|---|---|
| GJR-GARCH-*t* | 0.0145 | 13/24 |
| Hist-Sim | 0.0158 | 11/24 |
| **Chronos, analytic** | **0.0175 / 0.0178** | **8/24** |
| GARCH-N | 0.0193 | 8/24 |
| EWMA | 0.0208 | 5/24 |
| Lag-Llama | 0.0294 | 0/24 |

Corrected Chronos is indistinguishable from a Gaussian-innovation GARCH — same
number of passes, a marginally lower violation rate — and is beaten by a
Student-*t* GARCH and by historical simulation. Before the correction it was
worse than all of them by a factor of twenty. Neither the original result nor
its correction supports a claim that foundation models are competitive at the 1%
tail; what changes is that the original result was an artefact and the corrected
one is a measurement.

---

## 5. Why the standard diagnostics did not reveal this

The defect survived because the tests that should have caught it are least
informative exactly where it lives. This is measurable, and it is a property of
the instruments rather than of this defect.

**The independence test is usually undefined.** Christoffersen's (1998) test
conditions on the transition counts of the exceedance indicator. At α = 0.01 a
test window typically contains too few exceedances to populate the 2×2 table,
and the statistic does not exist. Across 312 model–asset pairs it is undefined
on 53.5% at α = 0.01, 26.6% at 0.025, 4.5% at 0.05 and 0.0% at 0.10. Where it is
defined it rejects on 48.3% of pairs at α = 0.01.

A degenerate table is an absence of evidence, and the operational distinction
matters: the natural implementation returns "no rejection", which reads as a
pass. A test that cannot fail on half its applications, and that is reported as
passing on all of them, is not evidence of independence.

**Unconditional coverage rejects almost everything.** Kupiec's (1995) test
rejects 70% of pairs at α = 0.01 against 39% at α = 0.10, and no forecaster we
examine passes on more than 15 of 24 assets at the 1% level. When a test rejects
the majority of defensible models, its failures carry little information about
any one of them — and, crucially for this paper, a genuinely defective series is
not distinguishable from a merely imperfect one by that test.

Both instruments are weakest at α = 0.01, the level at which the Basel
traffic-light system operates. This is the deeper reason a structural check is
needed rather than a better backtest: the backtests are not being misapplied,
they are being applied where they have the least power.

**Nor does recalibration reveal it.** Split-conformal recalibration guarantees
finite-sample marginal coverage regardless of the forecaster (Vovk et al., 2005;
Lei et al., 2018), and it delivers. Applied to the truncated series it restores
π̂ = 0.0108 and places 19 of 24 assets in the Basel green zone. Every
coverage-based backtest downstream of it therefore passes. A correction that
guarantees coverage cannot diagnose the forecaster it corrects, because the
guarantee holds whatever the forecaster is.

---

## 6. Structural validation

If coverage cannot detect these defects, validation must not be based on
coverage. We propose ten checks applied to a forecast series before it is
calibrated or evaluated. Each is a property a Value-at-Risk series must have to
be a Value-at-Risk series, and none requires a benchmark, a test window, or
access to the hardware that produced it.

| check | condition |
|---|---|
| sign | median VaR_α < 0 at every α |
| monotonicity | VaR ordered across α on ≥ 99.9% of days |
| scale | median VaR_0.01 / realised σ ∈ [−3.5, −1.8] |
| α-response | π̂(0.10) / π̂(0.01) ≥ 3 |
| coverage | π̂ ∈ [0.2α, 5α] at every α |
| alignment | &#124;corr(VaR_t, r_t)&#124; < 0.30 |
| dispersion | predictive σ / realised σ ∈ [0.5, 2.0] |
| cardinality | distinct VaR values > 5% of observations |
| tail reach | ≥ 5 distinct sampled values below the α-quantile |
| extremes | max &#124;VaR&#124; ≤ 50× the asset's own median &#124;VaR&#124; |

Two require scoping rather than a verdict. An order-statistic estimator may
legitimately tie across adjacent α and hold constant for long stretches, so
monotonicity is weak and cardinality inapplicable for it; a discrete predictive
support may tie for the same reason. A check that cannot fail informatively
should say so rather than return a verdict.

**These checks were written after this defect was known**, several of them
because a specific defect had already got past the checks that existed at the
time. Their performance on it is in-sample by construction and is not evidence
of their general sensitivity. What can be said is narrower and still useful:
each is cheap, each is a necessary condition rather than a statistical test, and
each states a property that a practitioner would endorse before seeing any data.
On the truncated series, `dispersion`, `cardinality`, `α-response`, `scale` and
`coverage` all fail; on the analytic series none does.

---

*Numbers in §4–§6 come from a 24-asset panel; §2 and §3 reproduce from the
public checkpoints alone.*
