# What Backtests Cannot Detect: Structural Validation of Value-at-Risk Pipelines

*Draft abstract and introduction. Every number is marked **[panel]** (24 assets)
or **[SP500]** (single asset, provisional pending a panel run). Nothing here
requires the recalibration machinery of the parent project.*

---

## Abstract

Every tail-risk pipeline converts a model's predictive output into a threshold:
sampled paths, a grid of predictive quantiles, a fitted parametric law or an
order statistic, reduced to a single number. We show that this conversion step
is where such pipelines fail, that the failures are not detectable by the
backtests applied to them, and that post-hoc calibration conceals rather than
exposes them.

We audit a twenty-four asset, sixteen-forecaster Value-at-Risk panel. Five
defects that corrupted a reported result are traced to specific lines of code:
two in foundation models, one in a classical GJR-GARCH benchmark, two in data
handling. The classical one is the instructive case. A Student-*t* quantile with
its degrees of freedom hard-coded at five and the distribution left
unstandardised inflated every threshold by 45%, produced a 0.42% violation rate
against a nominal 1%, and placed the benchmark in the Basel green zone on all 24
assets. Correcting only that quantile map, changing nothing else, moves the rate
to 1.94%. A defect that flatters its owner attracts no scrutiny, and this one was
found last.

We develop the argument on the cleanest case, which is also the only one in
someone else's published code and therefore reproducible without our data.
Chronos-T5 emits a categorical distribution over 4093 quantised value bins.
Its packaged sampling routine defaults to `top_k = 50`, truncating the support
to 1.2% of its bins before any quantile is computed. The consequence is
invisible where forecasting is usually evaluated and fatal where risk is
measured: predictive dispersion falls to 0.117 of realised volatility **[SP500]**,
and the empirical violation rate of a nominal 1% Value-at-Risk forecast reaches
0.3884 for Chronos-Small and 0.4188 for Chronos-Mini across 24 assets **[panel]**
— a thirty-nine-fold exceedance that would place every asset in the Basel red
zone.

We establish the mechanism rather than infer it. A controlled dose–response on
identical contexts and seeds recovers dispersion monotonically as the truncation
is relaxed — 0.110, 0.336, 0.880, 0.925 at `top_k` of 50, 200, 1000 and 4094
**[SP500]**. One thousand draws contain exactly 50 distinct values.

A practitioner conducting a responsible sensitivity analysis would not find
this. Varying temperature across a fourfold range moves dispersion from 0.110 to
0.110; nucleus sampling moves it to 0.105. The parameters that are documented,
discussed and routinely tuned are precisely the ones that do not matter here,
and the analysis would correctly report the model as robust.

Because the predictive distribution at horizon one is categorical and available
in closed form, sampling is unnecessary. We give an estimator that reads
quantiles directly from the model's logits: softmax, map token identifiers to
bin centres, scale, and invert the cumulative distribution. It agrees with
full-vocabulary sampling to within Monte Carlo error (dispersion 1.041 against
1.038; identical π̂ at every level **[SP500]**), has no `num_samples`, no
`top_k`, no seed, and costs one forward pass. Under it the violation rate falls
to 0.0175 and 0.0178 **[panel]** — a factor of twenty-two.

**The default was most of the anomaly, but not all of it.** At 1.75× nominal the
corrected forecasts still fail unconditional coverage on 16 of 24 assets, and we
state this rather than let it be found: Chronos passes the Kupiec test on 8 of
24 after correction. That number is unremarkable in context — GARCH-N also
passes on 8, EWMA on 5, and the best forecaster in the panel, a 2004 dynamic
quantile regression fitted per asset, passes on 15. Correcting the default does
not make Chronos well calibrated. It makes Chronos statistically
indistinguishable from a standard GARCH benchmark, having been worse than one by
a factor of twenty before.

Two features of this failure make it more general than one parameter in one
library. First, it is invisible to the diagnostics normally applied — and we
show those diagnostics are weak precisely where regulation relies on them. On
our 312-pair panel the Christoffersen independence test is *undefined* on 53.5%
of pairs at α = 0.01, 26.6% at 0.025, 4.5% at 0.05 and 0.0% at 0.10, because a
1% tail generates too few exceedances to populate a transition table. Over the
same panel unconditional coverage rejects 70% of model–asset pairs at α = 0.01
and 39% at α = 0.10, and no forecaster of the sixteen passes on more than 15 of
24 assets at the 1% level. A test that is undefined on half its applications and
one that rejects the majority of reasonable models are both statements about the
instrument, not about the forecasters. Second, the defect is concealed rather
than exposed by post-hoc calibration. Applying split-conformal recalibration to the
defective series restores near-nominal marginal coverage (π̂ = 0.0108, 19 of 24
assets in the Basel green zone **[panel]**), because the conformal guarantee is
agnostic to the quality of the forecaster it corrects. A defect that survives
every coverage-based backtest is not detectable by coverage.

We therefore propose validating predictive distributions structurally before
they are calibrated or evaluated, and give ten such checks. They are cheap, need
no re-inference, and no access to the original hardware. On our panel they
identify the truncation, and separately identify a sign inversion and an
optimiser failure that no backtest flagged. We also document a data-quality
regime in which the same underlying defect — an era in which an asset did not
trade — propagates through a nonparametric estimator, a parametric GARCH
variant and a foundation model as three unrelated-looking pathologies.

---

## 1. Introduction

Consider a Value-at-Risk study across 24 assets and four quantile levels, in
which the forecasters differ in how they express a predictive distribution. Some
return sampled paths; some return a grid of predictive quantiles; some return a
fitted parametric law. The sample-based forecasters come back with a 39%
violation rate at a nominal 1%. The grid-based forecasters come back with 99%.
One of the parametric benchmarks comes back at 0.4%, comfortably inside the
Basel green zone on all 24 assets. Every one of these results is stable across
assets, survives the robustness checks applied to it, and supports an immediate
and interesting conclusion about the family it belongs to.

All three conclusions are wrong. The 39% is a sampling parameter defaulting to a
value inherited from text generation, which discards 98.8% of the model's
predictive support before any quantile is read. The 99% is a sign convention
applied where a quantile grid is converted into a threshold, in two of ten
forecast files, consumed by an evaluation routine that assumed the opposite
convention. The 0.4% is a Student-*t* quantile with its degrees of freedom
hard-coded at five and the distribution left unstandardised, which inflates
every threshold by 45% and buys a green zone the model has not earned. None is a
property of any model, and the third flatters whoever built it rather than
embarrassing them, which is why it was found last.

What makes this worth a paper is not that a pipeline had bugs. It is that the
study design could not have distinguished a pipeline defect from a model
property, and that the evidence which normally increases confidence in a finding
is exactly what a systematic defect produces. A defect at an interface applies to
every model sharing that interface and to every asset equally, so it presents as
a clean, consistent, mechanistically plausible result. Cross-asset consistency,
the usual defence against a spurious finding, is here the signature of the
artefact. Each defect also had a ready economic story: sample-based forecasters
are overconfident because pretraining on non-financial series fails to transfer;
grid-based forecasters are miscalibrated because a discretised quantile grid
cannot represent extreme tails. Both stories are publishable. Both are false.

It would be convenient to conclude that this is a foundation-model problem —
that new architectures arrive with unfamiliar defaults and the remedy is caution
with unfamiliar tools. The classical benchmarks in the same panel refute that.
The GJR-GARCH case above is not a foundation model and not new: a hard-coded
Student-*t* quantile, unstandardised, in a specification that has been standard
since Glosten, Jagannathan and Runkle (1993).

We are deliberately careful about the count. Five defects corrupted a reported
number; two of those are in foundation models, one in a classical benchmark and
two in data handling. Four further problems surfaced in the same audit and did
NOT corrupt a reported number, and we separate them rather than inflate the
tally with them: an EWMA whose documentation describes a truncated weighted sum
while its data came from the RiskMetrics recursion (a documentation defect worth
1.2e-10 in VaR); a GARCH benchmark that does not reproduce across library
versions to better than 4% of its own Value-at-Risk (a limit on reproducibility,
not an error); and two failures in benchmarks we constructed during this audit —
an unguarded optimiser returning a Value-at-Risk of 7.6e8 on an asset with 0.3%
daily returns, and a skewed-*t* variant diverging on the same windows — both of
which the validation procedure of Section 6 blocked before promotion. Those last
two are evidence that the procedure works, not evidence that a pipeline failed,
and reporting them as found defects would repeat precisely the error this paper
is about.

The common factor is not the model class. It is the step at which a predictive
object — a sample, a quantile grid, a fitted law, an order statistic — is
converted into a threshold. Every pipeline performs that conversion, each output
representation invites its own convention, and nothing in the standard
evaluation apparatus checks that the convention used to write a series matches
the one used to read it. Foundation models matter here only because they
introduce new output representations faster than conventions form around them,
which makes them a leading indicator rather than the subject.

We develop the argument on the truncation defect, for two reasons. It is the
only one located in someone else's published code, so it reproduces on any
machine from a public checkpoint without access to our data. And its magnitude
is large enough to be unambiguous, which the subtler defects are not.

### 1.1 The defect

Chronos-T5 (Ansari et al., 2024) tokenises a real-valued series into a fixed
vocabulary and forecasts by autoregressive sampling over it. At a horizon of one
step the model emits a single categorical distribution over 4093 quantised value
bins, which is exactly the object a tail-risk forecast requires. The packaged
`predict` routine, however, samples that distribution under a default
`top_k = 50` inherited from text generation, where restricting attention to the
fifty most probable continuations is a sensible fluency heuristic. Applied to a
distribution over numerical magnitudes it discards 98.8% of the support, and it
discards it from the tails inward.

The effect is precisely localised. The median is unaffected; a point-forecast
evaluation on this configuration is entirely normal. The damage is confined to
the region that risk measurement reads.

### 1.2 What we do

We treat the anomaly as a defect to be traced rather than a property to be
interpreted, and this distinction is the methodological point of the paper. The
observed behaviour — a predictive distribution far narrower than realised
volatility — admits an economic reading: that the model is overconfident, that
zero-shot forecasters underestimate tail risk, that pretraining on
predominantly non-financial series fails to transfer. Each is plausible, each is
publishable, and each is wrong here.

Section 3 fixes the readings in advance and runs a controlled experiment.
Contexts, seeds and model weights are held fixed and only the sampling
configuration varies. Dispersion recovers monotonically with the truncation
parameter and is unmoved by temperature or nucleus sampling. One thousand draws
under the default contain exactly fifty distinct values, which is the signature
of the mechanism rather than evidence about the model.

Section 4 removes the sampler. At horizon one no sampling is needed: the
predictive law is categorical and its quantiles are available in closed form.
The analytic estimator is validated against full-vocabulary sampling before it
is used, on the principle that a reconstruction of a model's support must be
checked against the model before conclusions are drawn from it.

Section 5 asks why the defect survived. It survived because it was hidden by the
correction applied on top of it. Split-conformal recalibration guarantees
finite-sample marginal coverage regardless of the forecaster (Vovk et al., 2005;
Lei et al., 2018), and it delivers: applied to the truncated series it restores
near-nominal coverage and Basel green-zone status. Every coverage-based backtest
in our own earlier work therefore passed. We show the same holds in a more
extreme case still — a forecaster whose sign was inverted, so that its
thresholds pointed at the wrong tail, is recalibrated to π̂ = 0.0146 and 19 of
24 green **[panel]** — while its response to volatility remains exactly
inverted, corr(VaR, σ) = +0.530 against −0.530 for the corrected series,
unanimously across assets. Marginal coverage is restored; conditional coverage
is not merely absent but reversed.

Section 6 draws the operational conclusion. If coverage cannot detect these
defects, validation must not be based on coverage. We give ten structural checks
on a forecast series — sign, monotonicity across levels, scale against realised
volatility, response across quantile levels, information alignment, dispersion,
distinct-value cardinality, tail reach, extremes, and coverage bands — with the
scoping each requires (an order-statistic estimator may legitimately tie across
adjacent levels; a discrete predictive support may too). Each check in the
resulting gate exists because something got past its absence.

### 1.3 The standard tail diagnostics are weak instruments

The defect survived because the tests that should have caught it are least
informative exactly where it lived. This is measurable on our own panel and is a
contribution independent of any defect.

The Christoffersen (1998) independence test conditions on the transition counts
of the exceedance indicator. At α = 0.01 a typical test window contains too few
exceedances to populate the two-by-two table, and the statistic is undefined —
on 53.5% of the 312 model–asset pairs after recalibration, and 34.6% before.
The proportion falls monotonically with the tail probability: 26.6% at α = 0.025,
4.5% at 0.05, and 0.0% at 0.10. A degenerate table is an absence of evidence, and
the distinction matters operationally, because the natural implementation counts
it as a pass. The submitted version of our own earlier work did exactly that in
one table and not in another, and the two tables disagreed for that reason alone.

Unconditional coverage is better defined but no more discriminating. Kupiec's
(1995) test rejects 70% of model–asset pairs at α = 0.01 against 39% at α = 0.10,
and the best of sixteen forecasters passes on only 15 of 24 assets at the 1%
level — a per-asset dynamic quantile regression from 2004, ahead of every
foundation model in the panel. When a test rejects the majority of defensible
models, its failures carry little information about any one of them.

Both instruments are therefore weakest at α = 0.01, which is the level at which
the Basel traffic-light system operates. This is the deeper reason a structural
check is needed rather than a better backtest: the backtests are not being
misapplied, they are being applied where they have the least power.

### 1.4 One defect can present as several

The converse of a defect that mimics a finding is a defect that fragments into
several. Beyond the two interface failures above we trace a third pattern, in
which a single data-quality fact appears as three unrelated pathologies. We document a period in which two of our
assets barely traded, with up to 97.7% of daily returns exactly zero. The same
underlying fact appears as a foundation model correctly forecasting no movement
and taking a 41% violation rate; as a GJR-GARCH-*t* optimiser failing to
converge on 41% of estimation windows; and as a skewed-*t* variant returning
quantiles of order 10⁶. Three model families, three apparently unrelated
pathologies, one data-quality defect, all vanishing once the estimation window
clears the period.

### 1.5 What this paper does not claim

We do not claim that foundation models are well calibrated for tail risk. Read
through their own predictive distributions they under-cover at the 1% level by a
factor of roughly 1.4 to 2.9 on our panel, as do the classical benchmarks. We do
not claim the truncation is the only such defect, nor that our checks are
complete — three of the ten exist because a defect got past the other seven. We
do not claim `top_k = 50` is wrong as a library default; it is a reasonable
default for the task it was chosen for, and unreasonable only for the task it is
being borrowed into.

The claim is narrower and, we think, harder to dismiss: a default that is
invisible under standard evaluation, and that post-hoc calibration will conceal
rather than reveal, changes a tail-risk conclusion by a factor of twenty-two.

---

## Data note (for §2), stated in full

Two series contain pre-2015 periods in which the asset barely traded. Measured
over **all return observations in the calendar year**, CBU0 is 78.8% exactly-zero
in 2013 and 45.5% in 2014; IBGL is 22.5% and 20.2%. Measured instead over **only
those dates carrying a Chronos forecast** — which begin 2013-05-01, after the
512-observation context requirement — CBU0's 2013 figure is 97.65%, because the
staleness is concentrated in the second half of that year (53.6% January–June,
97.7% July–December). Both figures are reported in this paper where each is the
relevant one; the screen below is defined on the calendar-year measure, since it
is a property of the returns and must not depend on which model consumes them.
These periods are excluded by a rule stated on the input data — trailing
250-day zero-return fraction above 0.20 — applied identically to every asset and
every model, removing 719 of 134,211 observations (0.54%).

The rule was written after observing that CBU0 failed a structural coverage
check, and is therefore not pre-registered; it is disclosed as post-hoc. It does
land in an empty region of the panel's own distribution: across 525 asset-years,
the 10–20% band contains none, and on the trailing measure the nearest excluded
series sits at 0.404 and the nearest retained series at 0.112.

The margin is real but the symptoms are a gradient rather than a cliff, and the
boundary does not partition them cleanly:

| | GJR-*t* degenerate windows | divergences | Chronos π̂(1%) |
|---|---|---|---|
| CBU0 (excluded) | 214/3515 | 52 | 0.0559 ✗ |
| IBGL (excluded) | 109/4348 | 1 | 0.0135 ✓ |
| DJCI (retained) | 9/2572 | 1 | 0.0169 ✓ |

Only CBU0 is severely affected across all three model families. IBGL's exclusion
is not corroborated by its Chronos coverage, which is normal, and DJCI — which
is retained — shows the same pathologies as IBGL at lower intensity. We keep the rule as stated rather than narrowing it to CBU0, because a
criterion defined on the data and applied blind to symptoms is defensible in a
way that "exclude the asset that showed problems" is not. In the event the
choice does not matter, which is the reason to report it three ways rather than
argue for one:

| forecaster | full panel | stated screen | CBU0 dropped entirely |
|---|---|---|---|
| Chronos-Small, `top_k=50` | 0.3884, 0/24 | 0.3882, 0/24 | 0.3899, 0/23 |
| Chronos-Mini, `top_k=50` | 0.4188, 0/24 | 0.4192, 0/24 | 0.4188, 0/23 |
| Chronos-Small, analytic | 0.0175, 8/24 | 0.0174, 8/24 | 0.0178, 7/23 |
| Chronos-Mini, analytic | 0.0178, 8/24 | 0.0177, 8/24 | 0.0181, 7/23 |
| GARCH-N | 0.0193, 8/24 | 0.0192, 8/24 | 0.0195, 7/23 |

(π̂ at α = 0.01, Kupiec passes out of assets.) Every entry is stable to the
third decimal and no Kupiec count moves except by the removal of CBU0 from the
denominator. No conclusion in this paper depends on the exclusion.

## Provisional items requiring a panel run

| finding | current basis | needed |
|---|---|---|
| dose–response across `top_k` | SP500, 20 dates | 3–4 assets, ~200 dates |
| temperature / top-p invariance | SP500, 20 dates | same run |
| exactly 50 distinct values | SP500 | same run (mechanically necessary, but show it) |
| analytic vs sampled agreement | SP500, 40 dates | 3–4 assets |
| violation rates, R̄, Kupiec | **24 assets — final** | — |
| recalibration masking | **24 assets — final** | — |
| ±0.530 volatility response | **24 assets — final** | — |
| stale-period propagation | **CBU0, IBGL — final** | — |

## References to verify before they enter the .bib

Ansari et al. (2024) Chronos; Vovk, Gammerman & Shafer (2005); Lei, G'Sell,
Rinaldo, Tibshirani & Wasserman (2018). The impossibility of distribution-free
conditional coverage needs a primary citation for §5 — candidates are Vovk
(2012), Lei & Wasserman (2014) and Foygel Barber, Candès, Ramdas & Tibshirani
(2021), none of which have been checked yet. **No citation enters the .bib
without passing `scripts/audit_bib.py` first.**
