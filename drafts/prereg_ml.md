# Pre-registration: an ML family as a second reduction-step contrast

Written and committed **before any model is fitted**. Nothing below is revised
after a result is seen; revisions, if any, are appended with a timestamp and a
reason, never edited in place.

The object of this exercise is **not** where gradient boosting ranks against
GARCH or against a foundation model. It is whether a second, non-generative model
family exhibits the same structure as Section 4: a library default in the
reduction step that moves the 1% tail while leaving the centre of the predictive
object alone. The row that matters is the **pair**, not the forecaster.

---

## P1. Unit of analysis, and what varies between rows

PROTOCOL.md Rule 1 requires both fields. Declared here, checked against the data
before any result is read.

| table produced | unit | expected rows | what varies between rows |
|---|---|---|---|
| dose–response (E1) | model-config × asset × date cell | 2 models × 5 grid points × 4 assets × 200 dates = **8,000** | the leaf-size parameter and the asset; context, seed, features and split held identical |
| dose–response summary | model-config × asset | 2 × 5 × 4 = **40** | as above, aggregated over dates |
| full panel (E2) | series × asset | 2 models × 2 configs × 24 assets = **96** | the model, the config, and the asset; nothing else |
| blind gate (E3) | series | 2 models × 2 configs = **4** | the model and the config |
| backtests (E3) | series × asset | **96** | as the panel |

The four ML series join the sequence panel, taking it from 13 to **17** series
and from 312 to **408** cells. The main panel goes from 16 to **20** forecasters
and from 384 to **480** pairs. Every "N of M" in the manuscript is recounted
against these numbers (R1) — nothing is carried over.

## P2. Models, and what is excluded

**In:** LightGBM pinball quantile regression (`objective="quantile"`), and
quantile random forest (scikit-learn `RandomForestRegressor` with the empirical
conditional distribution of in-leaf training residuals).

**Excluded, with the reason written now:**

- **XGBoost.** Same class as LightGBM — gradient-boosted trees under a pinball
  objective. A second member adds a row and no contrast, which is the design this
  paper explicitly declined.
- **Neural forecasters (LSTM, TCN, N-BEATS, DeepAR).** They would be a third
  contrast, not a second, and the reduction step they expose is a sampler —
  already covered by Chronos, on a public checkpoint, without our data. Adding
  them lengthens the panel and repeats an argument already made cleanly.
- **Any model requiring a tuning search.** P3 permits exactly one parameter to
  differ between the paired configurations. A tuned model is a different object
  from its default, and the pair would then vary in more than one place.

## P3. The pair is the object

Each model enters **twice**, and the two entries differ in exactly one parameter:

| model | default configuration | tail configuration | the single parameter |
|---|---|---|---|
| LightGBM | library default | leaf size lowered | `min_data_in_leaf` (default 20) |
| Quantile RF | library default | leaf size lowered | `min_samples_leaf` (default 1) |

Features, rolling windows, seeds, re-estimation schedule and the calibration/test
split are **identical** between the two members of a pair. Any difference in
tail behaviour is therefore attributable to that one parameter, in the same way
that `top_k` is the only difference between the two Chronos configurations.

Note the asymmetry with Chronos, and it is stated rather than smoothed: for
LightGBM the library default is the *coarser* setting, while for the quantile
forest the default (`min_samples_leaf = 1`) is the *finest*. If the mechanism is
leaf resolution, the two families should move in **opposite** directions from
their defaults. That is a sharper prediction than "the default is bad" and it is
recorded here as such.

## P4. Features, fixed now

Lagged returns at 1–5 days; realised volatility over 5, 22 and 250 days; the sign
of the lagged return. Nine features. No selection, no engineering, no tuning of
anything except the parameter named in P3.

## P5. Estimation regime, and what it does to the paper's asymmetry

Per asset, rolling window of 1,000 observations, re-estimated every **K = 25**
trading days. Refitting at every step is not feasible for 96 series and is not
needed: the window moves by 2.5% between refits.

**These models join the FITTED group**, alongside GARCH and CAViaR. That
**deepens** the zero-shot-versus-fitted asymmetry Reviewer 1 objected to; it does
not resolve it. Section 3.3.3 is updated to say so explicitly, and no claim is
made that adding them answers that objection. Cross-family rankings stay in the
supplement (R3).

## P6. Dose–response, readings fixed before the run

Vary the leaf-size parameter over **{1, 5, 20, 100, 500}** at identical features,
seeds and windows, on four assets of unlike character (SP500, GOLD, BTC, EURUSD),
200 evenly spaced dates. Three readings, fixed now:

1. **Monotone response.** If the mechanism is leaf resolution, predictive
   dispersion and π̂ move monotonically in the parameter.
2. **Flat response to the documented knobs.** Learning rate, number of trees and
   maximum depth — the parameters an analyst would sweep — do not move the tail.
3. **Centre unchanged.** The predictive median is materially unchanged across the
   whole grid.

## P7. Interpretation, committed in both directions

**If the default degrades the 1% tail materially** (π̂ at least twice nominal at
the default and near nominal at the tail configuration, holding the median fixed):
a second instance of the paper's thesis, from a non-generative family that ships
no sampler. It enters Section 4 or 5 as a second exhibit, and the claim
strengthens from "one public checkpoint" to "two families, two mechanisms".

**If the default is clean at 1%** (π̂ within a factor of 1.5 of nominal at every
grid point): a **negative result**, reported as such, and it *bounds* the thesis
rather than weakening it. The bounded claim is: not every library default reaches
the tail — only those governing the **support or the resolution** of the
predictive object. A pinball-loss learner that estimates the quantile directly
has no support to truncate, and the absence of an effect is evidence for that
reading. **The section is not cut in this case.** A negative result obtained
under a pre-registered design is worth more than a positive one obtained by
searching.

**Intermediate outcomes** — a monotone but small effect, or an effect that does
not survive the tail configuration — are reported as intermediate. No threshold
is moved after the fact.

## P8. The gate runs blind, and this is its first out-of-sample test

The structural gate of Section 7 is run on the four new series **before** any
coverage, Kupiec or Basel figure is computed for them. Its verdict is written to
`analysis/ml/GATE_BLIND.md` with a timestamp, and that file is committed before
the backtests are run. This is the only way the gate's performance on these
series is out-of-sample: every check it contains was written against failure
modes found in 2026, and these series did not exist then.

Scoping, declared in advance:

- **Tail reach is inapplicable.** There are no sample paths; a pinball learner
  returns a quantile and a forest returns an in-leaf empirical distribution. It
  is recorded *inapplicable* in the manifest, not as a pass.
- **Support cardinality is expected to be inapplicable for the same reason it is
  for Historical Simulation**: a leaf-based estimator takes finitely many values
  by construction, bounded by the number of distinct leaves. This is checked
  before the run and, if confirmed, declared inapplicable rather than failed.
- The remaining eight checks apply and are reported as verdicts.

## Execution order

E1 dose–response on four assets → readings (i)–(iii) checked → **stop and report
if they do not hold**, because then the assumed mechanism is wrong and the design
is rebuilt rather than extended. E2 full panel. E3 blind gate, then backtests.
E4 manifest entry per series.

## Reporting

Dose–response report and panel report are **separate documents**, so that the
readings can be seen to have been fixed before the panel was run. Everything
recounted (R1); every number a macro (R2); cross-family rankings in the
supplement (R3); the in-pair contrast is the body result.
