# Technical pilot for native-tail candidates

Author requested testing other models on 10 September 2026 after excluding
central-grid completion and half-normal tail constructions. Record this
protocol before producing any candidate forecast. This is an implementation
and runtime pilot, not a coverage or model-ranking experiment.

- Candidates: amazon/chronos-2 and thuml/sundial-base-128m, resolved to
  immutable checkpoint revisions before loading. Save code, configuration,
  weight hashes, package versions and seeds.
- Inputs: the existing archived, unfiltered SP500 and BTC log returns.
- Forecast dates: the last 32 observed dates through 31 August 2026, for
  each asset; 512 preceding observations, excluding the target return.
- Horizon: one next observed return. No cross-series inputs, fine-tuning,
  new return downloads, distributional completion, clipping or data filtering.
- Chronos-2: use native output quantiles and select its exact 0.01 entry.
  Save the entire native quantile vector for each date.
- Sundial: save 1,000 generated one-step samples per date. Read the first
  predicted position from its native patch output; calculate the 0.01
  quantile with the same linear empirical-quantile convention as the paper.
  On the last two dates per asset additionally save pools of 10,000 draws,
  for a descriptive sampling-stability check. A pool is not ground truth.
- Deterministic seeds: 20260910 plus asset/date/batch offsets recorded in
  output metadata. Repeat first and last contexts to check deterministic
  replay under each model's recorded backend and seed.
- Validate finite outputs, native quantile ordering, shape, inverse
  preprocessing, use of past-only observations and repeatability. Record
  all failures and runtime; do not change model settings in response to QS.
- A 32-date sample cannot assess 1% coverage. Do not use its hit counts or
  loss to admit/reject a candidate. Full evaluation needs the original
  complete per-asset forecast/calibration/test design before comparison.

TabPFN variants with a FullSupportBarDistribution half-normal construction
are excluded under the author's criterion; no hand-patched tail is tested.

## Full Chronos-2 evaluation after the technical pilot

Before inspecting any full-history candidate losses, extend Chronos-2 to
all 24 archived assets. Its technical pilot passes and indicates feasible
runtime. Forecast every date after the 512-observation warm-up through
each asset's August endpoint, using batch size 16, separate univariate
tasks and `cross_learning=False`. Save all 21 native quantiles; use the
native 0.01 entry. Record crossings and failures without dropping rows.

The first 70% of available forecast-return pairs calibrates a static
conformal shift; evaluate on the remaining 30%. Also evaluate the original
250-score rolling shift, using only preceding scores. No tuning, new gates
or added tail fit. Use existing strict-hit and pinball-loss definitions.
Report per-asset and equal-asset summaries separately from the canonical
168-pair paper until the new model's remaining comparator evaluations are
complete. Repeat full inference in a fresh process and require exact replay
on the recorded backend. This is retrospective evidence with unresolved
pretraining overlap, not an independent prospective test.
