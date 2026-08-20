# Phase 0e pre-registration — does q̂_V detect what backtests cannot?

Written and committed **before** any detection rate is computed. Nothing below
is tuned on the labelled set.

## The question

The drafted title asserts that backtests cannot detect these defects. Three
observations suggest the conformal statistic could:

- the sign-flipped series came out at R = 3.18, outside every other forecaster
- after the sign fix, the R > 1 partition contained exactly the two Chronos
  models — the `top_k` truncation, before anyone knew it existed
- GJR-GARCH was the sign outlier, q̂_V < 0 on 23 of 24 assets

Each is a retrospective observation on a set of defects already known. That is
precisely why the rules below are fixed in advance.

## Decision rules (fixed, not tuned)

| Detector | Rule | Predates discovery? |
|---|---|---|
| q̂_V magnitude | R > 1, the paper's existing threshold, unchanged | **yes** |
| q̂_V sign | q̂_V < 0 on a majority of assets for that forecaster | **yes** |
| Kupiec | p < 0.05 | yes |
| Christoffersen | p < 0.05; degenerate cells UNDEFINED, never counted as pass | yes |
| Gate checks | each its own stated condition, one column per check | **no — see below** |

### The asymmetry, carried into every table

The ten gate checks were **written after these defects were known**, several of
them specifically because a particular defect got past the checks that existed
at the time. Their sensitivity on this labelled set is in-sample by construction
and is evidence of nothing. They are reported for completeness and every table
that contains them says so in its header.

The only fair three-way comparison is **q̂_V against Kupiec and Christoffersen**,
all three of which predate the discovery of every defect in the set.

Two further asymmetries, stated so they are not discovered later:

- R > 1 is "the paper's existing threshold", but the paper chose it while the
  defective series were in the panel. It is not tuned on the labels, and it is
  not innocent of them either. A sensitivity curve over R ∈ [0.5, 5] is reported
  alongside the fixed rule.
- The defects were found by a process that included looking at q̂_V. If q̂_V is
  the instrument that found them, high sensitivity is partly circular. Task 4 is
  the test that separates these: whether the flags were legible in the
  *published* artefacts, before anyone was looking.

## Labelling

`defect_label` ∈ {`sign_flip`, `top_k_truncation`, `gjr_quantile_map`,
`stale_price`, `cact_alias`, `ewma_estimator_mismatch`,
`garch_n_irreproducible`, `none`}

`none` means **not known to be defective**. It does not mean clean. Seven
defects were found because someone looked, in one codebase, after a rejection.
The unlabelled forecasters have never been audited to the same depth, so
specificity computed against them is an upper bound on the true false-positive
rate, not an estimate of it.

Consequently: **every flagged-but-unlabelled case is traced, not counted.** An
untraceable flag is a false positive. A traceable one is defect number eight.

`defect_family` ∈ {foundation, classical, data}. Reported separately, because a
statistic that catches foundation-model defects and misses classical ones is a
different claim from one that catches both.

## Branches, and what each does to the paper

Fixed now, before Task 3 runs.

### C1 — q̂_V dominates
Higher sensitivity than Kupiec and Christoffersen at comparable specificity, in
**both** defect families.

- **Title** changes: the assertion becomes an answer. Candidate — *Detecting
  Pipeline Defects in Value-at-Risk Forecasts with Conformal Diagnostics*.
- **Abstract** leads with the detection result; the truncation becomes the
  worked example rather than the subject.
- **§1.3** keeps the weak-instrument evidence but ends on the positive claim:
  the instrument that works is the one already in the pipeline.
- Conformal recalibration becomes the method, with a validated second use —
  screening for structural defects rather than sizing a correction. This also
  reframes the AE's uMCB objection: uMCB measures miscalibration against a
  calibrated ideal, defect screening is a different task, and the comparison
  stops being a threat.
- Recalibration results return as the second contribution: the gate rule, the
  Basel upgrades retained, the rolling degradations avoided.

### C2 — comparable
No detector dominates in both families.

- Title unchanged. All three reported side by side, no dominance claimed.
- Conformal is a co-equal component of a validation battery, not its centre.
- §1.3 gains one paragraph: the existing statistics are weak, and so is this one.

### C3 — q̂_V does not dominate
Sensitivity at or below the backtests, or dominance in only one family.

- **The structural gate is the contribution.** Conformal prediction is genuinely
  background: it sizes a correction, it does not diagnose.
- Title stands. Abstract and §1 survive unchanged.
- The three retrospective observations get reported as what they are —
  suggestive, and not borne out under a fixed rule. Reporting a negative result
  here costs one paragraph and buys the paper its credibility on every other
  claim.

**C3 is the outcome most consistent with what is already known**, because R is
a magnitude and two of the defects (the sign flip, the GJR quantile map) produce
large R for reasons that have nothing to do with detection — a wrong-signed or
mis-scaled forecast needs a large correction by arithmetic necessity. That is a
mechanism, not a diagnostic. Sensitivity that comes from arithmetic necessity
will also fire on any forecaster that is merely badly scaled, which is why
specificity against the unlabelled set matters more than sensitivity here.

## Sample sizes, stated in advance

Forecaster level: 16 forecasters, of which 5–7 carry a label depending on
whether data-handling defects are counted at forecaster level. Fisher exact
tests, exact counts, no asymptotic approximations, no summary score that hides
which defects a detector missed.

Pair level: 312 (forecaster, asset) cells at α = 0.01, 1248 across all four
levels. Pair-level counts are not independent within a forecaster and are
reported as descriptive only.
