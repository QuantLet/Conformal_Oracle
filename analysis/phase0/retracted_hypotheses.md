# Retracted hypotheses

Standing rule 5. Every diagnosis proposed and then killed, with the evidence.

## R1 — "R-bar does not reproduce from the raw series" — RETRACTED

**Proposed.** Recomputing R-bar for Chronos-Small (default) from
`cfp_ijf_data/chronos_small/*.parquet` gave 17.057 against a published 17.3, and
0.129 against 0.145 for the analytic series. I proposed that the aggregation
convention (mean of ratios vs ratio of means) was unstated in the manuscript and
that the published value could not be reproduced.

**Killed by.** My conformal shift used `np.quantile(S, 1-alpha)`. The pipeline
uses the finite-sample split-conformal order statistic, the
`ceil((n+1)(1-alpha))`-th of the sorted calibration scores. Substituting it
reproduces every published `qV` to 9.9e-17 and every `raw_width` to 9.7e-17,
across all 24 assets. R-bar then agrees exactly. Verified additionally on all
five classical benchmarks: pi-hat and R-bar reproduce to <1e-16 for GJR-GARCH,
GJR-GARCH-t, GARCH-N, EWMA and Hist-Sim.

**What survives.** Not a reproduction failure, but a **definition defect** in the
manuscript: equation (7) defines `qV_stat = Q_{1-alpha}({S_t})`, "the empirical
(1-alpha)-quantile of the calibration scores". That is not what the code
computes, and the two differ materially -- on DJCI the plain empirical quantile
is -0.000176 and the conformal order statistic is +0.000708, opposite signs. The
manuscript states the order-statistic convention only for the *rolling*
estimator, in the Data and Code section. Carried to the recompute table as
DEFINITION_MISMATCH.

## R2 — "The multi-level Kupiec counts are wrong" — RETRACTED

**Proposed.** Section 5.6 states that Chronos-Small read analytically is
"rejected on 5, 9, 5 and 4 of 24 assets" and the default gives "10, 14, 16 and
15". Recomputing raw Kupiec rejections gave 16, 14, 10, 4 for the analytic series
and 24, 24, 24, 24 for the default. I proposed a DIFFERS, and noted it would
contradict Table 2, which shows the analytic series passing raw Kupiec on 8 of 24
at alpha = 0.01.

**Killed by.** The counts are on the **corrected** series, not the raw one.
Recomputing `p_kup_cp < 0.05` gives exactly 5, 9, 5, 4 and 10, 14, 16, 15.
MATCHES.

**What survives.** An ambiguity defect. The sentence does not say the counts are
post-correction, and a reader checking them against the raw Kupiec column of
Table 2 finds an apparent contradiction. Carried to Phase 5 as a writing fix, not
a numeric one.

## R3 — "The Diebold--Mariano count of 18 of 30 is wrong" — RETRACTED

**Proposed.** Extracting the upper triangle of `tab_dm_pvalues.csv` gave 15
pairs with 12 significant at 5%, against a claimed 18 of 30.

**Killed by.** The matrix is rectangular, 5 benchmarks x 6 TSFM series, not
symmetric. All 30 cells are populated and 18 are below 0.05. My triangular
extraction was wrong. MATCHES.

## R4 — "Pre-registered expectation 6 was wrong" — RECORDED AS A MISS

**Pre-registered.** I expected "between 10 and 30 literals that are NOT_EMITTED".

**Outcome.** Four, not ten to thirty: the forward-pass count, the GPU timing, the
absolute dispersion pair, and the panel-observation count. The prose is better
sourced than I predicted. The failures that do exist are of a different kind than
I expected -- wrong cross-references, a panel-mixing sentence, and a
definition/implementation mismatch -- none of which a literal-provenance screen
detects. Recorded because the pre-registration was directionally wrong and the
screen I built for it was aimed at the wrong failure mode.

## R5 — "Substring presence in the artefact tree is evidence of provenance" — ABANDONED

**Proposed.** An index over 276 artefact files, checking whether each prose
literal appears anywhere in it.

**Killed by.** It returned "found" for every claim including tokens like `38`,
`76`, `59`, `84`, `48`, `77`, `161`, where coincidental matches in a 6 MB corpus
are near-certain. It also returned "found" for the ACI figures 0.0750 and 0.0443,
which are real but come from an undeclared third panel -- the screen cannot see
that. Replaced by targeted recomputation from the emitting artefact. The existing
`scripts/audit_prose_numbers.py` has the same weakness and its "0 unsourced"
verdict should not be read as provenance.

---

# Standing defect log (not retractions)

## C1 — the conformal quantile convention has now produced a defect twice, in opposite directions

The convention at issue: whether the shift is the plain empirical quantile
`Q_{1-alpha}({S_t})` or the finite-sample split-conformal order statistic
`S_(k)`, `k = ceil((n+1)(1-alpha))`. They differ by one order statistic, `O(1/n)`,
and on this panel that gap changes the sign of `qV` on at least one asset
(DJCI: -0.000176 against +0.000708).

**First occurrence — code drifted from the text.** Package versions before 0.3.1
implemented the *rolling* shift as the plain empirical quantile while the
manuscript and the replication pipeline used the order statistic. Recorded in the
Data and Code section: 0.3.1 "aligns its packaged rolling implementation with
this same convention".

**Second occurrence — text drifted from the code, in the opposite direction.**
Equation (7) of the manuscript defined the *static* shift as the plain empirical
quantile while the pipeline computed the order statistic. Found in this session
by attempting to reproduce `qV` from the raw series and failing; substituting the
order statistic reproduced every published value to 1e-16. Fixed in the text,
not the code, per instruction.

**Third site, unfixed at the time of writing.** The same wrong description sits
in the package docstring, in *both* trees:
`conformal/static.py` line 19 reads "qV_stat = empirical (1-alpha)-quantile of
{S_t}" while calling `conformal_quantile()`, which is the order statistic. The
implementation file `conformal/quantile.py` is byte-identical across 0.3.1 and
0.3.2 and is correct; only its callers' docstrings are wrong.

**Why this keeps happening.** The two conventions produce series that are
indistinguishable to every check in this project. The structural gate cannot see
the difference -- both yield well-formed series that pass all ten checks -- and
neither can any backtest, because an `O(1/n)` shift in a threshold is far below
the resolution of a coverage test at `alpha = 0.01`. The only instrument that
separates them is reading the code. This is now stated in Section 7 as a scope
limit on the gate.

## R6 — "The equivalence is infeasible under unimodality" — RETRACTED (my bug)

**Proposed.** Adding a unimodality constraint to the linear program made it
infeasible at delta = 0.05, and I was about to report that the construction
requires a non-unimodal return law.

**Killed by.** A grid artefact of my own making. The candidate support was capped
at |x| <= 4, too narrow to place the compensating mass. Widening the ceiling to 8
makes delta = 0.05 feasible, and the answer is then stable at ceilings 8, 16, 32
and 64. I had built the negative-control habit for LaTeX checks this session and
did not apply it to my own optimiser: I never tested whether an infeasibility
verdict could be produced by a correct model under a bad discretisation.

**What survives, and it is better than what I set out to build.** The boundary is
real once the grid is adequate: bisection gives a critical truncation of
**delta = 0.0655** per tail. Below it an exactly equivalent, unimodal,
variance- and MAD-matched return law exists; above it none does. At the boundary
the reported 1% quantile is -1.323 against a true -2.606, so a VaR understated by
**49.3%** is exactly unidentifiable from the exceedance path. Beyond the
boundary the exceedance path can in principle detect the truncation -- which is
why Kupiec rejects the real Chronos default, whose implied delta is about 0.388,
on 24 of 24 assets. The proposition therefore acquires a quantitative boundary
that explains the empirical pattern instead of merely asserting blindness.

## R7 — "36,600 panel observations / 366 expected violations is wrong (D2)" — RETRACTED

**Proposed.** Phase 0 recomputed 485,069 test observations over 13 models, i.e.
37,313 per model and 373 expected violations, against a manuscript claim of
"roughly 36,600 ... and 366". I filed it as DIFFERS.

**Killed by.** The panel is not balanced in sample length. Foundation-model
series carry 36,588 test observations each (512-observation warm-up) and the
benchmarks 38,473 (250-observation warm-up). My 37,313 was the mean over a mixed
panel and corresponds to no forecaster. The manuscript sentence sits in Section
3.3.3, which describes the foundation-model design, and 36,588 rounds to 36,600
with 365.9 expected violations. **The manuscript is right and my recomputation
was the wrong statistic.**

**What survives.** Section 5.6 quoted "36,000 observations" for a pooled test
applied to benchmarks as well as foundation models, where the correct figures are
36,588 and 38,473. That sentence is now stated with both.

## C2 — Section 5.6's cluster-robust claim was false (confirmed defect, found via a failing unit test)

`test_clustered_se_differs_from_ols` asserted that clustered standard errors
differ from OLS by more than 10% on the package fixture. Tracing it produced two
results.

**The estimator is correct.** `_cluster_se` reproduces `statsmodels`
cluster-robust standard errors to 8.7e-16 on data built with genuine
within-cluster correlation. On data without it, clustered and OLS standard errors
agree — which is the correct answer. The test asserted a property of its fixture
rather than of the estimator and failed on working code. Replaced by a check
against an independent reference plus a negative control that pins the
no-clustering case.

**The manuscript claim it sits under was false.** Section 5.6 read: "EWMA is the
only forecaster rejected by the cluster-robust test (p = 0.035). Three others are
rejected by the pooled unconditional test and not by the cluster-robust one."
`Quantlets/CO_panel_wildcluster/wild_cluster_kupiec.csv` gives EWMA
`p_boot = 0.058`, and **the wild-cluster bootstrap rejects nothing at 5%**. The
value 0.035 occurs in no artefact. The pooled asymptotic test rejects four
forecasters, EWMA among them, and the bootstrap rejects none. Corrected in the
text, where it now makes a cleaner claim than the one it replaces.

## R8 — "The critical delta is stable from grid ceiling 8 upward" — SUPERSEDED

**Claimed.** After the R6 grid bug I reported that the unimodal boundary was
"stable from ceiling 8 up", citing delta* = 0.0655 at ceilings 8, 16, 32 and 64.

**Superseded by.** The ceiling was never the controlling parameter; the grid
*spacing* is. Holding spacing at 0.016, ceilings 32, 64 and 128 all return
delta* = 0.0690. Holding the ceiling at 32 and refining spacing to 0.008 and
0.004 returns 0.0669 and 0.0659. My earlier sweep varied both at once and read
agreement along a diagonal as stability in one argument.

**Consequence, which matters for how the number is reported.** delta* converges
**from above** as the grid is refined: 0.0690, 0.0669, 0.0659 at spacings 0.016,
0.008, 0.004, with differences halving. The reported 0.066 is therefore an
**upper bound**, and the bias runs in the direction that *overstates* the blind
spot. Every understatement figure derived from it -- 49.4% for the unimodal
class and the rest of the class table -- is likewise an upper bound. The paper
must say so; a discretisation that flatters the paper's own thesis is exactly the
kind of error this project exists to catch.

## C3 — three further unsourced literals, same class as the fabricated p = 0.035

Found by the literal ledger, not by any existing check.

1. **Supplement S.3: "the mean absolute difference from the canonical
   formulation is 0.035 Z2 units, and the modified and canonical versions agree
   on the 5% pass/fail classification for 97% of model--asset pairs."**
   `analysis/provenance/z2_verification.csv` holds three models and a median
   canonical Z2 per model; it carries no per-pair canonical-versus-modified
   comparison. Neither figure is recomputable from any artefact. **UNSOURCED.**
2. **Supplement S.2.2: "5/9 Kupiec rejections (vs. 0/9), and 88.9% Green
   (vs. 100%)."** The denominators are 9; the tuning grid
   `tuned_gbm_qr_grid.csv` covers 13 models, and its green rates are 82.7% at
   eta = 0.01 and 76.9% at eta = 0.05, not 100% and 88.9%. The paragraph reports
   an undeclared 9-model subset or a superseded vintage. **UNSOURCED as printed.**
3. **Main text S5.7: the 216-pair ACI figures 0.0750 and 0.0443**, already filed
   as P1 in Phase 0 and slated for replacement.

The p = 0.035 in Section 5.6 was not an isolated case. It was the first one
found.

## R9 — "The scale band should be tightened to -2.05" — RETRACTED

**Proposed.** Sweeping the band edge against per-series median scale ratios, I
found the eleven well-specified series in [-2.528, -2.085] and the two truncated
Chronos defaults at -0.208 and -0.127, and recommended moving the edge from -1.8
to -2.05: residual understatement 30.9% -> 21.3% at no false-positive cost.

**Killed by.** The gate does not operate on per-series medians. It evaluates each
**series-asset cell**, and cells are far more dispersed than their medians. At
cell level the well-specified population runs to **-1.947**, not -2.085, and the
proposed edge of -2.05 blocks **11 well-specified cells**. -2.00 blocks 2. The
recommendation was an artefact of aggregating before testing -- the same class of
error as R6 (grid too narrow) and R3 (wrong matrix geometry): a correct
computation on the wrong object.

**What survives, and it is the stronger statement.** The two populations are
separated by **1.664 sigma-hat units** at cell level: worst well-specified cell
-1.947, best truncated cell -0.283. Any edge strictly inside that interval blocks
all 48 truncated cells and no well-specified ones, so **the edge is
underdetermined within the gap** and the gate's verdict on this panel does not
depend on where it is placed. That is a better position than a calibrated
constant, because it does not rest on a choice.

The real safety margin is not the gap. At the current -1.8 it is **0.147** to the
nearest well-specified cell; pushing to -1.94 buys 30.9% -> 25.6% residual and
leaves a margin of 0.007. Both numbers belong in the paper; a referee who
computes the margin from the gap alone will conclude the gate is more robust than
it is.

## C4 — the lower band edge has never bound

`-3.5` blocks **0 of 312** series-asset cells. The most negative cell anywhere in
the panel is -2.735 (TimesFM 2.5 on BTC), a margin of 0.765 sigma-hat units. The
band is therefore strongly asymmetric in risk: 0.147 of margin above, 0.765
below, and the lower edge was never calibrated against anything. An edge that
cannot bind carries no evidence either way -- the same objection this paper makes
to a degenerate transition table, applied to its own gate. To be reported, not
quietly widened or narrowed.

---

# Round-4 ledger

## VERIFIED (checked against artefacts, no change needed)
- Macro-generated arithmetic closes everywhere tested: 123/273 Kupiec passes,
  335 Green, 97/180 CC, 278 + 57 = 335 across panels, 309 = 12x24 + 21 rolling.
- `3.3` and `76` closure factors reproduce on unrounded data (3.253, 75.903);
  the printed table rounds R to 2 dp, which is what made an earlier parse give 39.
- The `\bar R` span 23264 and the 84x consecutive ratio are correct on unrounded
  values and now reproduce from the print, after moving R to four significant
  figures.

## CHANGED
- **A1** Two numbered propositions added to Section 6 (characterisation;
  depth of the blind spot), with proofs at Supplement S.9.6 and S.9.7. Section 7
  no longer points at a construction that did not exist, and Table 2 is no longer
  the first appearance of delta*.
- **A2** Z2 now carries panel numbers in Section 6 (0/24 against 17/24) together
  with the statement that a coverage-and-mean-ES matched pair remains LP-feasible,
  so joint VaR-ES evaluation is not a general escape.
- **A3** DQ **kept**, because it ran: 312 cells in 2 seconds. It rejects 81.7% of
  raw cells against Kupiec's 69.9%, and 19/24 for GARCH-N, EWMA and GJR-GARCH.
  Pre-registered falsification condition fired -- it does not discriminate -- and
  that is now a subsection: the most elaborate of the three backtests is the least
  discriminating.
- **A4** Equation (10) corrected to the conformal order statistic. No third
  variant found on grep of either document.
- **B1** Zone decomposition restated: 99 already Green, one not-already-Green with
  no change, 2 Green-to-Yellow, summing to 102.
- **B3** R printed at four significant figures at the producer.
- **B4** "four and seventy" -> "five and seventy" (minimum is 0.0200/0.004 = 5).
- **B5** The negative-shift count is now a printed column of Table 4; both
  references repaired.
- **B6** Baseline count reconciled to ten; gate counts restated as 7 + 1 + 2;
  Table 3 marks which two checks are scoped out; gate wording aligned.
- **B7** Section 1 roadmap rewritten against the real structure.
- **B8** Equation (5) removed in favour of VaR^raw; the two Chronos rates now
  carry their alpha level; delta* separated in text from the drift diagnostic.
- **C1** `scripts/audit_structural_claims.py` added, four checks, each with a
  negative control. It found one real defect on first use: a sentence describing
  the last *two* columns of Table 1 as "the last column".
- **C2** PROTOCOL.md Rule 1 gains a second mandatory field, what varies between
  rows and over what range, because a row count does not catch R6.

## BLOCKED
- **D1** IRFA submission fee and highlights requirement: cannot be read offline.
  Keywords cut to five is applied; the 208-word abstract is within a 250 limit if
  that limit is confirmed. **Needs the Guide for Authors.**
- **ML dose-response** running at the pre-registered 8,000 cells, ~145 minutes.
  Nothing read yet.

## Defects found in my own new instruments, before trusting them
1. The item-count check could not fire: its regex did not cross a line break, so
   it silently matched nothing and reported a pass.
2. The decomposition check could not read totals written as macros -- the paper's
   own discipline of emitting numbers would have made the checker blind. Fixed by
   expanding `numbers.tex` before reading.
3. The column check could not resolve a claim inside a caption, and then matched
   on prose keywords rather than identifiers, producing a false positive on a
   column headed by a symbol.
All three were found by re-injecting the historical defects rather than by
reading the code.


## R11 — "p = 0.035 appears in no artefact and the paragraph's conclusion inverts" — RETRACTED

**Asserted.** Section 5.6 read "EWMA is the only forecaster rejected by the
cluster-robust test (p = 0.035)". I searched
`Quantlets/CO_panel_wildcluster/`, found EWMA at `p_boot = 0.058` and no
forecaster rejected by the bootstrap at 5%, reported the value as produced by no
artefact, called the conclusion inverted, **and rewrote the paragraph**. Logged
as C2, a confirmed defect found via a failing unit test.

**Killed by.** `Quantlets/CO_multi_quantile_panel/panel_pooled_reproduced.csv`
carries a `p_cluster` column. EWMA's value is **0.035192**, and it is the only
series below 0.05 in that column. The manuscript's own wording said
`p_{cluster}`, not `p_boot`. **The paper has two distinct cluster-robust
procedures** -- a Driscoll-Kraay cluster statistic and a Rademacher wild-cluster
bootstrap -- and I compared the text against the wrong one.

The original sentence was **correct in every part**: EWMA is the only rejection
under the cluster statistic; the other three pooled rejections (Moirai 2.0,
Chronos-Small-A, GJR-GARCH) are not rejected by it; the pooled rates 0.0114,
0.0113 and 0.0111 all check out.

**The fourth instance of the same pattern.** R3 read a rectangular matrix as
symmetric; R6 tested a correct model on a truncated grid; R9 calibrated on series
medians where the gate uses cells; R11 compared a claim against a different
statistic with a similar name. Correct arithmetic on the wrong object, and this
time I did not merely mis-report it -- **I rewrote a correct paragraph on the
strength of it**, and the rewrite was wrong.

**What survives.** The two procedures disagree at the boundary, 0.035 against
0.058 on the same series. That is worth reporting and now is, framed as a
resolution limit at 24 clusters rather than as a verdict either way. Restored to
Section 8.4 in corrected form.

## R12 — "88.9% Green is computed on an undeclared nine-model subset" — RETRACTED

**Asserted.** Supplement S.2.2 quoted "5/9 Kupiec rejections (vs. 0/9), and 88.9%
Green (vs. 100%)". I compared against `tuned_gbm_qr_grid.csv`, which covers 13
models with green rates of 82.7% and 76.9%, concluded the denominator was
undeclared or the vintage superseded, **and deleted the sentence**.

**Killed by.** `Quantlets/CO_baseline_comparison_tuned/REPRO_NOTES_E1.md`, dated
2026-05-08, line 31: "Best QS config (n=100, d=3, lr=0.05): QS=4.40e-4, pi=.015,
5/9 Kupiec rejections, 88.9% Green." Every figure in the sentence is in a
provenance note that predates the audit. Same failure as R11: I searched one
artefact directory and declared absence.

**What survives.** The nine-series denominator was not declared *in the
supplement*, only in the provenance note. The sentence is restored with the
denominator stated in the text.

## Consequence for the "unsourced literals" finding

The claim was five. **Two of the five were my errors.** What stands:

| claim | status |
|---|---|
| Z2: mean absolute difference 0.035 units, 97% classification agreement | **UNSOURCED.** `verify_z2.py` computes neither a mean absolute difference nor a classification-agreement rate. In the IJF submission, so it survived review. |
| Raw rates 0.019 / 0.075 under two closures | **UNSOURCED.** `run_inner7_tail_closure.py` emits `pi_corr` only. |
| ACI 0.0750 / 0.0443 | Sourced, but from an undeclared 216-pair panel. A panel-declaration issue, not a sourcing one. |
| p_cluster = 0.035 | **SOURCED** (R11). |
| 88.9% Green | **SOURCED** (R12). |

Section 7 now says two, not five, and describes the two that stand. The cover
letter must be corrected the same way before it is sent.

## R13 — the sign inversion is real; the diagnostic Table 1 claimed for it is not

**Challenged.** Whether the two grid series were sign-inverted at all.

**The conclusion holds, and on better evidence than I had cited.**
`analysis/interface/sign_diagnostic.csv` carries 240 per-cell rows, 10 models by
24 assets. For TimesFM-2.5 and Moirai-2.0 the stored lower quantile is positive
in **48 of 48 cells**, `pct_positive = 1.0000` in every one, and the ordering
across alpha is reversed on **100% of days in every cell**. Every other series:
median negative, positive on at most 10% of observations, reversed on 0% of days.
The implied violation rate from the stored threshold is 0.9748-0.9940, matching
the 0.990 and 0.988 the paper published. Two saturated signatures, both derived
from the data rather than from a label.

**But Table 1 described a diagnostic that does not exist.** It read: "the stored
series is reproduced to ten decimal places by $-F^{-1}(\alpha)$ from the
distributional parameters recorded in the same file." `sign_diagnostic.py`
performs no such reconstruction. It measures sign, monotonicity and the implied
rate. Its `code_path` classification -- the label `student_t_negated` -- is a
**hard-coded dictionary** with the comment "read from the pipeline notebooks",
which is an author's reading of source, not a computation.

**And it cannot now be run.** No vintage with a positive stored lower quantile
remains in the repository: both `cfp_ijf_data/timesfm25` and the quarantined
six-column vintage have median VaR at -0.019. The defective input was replaced by
the correction, so the strongest-sounding diagnostic in the table is the one that
can least be re-executed.

**Fixed** by stating what was measured. This is the class of defect the paper is
about, found in the paper's own audit table: a claim about verification stronger
than the verification performed.

## R14 — the analytic Chronos estimator is off by one quantisation bin — STANDS

Registered as R14, not R11: the R11 slot is occupied by the `p_cluster = 0.035`
retraction above. This entry is a defect that survived challenge, in the class of
R13 rather than of R1–R12.

**Proposed.** Section 4.4 describes the analytic estimator in five numbered steps.
Rebuilt from that prose alone — no file in `scripts/`, `pipeline/` or
`Quantlets/` opened — it disagreed with the shipped series on 200 contiguous
SP500 dates: median relative deviation 2.5e-03 on VaR₀.₀₁, rising to 5.0e-03 at
the 10% level.

**Survived its own negative control, which is how it was found.** Three controls
were declared in advance. The one built to fail — *the stored series shifted by
one bin width* — **agreed**, to a median relative deviation of 2.4e-09 and a
maximum of 8.1e-08. A control that agrees is a comparison rule that cannot detect
a wrong answer, and reporting that rather than the headline disagreement is what
located the defect.

**Adjudicated, without Monte Carlo error.** The two candidates are:
- A: pair `p[id]` with `centres[id − n_special − 1]` — the map the library's own
  `MeanScaleUniformBins.output_transform` uses to decode a sampled token;
- B: pair `p[id]` with `centres[id − n_special]` — one bin higher.

Sampling cannot separate them: at 2,000 draws the Monte Carlo error on the 1%
quantile is about five bins, and |A − MC| and |B − MC| differ in the fifth decimal.
Forcing `top_k = 1` removes the error entirely — every draw is the arg-max token,
and what the library decodes it to is a fact, not an estimate. **On 12 of 12
cells across SP500, GOLD and EURUSD the library's decoder returns A exactly
(gap 0.000 bins) and B is one full bin above (gap −1.000).** The shipped analytic
series is B.

**Confirmed as systematic, not local.** stored − rebuilt = 0.999999 bin widths,
sd ≈ 1e-16, on SP500, GOLD and EURUSD for `chronos-t5-small` and SP500 for
`chronos-t5-mini`.

**Why the paper's own validation could not have caught it.** Section 4.4 reports
two agreements against full-vocabulary sampling on 40 SP500 dates. Neither is
sensitive to this defect, and not because the tolerances were loose:

- The defect is a **uniform translation of the support**, which leaves the
  predictive standard deviation *exactly* invariant. The "0.3%" dispersion
  agreement is a tolerance on a quantity the defect does not move.
- The violation-rate agreement "to four decimal places" ran on 40 dates at
  α = 0.01, expected count 0.4, on a rate grid of 1/40 = 0.025. The offset is
  0.249% of VaR₀.₀₁ and 0.591% of the predictive standard deviation. Nothing on
  a 0.025 grid resolves either.

Comparing the per-date quantiles — the object, rather than two summaries of it —
would have found this in one line.

**What it moves.** The offset is deterministic: stored = correct + binwidth ×
scale_t, and scale_t is the mean absolute value of the 512-day context, which
needs no model. The reconstruction rule was checked against the 200 re-run dates
(max relative error 8.1e-08) before being applied to all 24 assets.

| | published | corrected |
|---|---|---|
| Chronos-Small-A, π̂ at α = 0.01 | 0.0175 | **0.0173** |
| Chronos-Mini-A, π̂ at α = 0.01 | 0.0178 | **0.0177** |
| Chronos-Small-A, ratio π̂/α | 1.750 | **1.733** |
| Chronos-Mini-A, ratio π̂/α | 1.781 | **1.772** |
| Chronos-Small-A, π̂ at α = 0.10 | 0.1036 | **0.1027** |
| Chronos-Mini-A, π̂ at α = 0.10 | 0.0989 | **0.0981** |
| Chronos-Mini-A, Kupiec passes at α = 0.10 | 22/24 | **20/24** |
| Kupiec passes at α = 0.01, both | 8/24 | 8/24 (unchanged) |
| Quantile score at α = 0.01, both | — | −0.09% |

Nothing qualitative moves. The sampler is still removed, the residual is still a
tail deficiency that shrinks with α, and 1.73× nominal is the same finding as
1.75×.

**The pattern, which is new to this register.** R1, R2, R3, R7, R9 and R12 are
*correct arithmetic on the wrong object* — Rule 1 catches them by declaring the
unit. R6 is a correct model on a malformed support — Rule 1a catches it by
declaring the range. R14 is neither. The arithmetic is right, the object is
right, the unit is right, and the range is right. What failed is that **the
check written to exclude the defect was calibrated on a statistic the defect
leaves invariant.** It is the second instance in this project of a validation
that could not fail informatively — the first being the gate's lower band edge at
−3.500, which blocks 0 of 312 cells — and the two are now Rule 2's subsection in
`PROTOCOL.md`.

**And the location is the point.** This is not a defect in a benchmark or in an
input series. It is in the estimator Section 4.4 offers as the *remedy* for a
configuration failure, presented as having "no `num_samples`, no `top_k`, no
`top_p`, no temperature and no seed — there is no configuration to get wrong."
That claim stands: there was no configuration to get wrong, and the estimator was
got wrong anyway, in the arithmetic beneath the configuration. Section 9 says so.

**Disposition.** Both analytic panels are regenerated with the corrected map
before submission; the corrected figures enter wherever they appear, including
whatever remains in the body after the Chronos exhibit moves to the supplement.
Evidence: `analysis/k1_verify/` (`k1a_result.json`, `k1a_decide.json`,
`k1a_adjudicate.json`, `k1a_impact.json`).
