# K1 — independent verification, pre-registration

Written before any of the five checks below was run. Protocol Rule 1: each check
declares its unit of analysis, its row count, and what varies from row to row and
over what range. Protocol standing rule: **no check reports PASS until it has been
seen to FAIL on a case constructed to make it fail.** Each script therefore carries
a negative control, run in the same process, and the negative control's failure is
printed beside the real result.

---

## K1a — the analytic Chronos estimator, reimplemented from the manuscript

**What is being checked.** Section 4.4 of `main_R2.tex` describes the analytic
estimator in five steps. The stored series in `cfp_ijf_data/chronos_small_analytic/`
were produced by code in this repository. The panel numbers 0.0175 and 0.0178 come
from those series. The same source produced the five failure modes the paper
reports, so the estimator is reimplemented **from the prose alone**, without
reading `scripts/`, `pipeline/`, or any existing analytic-quantile code, and the
two are compared.

**Unit of analysis.** One row = one date on one asset. Asset = SP500.
**Row count.** The number of dates in `cfp_ijf_data/chronos_small_analytic/SP500.parquet`
that fall in a contiguous verification block; the block is 200 consecutive dates
drawn from the middle of the series, declared here so that a later choice of an
easier window cannot be made after seeing the answer.
**What varies from row to row.** The 512-observation context ending at t-1, and
nothing else. Weights (`amazon/chronos-t5-small`), tokenizer, dtype and device are
fixed.
**Range.** SP500 log returns, roughly [-0.13, 0.11]; VaR_0.01 in the stored file
runs to about -0.09 on the crisis dates.

**Comparison rule, fixed now.** The reimplementation agrees if, over the 200 dates,
the median absolute relative deviation of VaR_0.01 is below 1e-3 and the maximum
absolute relative deviation is below 1e-2. Anything larger is reported as a
disagreement, with the dates that produce it, and no attempt is made to reconcile
it by adjusting the reimplementation.

**Negative control.** The same comparison is run against the stored series shifted
by one bin width and against the stored **sampled** (default top_k) series. Both
must fail the comparison rule.

**Interpretation written in advance.** If the reimplementation agrees, 0.0175 and
0.0178 are supported by an estimator described sufficiently in the text for a
reader to rebuild, and that fact is worth one sentence in the paper. If it
disagrees, the disagreement is the finding, and the analytic column is marked
BLOCKED, not repaired.

---

## K1b1 — violation rates for TimesFM 2.5 and Moirai 2.0

**What is being checked.** `\nRawPiTimesFM` = 0.0143 and `\nRawPiMoiraiTwo` =
0.0178: the fraction of realised returns below the stored threshold.

**Unit of analysis.** Two readings, both reported.
(i) *cell*: one row = one forecaster on one asset. 24 rows per forecaster.
(ii) *panel*: one row = one forecaster, pooling all test-window dates across the
24 assets. 1 row per forecaster.
The manuscript's number is claimed to be the panel reading; if the two differ, both
are printed and the manuscript is checked against the one it actually asserts.
**What varies from row to row.** The asset, and with it the test-window length
(roughly 450 to 1,880 dates) and the sample period.
**Range.** Per-asset violation rates expected in [0.005, 0.04].

**Comparison rule.** Agreement to the fourth decimal, the precision printed.

**Negative control.** The same counter run with the inequality reversed
(`r_t > q_t`) and run on the full sample rather than the test window. Both must
produce a number that fails the comparison.

---

## K1b2 — 1000 draws under the default contain exactly 50 distinct values

**What is being checked.** The manuscript asserts this holds in all 1600
model-asset-date cells, without exception, and calls it the reading that makes the
diagnosis arithmetic rather than inferential.

**Unit of analysis.** One row = one checkpoint x asset x date cell.
**Row count.** 2 checkpoints x 2 assets x 10 dates = 40 cells. This is a subsample
of the 1600; it is declared as a subsample and the manuscript's claim over 1600
cells is **not** re-established by it. What 40 cells can establish is whether the
count is 50 or something else, since the claim is an exact equality that a single
counterexample refutes.
**What varies from row to row.** Checkpoint (small, mini), asset (SP500, EURUSD),
and the context date. Seed fixed at 0; `num_samples` = 1000; `top_k` at the
packaged default.
**Range.** Distinct-value counts in [1, 1000].

**Comparison rule.** Every cell returns exactly 50.

**Negative control.** The same counter at `top_k = 4094`, which must return a count
far above 50, and at `top_k = 10`, which must return 10.

---

## K1b3 — n11 = n10 = 0 against what the code counts as a Christoffersen pass

**What is being checked.** `\nSeqCCUndefRawOne` = 34.6% and `\nMainCCAsPassPct` =
78.4. The manuscript's claim is that the independence test is degenerate when the
transition table cannot be populated, and that the natural implementation returns
"no rejection", which reads as a pass.

**Unit of analysis.** One row = one pair (forecaster x asset) at alpha = 0.01, on
the sequence panel.
**Row count.** 13 x 24 = 312.
**What varies from row to row.** The forecaster and the asset, and therefore the
number of exceedances in the test window (expected count near 15 at alpha = 0.01
on a 1,500-day window, but ranging from 0 upward).
**Range.** n11 in [0, ~50]; the degeneracy condition of interest is the state in
which no exceedance is followed by an exceedance.

**The condition is stated precisely before counting**, because more than one
degeneracy exists and they are not the same set:
- (A) n11 = 0 and n10 = 0: no exceedance at all in the window, or a single
  exceedance on the final date. The transition table has an empty row.
- (B) n11 = 0 with n10 > 0: exceedances occur but never consecutively. pi_11 is
  estimated at the boundary 0.
- (C) n01 = n11 = 0: no transition into the exceedance state.
The manuscript's phrase is "too few exceedances to populate a transition table".
All three counts are reported, and the one the code's undefined-flag actually
implements is identified by reading the code after the counts are computed.

**Comparison rule.** The count of pairs the code flags undefined must equal the
count under exactly one of (A), (B), (C); if it equals none of them, that is the
finding.

**Negative control.** A constructed exceedance sequence with n11 > 0 must not be
flagged; a constructed all-zero sequence must be.

---

## K1c — CAViaR passes Kupiec on 15 of 24 assets

**What is being checked.** `\nMainBestKupiec` = 15 for CAViaR-AS. The manuscript
uses it to say a per-asset CAViaR fitted the ordinary way outperforms all six
foundation models. It is load-bearing for the "not superior" claim and has not
been verified against an independent implementation.

**Unit of analysis.** One row = one asset. 24 rows.
**What varies from row to row.** The asset's return series and its test window.
**Range.** Kupiec LR statistics in [0, ~200]; pass = p > 0.05.

**Independent implementation.** CAViaR asymmetric slope, estimated by the
\citet{engle2004caviar} recipe: the RQ criterion minimised over a set of starting
values drawn uniformly, the best m retained and refined. The estimation is written
from the published specification, not from this repository's routine, and the
comparison is on the pass/fail vector across the 24 assets, not only on the count
--- two implementations can both give 15 and disagree on which assets.

**Comparison rule.** The pass/fail vector agrees on at least 22 of 24 assets and
the count is 15. Fewer than 22 agreements is a disagreement and is reported as one.

**Negative control.** The Kupiec routine is run on a series constructed to violate
at 5% against a nominal 1%; it must reject on every asset.

---

## K1d — evidence grade

Table 3 and Table I.6 rest on reconstruction rather than on retrieved artefacts.
No computation; a written classification, entered in the ledger, and carried into
whatever section of the rebuilt manuscript inherits those tables.
