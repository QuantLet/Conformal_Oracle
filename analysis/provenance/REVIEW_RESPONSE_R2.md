# Response to the two internal reviews of the rebuilt manuscript

Both reviews are on file. This records what was changed, what was verified before
being changed, and what is still open. Every claim marked "verified" was checked
against the artefacts before the text was touched.

## Title

`What Backtests Cannot Detect` -> **`What Backtests Cannot Diagnose: Structural
Validation of Tail-Risk Forecasting Pipelines`**. The old title overclaimed in
exactly the way the second review identified: Kupiec's test *detects*
miscoverage on most of this panel, defective series included. What it cannot do
is say whether the source is the model, the reduction rule, a sign, or an
alignment fault. Section 6 is reframed on that distinction and now opens with it.

## Arithmetic: nine findings, nine confirmed

| finding | as printed | correct | status |
|---|---|---|---|
| Table 3 note | 260/312 Green (83.3%) | 335/384 (87.2%) | caption no longer duplicates the generated note |
| CC undefined | 53.1% (204/384) vs 53.5% (167/312) | two different panels | both tagged, never mixed |
| CC as-pass | 73.7% | **78.4%** | was my arithmetic error |
| CC pass rate | 53.9% vs 48.3% | different denominators | tagged |
| baseline footnote | "97.5% here" | 99.0% | rewritten |
| Appendix C.2 | n = 240 | 312 | corrected in the supplement |
| gate footnote | "all series pass" | gate blocks 4 of 13 | rewritten |
| W vs GJR | 1.60-1.75 | **1.02 and 1.00** | paragraph rewritten |
| GJR q_V sign | negative on 23 of 24 | **0 of 24** | paragraph rewritten |

The systemic fix, which is the point: **`scripts/paper_numbers.py` recomputes
every headline figure from the artefacts and emits `numbers.tex`, 145 LaTeX
macros the manuscript uses in place of literals.** `--check` fails if a value in
the text has drifted from the value in the data. Two panels are defined once, in
Section 3.3, and every macro name carries which one it belongs to:

- **main**: 16 forecasters x 24 assets = 384 pairs (Table 1)
- **sequence**: 13 x 24 = 312 pairs, the subset with stored per-date series

## Four substantive corrections

**1. The validation gate contradicted the paper's own definition of VaR.**
Equation (1) sets VaR = -q_lo, a positive threshold; the gate checked
`median(VaR) < 0` and an increasing order across alpha, which are conditions on
the *lower quantile*. The table is now written on q_lo throughout, with the
convention stated -- and the observation that the stored column is named
`VaR_alpha` while holding a quantile is exactly how two of the five defects
survived.

The gate is also no longer presented as ten "necessary properties". Two are
invariants, one is an invariant tested through a proxy with a threshold, and
seven are plausibility bands whose cutoffs we chose and state. Two of them
(alpha-response, coverage plausibility) need an evaluation window, so the claim
that no check needs one was false and is withdrawn; what separates them from a
backtest is the question, not the input, and that argument is now made explicitly.

**2. The Christoffersen claim was too strong.** A degenerate transition table
does not make the likelihood ratio non-existent -- the estimate sits at the
boundary and the chi-squared calibration fails. Restated: the test has
essentially no power there, implementations differ in what they return, and the
substantive point survives either way.

**3. The deployment rule had look-ahead bias, and correcting it costs a headline
claim.** The rule gated on the *test-window* backtest and was then scored on that
same window. Recomputed on the calibration window -- the only signal available at
the decision date -- at alpha = 0.01 under the rolling estimator:

| | oracle (test window) | deployable (calibration window) |
|---|---|---|
| pairs skipped | 94 | 53 |
| degradations avoided | 89 of 174 (51%) | **44 of 174 (25%)** |
| Basel zone upgrades kept | 205 of 205 | **185 of 205** |

"Removes half the damage while retaining every zone upgrade" was an in-sample
result. Deployable, it removes a quarter and costs 20 upgrades. Both are now
reported, the oracle labelled as such, in the abstract, introduction, Section
5.3, the practitioner guidance and the conclusion.

**4. Lemma E.1 was false as stated.** The proof needs
sigma(S_s : s >= t+k) contained in sigma(r_u : u >= t+k-1), which fails when the
forecast depends on the whole past. Repaired by assuming a finite context of
length m -- true of every forecaster here, m = 512 for the TSFMs and 250 for the
benchmarks -- giving beta_S(k) <= beta_r(k-m) for k > m. A remark states what was
wrong and why the repair costs nothing.

Two further theory statements are now explicit rather than implied: the theorem
requires a gap g_n > m while the empirical protocol uses a contiguous split
(g_n = 0), so it does not apply to the estimator as run -- the gap ablation
measures that discrepancy at <= 0.0058 in pi-hat -- and Proposition E.4 is
labelled a heuristic supporting bound, since its drift measure is not estimable
from one path.

## Presentation

Abstract 500 -> 339 words, leading with the diagnosis claim rather than the
near-tautological configuration one. Fourteen double periods from
`\paragraph{X.}` removed. Revision colouring off. Hyperlink boxes removed. The
model table no longer runs 278pt past the margin. The traffic-light figure is
larger. The tail-closure caption no longer asserts an invariance the table
refutes.

## Round three

**Proposition E.3 proved, not asserted.** Replaced the regret argument with an
exact telescoping identity. The update in the statement is
q_{t+1} = q_t + eta*(1{s_t > q_t} - alpha), so summing gives
sum_t (1{s_t > q_t} - alpha) = (q_{T+1} - q_1)/eta directly. Step 1 shows the
iterates cannot leave [-R - eta*a, R + eta*(1-a)] without projection -- once the
iterate passes R the indicator is 0 and the update pulls it back -- so the range
has width 2R + eta and

    |T^-1 sum_t 1{s_t > q_t} - alpha| <= (2R + eta) / (eta T),

which is <= 3/sqrt(T) at eta = R/sqrt(T). No density assumption, no Zinkevich,
explicit constant, and the statement is pathwise. Checked numerically on six
score paths at three sample sizes: the telescoping identity is exact to 1e-10,
the iterates stay in the claimed interval, and the bound holds
(`analysis/provenance/verify_prop_ogd.py`).

**Corrigenda consolidated.** One table at the head of Section 5, in the shape
suggested: defect, diagnostic, as reported, corrected. Fourteen rows in four
blocks -- pipeline defects, analysis-code defects, claims withdrawn, theory
repairs. The subsections that used to carry their own corrections now state the
current position and point at the table. Section 6.4 keeps its retrospective,
where the superseded tables are the evidence rather than an apology.

**The Z_2 result must not go into Section 6, and the caution was right.** Written
as a second implementation from the published definition, the canonical statistic
does NOT pass on the truncated series -- it rejects on all 24 assets, with median
Z_2 of -144 and -389. The 24/24 pass was an artefact of our own routine, which
divided by the stored ES column without negating it; the column is a lower-tail
quantity and negative, so every term's sign was reversed and the statistic came
out large and positive, where a one-sided lower-tail test can never reject.

This is the seventh defect of the same family in this project and the third in
analysis rather than generation code. The producer is fixed, the table
regenerated -- Chronos default now 0/24 raw passes against 24/24 before -- and
the episode is written up in Supplement S.1 rather than promoted into the body.
It also bounds the paper's own thesis: a structural gate would not have caught
this one, because the series were fine and the test was wrong.

**Public artefacts.** `drafts/ssrn_correction_notice.md` drafts the SSRN version
note, a shorter Quantinar note, and the argument for a version note over a silent
PDF replacement, with the sequence: co-author message, then SSRN, then Quantinar,
then submission.

## Open

- **Propositions E.3.** The telescoping proof the review suggests is shorter and
  cleaner than the regret argument in place. Not yet rewritten.
- **The two-paper "survey" claim.** Still in the text as a hedged observation.
  Either audit 15-20 papers or demote it to a footnote; not yet done.
- **The Z_2 result.** The review is right that "Z_2 passes 24/24 on a series
  running at pi-hat = 0.39" is the most vivid instrument-blindness evidence in
  the paper and is buried in an appendix. Not yet moved into Section 6.
- **Corrigenda are still scattered** across six places rather than gathered,
  with Section 6.4 correctly staying where it is.
- **ML forecaster benchmarks** (RF, GBM, LSTM as forecasters rather than as
  recalibration baselines) remain absent; this was an IJF objection and will
  recur at any forecasting journal.
- **SSRN 6757685 and the Quantinar course still carry 98.8% / 99.0%**, and the
  co-author message has not been sent. Both precede submission, not follow it.

---

# Round four: the consistency sweep

Eight items, all confirmed, all fixed.

| # | Item | Resolution |
|---|---|---|
| 1 | pp. 4 and 41 said Theorem 3.3's conditions hold here; p. 14 said correctly that it does not apply at g_n = 0 | Both now point at Remark 3.6: the contiguous split does not satisfy the separation the theorem needs, and neither guarantee covers the estimator exactly as run |
| 2 | p. 6 called all ten checks "necessary conditions" | Now "only three of the ten are necessary conditions in the strict sense; the other seven are plausibility bands" |
| 3 | p. 48 said almost every raw forecast fails at alpha >= 0.025 | Reversed: the rejection rate falls from 69.9% to 39.4% across the four levels, so at alpha = 0.10 a clear majority **pass** and the rule skips most of the panel |
| 4 | p. 28 called R-bar "signed" | R-bar is a ratio of absolute values and is non-negative by construction; the sign belongs to q_V, and the table's last column counts the assets where it is negative |
| 5 | Gap factors 97 and 8.9 stale | Replaced by the ordering the current table gives: 84x between the dynamic-quantile forecasters and the rest, 48x between Lag-Llama and the truncated Chronos pair |
| 6 | EWMA "Recursive" in Table B.1 vs the lemma saying truncated at 250 | **Table B.1 is right and the lemma remark was wrong.** The documentation described a truncated 250-day sum; the series came from the RiskMetrics recursion, whose memory is infinite. The finite-context assumption therefore does not hold for EWMA as implemented. Stated, with the number: at lambda = 0.94 the weight beyond lag k is lambda^k, 1.9e-7 at k = 250 and below double-precision epsilon at k = 583, so the lemma applies to the filter truncated at m = 600, indistinguishable from the recursion in the arithmetic that produced the series. An approximation argument, labelled as the only one in the paper |
| 7 | "Every coverage-based backtest passes" too absolute | Replaced by what the data support: after correction 335 of 384 pairs are green and every forecaster's violation rate clusters within a fraction of a point of nominal, defective series included |
| 8 | Supplement had boxed links; both documents inconsistent | Same hypersetup in both: black, unboxed. Revision colouring off in both |

A scripted sweep now checks for the whole class -- theorem conditions asserted
to hold, checks called necessary, R described as signed, stale gap factors,
absolute backtest claims, old pair counts, the old green figure, and the oracle
rule stated as deployable. It runs clean.

Final state: manuscript **72 pages**, supplement **24**. Both compile with no
undefined reference or citation, and `paper_numbers.py --check` reports the
registry current.

---

# Round five: length

Target 45--50 pages including the bibliography. **72 -> 50.**

| moved to the supplement | pages |
|---|---|
| All proofs (both lemmas, the theorem, the corollary, the two auxiliary propositions) | ~8 |
| Tail-completion analysis | 3 |
| Full statement of the validation gate | 2 |
| Asset and model universes | 1 |
| Static-vs-rolling, Diebold--Mariano, baselines and Monte Carlo tables; the rolling-threshold, COVID and capital figures | ~4 |
| Prose compressed: rolling and static-vs-rolling merged, simulation and robustness to one paragraph each, multi-quantile and baselines rewritten around absent tables, discussion halved | ~4 |

What stays in the body is what the review asked for: Table 3 (the audit trail),
Table 4 (the master panel), the compact gate table, the two mechanism tables that
carry the dose--response and the alpha-response, and Figure 1. The theory is
reduced to Assumption 3.2, Theorem 3.3, Corollary 3.5 and the remark stating that
the contiguous split does not satisfy the separation the theorem needs.

Section budget now: introduction 5, literature 2, methodology 10, mechanism 4,
results 12, blind spot 3, gate 2, discussion 2, conclusion and back matter 8
(of which the bibliography is about 6).

Supplement grows to 44 pages and carries S.1--S.12. Cross-document references
are plain text in both directions --- neither document needs the other to
compile. The contradiction sweep runs clean on all nine patterns.
