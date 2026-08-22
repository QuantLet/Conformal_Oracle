# Change log: rebuilding the manuscript as a standalone study

**Title.** *What Backtests Cannot Diagnose: Structural Validation of Tail-Risk
Forecasting Pipelines* (unchanged).

**Length.** Main text 50 pages including references (body 1--45, bibliography
46--50). Supplement 46 pages. Both compile clean; one residual overfull box of
0.69 pt, which is below a hairline and invisible.

**Build state.** `paper_numbers.py --check` reports `numbers.tex` current.
`audit_prose_numbers.py` examines 116 inline literals (down from 154, the
difference having been converted to generated macros) and finds 0 unsourced.
No undefined or multiply-defined references in either document; no `??` in
either PDF.

---

## 1. Claims removed

| removed | where it was | why |
|---|---|---|
| The entire corrigendum framing: that the paper corrects a working paper of its own, that the defects are the authors', that a defect "that flatters its owner" was found last | abstract, introduction, Sections 4--7, discussion, conclusion | The study is now presented on its own evidence. The history is preserved in `analysis/provenance/REVIEW_RESPONSE_R2.md`. |
| "the failures are neither rare nor exotic" | abstract | A prevalence claim the design does not identify. Replaced by an explicit statement that prevalence is not identified by this study. |
| "every backtest downstream of it passed" / "every one of them passed" | abstract, introduction, conclusion | Universal-pass claim. Replaced by the measured quantities: 273 of 384 pairs pass Kupiec after correction, and the truncated series reach 19 of 24 green zones at pi-hat = 0.0108. |
| Section 6.4, "The uncomfortable retrospective" | Section 6 | Rests on reading superseded tables as evidence, and on the "signal-preserving vs effective replacement" partition that is not restored. Its transferable point survives in Section 5.2.1. |
| Section 5.7, "The Moirai within-family comparison, withdrawn" | Section 5 | Retired as a standalone withdrawal. Its substance -- that the two releases differ on architecture, patching and pretraining corpus simultaneously, so no interface effect is attributable -- is now two sentences in Section 5.2.2, stated forward rather than as a retraction. |
| "R as an audit statistic" presented and then withdrawn | Section 5.2.1 | Now stated positively: R is a scale-interpretable restatement of the violation rate (Spearman 0.991 across the 14 well-specified forecasters), useful for magnitude, not an independent instrument. |
| The claim that the gating rule "does not bind at alpha >= 0.025 in the way an earlier draft claimed" | Section 5.4 | Reworded to state what the rule does at each level, without reference to a prior claim. |
| Naming one surveyed paper as "the clean case" of non-reporting | Section 2 | Reframed as an observation about a missing reporting convention, with an explicit disclaimer that prevalence is not inferred. Attribution is now generic (`\citep[e.g.][]{...}`). |

## 2. Material relocated

| item | from | to |
|---|---|---|
| Sampling dose--response table | main text Table 2 | Supplement Table S.26 (new S.12.1) |
| Alpha-response table | main text Table 3 | Supplement Table S.27 |
| The three non-forecast blocks of the old audit-trail table (analysis-code items, claims restated, theory repairs) | main text Table 3 | `analysis/provenance/REVIEW_RESPONSE_R2.md`, round six |
| The rolling-recalibration and static-versus-rolling subsections, which duplicated one another | Sections 5.5 and 5.6 | merged into one Section 5.5 |

**Main text now retains exactly three tables and one figure:** Table 1, the
five implementation failure modes; Table 2, the master forecast comparison;
Table 3, the compact structural gate; Figure 1, the Basel traffic-light heatmap.

## 3. Substantive rewrites

- **Abstract** rebuilt at 217 words, carrying only the identification problem,
  the 24 x 16 design, the reproducible Chronos mechanism, the limits of
  backtesting and post-hoc recalibration, the structural-validation
  contribution and the comparative conclusion. Three headline numbers
  (0.3884, 0.0175, 69.9%).
- **Table 1** restated as a neutral taxonomy: failure mode, the diagnostic that
  identifies it, and the violation rate measured with the failure mode present
  and absent on identical models, assets and dates.
- **The five failure modes are enumerated once,** in Table 1. The abstract,
  introduction and Section 6 now refer to them without re-listing.
- **Chronos** is presented as a controlled, independently reproducible case
  study on a public checkpoint and published code, explicitly not as a fault in
  Chronos and not as an attribution of blame.
- **Christoffersen boundary cases** are described as *degenerate*, with the
  transition probability estimated at the boundary of the parameter space and
  the chi-squared calibration invalid there -- not as mathematically undefined.
  Changed in six places including the Table 2 note.
- **R** is defined as a ratio of absolute values, non-negative by construction,
  with the sign of q_V reported separately and the count of negative assets
  given in the last column of Table 2.
- **Panels** are tagged throughout: 16 x 24 = 384 main, 13 x 24 = 312 sequence.
  One previously untagged "all 312 pairs" in Section 3.2.2 now carries its tag.
- **Gating result** reported at its deployable value everywhere: 44 of 174
  deteriorations avoided (25.3%), 20 of 205 Basel upgrades lost. The
  evaluation-window variant is labelled an oracle and an upper bound.
- **Conclusion** states the scientific result and does not retell the audit.

## 4. Two contradictions found and fixed

1. **Supplement S.1 contradicted its own table.** The closing paragraph asserted
   that Z2 "does not reject for any model in either the raw or corrected
   configuration", while Table S.2 shows 0/24 passing for both truncated Chronos
   series raw and 3/24 corrected. The paragraph was stale. Rewritten against the
   table, together with the table note, which described the Chronos values as
   "large" when they are large *negative*.
2. **EWMA was documented two ways.** Section 3.3.3 stated that all classical
   benchmarks use rolling windows of 250 observations; Table B.1 records EWMA as
   `lambda = 0.94, Recursive`, and Lemma S.9.1's remark states the series came
   from the RiskMetrics recursion. The exception is now carried in all three
   places, with the numerical argument (weight beyond lag k is lambda^k, which
   is 1.9e-7 at k = 250), and the warm-up sentence now says "the windowed
   classical benchmarks".

## 5. Presentation and typesetting

- Hyperlinks: `hidelinks`, `colorlinks=false`, `pdfborder={0 0 0}` in both
  documents. Verified in the compiled PDFs: 292 and 160 link annotations, all
  borders `0 0 0`, no coloured link text.
- Table overflow: the master table went from 52.2 pt over the measure to
  fitting, via `\tabcolsep` 4 pt -> 1.5 pt (fixed at the producer,
  `build_table1_r2.py`, so it survives regeneration) and by renaming the
  truncated rows `Chronos-* (top-k = 50)` -> `Chronos-* (default)`. Table 1 was
  rebuilt with ragged-right text columns and a spanning `pi-hat at alpha = 0.01`
  header; Table 3 narrowed. All three verified by rendering the pages.
- The Table 2 caption claimed a generated summary note "beneath it" that was
  never included. The note is now actually input beneath the table, and its own
  wording ("two series carrying a traced sampler defect") was neutralised at the
  producer.
- Two LaTeX artefacts repaired: `PropositionsProposition~S.9.4 andProposition`
  and `Equationequation~(1)`.
- The supplement now inputs `numbers.tex`, so a figure quoted there cannot drift
  from the same figure in the main text.

## 6. Unresolved scientific limitations, stated in the paper

These are carried in the text rather than resolved, and none is softened:

1. **Theorem 3.3 does not cover the estimator as run.** It requires a gap
   `g_n -> infinity` exceeding the context length m; the protocol is a
   contiguous 70/30 split with `g_n = 0`. The gap ablation (pi-hat moves by at
   most 0.0005 full-sample, 0.0058 within COVID) is offered as empirical
   evidence, explicitly not as proof.
2. **The rolling estimator has no conformal validity.** It is an operational
   heuristic supported by simulation and by the time-average and drift bounds of
   Supplement S.9, one of which is itself labelled heuristic because the drift
   measure is not estimable from one path.
3. **The gate is in-sample on the failures it was written for.** All ten checks
   were specified after the failure modes had been identified. Seven of the ten
   are plausibility bands with chosen cutoffs; two need an evaluation window.
4. **The discrimination exercise is small and closed.** Thirteen forecasters,
   five positives, and the positives are the failure modes this audit found.
5. **Prevalence is not identified.** Five failure modes in one pipeline says
   nothing about how often such failures occur in the literature.
6. **The comparison is asymmetric.** Benchmarks are fitted per asset and
   re-estimated at every step; foundation models are strictly zero-shot. No
   conclusion rests on a cross-group score ranking.
7. **Causal attribution is unavailable.** The cross-sectional association
   between q_V and asset volatility is consistent with pretraining domain
   mismatch, but the zero-shot design does not identify it.
8. **A structural check would not have caught the Z2 sign convention.** The
   series were well-formed and the statistic was misassembled. This is a limit
   on the paper's own thesis and is stated as one in Supplement S.1.
9. **Recalibration does not deliver a uniformly passing panel.** 273 of 384
   pairs pass Kupiec after correction; the truncated series pass on 14 and 11 of
   24 assets.
