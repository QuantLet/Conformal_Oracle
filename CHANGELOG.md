# Replication package — changelog

Corrections to the replication materials for *Recalibrating Tail Risk Forecasts
under Temporal Dependence*. Entries are append-only; superseded files are kept
rather than deleted so the provenance stays auditable.

---

## 2026-08-14 — Moirai-1.1 multi-α results table corrected

### What was wrong

`cfp_ijf_data/paper_outputs/tables/moirai11_full_results.csv` (96 rows,
Moirai-1.1 at α ∈ {0.01, 0.025, 0.05, 0.10}) was not produced by the same code
path as every other model. It differed in two ways:

1. **Quantile estimator.** The conformal shift was computed as
   `np.quantile(scores, 1 - α)` with linear interpolation, rather than the
   conformal order statistic S₍ₖ₎ with k = ⌈(n+1)(1−α)⌉ used everywhere else and
   analysed in Theorem 3.3. An interpolated quantile lies below that order
   statistic and carries no finite-sample coverage guarantee. Verified exactly on
   ASX200 at α = 0.01: the stored value 0.001566 reproduces `np.quantile` linear
   to machine precision; the conformal value is 0.001939.
2. **Calibration/test split.** The house convention is n_cal = ⌊0.70 n⌋, used by
   Moirai-2.0 on all 24 assets. The stale file used it on only 12 of 24 assets
   and ⌈0.70 n⌉ on the other 12, so it was not internally consistent either.

Both differences push q̂_V downward, so the file **understated** the correction
for Moirai-1.1 throughout.

### What it did and did not affect

**The published Table 1 is correct and unchanged.** It was built from
`moirai11_results.csv` (24 rows, α = 0.01), which matches the corrected
computation to machine precision on n_cal, n_test, q̂_V, π̂_raw, π̂_cp, Kupiec
p-values and Basel zones. Table 1's Moirai-1.1 row (Green 21/24, Kupiec 15/24,
R̄ = 0.11) and the aggregate figures (Panel A 127/144 = 88.2%; all pairs
203/240 = 84.6%) all reproduce from the corrected file.

**Worth stating plainly: the error flattered the result.** In Table 5, the stale
file records Moirai 1.1 as rejected on 10 of 24 assets at α = 0.01; the correct
computation gives 9. The stale file made the model look *better* calibrated than
it is, and the correction makes the reported outcome slightly worse. The same
direction holds throughout: both defects push q̂_V downward, understating the
size of the correction the model needs.

Two published outputs were built from the stale file and change:

| Output | Was | Now |
|---|---|---|
| Table 5 (`tab_multiquantile.tex`), Moirai 1.1 at α = 0.01, rejections | 10/24 | **9/24** |
| Figure `fig_qV_ranking`, Moirai-1.1 mean q̂_V | 0.003489 | **0.003897** |

The ranking figure's ordering changes accordingly: Moirai-1.1 moves from 2nd to
3rd lowest mean q̂_V, swapping with Hist-Sim. No other cell of Table 5 changes;
`diff` against the submitted version shows exactly one differing line.

`moirai11_results.csv` (24 rows) was **not** modified — it was already correct.
Note that its `QS_raw`/`QS_cp` columns follow a different sign/scale convention
from the rest of the pipeline and are not the quantile scores reported in
Table 1; they are not consumed by any published number.

### Files

- `cfp_ijf_data/paper_outputs/tables/moirai11_full_results.csv` — replaced.
- `analysis/moirai11_reconciliation/moirai11_full_results.SUPERSEDED-20260814.csv`
  — the stale file, preserved verbatim.
- `analysis/moirai11_reconciliation/rebuild_moirai11.py` — regenerates the
  corrected table and the full diff.
- `analysis/moirai11_reconciliation/RECONCILIATION.md` — diagnosis and
  per-cell comparison.

`cfp_ijf_data/` is distributed as a GitHub Release asset and is excluded from
version control, so **the Release asset must be re-uploaded** for this correction
to reach users. If the package has been deposited on Quantinar or attached to the
SSRN entry, those copies need the same replacement and this note.

### A larger reproducibility gap found while verifying the above

Checking which pipeline path produced the aggregate Green count turned up
something wider than the Moirai-1.1 defect, and it should be fixed before the
package is redistributed.

**`Quantlets/CO_full_evaluation/run_master_table.py` does not produce the
published Table 1.** The script emits a 10-column table over **nine** models,
with TimesFM 2.5 and Moirai 2.0 in Panel A (signal-preserving). The published
table is 12 columns over **ten** models, adds a CC-pass column and an R̄ column,
and places TimesFM 2.5 and Moirai 2.0 in Panel B (replacement) — the assignment
the paper argues for. The script predates the finding that those two models have
~99% raw violation rates, and it never reads any Moirai-1.1 input.

Consequently:

- **Table 1 is not regenerable from the shipped scripts.** Its values are correct
  — they reproduce from `moirai11_results.csv` and `all_results.csv` — but no
  script in the package assembles them into the published table.
- **The aggregate "Overall: 203/240 Green (84.6%), 32 Yellow, 5 Red"** in the
  table note, repeated in Section 4.3, is not emitted by any script. It is
  consistent with the correct inputs (182/216 from `all_results.csv` plus 21/24
  for Moirai-1.1), but it is a hand-entered figure.

This violates the project's own rule that every number in the text be traceable
to a pipeline script. `run_master_table.py` needs to be brought up to the
published specification — ten models, correct panel assignment, CC and R̄
columns, and the aggregate emitted rather than typed — before the replication
package is redistributed.

An earlier draft of this entry claimed the stale file yielded 205/240 Green
against the correct 203/240. That comparison is withdrawn: 205/240 is what one
obtains by concatenating `all_results.csv` with the stale file, but **no script
in the package takes that path**, so the figure was not reproducible and should
not have been offered as evidence. The Table 5 rejection count above is the
demonstrable version of the same point.

### Related disclosure

This is the second occurrence of the same class of error — an empirical quantile
substituted for the conformal order statistic. The first, already disclosed in
*Data and Code Availability*, affected the rolling path in the distributed
package. The disclosure paragraph should be widened to cover both: the rolling
path in the code package, and the static path in this results table. A partial
disclosure that names one incident while a second is discoverable is worse than
none.

---

## Convention notes recorded during this audit

These are not corrections — they are undocumented conventions that the paper
should state explicitly, recorded here so the package and the text agree.

- **"CC pass" columns.** Table 1 counts a *degenerate* Christoffersen test as a
  pass (n₁₁ = n₁₀ = 0, statistic undefined, stored as NaN); Tables 2 counts only
  informative passes. The same statistic therefore appears as 15/24 (Table 1) and
  4/24 (Table 2) for Chronos-Small. Table 2's convention is the defensible one.
  See `analysis/cc_column/`.
- **The statistic itself** is the Christoffersen *independence* LR ~ χ²₁, not the
  joint LR_CC = LR_POF + LR_IND ~ χ²₂ that Appendix G defines. Independence-only
  reproduces Table 2 in 9/9 rows; the joint statistic reproduces 3/9.
- **Bibliography.** Eight entries carried defective metadata (non-existent DOIs,
  wrong venue, invented co-authors) and were corrected against Crossref/arXiv.
  `scripts/audit_bib.py` and `scripts/audit_refs.py` verify a bibliography or a
  formatted reference list against Crossref, arXiv and doi.org; see
  `reports/bib_audit.md` and `reports/phase0_memo.md`.
