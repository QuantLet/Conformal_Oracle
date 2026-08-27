# What is left to do

Written 27 August 2026, to be picked up on another machine. The ledger beside
this file records what was decided and why; this file records what is not done.
Standing rules are unchanged: no check reports PASS until it has been seen to
fail on a case built to make it fail; every number touched is recomputed from
artefacts; decisions are logged, not asked; what is missing is marked BLOCKED and
passed over.

---

## 0. Where the manuscript currently stands

Sections written and integrated: **§1 abstract and title, §3 methodology
patches, §4 theory, §5 Monte Carlo.** Both documents compile with zero undefined
references and zero undefined citations. Four audits pass:

    .venv_forecasts/bin/python scripts/paper_numbers.py --check
    .venv_forecasts/bin/python scripts/audit_structural_claims.py
    .venv_forecasts/bin/python scripts/audit_qv_convention.py
    .venv_forecasts/bin/python scripts/audit_supplement_targets.py

Current numbering, which the remaining sections must be written against:
1 Introduction · 2 Related Literature · 3 Methodology · **4 Theory** ·
**5 Monte Carlo** · 6 Controlled Case (Chronos) · 7 The Audit ·
8 What the Diagnostics Cannot Diagnose · 9 Structural Validation ·
10 Recalibration · 11 Discussion · 12 Conclusion.

The target structure of K2 renumbers these. §6–§10 have not yet been rebuilt, so
the target §6/§7/§8 are the current §10/§10/§8–9 in content.

---

## 1. The one outstanding item that changes published numbers

**R14 — regenerate both analytic Chronos panels with the corrected
token-to-bin map.** `analysis/phase0/retracted_hypotheses.md` has the full entry.
The shipped analytic series pair each bin's probability with the next bin up;
the library's own decoder settles it 12/12 at `top_k = 1`.

Pending changes, already measured exactly (`analysis/k1_verify/k1a_impact.json`):

| macro | published | corrected |
|---|---|---|
| `\nPiSmallAnalyticOne` | 0.0175 | **0.0173** |
| `\nPiMiniAnalyticOne` | 0.0178 | **0.0177** |
| `\nRatioSmallAnalyticOne` | 1.750 | **1.733** |
| `\nRatioMiniAnalyticOne` | 1.781 | **1.772** |
| `\nPiSmallAnalyticTen` | 0.1036 | **0.1027** |
| `\nPiMiniAnalyticTen` | 0.0989 | **0.0981** |
| `\nKupMiniAnalyticTen` | 22 | **20** |
| `\nKupSmallAnalyticOne`, `\nKupMiniAnalyticOne` | 8 | 8 (unchanged) |
| quantile score, both | — | −0.09%, unchanged at printed precision |

Two routes, and the second is preferred:

1. Exact and model-free: `corrected = stored − binwidth × scale_t`, with
   binwidth = 30/4092 and scale_t the mean absolute value of the 512-day
   context. The rule was verified against 200 re-run dates to a maximum relative
   error of 8.1e-08.
2. Re-run the estimator from the weights with the corrected map. Roughly ten
   minutes per checkpoint on a laptop GPU, 121,923 dates per checkpoint. This is
   the job the faster machine is for.

Everything downstream of the analytic series must then be rebuilt: `all_results`,
the violation and QS sequences, and every table that consumes them.

---

## 2. Sections still to write, in order

**§6 — what recalibration restores.** K0a is decided
(`analysis/k0a_mcb/VERDICT.md`): q̂_V is written as a **measurement, not a
statistic**. It is the out-of-sample estimate of the argmin of the
Gneiting–Resin unconditional miscalibration term, reported in the units of the
forecast. It does **not** support an out-of-sample ordering between forecasters —
rank correlation 0.52 with the evaluation-window optimum on well-specified
series — and that claim says nothing about δ* itself. Content: 335/384 green,
123 → 273 Kupiec, corrected rates tight to nominal for every forecaster, and
therefore the corrected column cannot order anything. Abstract does not change.

**§7 — what it costs and when.** Three things it owes:

- The forward reference §3.2.1 now makes: at α = 0.01 and w = 125 the rolling
  shift **is the window maximum**, because k ≥ n whenever n < 2/α − 1
  (n ≤ 198). w ∈ {125, 250, 500} is not a variance curve; one of the three points
  is a degenerate estimator. This is a better answer to R2-1 than the ablation
  requested, and it is exact rather than empirical.
- The indication rule repositioned as a **consequence of Proposition 5.1**: a
  rule keyed on the Basel zone inherits the zone's resolution, which is a sign.
- The corrected ledger (`analysis/k2_indication/RESULTS.md`). Benefit reported in
  two measures that are not functions of the zone — Δ|π̂ − α| and ΔQS. The 205
  and the 20 stay, as counts of zone changes. **Of the 20 upgrades the deployable
  rule gives up, 11 are pairs the correction degrades on the score; net cost 9.**
  **72 of the 205 upgrades are simultaneously among the 174 deteriorations**, so
  the two columns cannot be traded off as printed.

Also: the full ΔQS distribution, the concentration of deteriorations on
already-calibrated pairs, the gated rule, the oracle, and the distance between
them.

**§8 — why structural validation precedes correction.** K0b is decided
(`analysis/k0b_precedent/VERDICT.md`): the 2012 precedent is **specific**, so
§8 **keeps δ\*** and Table `tab:gate_residual`. It **loses any claim of priority**
over the observation that exceedance-based tests are weak. Non-identification
enters here as the motivation for the ordering, not as the thesis. The truncated
series reaching π̂ = 0.0108 and 19/24 green after correction is the panel fact.

**§2 — related literature.** Four attributions, and **the w\* attribution is the
first sentence of the delimitation paragraph, not the last**: Escanciano & Pei's
optimal weight w\*(I_{t−1}) = F_{I_{t−1}}(m_α(I_{t−1},θ₀)) *is* the manuscript's
u_t. Then: their Theorem 1 is specific to HS/FHS and to the unconditional test,
and their Lemma 1 exhibits a consistent weighted backtest, which is the clean
boundary. Then Gordy–McNeil and Kratz–Lok–McNeil as the class that reads more
than the exceedance, with what they require of the reporting regime. Then
Cont–Deguest–Scandolo on the estimation procedure as part of the risk measure.
Also the multiplication-factor precedent: their factors sit at the 3.0 floor
while the D-test rejects.

**§9 — limitations.** Must carry the pair §5.3 points forward to: the one-bin
offset in the estimator this paper offers as a remedy (R14), and the two defects
in the verification harness itself. Both found by reimplementing from the
description. Plus the nine limitations of the original plan: scalar shift,
unrepaired conditional dynamics, unstable small calibrations, lag at structural
breaks, zero-shot versus fitted asymmetry.

---

## 3. K3 — what moves to the supplement (move, not delete)

- The full Chronos exhibit. The body keeps only the sentence showing the
  guarantee is indifferent to the forecaster. **§4.4, the analytic estimator,
  moves with it** — but the corrected R14 figures still enter the body wherever
  the body cites them.
- The ten-check validation gate, in full.
- The TSFM-versus-GARCH panel.
- The five failure modes, as a pipeline-hygiene note.

All with numbers intact and pointers from the body.

## 4. K4 — the two rejection grounds that structure cannot fix

**K4a.** Zero-shot against per-asset fitted. §3.3.3 declares the asymmetry;
it must now be **defended, or the comparison restricted to within-group**.

**K4b.** Each hyperparameter gets a justification or an ablation: the 70/30
split, the 250-day window, the 512 context, the 1,000 samples, the Student-t
closure. R2-1's w ∈ {125, 250, 500} is partly answered already, analytically,
by the n ≤ 198 result above; the empirical sweep still has to run.

## 5. K5 — recounts and hygiene

Every "N of M" claim against its object. The §1 roadmap against the new
structure. Baseline counts. **Table 4's legend — note that Table 4 is now
`tab:gate_compact`, not `tab:master`.** Remark 3.1 if the ML exhibit lands beside
it. `audit_structural_claims.py` over everything including the new sections.
`paper_numbers --check` clean.

## 6. The final deliverable

A list of every published number that changed against the current PDF. R14's
seven macros are the only ones known to change so far; the §5 regeneration
already moved Tables S.5–S.8 and S.10, and those are recorded in
`analysis/provenance/QV_CONVENTION.md`.

---

## 7. Known open items, carried rather than closed

| item | status |
|---|---|
| `tab_baselines` (Table S.25) | **NOT_EMITTED.** Recovered intermediates carry 9 forecasters and 216 pairs; the published table reflects ten. No evidence the printed values are wrong, and no artefact that produces them. |
| The six TSFM series | **UNVERIFIABLE_HERE.** GPU inference on an A30; sampling is not bit-reproducible across backends. A statement about the machine, not a pass. |
| GARCH-N | **data revised on 24 of 24 assets**, max absolute difference 1.8e-01. Upstream dividend-adjusted histories were restated after the forecasts were written. Cannot be reconciled without the original vintage. Grade WEAKER. |
| "Table I.6" | **BLOCKED.** No referent in the current document; the submitted version's I.6 carries no label. Both candidates by position are panel-independent, so the requested grade does not apply either way. Name the table if a specific one was meant. |
| Eight `LEVEL_K_OVER_N` sites | Declared, not migrated. Measured harmless: 0 of 312 pairs change a violation count. `QV_CONVENTION_SITES.tsv` carries each with its measurement. |
| Table S.26 | Superseded by Table 1 of the manuscript, retained because Table S.6's note contrasts against it. |

## 8. Rhythm to keep

Three self-corrections in three rounds, each caught by recomputing rather than by
rereading: "four times too coarse" when the statistic was invariant to the
defect; "estimation error, not independent content" when the measurement
supported only the narrower claim; the six-pair ρ̂ range attached to a four-pair
ablation. Keep recomputing the thing being asserted.
