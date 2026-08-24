# Rebuild ledger — one entry per item

Reconstruction of the manuscript as a recalibration paper. Theory at the centre;
non-identification demoted from thesis to the reason for the ordering. Nothing is
deleted from the repository; material leaving the body moves to the supplement
with its numbers intact.

Status values: DONE, IN PROGRESS, BLOCKED, NOT STARTED.

| item | status | decision / finding | artefact |
|---|---|---|---|
| K1a Chronos analytic rebuilt from prose | DONE, **R14** | **FAILS.** Stored analytic series is one quantisation bin above the library's own decoder, on 4 asset×checkpoint combinations. Adjudicated at top_k=1 with no Monte Carlo error, 12/12. Registered as R14 (R11 slot taken); new pattern, and PROTOCOL.md Rule 2 gains the tolerance subsection. The announced validation was **blind, not coarse**: a uniform support translation leaves predictive std exactly invariant, and the coverage check ran on 40 dates at α=0.01 (grid 1/40). | `analysis/k1_verify/k1a_*`, `analysis/phase0/retracted_hypotheses.md` |
| K1b1 π̂ for TimesFM 2.5, Moirai 2.0 | DONE | VERIFIED to 6 dp from returns + series. Printed figures are the **cell mean**, not the pooled panel rate (0.0147 / 0.0182); the manuscript must say which, once. | `analysis/k1_verify/k1b1_*` |
| K1b2 distinct values at default top_k | DONE | VERIFIED on the 40-cell subsample: exactly 50 in 40/40. Controls fire at top_k=10 (→10) and 4094 (→331–648). Packaged defaults are top_k=50, top_p=1.0, temp=1.0, num_samples=20. | `analysis/k1_verify/k1b2_*` |
| K1b3 n₁₁ = n₁₀ = 0 vs code's CC pass | DONE | Percentages 34.6 / 53.5 VERIFIED. **Stated mechanism false**: zero pairs have n₁₁ = n₁₀ = 0; all 108 (and 167) are n₁₁ = 0 < n₁₀. "Too few exceedances to populate a transition table" must be rewritten. | `analysis/k1_verify/k1b3_*` |
| K1c CAViaR-AS 15/24 | DONE | VERIFIED. Second implementation agrees on the count and on the per-asset vector, 24/24, both windows. Task premise ("unverified") was out of date. | `analysis/k1_verify/k1c_*` |
| K1d evidence grade | DONE (1 BLOCKED) | `tab_baselines` NOT_EMITTED; six TSFM series UNVERIFIABLE_HERE; GARCH-N data-revised on 24/24. "Table I.6" has no referent — BLOCKED, and both candidates are panel-independent anyway. | `analysis/k1_verify/RESULTS.md` |
| K0a q̂_V vs uMCB | DONE | q̂_V is a rewriting of the **argmin** δ*, not of uMCB (the value). ρ = 0.993 on the calibration window, **0.52 on the evaluation window among well-specified series**. Conclusion narrowed: **q̂_V does not support an out-of-sample ordering between forecasters** — this says nothing about δ*. §6 writes q̂_V as measurement, not statistic. Abstract unchanged. AE-3 verified: in-sample improvement / uMCB = 0.99987 median. Independently reproduces `\nDegradedStatic` = 66. | `analysis/k0a_mcb/VERDICT.md` |
| K0b Escanciano & Pei and the precedent | DONE | Precedent is **specific** — to HS/FHS estimators and to the unconditional test; their own Lemma 1 gives a *consistent* weighted backtest, running opposite to Prop 6.1. §8 keeps δ*. §2 gains four attributions; their optimal weight w* **is** the manuscript's u_t. | `analysis/k0b_precedent/VERDICT.md` |
| **R14 registered** | DONE | new pattern: a check calibrated on a statistic the defect leaves invariant. Second instance in this project (first: the −3.500 band edge blocking 0 of 312 cells). `PROTOCOL.md` Rule 2 gains "The tolerance is part of the check". | `analysis/provenance/PROTOCOL.md`, `analysis/phase0/retracted_hypotheses.md` |
| **corrected analytic figures** | PENDING REGEN | π̂ 0.0175→**0.0173**, 0.0178→**0.0177**; ratios 1.750→**1.733**, 1.781→**1.772**; π̂(α=0.10) 0.1036→**0.1027**, 0.0989→**0.0981**; `\nKupMiniAnalyticTen` 22→**20**. Kupiec at α=0.01 unchanged at 8/24. QS −0.09%, unchanged at printed precision. Enter wherever they appear, including what stays in the body after the Chronos exhibit moves to the supplement. | `analysis/k1_verify/k1a_impact.json` |
| §4.4 analytic estimator | NOT STARTED | moves to the supplement **with** the Chronos exhibit (K3); the corrected numbers still enter the body wherever the body cites them | |
| K2 §1 Introduction | NOT STARTED | | |
| K2 §2 Related Literature | NOT STARTED | after K0b | |
| K2 §3 Methodology, eq. (10) | PARTLY DONE | **Three** further variants found, two now fixed. (a) Supplement `prop:rolling_drift` and its proof wrote the plain $Q_{1-\alpha}$ — **fixed**, now states eq. (9) with the O(1/w) gap named. (b) Supplement `lem:quantile_mixing` wrote $\hat F_n^{-1}(1-\alpha)$ — **fixed**, the body's Lemma 4.4 states eq. (8) and the O(1/n) gap is absorbed into the remainder explicitly. (c) `run_simulation_study.py` uses `np.quantile` at level k/n with linear interpolation, a **third convention** — gap 2.8e-06 (0.01%), numerically negligible but contradicts "every static and rolling result in this paper uses equation (8)". Still to reconcile in §3. | `analysis/k2_sim/GATE_REVISION.md` |
| K2 §4 Theory | **DONE** | Theory moved from Supplement S.9 into the body as §4, at full length: Assumption 4.1, Lemmas 4.2/4.4, Theorem 4.5, **Corollary 4.6 — newly stated**, Propositions 4.7/4.8, Remarks 4.3/4.9. The GARCH corollary previously had a proof and **no statement anywhere in either document**. Supplement S.9 reduced to proofs only; every S.9.x back-reference repointed and verified against its heading. | `sections/sec4_theory.tex` |
| K2 §5 Monte Carlo | **DONE** | Grid extended from 2 to 5 sample sizes, T ∈ {500…10000}, 12,500 replications. Published Table S.26 **reproduces exactly** from its per-replication artefact. All four pre-registered predictions hold, including the fourth: **the raw Basel classification diverges rather than converging** — Student-t(5), miscovering by 51%, is green 61.6% → **74.2%** as T grows twentyfold. New: `figures/fig_mc_convergence`. | `sections/sec5_montecarlo.tex`, `analysis/k2_sim/` |
| K2 §6 What recalibration restores | NOT STARTED | after K0a | |
| K2 §7 What it costs and when | NOT STARTED | | |
| K2 §8 Why structural validation precedes correction | NOT STARTED | after K0b; keeps δ* | |
| K2 §9 Limitations | NOT STARTED | | |
| K3 moves to supplement | NOT STARTED | | |
| K4a zero-shot vs fitted | NOT STARTED | | |
| K4b hyperparameters, w ∈ {125,250,500} | NOT STARTED | | |
| K5 recounts and hygiene | IN PROGRESS | New audit `scripts/audit_supplement_targets.py`: the existing reference check passed on *resolution*, which is satisfied by a reference pointing at the wrong subsection — and that is what the renumbering produced. The new one prints the target heading beside the citing sentence. Orphan logged: `cfp_ijf_data/paper_outputs/tables/simulation_results.csv` (6 April) disagrees with the published table on every number and backs nothing. | `scripts/audit_supplement_targets.py` |
