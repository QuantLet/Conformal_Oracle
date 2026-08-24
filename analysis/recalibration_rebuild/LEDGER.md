# Rebuild ledger — one entry per item

Reconstruction of the manuscript as a recalibration paper. Theory at the centre;
non-identification demoted from thesis to the reason for the ordering. Nothing is
deleted from the repository; material leaving the body moves to the supplement
with its numbers intact.

Status values: DONE, IN PROGRESS, BLOCKED, NOT STARTED.

| item | status | decision / finding | artefact |
|---|---|---|---|
| K1a Chronos analytic rebuilt from prose | DONE | **FAILS.** Stored analytic series is one quantisation bin above the library's own decoder, on 4 asset×checkpoint combinations. Adjudicated at top_k=1 with no Monte Carlo error, 12/12. π̂ 0.0175→0.0173 and 0.0178→0.0177; Kupiec at α=0.10 for Chronos-Mini-A 22/24→20/24. Nothing qualitative moves. | `analysis/k1_verify/k1a_*` |
| K1b1 π̂ for TimesFM 2.5, Moirai 2.0 | DONE | VERIFIED to 6 dp from returns + series. Printed figures are the **cell mean**, not the pooled panel rate (0.0147 / 0.0182); the manuscript must say which, once. | `analysis/k1_verify/k1b1_*` |
| K1b2 distinct values at default top_k | DONE | VERIFIED on the 40-cell subsample: exactly 50 in 40/40. Controls fire at top_k=10 (→10) and 4094 (→331–648). Packaged defaults are top_k=50, top_p=1.0, temp=1.0, num_samples=20. | `analysis/k1_verify/k1b2_*` |
| K1b3 n₁₁ = n₁₀ = 0 vs code's CC pass | DONE | Percentages 34.6 / 53.5 VERIFIED. **Stated mechanism false**: zero pairs have n₁₁ = n₁₀ = 0; all 108 (and 167) are n₁₁ = 0 < n₁₀. "Too few exceedances to populate a transition table" must be rewritten. | `analysis/k1_verify/k1b3_*` |
| K1c CAViaR-AS 15/24 | DONE | VERIFIED. Second implementation agrees on the count and on the per-asset vector, 24/24, both windows. Task premise ("unverified") was out of date. | `analysis/k1_verify/k1c_*` |
| K1d evidence grade | DONE (1 BLOCKED) | `tab_baselines` NOT_EMITTED; six TSFM series UNVERIFIABLE_HERE; GARCH-N data-revised on 24/24. "Table I.6" has no referent — BLOCKED, and both candidates are panel-independent anyway. | `analysis/k1_verify/RESULTS.md` |
| K0a q̂_V vs uMCB | DONE | q̂_V is a rewriting of the **argmin** δ*, not of uMCB (the value). ρ = 0.993 on the calibration window, **0.52 on the evaluation window among well-specified series**. §6 writes q̂_V as measurement, not statistic. Abstract unchanged. AE-3 verified: in-sample improvement / uMCB = 0.99987 median. Independently reproduces `\nDegradedStatic` = 66. | `analysis/k0a_mcb/VERDICT.md` |
| K0b Escanciano & Pei and the precedent | DONE | Precedent is **specific** — to HS/FHS estimators and to the unconditional test; their own Lemma 1 gives a *consistent* weighted backtest, running opposite to Prop 6.1. §8 keeps δ*. §2 gains four attributions; their optimal weight w* **is** the manuscript's u_t. | `analysis/k0b_precedent/VERDICT.md` |
| K2 §1 Introduction | NOT STARTED | | |
| K2 §2 Related Literature | NOT STARTED | after K0b | |
| K2 §3 Methodology, eq. (10) | NOT STARTED | third variant found: Supplement `prop:rolling_drift` and eq. (S.31) define q_V^roll as the plain $Q_{1-\alpha}$, contradicting §3.2.1 and Data & Code | |
| K2 §4 Theory | NOT STARTED | uncontested; write at full length | |
| K2 §5 Monte Carlo | NOT STARTED | | |
| K2 §6 What recalibration restores | NOT STARTED | after K0a | |
| K2 §7 What it costs and when | NOT STARTED | | |
| K2 §8 Why structural validation precedes correction | NOT STARTED | after K0b; keeps δ* | |
| K2 §9 Limitations | NOT STARTED | | |
| K3 moves to supplement | NOT STARTED | | |
| K4a zero-shot vs fitted | NOT STARTED | | |
| K4b hyperparameters, w ∈ {125,250,500} | NOT STARTED | | |
| K5 recounts and hygiene | NOT STARTED | | |
