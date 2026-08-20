# Referee and AE comments: what was done, and what the corrections invalidate

Every comment from the IJF round, the analysis that answers it, and its status
after the four defect corrections.

**The governing fact on this page:** Phases 1, 2 and 3 were computed *before*
the defects were found, on the series that contained them. Their conclusions are
therefore provisional until re-run. Some will be unaffected, some will change
sign. Which is which is not a matter of judgement and must not be guessed —
each is re-run and reported. An analysis whose input changed and whose output
did not is a claim requiring evidence, not a default.

Corrections applied to the series since those analyses ran:

| defect | models affected | effect on the series |
|---|---|---|
| sign inversion | Moirai-2.0, TimesFM-2.5 | π̂ 0.988/0.990 → 0.0166/0.0141 |
| raw t₅ quantile, df hard-coded | GJR-GARCH | π̂ 0.0042 → 0.0194, width −33% |
| `top_k = 50` sampler default | Chronos-Small, Chronos-Mini | π̂ 0.39/0.42 → 0.0175 (analytic) |
| `CACT`/`FCHI` alias | 6 scripts, 5 stale files | quarantined |

---

## AE point 4 — does recalibration help every model–asset pair?

**Analysis:** `analysis/ae_point4/run_ae_point4.py` — ΔQS distributions, the
cross-tabulation of deterioration against raw calibration quality, ΔQS
regressed on |π̂_raw − α|, and the Green→Green count.

**Status: RE-RUN 2026-08-17, and the answer changed.** The original finding —
84 pairs degraded, 48 of them already Green — was computed with two forecasters
at π̂ ≈ 0.99, where "deterioration" is not a meaningful notion, on a 10 × 24
panel. The panel is now 13 × 24 and those two forecasters run at 0.0143 and
0.0178.

| α | estimator | pairs | deteriorate |
|---|---|---|---|
| 0.01 | single split | 312 | 66 (21%) |
| 0.01 | rolling | 312 | **174 (56%)** |
| 0.025 | single split | 312 | 64 (21%) |
| 0.025 | rolling | 312 | 160 (51%) |
| 0.05 | single split | 312 | 90 (29%) |
| 0.05 | rolling | 312 | 194 (62%) |
| 0.10 | single split | 312 | 117 (38%) |
| 0.10 | rolling | 312 | **202 (65%)** |

The answer to the AE is now direct rather than hedged: **no, and the pairs it
harms are identifiable in advance.** Deterioration concentrates on pairs whose
raw forecast already passes Kupiec comfortably — GJR-GARCH-t/BTC at π̂ = 0.0110
(p = 0.741), Chronos-Mini-A/ICLN at 0.0093 (p = 0.802), TimesFM-2.5/BTC at
0.0063 (p = 0.187). Recalibrating an already-calibrated forecaster adds the
estimation noise of q̂_V and has nothing to remove, so ΔQS is negative by
construction. Under the rolling estimator, where q̂_V is re-estimated on a
250-day window and is correspondingly noisier, the deterioration rate roughly
doubles at every α.

That is a usable decision rule and it is what the AE was asking for. It was not
visible before because two series at π̂ ≈ 0.99 dominated every pooled statistic.

`GRID_FAILURES` in the script has been updated accordingly: it used to name
TimesFM-2.5 and Moirai-2.0 as "interface failures, not forecasters", which was
the sign defect being mistaken for a property of the quantile-grid interface. It
now names the two Chronos series sampled at `top_k = 50`, whose R̄ is 17.3 and
23.5 against 0.09–0.36 for every other forecaster.

The re-run also changes the *interpretation* rather than only the number. With
the defects in place, deterioration looked like a property of recalibration.
With them removed, every forecaster under-covers modestly (1.4×–2.9× nominal),
which is the regime where the marginal/conditional distinction of
`CO_marginal_vs_conditional` bites — deterioration in QS is what a marginal
guarantee buys at the cost of conditional response.

## AE point 7 — sensitivity to the GARCH estimation window

**Analysis:** `analysis/phase3_windows/run_window_sensitivity.py`, w ∈ {250, 500, 1000}.

**Status: MUST BE RE-RUN for GJR-GARCH.** The window sweep was run against the
old quantile map, so its GJR rows describe a model that used a raw t₅ multiplier
with hard-coded df. GARCH-N, EWMA and Hist-Sim rows are unaffected — their
series are unchanged and verified reproducible by
`scripts/verify_producers.py`.

## Referee 1, point ix — dynamic quantile benchmarks

**Analysis:** `analysis/phase3_dynamic/run_dynamic_var.py` — CAViaR-SAV,
CAViaR-AS, GAS-t. Entered Table 1 as rows.

**Status: UNAFFECTED, and now better motivated.** These are estimated directly
from returns and touch none of the corrected series. Their standing improves:
with GJR-GARCH no longer accidentally over-conservative, the dynamic-quantile
benchmarks are competing against a correctly specified parametric family rather
than one inflated by 45%.

## Referee — relation of q̂_V to the Gneiting–Resin decomposition

**Analysis:** `analysis/umcb/run_umcb.py`. Verdict (a): q̂_V is the
unconditional miscalibration component, not a new object.

**Status: RE-RUN, conclusion expected to stand.** Verdict (a) is an identity
between estimators, not an empirical finding, so it does not depend on which
series it is evaluated on. The *magnitudes* reported alongside it do, and are
recomputed. Reporting verdict (a) plainly was right and stays right.

## AE — over-claiming from the binary regime classification

**Analysis:** Phase 4. The signal-preserving/replacement split, the R > 1
threshold, the 20-day persistence rule and Table I.1 were removed; R is kept
continuous and Table 1 is ordered by it.

**Status: STRUCTURALLY DONE, values MUST BE RE-RUN.** R̄ = mean over assets of
|q̂_V| / |VaR_raw| is a ratio in which *both* terms move for four forecasters.
The ordering of Table 1 is by R, so the corrections can reorder the table
itself, not merely its entries.

## New material the comments did not ask for, and why it is here

**`scripts/promotion_gate.py`** — ten structural checks on a forecast series.
Not a response to a comment; a response to the fact that four defects passed
every backtest in the submitted paper. Recalibration *restored coverage on a
sign-inverted forecaster* (π̂ 0.0146, 19/24 Basel Green), so no coverage-based
test could have caught them. Something that does not use coverage was required.

**`Quantlets/CO_marginal_vs_conditional/`** — measures the gap the theorem does
not cover: corr(VaR_cp, σ_t) = +0.530 on the inverted input against −0.530 on
the corrected one, unanimous across 24 assets, with marginal coverage attained
either way. This is the empirical content of "marginal, not conditional".

**GJR-GARCH-t** — added because correcting GJR to the Gaussian innovation the
manuscript describes leaves it under-covering (Kupiec 0/24), which makes the
absence of a fat-tailed parametric benchmark the obvious next question.

---

## Re-run order

Nothing downstream is regenerated until the series are final, so that the paper
moves once rather than three times.

1. Chronos analytic series, both sizes → gate → promote *(pending)*
2. GJR-GARCH-t → gate → promote *(rebuilding with the convergence guard)*
3. `Quantlets/CO_full_evaluation/run_full_evaluation.py --write`
4. AE point 4, AE point 7, uMCB, R̄, rolling recalibration — all re-run
5. Table 1 and every dependent table and figure, with a before/after table
