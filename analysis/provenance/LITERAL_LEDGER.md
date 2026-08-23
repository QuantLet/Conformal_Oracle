# Literal ledger

Every numeric literal in the prose of `main_R2.tex` and `supplement.tex` that is
not emitted by `paper_numbers.py` and not on the declared-constants list.

The occasion for this ledger: a p-value of `0.035` was typed into Section 5.6,
matched no artefact, and reported a statistical conclusion that inverts once the
artefact is read. `paper_numbers.py` never saw it, because a literal that is
never emitted cannot be checked by a tool that checks emitted values. Every
remaining literal is treated as a suspect until traced.

Status values: **VERIFIED** — recomputed this session against a named artefact.
**UNSOURCED** — no artefact produces it. **REPLACE** — sourced but drawn from an
undeclared panel or mislabelled, already scheduled for change.

Starting population 102. After stripping typesetting lengths, citation locators
and grant contract numbers (not claims), and after the declared-constants list,
**79 remain**.

## UNSOURCED — the class that produced the 0.035 defect

| where | literal | claim | finding |
|---|---|---|---|
| Supp. S.3 | `0.035`, `97` | "mean absolute difference from the canonical formulation is 0.035 Z2 units ... agree on the 5% pass/fail classification for 97% of model-asset pairs" | `z2_verification.csv` holds three models and a median canonical Z2 each. No per-pair canonical-versus-modified comparison exists anywhere. Neither figure is recomputable. |
| Supp. S.2.2 | `88.9`, and the `5/9`, `0/9` denominators | GBM-QR tuning at eta = 0.05 "pays in coverage ... 88.9% Green (vs. 100%)" | `tuned_gbm_qr_grid.csv` covers 13 models; its green rates are 82.7% at eta = 0.01 and 76.9% at eta = 0.05. The paragraph reports a 9-model subset that is nowhere declared, or a superseded vintage. |
| Main S5.7 | `0.0750`, `0.0443` | ACI mean absolute VaR against the rolling estimator | Real, but from `CO_aci_baseline`, a **216-pair** panel neither document prints. Filed as P1 in Phase 0. |
| Main S5.7 | `97.4` | "ACI reaches higher **coverage** (97.4%)" | The column is Basel Green percentage, not coverage. ACI coverage is 98.7%. Filed as D4. |
| Main S4.4 | `1.041`, `1.038` | analytic against full-vocabulary dispersion "in units of realised volatility" | The normalisation is not stored with the artefact; only the 0.3% agreement reproduces (at 0.27%). Filed as N3. |

## VERIFIED this session

Grouped by the artefact that produced them. Each was recomputed, not
substring-matched.

| artefact | literals verified |
|---|---|
| raw parquet series + returns, reimplemented protocol | `0.4` (GJR rate with the failure mode present), and every macro-backed rate |
| `all_results.csv` | `0.968`, `0.891`, `98.9`, `0.0119`, `0.0114`, `0.0113`, `0.0111`, `0.010`, `67.3`, `89.7` |
| `tab_master_results_r2.csv` | `4.67`, `5.83`, `17.3`, `23.5`, `0.001`, `0.18`, `0.98`, `1.14` |
| `tab_dose_response.csv` | `0.001`, `0.006`, `98.8`, `0.128`, `0.3942`, `1.08` |
| `tab_alpha_response.csv` | `1.35`, `1.15`, `1.33`, `1.12`, `0.3` |
| `tab_dm_configuration.csv` | `5.03`, `5.81`, `5.00`, `5.69`, `0.001` |
| `tab_dm_pvalues.csv` | the 18-of-30 count |
| `wild_cluster_kupiec.csv` | `0.058`, and the four pooled rates |
| `tab_baselines.tex` | `5.14`, `89.1`, `50.6`, `0.078` |
| `analysis/detection/DETECTION.md`, `VERDICT.md` | `0.49`, `0.00078` |
| `covid_response_lags.csv` | the 7 / 4 / 2 lag counts |
| `simulation_study_results.csv` | the 76-81 and 96-98 ranges |
| `gap_ablation` | `0.0005`, `0.0058` |
| bound-validation artefacts | `0.18`, `0.67`, `0.109`, `0.138` |
| `tab_tail_closure_extended.tex` | `1.70`, and the closure range |
| derived in text, arithmetic shown | `1.9` (lambda^250), `1.22` (3.65/3.00), `0.24` (macro difference) |
| computed this session, Phase 2 | `2.61`, `1.32`, `0.008`, `0.399`, `0.69`, `9.3` |

## Not yet traced

| where | literal | note |
|---|---|---|
| Supp. S.10 | `3.3`, `76`, `0.019`, `0.075` | the closure-factor extremes and the two TimesFM rates. The table has per-asset sub-blocks that my parse does not separate cleanly; a proper per-asset extraction is needed before either confirming or disputing them. Flagged rather than asserted in either direction. |

## Guard status

`scripts/build_guards.py` fails the build on any literal outside this ledger's
declared list. It is currently **red at 65 (main) + 14 (supplement)**: the
verified literals above are traced but not yet converted to generated macros.
Conversion is the remaining mechanical work and is what turns the ledger from a
one-off audit into a standing check. The guard runs its negative control first
and reports BROKEN if the control does not fail.


## B6 recount list, extended

Added after the ML placement decision:

- **Remark 3.1** — title and first sentence are scoped to the recalibration
  layer and become a two-layer statement if the LightGBM result is placed there.
- **Section 3.2.4** — cites Remark 3.1 in its single-layer form
  ("the comparison tests Remark 3.1: at alpha = 0.01, additional parameters
  increase variance faster than they reduce bias").

Both are tracked in `analysis/phase0/CONDITIONAL_PASSAGES.md` and neither is
touched until the placement is settled.
