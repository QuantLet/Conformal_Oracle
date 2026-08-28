# Provenance manifest

For every table and figure in `main_R1.tex`: does a script emit it, and does the emitted value match what was submitted?

| Status | Meaning |
|---|---|
| `OK` | a generator exists and reproduces the submitted artefact |
| `DIFFERS` | a generator exists, output differs — **erratum** |
| `NOT_EMITTED` | no generator — **reproducibility gap**, not an erratum |
| `RUN_FAILED` | generator exists but does not execute |
| `COSMETIC` | regenerates with identical values; only formatting differs |
| `NOT_WRITTEN` | generator exited 0 but did not touch the file — **no verdict** |
| `PENDING` | generator found, not yet executed |

## Summary

| Status | Count |
|---|---|
| DIFFERS | 22 |
| NOT_EMITTED | 1 |
| RUN_FAILED | 1 |
| COSMETIC | 1 |
| FIGURE | 9 |

## Detail

| Artefact | Status | Generator | Note |
|---|---|---|---|
| `Quantlets/CFP_ES_Correction_Z2/tab_es_correction.tex` | **DIFFERS** | CFP_ES_Correction_Z2.py | CFP_ES_Correction_Z2.py: 58 numeric token(s) differ from the submitted copy (83 vs 58 tokens) |
| `Quantlets/CO_baseline_comparison/tab_baselines.tex` | **DIFFERS** | compile_tab_baselines.py | compile_tab_baselines.py: 59 numeric token(s) differ from the submitted copy (86 vs 84 tokens) |
| `Quantlets/CO_baseline_comparison_tuned/tab_gbm_tuned.tex` | **DIFFERS** | run_tuned_gbm_qr.py | run_tuned_gbm_qr.py: 30 numeric token(s) differ from the submitted copy (66 vs 66 tokens) |
| `Quantlets/CO_cross_sectional/tab_cross_sectional.tex` | **DIFFERS** | run_cross_sectional.py | run_cross_sectional.py: 44 numeric token(s) differ from the submitted copy (47 vs 40 tokens) |
| `Quantlets/CO_diagnostic_regression/tab_diag_regression.tex` | **DIFFERS** | run_diag_regression.py | run_diag_regression.py: 27 numeric token(s) differ from the submitted copy (32 vs 32 tokens) |
| `Quantlets/CO_full_evaluation/tab_master_results.tex` | **DIFFERS** | build_table1_r2.py, rebuild_master_table.py, run_master_table.py | rebuild_master_table.py reproduces 109/110 cells; Moirai 1.1 W/GJR printed 1.00 vs computed 0.99. The shipped run_master_table.py does not emit this table at all (9 models, wrong panels) and is marked SUPERSEDED. |
| `Quantlets/CO_fz_scores/tab_fz_scores.tex` | **DIFFERS** | run_fz_scores.py | run_fz_scores.py: 35 numeric token(s) differ from the submitted copy (40 vs 23 tokens) |
| `Quantlets/CO_garch_conformal/tab_rolling_vs_static.tex` | **DIFFERS** | run_rolling_vs_static.py | run_rolling_vs_static.py: 102 numeric token(s) differ from the submitted copy (133 vs 92 tokens) |
| `Quantlets/CO_model_overview/tab_models.tex` | **DIFFERS** | run_model_overview.py | run_model_overview.py: 50 numeric token(s) differ from the submitted copy (53 vs 35 tokens) |
| `Quantlets/CO_multi_quantile_panel/tab_multiquantile.tex` | **DIFFERS** | run_multiquantile.py | Moirai 1.1 at alpha=0.01: 10/24 rejections printed, 9/24 correct. Built from the stale moirai11_full_results.csv, now replaced. |
| `Quantlets/CO_multi_quantile_panel/tab_panel_by_class.tex` | **DIFFERS** | run_panel_by_class.py | run_panel_by_class.py: 28 numeric token(s) differ from the submitted copy (35 vs 35 tokens) |
| `Quantlets/CO_multi_quantile_panel/tab_panel_pooled.tex` | **DIFFERS** | run_panel_pooled.py | run_panel_pooled.py: 58 numeric token(s) differ from the submitted copy (94 vs 73 tokens) |
| `Quantlets/CO_panel_wildcluster/tab_panel_wildcluster_dm.tex` | **DIFFERS** | run_wild_cluster_bootstrap.py | run_wild_cluster_bootstrap.py: 25 numeric token(s) differ from the submitted copy (32 vs 20 tokens) |
| `Quantlets/CO_panel_wildcluster/tab_panel_wildcluster_kupiec.tex` | **DIFFERS** | run_wild_cluster_bootstrap.py | run_wild_cluster_bootstrap.py: 62 numeric token(s) differ from the submitted copy (68 vs 47 tokens) |
| `Quantlets/CO_quantile_scores/tab_dm_pvalues.tex` | **DIFFERS** | run_dm_pvalues.py | run_dm_pvalues.py: 28 numeric token(s) differ from the submitted copy (32 vs 26 tokens) |
| `Quantlets/CO_regime_sensitivity/tab_regime_sensitivity.tex` | **DIFFERS** | run_regime_sensitivity.py | run_regime_sensitivity.py: 35 numeric token(s) differ from the submitted copy (135 vs 135 tokens) |
| `Quantlets/CO_robustness/tab_h14_small_sample.tex` | **DIFFERS** | run_robustness_mc.py | run_robustness_mc.py: 94 numeric token(s) differ from the submitted copy (168 vs 168 tokens) |
| `Quantlets/CO_robustness/tab_h15_fc_sensitivity.tex` | **DIFFERS** | run_robustness_mc.py | run_robustness_mc.py: 120 numeric token(s) differ from the submitted copy (194 vs 194 tokens) |
| `Quantlets/CO_robustness/tab_h16_regime_stability.tex` | **DIFFERS** | run_robustness_mc.py | run_robustness_mc.py: 16 numeric token(s) differ from the submitted copy (38 vs 38 tokens) |
| `Quantlets/CO_robustness/tab_qV_bootstrap_ci.tex` | **DIFFERS** | run_qV_bootstrap_table.py | run_qV_bootstrap_table.py: 262 numeric token(s) differ from the submitted copy (270 vs 241 tokens) |
| `Quantlets/CO_robustness/tab_robustness_summary.tex` | **DIFFERS** | run_robustness_summary.py | run_robustness_summary.py: 21 numeric token(s) differ from the submitted copy (38 vs 38 tokens) |
| `Quantlets/CO_robustness_inner7/tab_tail_closure_extended.tex` | **DIFFERS** | run_inner7_tail_closure.py | run_inner7_tail_closure.py: 91 numeric token(s) differ from the submitted copy (99 vs 97 tokens) |
| `Quantlets/CO_robustness/tab_gap_ablation.tex` | **NOT_EMITTED** | — | no script in the artefact's directory writes it |
| `Quantlets/CO_simulation_study/tab_simulation_extended.tex` | **RUN_FAILED** | run_simulation_study.py | run_simulation_study.py: TypeError: Axes.boxplot() got an unexpected keyword argument 'labels'. Did you mean 'label'? |
| `Quantlets/CO_asset_overview/tab_assets.tex` | **COSMETIC** | run_asset_overview.py | run_asset_overview.py: every reported value is identical; only formatting differs |

## Figures

| Artefact | Note |
|---|---|
| `capital_charge_cumulative` | figure; checked separately |
| `fig_covid_response` | figure; checked separately |
| `fig_drift_diagnostic` | figure; checked separately |
| `fig_forensic_tsfm` | figure; checked separately |
| `fig_frontier_killer` | figure; checked separately |
| `fig_rolling_qv.pdf` | figure; checked separately |
| `fig_traffic_light` | figure; checked separately |
| `ql_logo.png` | figure; checked separately |
| `qr_logo.png` | figure; checked separately |

