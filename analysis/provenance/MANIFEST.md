# Provenance manifest

For every table and figure in `main_R1.tex`: does a script emit it, and does the emitted value match what was submitted?

| Status | Meaning |
|---|---|
| `OK` | a generator exists and reproduces the submitted artefact |
| `DIFFERS` | a generator exists, output differs — **erratum** |
| `NOT_EMITTED` | no generator — **reproducibility gap**, not an erratum |
| `RUN_FAILED` | generator exists but does not execute |
| `PENDING` | generator found, not yet executed |

## Summary

| Status | Count |
|---|---|
| DIFFERS | 11 |
| NOT_EMITTED | 3 |
| RUN_FAILED | 5 |
| OK | 6 |
| FIGURE | 9 |

## Detail

| Artefact | Status | Generator | Note |
|---|---|---|---|
| `Quantlets/CFP_ES_Correction_Z2/tab_es_correction.tex` | **DIFFERS** | CFP_ES_Correction_Z2.py | CFP_ES_Correction_Z2.py output differs from the submitted copy |
| `Quantlets/CO_asset_overview/tab_assets.tex` | **DIFFERS** | run_asset_overview.py | run_asset_overview.py output differs from the submitted copy |
| `Quantlets/CO_full_evaluation/tab_master_results.tex` | **DIFFERS** | rebuild_master_table.py, run_master_table.py | rebuild_master_table.py reproduces 109/110 cells; Moirai 1.1 W/GJR printed 1.00 vs computed 0.99. The shipped run_master_table.py does not emit this table at all (9 models, wrong panels) and is marked SUPERSEDED. |
| `Quantlets/CO_fz_scores/tab_fz_scores.tex` | **DIFFERS** | run_fz_scores.py | run_fz_scores.py output differs from the submitted copy |
| `Quantlets/CO_garch_conformal/tab_rolling_vs_static.tex` | **DIFFERS** | run_rolling_vs_static.py | run_rolling_vs_static.py output differs from the submitted copy |
| `Quantlets/CO_model_overview/tab_models.tex` | **DIFFERS** | run_model_overview.py | run_model_overview.py output differs from the submitted copy |
| `Quantlets/CO_multi_quantile_panel/tab_multiquantile.tex` | **DIFFERS** | run_multiquantile.py | Moirai 1.1 at alpha=0.01: 10/24 rejections printed, 9/24 correct. Built from the stale moirai11_full_results.csv, now replaced. |
| `Quantlets/CO_multi_quantile_panel/tab_panel_by_class.tex` | **DIFFERS** | run_panel_by_class.py | run_panel_by_class.py output differs from the submitted copy |
| `Quantlets/CO_quantile_scores/tab_dm_pvalues.tex` | **DIFFERS** | run_dm_pvalues.py | run_dm_pvalues.py output differs from the submitted copy |
| `Quantlets/CO_robustness/tab_robustness_summary.tex` | **DIFFERS** | run_robustness_summary.py | run_robustness_summary.py output differs from the submitted copy |
| `Quantlets/CO_simulation_study/tab_simulation_extended.tex` | **DIFFERS** | run_simulation_study.py | run_simulation_study.py output differs from the submitted copy |
| `Quantlets/CO_baseline_comparison_tuned/tab_gbm_tuned.tex` | **NOT_EMITTED** | — | no script in the artefact's directory writes it |
| `Quantlets/CO_robustness/tab_gap_ablation.tex` | **NOT_EMITTED** | — | no script in the artefact's directory writes it |
| `Quantlets/CO_robustness/tab_qV_bootstrap_ci.tex` | **NOT_EMITTED** | — | no script in the artefact's directory writes it |
| `Quantlets/CO_baseline_comparison/tab_baselines.tex` | **RUN_FAILED** | compile_tab_baselines.py | compile_tab_baselines.py: FileNotFoundError: [Errno 2] No such file or directory: '/Users/danpele/Library/Mobile Documents/com~apple~CloudDocs/Documents/2026 CFP LLM VaR/legacy/results/rolling_w250_pooled.csv' |
| `Quantlets/CO_robustness/tab_h14_small_sample.tex` | **RUN_FAILED** | run_robustness_mc.py | run_robustness_mc.py: timeout after 600s |
| `Quantlets/CO_robustness/tab_h15_fc_sensitivity.tex` | **RUN_FAILED** | run_robustness_mc.py | run_robustness_mc.py: timeout after 600s |
| `Quantlets/CO_robustness/tab_h16_regime_stability.tex` | **RUN_FAILED** | run_robustness_mc.py | run_robustness_mc.py: timeout after 600s |
| `Quantlets/CO_robustness_inner7/tab_tail_closure_extended.tex` | **RUN_FAILED** | run_inner7_tail_closure.py | run_inner7_tail_closure.py: timeout after 600s |
| `Quantlets/CO_cross_sectional/tab_cross_sectional.tex` | **OK** | run_cross_sectional.py | reproduced by run_cross_sectional.py |
| `Quantlets/CO_diagnostic_regression/tab_diag_regression.tex` | **OK** | run_diag_regression.py | reproduced by run_diag_regression.py |
| `Quantlets/CO_multi_quantile_panel/tab_panel_pooled.tex` | **OK** | run_panel_pooled.py | reproduced by run_panel_pooled.py |
| `Quantlets/CO_panel_wildcluster/tab_panel_wildcluster_dm.tex` | **OK** | run_wild_cluster_bootstrap.py | reproduced by run_wild_cluster_bootstrap.py |
| `Quantlets/CO_panel_wildcluster/tab_panel_wildcluster_kupiec.tex` | **OK** | run_wild_cluster_bootstrap.py | reproduced by run_wild_cluster_bootstrap.py |
| `Quantlets/CO_regime_sensitivity/tab_regime_sensitivity.tex` | **OK** | run_regime_sensitivity.py | reproduced by run_regime_sensitivity.py |

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

