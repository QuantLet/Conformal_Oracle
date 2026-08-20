# Which of the paper's tables were computed on the defective series

Established 20 August 2026 by comparing the modification time of every artefact
`main_R2.tex` `\input`s against the date the corrected series were promoted
(17 August). A table older than that date was computed on at least one of the
four defective series unless it is independent of the panel.

**Only Table 1 is current.** `tab_master_results_r2` was regenerated on
17 August at 16:59. The other 24 inputs predate the corrections.

## Stale and panel-dependent — must be regenerated

| artefact | last built | consumes |
|---|---|---|
| `CO_quantile_scores/tab_dm_pvalues` | 15 Aug | QS sequences |
| `CO_panel_wildcluster/tab_panel_wildcluster_kupiec` | 15 Aug | violation sequences |
| `CO_panel_wildcluster/tab_panel_wildcluster_dm` | 15 Aug | QS sequences |
| `CO_multi_quantile_panel/tab_multiquantile` | 14 Aug | per-pair results, 4 levels |
| `CO_multi_quantile_panel/tab_panel_pooled` | 30 Apr | violation sequences |
| `CO_multi_quantile_panel/tab_panel_by_class` | 30 Apr | per-pair results |
| `CO_cross_sectional/tab_cross_sectional` | 8 May | per-pair q_V, volatility |
| `CO_diagnostic_regression/tab_diag_regression` | 8 May | per-pair results |
| `CO_baseline_comparison/tab_baselines` | 14 Aug | alternative recalibrations |
| `CO_baseline_comparison_tuned/tab_gbm_tuned` | 8 May | GBM-QR sweep |
| `CO_garch_conformal/tab_rolling_vs_static` | 30 Apr | rolling and static series |
| `CO_fz_scores/tab_fz_scores` | 15 Aug | ES/FZ evaluation |
| `CFP_ES_Correction_Z2/tab_es_correction` | 15 Aug | ES correction |
| `CO_robustness/tab_qV_bootstrap_ci` | 10 May | block bootstrap of q_V |
| `CO_robustness/tab_robustness_summary` | 7 May | robustness battery |
| `CO_robustness/tab_h14_small_sample` | 15 Aug | subsample splits |
| `CO_robustness/tab_h15_fc_sensitivity` | 15 Aug | calibration-fraction sweep |
| `CO_robustness/tab_h16_regime_stability` | 15 Aug | regime splits |
| `CO_robustness/tab_gap_ablation` | 30 Apr | gap-parameter ablation |
| `CO_robustness_inner7/tab_tail_closure_extended` | 15 Aug | tail-closure rules |

## Stale but panel-independent — check, do not necessarily rebuild

| artefact | why it may stand |
|---|---|
| `CO_simulation_study/tab_simulation_extended` | Monte Carlo on synthetic data; touches no forecast series |
| `CO_asset_overview/tab_assets` | asset metadata |
| `CO_model_overview/tab_models` | model metadata, but must be extended from 13 to 16 forecasters |

## Progress, 20 August

Rebuilt on the corrected 13-forecaster panel and verified:

| artefact | note |
|---|---|
| `qs_sequences/`, `violation_sequences/` | rebuilt by `scripts/build_qs_sequences.py` for all 13 forecasters, raw and corrected; reproduces `all_results.csv` with counts identical and floats within 1e-12 |
| `tab_dm_pvalues` | 5 benchmarks x 6 TSFM series; adds the default-versus-analytic comparison |
| `tab_panel_wildcluster_kupiec`, `tab_panel_wildcluster_dm` | 13 forecasters, 999 Rademacher draws |
| `tab_panel_pooled` | refactored to read the committed sequences instead of rebuilding the correction itself |
| `tab_multiquantile` | both Chronos configurations, four levels |
| `tab_panel_by_class` | reads `all_results.csv` directly; no change needed beyond the input |

`compute_panel_pooled.py` was converted from a second writer of
`violation_sequences/` into an independent cross-check of them. It agrees with
the committed sequences on every cell of all 13 forecasters, which is worth more
than the duplicate producer was: the producer verifies against a summary it also
derives, while this path shares no code with it.

## Final state, 20 August

Every panel-dependent artefact the manuscript inputs has been rebuilt on the
corrected 13-forecaster panel:

| artefact | note |
|---|---|
| `tab_dm_pvalues` | 5 benchmarks x 6 TSFM series, plus the default-versus-analytic comparison |
| `tab_panel_wildcluster_kupiec`, `..._dm` | 13 forecasters, 999 Rademacher draws |
| `tab_panel_pooled`, `tab_panel_by_class`, `tab_multiquantile` | |
| `tab_cross_sectional` | GJR-GARCH's correlation with volatility: -0.786 to +0.816 |
| `tab_diag_regression` | R2 0.828; partial R2 of q_V falls to 0.007 without the truncated pair |
| `tab_models` | rewritten from primary sources; Moirai 2.0 was described as Moirai 1.1 |
| `tab_rolling_vs_static` | static Green counts match all_results.csv on all 312 pairs |
| `tab_baselines` | ACI, GBM-QR, GAMLSS, EVT-POT, FHS all recomputed; denominators derived |
| `tab_gbm_tuned` | given a producer; it had none and still printed denominators of 9 |
| `tab_fz_scores`, `tab_es_correction` | 13 forecasters |
| `tab_qV_bootstrap_ci` | given a producer; Chronos enters analytically |
| `tab_robustness_summary`, `tab_gap_ablation` | re-based off the truncated series |
| `tab_tail_closure`, `tab_tail_closure_extended` | sign defect fixed first; see SIGN_DEFECT_SIXTH.md |

**Panel-independent, verified rather than assumed.** `tab_h14_small_sample`,
`tab_h15_fc_sensitivity` and `tab_h16_regime_stability` come from
`run_robustness_mc.py`, and `tab_simulation_extended` from the simulation study.
Neither script reads anything from `cfp_ijf_data/` -- both generate their own
GARCH paths -- so their contents cannot depend on the defective series. They
stand as built.

## The blocking dependency

`cfp_ijf_data/paper_outputs/qs_sequences/` and `violation_sequences/` date from
24 April and cover nine forecasters. They are the input to the
Diebold--Mariano, wild-cluster and panel-pooled tables, and no script in the repository produced the QS ones --- they were consumed by
`CO_quantile_scores/run_dm_pvalues.py`,
`CO_panel_wildcluster/run_wild_cluster_bootstrap.py` and
`python/src/conformal_oracle/panel/diebold_mariano.py`, and written by nothing.
(The violation sequences did have a producer, `compute_panel_pooled.py`, which
wrote them for nine models as a side effect of a verification run. That is worse
than no producer in one respect: re-running it would have silently replaced a
thirteen-model artefact with a nine-model one.)

Regenerating them is not optional and not hard: a quantile-loss sequence per
(model, asset) on the test window is exactly what `run_full_evaluation.py`
already computes internally to report `QS_raw` and `QS_cp`. It has to be written
once, for all 16 forecasters, and then the tables above rebuild from it.

## Order

1. `scripts/build_qs_sequences.py` --- QS and violation sequences for all 16
   forecasters from the promoted series, with a check that the per-pair means it
   implies reproduce `all_results.csv` to floating point.
2. The four panel tables that consume sequences.
3. The remaining panel-dependent tables, each against its own producer.
4. `tab_models`, extended to 16 forecasters.
5. A before/after table for the paper, so a referee can see what moved.
