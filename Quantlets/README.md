# Quantlets

Reproducible code units for all tables and figures in:

> **Recalibrating Tail Risk Forecasts under Temporal Dependence**
> Daniel Traian Pele, Vlad Bolovăneanu, Andrei Theodor Ginavar,
> Stefan Lessmann, Wolfgang Karl Härdle (2026)

Each Quantlet is a self-contained directory with a `Metainfo.txt` (QuantNet standard),
a Python script (`.py`), a Jupyter notebook (`.ipynb`), and one or more outputs
(`.tex`, `.csv`, `.pdf`, `.png`).

## Data prerequisites

All Quantlets read from the canonical dataset in `cfp_ijf_data/`:

| Path | Content |
|------|---------|
| `cfp_ijf_data/returns/*.csv` | Daily log-returns for 24 assets |
| `cfp_ijf_data/{model}/*.parquet` | TSFM quantile forecasts (Chronos-Small, Chronos-Mini, TimesFM 2.5, Moirai 1.1, Moirai 2.0, Lag-Llama) |
| `cfp_ijf_data/benchmarks/*.parquet` | Parametric benchmark forecasts (GJR-GARCH, GARCH-N, Hist-Sim, EWMA) |
| `cfp_ijf_data/paper_outputs/tables/*.csv` | Pre-computed summary tables |
| `cfp_ijf_data/paper_outputs/qs_sequences/*.parquet` | Quantile score sequences for DM tests |

**Only the 24 return series are committed.** The forecast parquets and
pre-computed `paper_outputs/` tables (~126 MB) are published as a **GitHub
Release asset** and are **not** stored in git (keeping the code repo lean). Fetch
them once with `python download_data.py` (from the repo root) before running the
table/figure Quantlets — the Quantlets consume these inputs and do **not**
regenerate them. The committed Quantlet outputs let you inspect every result
without rerunning. To rebuild the forecasts from scratch, run the upstream
pipeline (`pipeline/`); this requires the foundation models and a GPU.

## Quick start

```bash
# All tables + figures (~10 min)
bash make.sh all

# Only tables
bash make.sh tables

# Only figures
bash make.sh figures

# Monte Carlo robustness — Tables D.16-D.18 (~30 min)
bash make.sh mc

# Rebuild and diff against committed outputs
bash make.sh verify
```

Python >= 3.10 required. Install dependencies with `pip install -r requirements.txt`.

## Execution order

The table below shows the exact execution order used by `make.sh`.
Dependencies run first: the three Table 12 sub-Quantlets (GBM-QR, GAMLSS, EVT/FHS)
must complete before the composite `CO_baseline_comparison`.

### Layer 0 — Data

| Step | Quantlet | Script | Output | Description |
|------|----------|--------|--------|-------------|
| D0 | CO_data_returns | `download_returns.py` | `cfp_ijf_data/returns/*.csv` | Download 24 asset return series from Yahoo Finance |

### Layer 1 — Tables

| Step | Quantlet | Script | Output | Description |
|------|----------|--------|--------|-------------|
| T1 | CO_asset_overview | `run_asset_overview.py` | Table 1 | Asset universe (24 assets, 5 classes) |
| T2 | CO_model_overview | `run_model_overview.py` | Table 2 | Model overview (6 TSFMs + 4 benchmarks) |
| T3 | CO_cross_sectional | `run_cross_sectional.py` | Table 3 | Cross-sectional correlations of q&#x302;_V |
| T4 | CO_full_evaluation | `run_master_table.py` | Table 4 | Master results: violation rates, Kupiec, Basel, QS |
| T5 | CO_multi_quantile_panel | `run_multiquantile.py` | Table 5 | Multi-quantile evaluation (α = 1%, 2.5%, 5%, 10%) |
| T6 | CO_multi_quantile_panel | `run_panel_pooled.py` | Table 6 | Panel-pooled backtest (Driscoll-Kraay HAC) |
| T7 | CO_multi_quantile_panel | `run_panel_by_class.py` | Table 7 | Panel by asset class |
| T8 | CO_quantile_scores | `run_dm_pvalues.py` | Table 8 | Diebold-Mariano p-values (HLN correction) |
| T9 | CO_garch_conformal | `run_rolling_vs_static.py` | Table 9 | Rolling vs static conformal correction |
| T11 | CO_bound_validation | `run_bound_validation.py` | Table 11 | Coverage bound evaluation (Theorem 3.5) |
| T12a | CO_gbm_qr | `baseline_gbm_qr.py` | — | GBM-QR baseline (prerequisite for T12) |
| T12b | CO_gamlss | `baseline_gamlss.py` | — | GAMLSS-SST baseline (prerequisite for T12) |
| T12c | CO_baselines_evt_fhs | `run_baselines_evt_fhs.py` | — | EVT-POT + FHS baselines (prerequisite for T12) |
| T12 | CO_baseline_comparison | `compile_tab_baselines.py` | Table 12 | Composite recalibration method comparison |
| T12t | CO_baseline_comparison_tuned | `run_tuned_gbm_qr.py` | Table 12 (tuned row) | Grid-searched GBM-QR baseline |
| T13 | CO_fz_scores | `run_fz_scores.py` | Table 13 | Fissler-Ziegel joint VaR-ES scores |
| TC14 | CFP_ES_Correction_Z2 | `CFP_ES_Correction_Z2.py` | Table C.14 | ES correction + Acerbi-Szekely Z₂ backtest |
| TE4 | CO_diagnostic_regression | `run_diag_regression.py` | Table E.4 | OLS diagnostic regression of ΔQS (clustered SEs) |
| TD15 | CO_robustness | `run_robustness_summary.py` | Table D.15 | Robustness: WCP, calibration fraction, rolling |
| TDr | CO_regime_sensitivity | `run_regime_sensitivity.py` | Table (App. D) | Regime classification sensitivity |
| TE2 | CO_robustness_inner7 | `run_inner7_tail_closure.py` | Table (App. D) | Extended tail-closure (inner-7) ablation |
| TE3 | CO_panel_wildcluster | `run_wild_cluster_bootstrap.py` | Tables (App. E) | Wild-cluster bootstrap panel (Kupiec + DM) |

### Layer 2 — Figures

| Step | Quantlet | Script | Output | Description |
|------|----------|--------|--------|-------------|
| F1 | CO_rolling_qV | `run_rolling_qV.py` | Figure 1 | Rolling q&#x302;_V on S&P 500 + realised volatility |
| F2 | CO_heatmap | `run_heatmap.py` | Figure 2 | Basel Traffic Light heatmap (9 × 24) |
| F3 | CFP_Calibration_Efficiency_Frontier | `run_frontier.py` | Figure 3 | Calibration-efficiency frontier |
| F4 | CO_violation_rates | `run_violation_rates.py` | Figure 4 | Raw vs corrected violation rates |
| F5 | CO_simulation_study | `run_simulation_study.py` | Figure 5 + Table 10 | Monte Carlo q&#x302;_V distribution (5 DGPs, 500 reps) |
| F6 | CO_covid_response_lag | `run_covid_response_lag.py` | Figure 6 | COVID-19 response lag |
| F7 | CO_drift_diagnostic | `run_drift_diagnostic.py` | Figure 7 | Distributional drift diagnostic (TV distance) |
| F8 | CFP_Capital_Charge | `CFP_Capital_Charge.py` | Figure 8 | Cumulative capital charge comparison |
| F9 | CO_qV_ranking | `run_qV_ranking.py` | Figure (q̂_V ranking) | Conformal correction magnitude ranking (10 models) |
| FF | CO_forensic_tsfm | `run_forensic_tsfm.py` | Figure (App.) | Forensic checks: TimesFM 2.5 + Moirai 2.0 |

### Layer 3 — Monte Carlo robustness (slow)

| Step | Quantlet | Script | Output | Description |
|------|----------|--------|--------|-------------|
| TD16-18 | CO_robustness | `run_robustness_mc.py` | Tables D.16-D.18 | Small-sample MC, calibration sensitivity, regime stability |

## Dependency graph

```
cfp_ijf_data/  (canonical data — Layer 0)
    │
    ├─── Independent Quantlets (no inter-Quantlet dependencies)
    │    ├── CO_asset_overview          → Table 1
    │    ├── CO_model_overview          → Table 2
    │    ├── CO_cross_sectional         → Table 3
    │    ├── CO_full_evaluation         → Table 4
    │    ├── CO_multi_quantile_panel    → Tables 5, 6, 7
    │    ├── CO_quantile_scores         → Table 8
    │    ├── CO_garch_conformal         → Table 9
    │    ├── CO_bound_validation        → Table 11
    │    ├── CO_fz_scores               → Table 13
    │    ├── CFP_ES_Correction_Z2       → Table C.14
    │    ├── CO_robustness              → Tables D.15-D.18
    │    ├── CO_regime_sensitivity      → Table (App. D)
    │    ├── CO_robustness_inner7       → Table (App. D)
    │    ├── CO_panel_wildcluster       → Tables (App. E)
    │    ├── CO_diagnostic_regression   → Table E.4
    │    ├── CO_rolling_qV              → Figure 1
    │    ├── CO_heatmap                 → Figure 2
    │    ├── CFP_Calibration_Efficiency_Frontier → Figure 3
    │    ├── CO_violation_rates         → Figure 4
    │    ├── CO_simulation_study        → Table 10 + Figure 5
    │    ├── CO_covid_response_lag      → Figure 6
    │    ├── CO_drift_diagnostic        → Figure 7
    │    ├── CFP_Capital_Charge         → Figure 8
    │    ├── CO_qV_ranking              → Figure (q̂_V ranking)
    │    └── CO_forensic_tsfm           → Figure (App.)
    │
    └─── Chained Quantlets (T12a-c must run before T12)
         ├── CO_gbm_qr          ──┐
         ├── CO_gamlss           ──┼──→ CO_baseline_comparison → Table 12
         └── CO_baselines_evt_fhs ─┘
              (CO_baseline_comparison_tuned is standalone → Table 12 tuned row)
```

## Running individual Quantlets

Each Quantlet can be run standalone from the repository root:

```bash
# Run the Python script directly
python Quantlets/CO_full_evaluation/run_master_table.py

# Or open the Jupyter notebook for interactive exploration
jupyter notebook Quantlets/CO_full_evaluation/CO_full_evaluation.ipynb
```

The `.py` scripts produce publication-ready `.tex` / `.pdf` / `.png` outputs.
The `.ipynb` notebooks contain the same logic with inline commentary and visualisation.

## Quantlet structure

Each directory follows the QuantNet standard:

```
CO_full_evaluation/
├── Metainfo.txt                    # QuantNet metadata (name, keywords, description)
├── run_master_table.py             # Standalone Python script
├── CO_full_evaluation.ipynb        # Jupyter notebook (same logic, interactive)
└── tab_master_results.tex          # Output (table or figure)
```

## Supplementary diagnostics

Three Quantlets driving the standalone supplement `rr_material.tex`. Build them
all (and compile the supplement) with `bash make.sh rr`.

| Quantlet | Topic | Output |
|----------|-------|--------|
| `CO_identification` | Shift is information, not scale/bias (Prop. `prop:decomposition`) | `tab_identification.tex`, `fig_identification.pdf/.png` (synthetic; ~20 s, CPU) |
| `CO_diagnostic_scale` | Empirical two-parameter scale check (216 pairs) | `tab_scale_diagnostic.tex`, `tab_scale_regime_crosstab.tex` (uses `conformal_oracle.recalibration.diagnose_scale`) |
| `CO_aci_baseline` | ACI vs rolling (216 pairs) | `tab_baselines_aci.tex`, `tab_aci_gamma_sensitivity.tex`, `verdict_aci.tex` (uses `ACICalibrator`) |

`CO_diagnostic_scale` and `CO_aci_baseline` consume the cached forecast parquets
in `cfp_ijf_data/` (same inputs as `CO_baseline_comparison`); `CO_identification`
is fully synthetic.
