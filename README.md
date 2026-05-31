# Recalibrating Tail Risk Forecasts under Temporal Dependence

**Recalibrating Tail Risk Forecasts under Temporal Dependence**

Daniel Traian Pele, Vlad Bolovăneanu, Andrei Theodor Ginavar, Stefan Lessmann, Wolfgang Karl Härdle (2026)

A scalar conformal correction that recalibrates any black-box tail quantile
forecast to achieve valid finite-sample coverage under beta-mixing temporal
dependence. Applied to six time-series foundation models (Chronos-Small,
Chronos-Mini, TimesFM 2.5, Moirai 1.1, Moirai 2.0, Lag-Llama) and four
parametric benchmarks (GJR-GARCH, GARCH-N, Historical Simulation, EWMA)
across 24 financial assets at the 1% VaR level.

## Quantlets

All 25 Quantlets live in the [`Quantlets/`](Quantlets/) directory with a
dedicated [README](Quantlets/README.md) covering execution order, dependencies,
and the data flow graph.

| Quantlet | Output | Description |
|----------|--------|-------------|
| CO_data_returns | cfp_ijf_data/returns/*.csv | Download 24 asset log-return series (Layer 0) |
| CO_asset_overview | Table 1 | Asset universe (24 assets, 5 classes) |
| CO_model_overview | Table 2 | Model overview (6 TSFMs + 4 benchmarks) |
| CO_cross_sectional | Table 3 | Cross-sectional correlations of conformal threshold |
| CO_full_evaluation | Table 4 | Master results (violation rates, Kupiec, Basel, QS) |
| CO_multi_quantile_panel | Tables 5, 6, 7 | Multi-quantile, panel pooled, panel by class |
| CO_quantile_scores | Table 8 | Diebold-Mariano p-values for quantile score |
| CO_garch_conformal | Table 9 | Rolling vs static conformal correction |
| CO_simulation_study | Table 10 | Monte Carlo validation (5 DGPs, 500 reps) |
| CO_bound_validation | Table 11 | Coverage bound evaluation (Theorem 3.5) |
| CO_gbm_qr | Table 12 row | GBM-QR baseline (LightGBM quantile regression) |
| CO_gamlss | Table 12 row | GAMLSS-SST baseline (skewed-t location-scale) |
| CO_baselines_evt_fhs | Table 12 rows | EVT-POT and Filtered Historical Simulation |
| CO_baseline_comparison | Table 12 | Composite recalibration method comparison |
| CO_baseline_comparison_tuned | Table 12 (tuned row) | Tuned GBM-QR baseline (grid-searched) |
| CO_fz_scores | Table 13 | Fissler-Ziegel joint VaR-ES scores |
| CFP_ES_Correction_Z2 | Table C.14 | ES correction and Acerbi-Szekely Z2 backtest |
| CO_diagnostic_regression | Table E.4 | OLS diagnostic regression of ΔQS with clustered SEs |
| CO_robustness | Tables D.15-D.18 | Robustness: WCP, calibration fraction, Monte Carlo |
| CO_regime_sensitivity | Appendix D | Regime classification sensitivity |
| CO_robustness_inner7 | Appendix D | Extended tail-closure (inner-7) ablation |
| CO_panel_wildcluster | Appendix E | Wild-cluster bootstrap panel (Kupiec + DM) |
| CO_forensic_tsfm | Appendix figure | Forensic checks: TimesFM 2.5 + Moirai 2.0 |
| CO_rolling_qV | Figure 1 | Rolling conformal threshold on S&P 500 |
| CO_heatmap | Figure 2 | Basel Traffic Light heatmap (10 models x 24 assets) |
| CFP_Calibration_Efficiency_Frontier | Figure 3 | Calibration-efficiency frontier |
| CO_violation_rates | Figure 4 | Raw vs corrected violation rates |
| CO_qV_ranking | Figure 5 | Conformal correction magnitude ranking (10 models) |
| CO_covid_response_lag | Figure 6 | COVID-19 response lag |
| CO_drift_diagnostic | Figure 7 | Distributional drift diagnostic (TV distance) |
| CFP_Capital_Charge | Figure 8 | Cumulative capital charge comparison |

## Reproduction

```bash
python -m pip install -r requirements.txt
bash make.sh all        # tables + figures + manuscript (~10 min)
bash make.sh mc         # Monte Carlo robustness, Tables D.16-D.18 (~30 min)
bash make.sh verify     # rebuild and diff against committed outputs
```

Python >= 3.10 required. See `make.sh help` for all targets:
`all`, `tables`, `figures`, `mc`, `manuscript`, `clean`, `verify`.

Canonical inputs live in `cfp_ijf_data/`. **Only the 24 daily return series
(`cfp_ijf_data/returns/*.csv`) are committed.** The ~126 MB of TSFM/benchmark
quantile-forecast parquets and pre-computed `paper_outputs/` tables are published
as a **GitHub Release asset** to keep this code repo lean — fetch them with
`python download_data.py` before running the table/figure Quantlets, or download
the archive directly from the
[data release](https://github.com/QuantLet/Conformal_Oracle/releases/tag/data-v1). The
committed Quantlet outputs (`tab_*.tex`, `*.csv`, `*.png`) let you inspect every
result without rerunning anything. The upstream `pipeline/` regenerates the
forecasts from scratch (requires the foundation models / a GPU).

## Python package

`python/` contains `conformal-oracle`, a pip-installable implementation of the
scalar conformal recalibration / tail-audit method (static, rolling, and
bootstrap variants) with TSFM and GARCH forecaster wrappers. See
`python/README.md` for the API and runnable examples.

```bash
pip install conformal-oracle          # from PyPI
# or, from this repo:
pip install ./python
```

The `Quantlets/` reproduce the paper's tables and figures; the package exposes
the same method as reusable, documented software.

## Supplementary material

`legacy/auxiliary/` contains exploratory analyses from the predecessor paper
(Pele et al. 2026, *Expert Systems with Applications*). These are retained for
reproducibility of that earlier work and are not part of the current manuscript.

## Citation

```bibtex
@article{pele2026conformal,
  title   = {Recalibrating Tail Risk Forecasts
             under Temporal Dependence},
  author  = {Pele, Daniel Traian and Bolov{\u{a}}neanu, Vlad
             and Ginavar, Andrei Theodor and Lessmann, Stefan
             and H{\"a}rdle, Wolfgang Karl},
  journal = {Working Paper},
  year    = {2026}
}
```

## License

MIT
