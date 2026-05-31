# pipeline/ — Evaluation Pipeline Notebooks

These notebooks form the full evaluation pipeline for the paper
"Distribution-Free Recalibration of Tail Quantile Forecasts under Temporal
Dependence" (Pele, Lessmann, Härdle, 2026).

They are **not** Quantlets — they are the upstream computation pipeline that
produces the intermediate data artifacts consumed by the Quantlets in the
repo root.

## Execution order

1. `CFP_Data_Download.ipynb` — downloads raw price data to `cfp_ijf_data/returns/`
2. `CFP_Chronos_Forecasts.ipynb` — Chronos model inference
3. `CFP_TimesFM_Forecasts.ipynb` — TimesFM model inference
4. `CFP_Moirai_Forecasts.ipynb` — Moirai model inference
5. `CFP_LagLlama_Forecasts.ipynb` — Lag-Llama model inference
6. `CFP_Parametric_Benchmarks.ipynb` — GARCH-N, GJR-GARCH, EWMA, Hist-Sim
7. `CFP_Conformal_Calibration.ipynb` — conformal correction + rolling variant
8. `CFP_Scoring_Rules.ipynb` — quantile score and coverage evaluation
9. `CFP_Traffic_Light.ipynb` — Basel Traffic Light classification
10. `CFP_Multi_Quantile.ipynb` — multi-quantile panel (alpha = 0.01, 0.025, 0.05, 0.10)
11. `CFP_Panel_Pooled.ipynb` — pooled panel analysis
12. `CFP_qV_Diagnostic.ipynb` — conformal threshold diagnostics
13. `CFP_Paper_Tables_Figures.ipynb` — assembles final tables and figures
14. `CFP_Practitioner_Guide.ipynb` — practitioner summary table

## Key outputs

- `cfp_ijf_data/paper_outputs/tables/all_results.csv` — 864 rows
  (9 models × 24 assets × 4 alpha levels), the central data artifact
  consumed by most Quantlets
- `cfp_ijf_data/paper_outputs/tables/rolling_qv_SP500.csv` — rolling
  conformal threshold series for the SP500 case study
- `cfp_ijf_data/paper_outputs/tables/master_table.csv` — aggregated
  model-level summary

These notebooks are **not typically re-run** — the outputs are checked
into `cfp_ijf_data/paper_outputs/` and consumed directly by the Quantlets.
