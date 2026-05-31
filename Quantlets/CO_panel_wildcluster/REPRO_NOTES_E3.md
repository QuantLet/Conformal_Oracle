# REPRO_NOTES — E3: Wild-Cluster Bootstrap

**Quantlet:** `CO_panel_wildcluster`
**Script:** `run_wild_cluster_bootstrap.py`
**Date:** 2026-05-08

## Reproduction

```bash
cd "/Users/danielpele/Documents/2026 CFP LLM VaR"
python3 Quantlets/CO_panel_wildcluster/run_wild_cluster_bootstrap.py
```

## Dependencies

- Python 3.10+, NumPy, Pandas, SciPy
- Violation sequences: `cfp_ijf_data/paper_outputs/violation_sequences/{model}_violations.parquet`
- QS sequences: `cfp_ijf_data/paper_outputs/qs_sequences/{model}_qs.parquet`

## Method

**Part 1: Kupiec LR bootstrap**
- Panel-pooled violation count across J = 24 asset clusters
- Rademacher weights w_j ∈ {−1, +1} applied to centered cluster-level counts
- B = 999 replications, seed = 42
- Bootstrap p-value = (#{|LR*_b| ≥ LR_obs} + 1) / (B + 1)

**Part 2: DM t-statistic bootstrap**
- Cluster means of QS differences (parametric benchmark vs TSFM)
- Rademacher weights applied to centered cluster means
- 4 × 5 = 20 pairwise comparisons (parametric vs TSFM)
- Bootstrap p-value = (#{|t*_b| ≥ |t_obs|} + 1) / (B + 1)

## Outputs

- `wild_cluster_kupiec.csv` — 9 rows (one per model), bootstrap p-values
- `wild_cluster_dm.csv` — 20 rows (parametric vs TSFM pairs), bootstrap p-values
- `tab_panel_wildcluster_kupiec.tex` — LaTeX table with asymptotic and bootstrap p-values
- `tab_panel_wildcluster_dm.tex` — LaTeX matrix of bootstrap DM p-values (bold < 0.01, italic < 0.05)
