# Conformal Oracle — Replication Package

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuantLet/Conformal_Oracle/blob/main/CO_conformal_garch_ewma/CO_conformal_garch_ewma.ipynb)

**Paper:** *Distribution-Free Recalibration of Tail Quantile Forecasts under Temporal Dependence*

## Quick Start

### Google Colab (no installation)

- [GARCH/EWMA Conformal Analysis](https://colab.research.google.com/github/QuantLet/Conformal_Oracle/blob/main/CO_conformal_garch_ewma/CO_conformal_garch_ewma.ipynb)
- [LLM Conformal Analysis](https://colab.research.google.com/github/QuantLet/Conformal_Oracle/blob/main/CO_conformal_llm/CO_conformal_llm.ipynb)

### Local Reproduction

```bash
git clone https://github.com/QuantLet/Conformal_Oracle.git
cd Conformal_Oracle
pip install numpy pandas matplotlib scipy openpyxl

python run_conformal_analysis.py       # LLM conformal analysis
python run_conformal_garch_ewma.py     # GARCH/EWMA conformal analysis
python run_simulation_study.py         # Monte Carlo simulation (5 DGPs)
```

## Repository Structure

```
Conformal_Oracle/
├── README.md
├── run_conformal_analysis.py          # Main LLM conformal analysis
├── run_conformal_garch_ewma.py        # GARCH/EWMA conformal analysis
├── run_simulation_study.py            # Monte Carlo simulation (5 DGPs)
├── run_all_three_tasks.py             # Extended sims + quantile scores
├── CO_conformal_agnostic/             # Cross-method q_V bar chart
├── CO_universality/                   # Universality figure (q_V vs pi)
├── CO_simulation_study/               # Monte Carlo simulation
├── CO_calibration_plot/               # Calibration plot
├── CO_garch_conformal/                # GARCH/EWMA conformal correction
├── CO_quantile_scores/                # Quantile score evaluation
├── CO_sharpness_penalty/              # Calibration-efficiency table
├── CO_conformal_garch_ewma/           # Notebook: GARCH/EWMA
├── CO_conformal_llm/                  # Notebook: LLM analysis
├── CO_dual_correction/                # VaR correction time series
├── CO_coverage/                       # Coverage comparison
├── CO_cross_model/                    # Cross-model thresholds
├── CO_GARCH_comparison/               # GARCH benchmark
├── CO_frontier/                       # Coverage-efficiency frontier
├── CO_freq_magnitude/                 # q_V diagnostic
├── CO_heatmap/                        # q_V heatmap
├── CO_rolling_qV/                     # Rolling q_V stability
├── CO_traffic_light/                  # Traffic light matrices
└── CO_*/                              # Other QuantLet figures
```

## Key Results

| Method | Mean q_V | SD(q_V) | Green/9 |
|--------|----------|---------|---------|
| GARCH-N(250) | +0.004 | 0.009 | 7/9 |
| GAS-t(250) | -0.007 | 0.008 | 6/9 |
| EWMA-DCS(120) | +0.002 | 0.004 | 9/9 |
| GPT-3.5+CP | +0.002 | 0.007 | 8/9 |
| GPT-4+CP | **+0.024** | 0.015 | 9/9 |
| GPT-4o+CP | **+0.020** | 0.014 | 9/9 |

### Monte Carlo Simulation (5 DGPs)

| DGP | T=1000 q_V | T=5000 q_V | T=5000 Green% |
|-----|------------|------------|---------------|
| Normal | 0.0001 | 0.0000 | 96.2% |
| Student-t(5) | 0.0014 | 0.0013 | 98.2% |
| Student-t(3) | 0.0016 | 0.0013 | 97.2% |
| Stoch. Vol. | 0.0009 | 0.0006 | 93.2% |
| Regime Switch | **0.0122** | **0.0114** | 95.8% |

## Citation

```bibtex
@article{pele2026conformal,
  author  = {Pele, Daniel Traian},
  title   = {Distribution-Free Recalibration of Tail Quantile
             Forecasts under Temporal Dependence},
  year    = {2026},
  url     = {https://github.com/QuantLet/Conformal_Oracle}
}
```

## License

MIT
