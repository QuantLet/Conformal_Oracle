# Conformal Oracle — Replication Package

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuantLet/Conformal_Oracle/blob/main/CO_conformal_garch_ewma/CO_conformal_garch_ewma.ipynb)

**Paper:** *Distribution-Free Recalibration of Tail Quantile Forecasts under Temporal Dependence*

## Quick Start

### Option 1: Google Colab (no installation)

- [GARCH/EWMA Conformal Analysis](https://colab.research.google.com/github/QuantLet/Conformal_Oracle/blob/main/CO_conformal_garch_ewma/CO_conformal_garch_ewma.ipynb)
- [LLM Conformal Analysis](https://colab.research.google.com/github/QuantLet/Conformal_Oracle/blob/main/CO_conformal_llm/CO_conformal_llm.ipynb)

### Option 2: Local reproduction

```bash
git clone https://github.com/QuantLet/Conformal_Oracle.git
cd Conformal_Oracle

# Install dependencies
pip install numpy pandas matplotlib scipy openpyxl

# Run LLM conformal analysis (requires simulation CSVs)
python run_conformal_analysis.py

# Run GARCH/EWMA conformal analysis (requires pickle files)
python run_conformal_garch_ewma.py
```

## Repository Structure

```
Conformal_Oracle/
├── README.md                          # This file
├── run_conformal_analysis.py          # Main LLM conformal analysis
├── run_conformal_garch_ewma.py        # GARCH/EWMA conformal analysis
├── CO_conformal_llm/                  # Notebook: LLM analysis
│   └── CO_conformal_llm.ipynb
├── CO_conformal_garch_ewma/           # Notebook: GARCH/EWMA analysis
│   └── CO_conformal_garch_ewma.ipynb
├── CO_dual_correction/                # Figure: VaR correction time series
├── CO_coverage/                       # Figure: Coverage comparison
├── CO_cross_model/                    # Figure: Cross-model thresholds
├── CO_GARCH_comparison/               # Figure: GARCH benchmark
├── CO_frontier/                       # Figure: Coverage-efficiency frontier
├── CO_freq_magnitude/                 # Figure: q_V diagnostic
├── CO_heatmap/                        # Figure: q_V heatmap
├── CO_rolling_qV/                     # Figure: Rolling stability
├── CO_traffic_light/                  # Figure: Traffic light matrices
├── CO_POF/                            # Figure: POF test results
├── CO_Z2_*/                           # Figures: Z2 ES diagnostics
└── CO_*/                              # Other QuantLet figures
```

## Data

- **LLM simulations**: CSV files with 1024 samples per day (available on request)
- **Asset data**: Excel files with daily close prices (9 assets, Oct 2021 — Mar 2024)
- **GARCH/EWMA**: Pickle files with pre-computed VaR/ES forecasts
- **Results**: CSV files with all 162 backtest results

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| alpha_VaR | 0.01  | VaR confidence level (99%) |
| alpha_ES  | 0.025 | ES confidence level |
| f_cal     | 0.70  | Calibration fraction (70/30 split) |
| n_samples | 1024  | LLM samples per day |
| tau       | 0.7   | LLM temperature |

## Key Results

| Method | Mean q_V | Green/9 | Interpretation |
|--------|----------|---------|----------------|
| EWMA-DCS | +0.002 | 9/9 | Near-perfect calibration |
| GARCH-N | +0.004 | 7/9 | Small thin-tail correction |
| GAS-t | -0.007 | 6/9 | Over-conservative |
| GPT-3.5 | +0.002 | 8/9 | Well-calibrated (pre-RLHF) |
| GPT-4 | +0.024 | 9/9 | Large RLHF correction |
| GPT-4o | +0.020 | 9/9 | Large RLHF correction |

## Citation

```bibtex
@article{pele2026conformal,
  author  = {Pele, Daniel Traian},
  title   = {Distribution-Free Recalibration of Tail Quantile
             Forecasts under Temporal Dependence},
  journal = {Working Paper},
  year    = {2026}
}
```

## License

MIT
