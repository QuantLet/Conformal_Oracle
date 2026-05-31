"""
CO_rolling_qV — run_rolling_qV.py
==================================
Rolling 250-day conformal threshold q̂_{V,t} for four representative
models on S&P 500, with realised volatility overlay. Produces Figure 1
of the paper.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent.parent
DATA = BASE / 'cfp_ijf_data' / 'paper_outputs' / 'tables'
FIG_DIR = BASE / 'figures'
OUT = SCRIPT_DIR

C_RED = '#E31E24'
C_BLUE = '#0066CC'
C_PURPLE = '#7B2FBE'
C_GRAY = '#666666'

plt.rcParams.update({
    'font.family': 'serif', 'axes.grid': False,
    'savefig.transparent': True, 'savefig.dpi': 600,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 18,
})

MODELS = ['Chronos-Small', 'TimesFM-2.5', 'Lag-Llama', 'GJR-GARCH']
DISPLAY = {'Chronos-Small': 'Chronos-Small', 'Chronos-Mini': 'Chronos-Mini',
           'TimesFM-2.5': 'TimesFM 2.5', 'Moirai-1.1': 'Moirai 1.1',
           'Moirai-2.0': 'Moirai 2.0', 'Lag-Llama': 'Lag-Llama',
           'GJR-GARCH': 'GJR-GARCH', 'GARCH-N': 'GARCH-N',
           'Hist-Sim': 'Hist-Sim', 'EWMA': 'EWMA'}
COLORS = [C_RED, C_BLUE, C_PURPLE, C_GRAY]

# ── Load data ──────────────────────────────────────────────────
rqv = pd.read_csv(DATA / 'rolling_qv_SP500.csv',
                  index_col=0, parse_dates=True)

print(f"Loaded rolling_qv_SP500.csv: {len(rqv)} rows, "
      f"{rqv.index[0].date()} to {rqv.index[-1].date()}")

# ── Figure ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

for m, c in zip(MODELS, COLORS):
    if m in rqv.columns:
        s = rqv[m].dropna()
        ax.plot(s.index, s.values, color=c, lw=2.5, label=DISPLAY[m], alpha=0.85)

ax.axhline(0, color='black', lw=0.8, ls=':')
ax.set_title(r'Rolling 250-day $\hat{q}_{V,t}$ on S&P 500',
             fontsize=22, fontweight='bold')
ax.set_ylabel(r'$\hat{q}_{V,t}$', fontsize=18)
ax.set_xlabel('Date', fontsize=18)

if 'rvol' in rqv.columns:
    ax2 = ax.twinx()
    rvol = rqv['rvol'].dropna()
    ax2.fill_between(rvol.index, 0, rvol.values, alpha=0.10, color=C_GRAY,
                     label='Realised vol.')
    ax2.set_ylabel('Realised vol.', color=C_GRAY, fontsize=16)
    ax2.tick_params(axis='y', labelcolor=C_GRAY, labelsize=14)
    ax2.spines['top'].set_visible(False)

ax.tick_params(axis='both', labelsize=14)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = (ax2.get_legend_handles_labels() if 'rvol' in rqv.columns
           else ([], []))
ax.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=5, fontsize=14, frameon=False)

FIG_DIR.mkdir(exist_ok=True)
for ext in ['pdf', 'png']:
    fig.savefig(OUT / f'fig_rolling_qv.{ext}', bbox_inches='tight')
    fig.savefig(FIG_DIR / f'fig_rolling_qv.{ext}', bbox_inches='tight')

plt.close(fig)
print("Saved: fig_rolling_qv.pdf/.png")
