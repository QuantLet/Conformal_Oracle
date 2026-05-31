"""
CFP_Calibration_Efficiency_Frontier — run_frontier.py
=====================================================
Two-panel calibration-efficiency frontier (Figure 3).
Left: raw forecasts (full y-range). Right: after conformal correction
(zoomed). Legend identifies models by unique color+marker.

Input:  cfp_ijf_data/paper_outputs/tables/all_results.csv
        cfp_ijf_data/paper_outputs/tables/moirai11_full_results.csv
Output: fig_frontier_killer.pdf/.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.grid': False,
    'savefig.transparent': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 12,
})

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent.parent
DATA = BASE / 'cfp_ijf_data' / 'paper_outputs' / 'tables'
FIG_DIR = BASE / 'figures'
SLIDE_DIR = BASE / 'ICFS 2026'
OUT = SCRIPT_DIR

TSFMS = ['Chronos-Small', 'Chronos-Mini', 'TimesFM-2.5',
         'Moirai-1.1', 'Moirai-2.0', 'Lag-Llama']
BENCHMARKS = ['GJR-GARCH', 'GARCH-N', 'Hist-Sim', 'EWMA']
SHOW = TSFMS + BENCHMARKS

COLORS = {
    'Chronos-Small': '#8B0000',
    'Chronos-Mini':  '#C8102E',
    'TimesFM-2.5':   '#DC143C',
    'Moirai-1.1':    '#228B22',
    'Moirai-2.0':    '#E06030',
    'Lag-Llama':     '#7B2FBE',
    'GJR-GARCH':     '#003DA5',
    'GARCH-N':       '#4A90D9',
    'Hist-Sim':      '#2CA02C',
    'EWMA':          '#555555',
}

MARKERS = {
    'Chronos-Small': 'o',
    'Chronos-Mini':  's',
    'TimesFM-2.5':   'D',
    'Moirai-1.1':    '^',
    'Moirai-2.0':    'v',
    'Lag-Llama':     'P',
    'GJR-GARCH':     'X',
    'GARCH-N':       'p',
    'Hist-Sim':      'h',
    'EWMA':          '*',
}

DISPLAY = {
    'Chronos-Small': 'Chronos-S',   'Chronos-Mini': 'Chronos-M',
    'TimesFM-2.5':   'TimesFM 2.5', 'Moirai-1.1':   'Moirai 1.1',
    'Moirai-2.0':    'Moirai 2.0',  'Lag-Llama':     'Lag-Llama',
    'GJR-GARCH':     'GJR-GARCH',   'GARCH-N':       'GARCH-N',
    'Hist-Sim':      'Hist-Sim',    'EWMA':          'EWMA',
}

df = pd.read_csv(DATA / 'all_results.csv')
m11 = pd.read_csv(DATA / 'moirai11_full_results.csv')
df = pd.concat([df, m11], ignore_index=True)
d01 = df[df['alpha'] == 0.01].copy()

data = {}
for model in SHOW:
    mdf = d01[d01['model'] == model]
    data[model] = {
        'raw_width':  abs(mdf['raw_width'].mean()),
        'raw_cov':    1 - mdf['pihat_raw'].mean(),
        'corr_width': abs(mdf['VaR_width'].mean()),
        'corr_cov':   1 - mdf['pihat_cp'].mean(),
    }

fig, (ax_raw, ax_corr) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True,
                                       gridspec_kw={'wspace': 0.08})

handles = []
for name in SHOW:
    d = data[name]
    c = COLORS[name]
    m = MARKERS[name]
    ms = 12 if m == '*' else 10
    ax_raw.plot(d['raw_width'], d['raw_cov'], m, color=c, ms=ms,
                mew=0.5, mec='black', zorder=5, clip_on=False)
    ax_corr.plot(d['corr_width'], d['corr_cov'], m, color=c, ms=ms,
                 mew=0.5, mec='black', zorder=5, clip_on=False)
    h = plt.Line2D([0], [0], marker=m, color=c, ms=ms, mew=0.5,
                   mec='black', ls='', label=DISPLAY[name])
    handles.append(h)

for ax in [ax_raw, ax_corr]:
    ax.axhline(y=0.99, color='grey', ls='--', lw=1.5, zorder=1)
    ax.set_xlabel(r'Mean VaR width', fontsize=13)
    ax.tick_params(axis='both', labelsize=12)

ax_raw.set_title('Raw Forecasts', fontsize=14, fontweight='bold', pad=10)
ax_raw.set_ylabel(r'Empirical coverage $1 - \hat{\pi}$', fontsize=13)
ax_raw.set_ylim(0.0, 1.08)
ax_raw.set_xlim(-0.002, 0.052)
ax_raw.text(0.048, 0.93, '99% target', fontsize=10, color='grey',
            style='italic', ha='right', va='top')

ax_corr.set_title('After Conformal Correction', fontsize=14,
                   fontweight='bold', pad=10)
ax_corr.set_xlim(0.033, 0.074)
ax_corr.text(0.072, 0.93, '99% target', fontsize=10, color='grey',
             style='italic', ha='right', va='top')

plt.tight_layout()
fig.subplots_adjust(bottom=0.22)
fig.legend(handles=handles, loc='lower center',
           bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=11,
           frameon=False, handletextpad=0.3, columnspacing=1.2)
FIG_DIR.mkdir(exist_ok=True)
for ext in ['pdf', 'png']:
    fig.savefig(OUT / f'fig_frontier_killer.{ext}',
                dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(FIG_DIR / f'fig_frontier_killer.{ext}',
                dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(SLIDE_DIR / f'fig_frontier_killer.{ext}',
                dpi=300, bbox_inches='tight', pad_inches=0.05)

plt.close(fig)
print("Saved: fig_frontier_killer.pdf/.png")
