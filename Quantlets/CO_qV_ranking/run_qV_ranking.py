"""
CO_qV_ranking — run_qV_ranking.py
==================================
Horizontal bar chart of mean q_V across 10 forecasters (24 assets,
alpha = 0.01), color-coded by regime. Designed for a Beamer half-column.

Input:  cfp_ijf_data/paper_outputs/tables/all_results.csv
        cfp_ijf_data/paper_outputs/tables/moirai11_full_results.csv
Output: fig_qV_ranking.pdf/.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

IDA_RED = '#C8102E'
CRIMSON = '#DC143C'
MAIN_BLUE = '#003DA5'
FOREST = '#228B22'

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

df = pd.read_csv(DATA / 'all_results.csv')
m11 = pd.read_csv(DATA / 'moirai11_full_results.csv')
df = pd.concat([df, m11], ignore_index=True)
d01 = df[df['alpha'] == 0.01].copy()

MODEL_ORDER = ['Chronos-Small', 'Chronos-Mini', 'TimesFM-2.5',
               'Moirai-1.1', 'Moirai-2.0', 'Lag-Llama',
               'GJR-GARCH', 'GARCH-N', 'Hist-Sim', 'EWMA']
DISPLAY = {
    'Chronos-Small': 'Chronos-Small', 'Chronos-Mini': 'Chronos-Mini',
    'TimesFM-2.5': 'TimesFM 2.5', 'Moirai-1.1': 'Moirai 1.1',
    'Moirai-2.0': 'Moirai 2.0', 'Lag-Llama': 'Lag-Llama',
    'GJR-GARCH': 'GJR-GARCH', 'GARCH-N': 'GARCH-N',
    'Hist-Sim': 'Hist. Sim.', 'EWMA': 'EWMA',
}

mean_qv = d01.groupby('model')['qV'].mean().reindex(MODEL_ORDER)
sorted_models = mean_qv.sort_values(ascending=True).index.tolist()

def get_color(val):
    absv = abs(val)
    if absv > 0.05:
        return IDA_RED
    if absv > 0.02:
        return CRIMSON
    if absv > 0.005:
        return MAIN_BLUE
    return FOREST

fig, ax = plt.subplots(figsize=(7, 6))

y = np.arange(len(sorted_models))
vals = [mean_qv[m] for m in sorted_models]
colors = [get_color(v) for v in vals]

ax.barh(y, vals, height=0.7, color=colors, edgecolor='black', linewidth=0.5)

ax.set_yticks(y)
ax.set_yticklabels([DISPLAY[m] for m in sorted_models], fontsize=12)
ax.set_xlabel(r'Mean $\hat{q}_V$ (24 assets, $\alpha = 0.01$)', fontsize=13)
ax.tick_params(axis='x', labelsize=12)

ax.axvline(x=0, color='black', linewidth=1, linestyle='-')

ax.set_xlim(-0.025, 0.125)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.025))
ax.grid(axis='x', which='major', color='#cccccc', lw=0.6, alpha=0.3)
ax.set_axisbelow(True)

for i, (model, val) in enumerate(zip(sorted_models, vals)):
    offset = 0.003 if val >= 0 else -0.003
    ha = 'left' if val >= 0 else 'right'
    ax.text(val + offset, i, f'{val:+.3f}', va='center', ha=ha,
            fontsize=11, color='black')

legend_elements = [
    Patch(facecolor=IDA_RED, edgecolor='black', linewidth=0.5, label='Replacement'),
    Patch(facecolor=CRIMSON, edgecolor='black', linewidth=0.5, label='Under-covers'),
    Patch(facecolor=MAIN_BLUE, edgecolor='black', linewidth=0.5, label='Moderate/Mild'),
    Patch(facecolor=FOREST, edgecolor='black', linewidth=0.5, label='Near-calibrated'),
]
ax.legend(handles=legend_elements, loc='upper center',
          bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=10, frameon=False)

FIG_DIR.mkdir(exist_ok=True)
for ext in ['pdf', 'png']:
    fig.savefig(OUT / f'fig_qV_ranking.{ext}',
                dpi=200, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(FIG_DIR / f'fig_qV_ranking.{ext}',
                dpi=200, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(SLIDE_DIR / f'fig_qV_ranking.{ext}',
                dpi=200, bbox_inches='tight', pad_inches=0.05)

plt.close(fig)
print("Saved: fig_qV_ranking.pdf/.png")
