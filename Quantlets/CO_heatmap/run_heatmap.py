"""
CO_heatmap — run_heatmap.py
============================
Basel Traffic Light heatmap (10 models x 24 assets) at alpha = 0.01.
Two stacked panels: raw vs conformal-corrected. Cell color encodes
annualized violation counts with Basel zone thresholds.

Input:  cfp_ijf_data/paper_outputs/tables/all_results.csv
        cfp_ijf_data/paper_outputs/tables/moirai11_results.csv
Output: fig_traffic_light.pdf/.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.grid': False,
    'savefig.transparent': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
})

MODEL_ORDER = ['Chronos-Small', 'Chronos-Mini', 'TimesFM-2.5',
               'Moirai-1.1', 'Moirai-2.0', 'Lag-Llama',
               'GJR-GARCH', 'GARCH-N', 'Hist-Sim', 'EWMA']
MODEL_DISPLAY = ['Chronos-Small', 'Chronos-Mini', 'TimesFM 2.5',
                 'Moirai 1.1', 'Moirai 2.0', 'Lag-Llama',
                 'GJR-GARCH', 'GARCH-N', 'Hist-Sim', 'EWMA']

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent.parent
DATA = BASE / 'cfp_ijf_data' / 'paper_outputs' / 'tables'
FIG_DIR = BASE / 'figures'
SLIDE_DIR = BASE / 'ICFS 2026'
OUT = SCRIPT_DIR

df = pd.read_csv(DATA / 'all_results.csv')
m11 = pd.read_csv(DATA / 'moirai11_results.csv')
df = pd.concat([df, m11], ignore_index=True)
d01 = df[df['alpha'] == 0.01].copy()
d01['ann_raw'] = d01['viol_raw'] / d01['n_test'] * 250
d01['ann_cp'] = d01['viol_cp'] / d01['n_test'] * 250

print(f"Loaded {len(df)} rows "
      f"({df['model'].nunique()} models, "
      f"{df['symbol'].nunique()} assets)")

ASSETS = sorted(d01['symbol'].unique())

mat_raw = np.full((len(MODEL_ORDER), len(ASSETS)), np.nan)
mat_cp = np.full((len(MODEL_ORDER), len(ASSETS)), np.nan)
for i, m in enumerate(MODEL_ORDER):
    for j, a in enumerate(ASSETS):
        row = d01[(d01['model'] == m) & (d01['symbol'] == a)]
        if len(row) == 1:
            mat_raw[i, j] = row.iloc[0]['ann_raw']
            mat_cp[i, j] = row.iloc[0]['ann_cp']

boundaries = [0, 4.5, 9.5, 25, 50, 100, 250]
colors_list = [
    '#2ECC40',
    '#FFDC00',
    '#FF6B6B',
    '#E83030',
    '#C8102E',
    '#B22222',
]
cmap = ListedColormap(colors_list)
norm = BoundaryNorm(boundaries, cmap.N, clip=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                gridspec_kw={'hspace': 0.30})

for ax, mat, title in [
    (ax1, mat_raw, 'Before Correction'),
    (ax2, mat_cp,  'After Conformal Correction'),
]:
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect='auto',
                   interpolation='nearest')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=8)
    ax.set_xticks(range(len(ASSETS)))
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_DISPLAY, fontsize=11)
    ax.set_xticklabels(ASSETS, fontsize=11, rotation=30, ha='right')
    ax.tick_params(length=0)

cbar_ax = fig.add_axes([0.15, 0.02, 0.70, 0.025])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                    ticks=[2.25, 7, 17, 37, 75, 175])
cbar.ax.set_xticklabels([
    'Green\n(≤4)', 'Yellow\n(5–9)', 'Red\n(10–25)',
    'Red\n(25–50)', 'Red\n(50–100)', 'Deep Red\n(100–250)'],
    fontsize=9)
cbar.ax.tick_params(length=0)

FIG_DIR.mkdir(exist_ok=True)
for ext in ['pdf', 'png']:
    fig.savefig(OUT / f'fig_traffic_light.{ext}',
                dpi=200, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(FIG_DIR / f'fig_traffic_light.{ext}',
                dpi=200, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(SLIDE_DIR / f'fig_traffic_light.{ext}',
                dpi=200, bbox_inches='tight', pad_inches=0.05)

plt.close(fig)

for label, mat in [('Raw', mat_raw), ('Corrected', mat_cp)]:
    green = np.sum(mat <= 4)
    yellow = np.sum((mat > 4) & (mat <= 9))
    red = np.sum(mat > 9)
    total = green + yellow + red
    print(f"{label}: Green={int(green)} ({green/total*100:.0f}%), "
          f"Yellow={int(yellow)} ({yellow/total*100:.0f}%), "
          f"Red={int(red)} ({red/total*100:.0f}%)")
print("Saved: fig_traffic_light.pdf/.png")
