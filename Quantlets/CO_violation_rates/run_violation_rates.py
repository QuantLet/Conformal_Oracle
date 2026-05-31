"""
CO_violation_rates — run_violation_rates.py
============================================
Raw vs corrected violation rates for 10 forecasters at alpha = 0.01,
averaged across 24 assets. Side-by-side bar chart with 1% target
dashed line. Produces Figure 4 of the paper.

Input:  cfp_ijf_data/paper_outputs/tables/all_results.csv
Output: fig_violation_rates.pdf/.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

IDA_RED = '#C8102E'
FOREST = '#228B22'
MAIN_BLUE = '#003DA5'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.grid': False,
    'savefig.transparent': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 13,
})

MODEL_ORDER = ['Chronos-Small', 'Chronos-Mini', 'TimesFM-2.5',
               'Moirai-1.1', 'Moirai-2.0', 'Lag-Llama',
               'GJR-GARCH', 'GARCH-N', 'Hist-Sim', 'EWMA']
MODEL_SHORT = ['Chr-S', 'Chr-M', 'TFM 2.5',
               'Moi 1.1', 'Moi 2.0', 'Lag-Llm',
               'GJR', 'GARCH-N', 'Hist-Sim', 'EWMA']

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

print(f"Loaded {len(df)} rows, filtered to {len(d01)} at alpha=0.01")

summary = d01.groupby('model').agg(
    pi_raw=('pihat_raw', 'mean'),
    pi_cp=('pihat_cp', 'mean'),
).reindex(MODEL_ORDER)

print("\nMean violation rates (alpha=0.01, 24 assets):")
print(f"{'Model':20s} {'Raw':>10s} {'Corrected':>10s}")
print("-" * 45)
for m, short in zip(MODEL_ORDER, MODEL_SHORT):
    print(f"{m:20s} {summary.loc[m, 'pi_raw']:10.4f} "
          f"{summary.loc[m, 'pi_cp']:10.4f}")

x = np.arange(len(MODEL_ORDER))
w = 0.38

fig, ax = plt.subplots(figsize=(12, 5))

ax.bar(x - w / 2 - 0.02, summary['pi_raw'], w,
       color=IDA_RED, edgecolor='black', linewidth=0.5, label='Raw')
ax.bar(x + w / 2 + 0.02, summary['pi_cp'], w,
       color=FOREST, edgecolor='black', linewidth=0.5, label='Corrected')
ax.axhline(0.01, color='red', ls='--', lw=1.5,
           label=r'1% target ($\alpha=0.01$)')

ax.set_xticks(x)
ax.set_xticklabels(MODEL_SHORT, fontsize=11, rotation=35, ha='right')
ax.set_ylabel(r'Mean violation rate $\hat{\pi}$', fontsize=14)
ax.set_ylim(0, 0.5)
ax.tick_params(axis='y', labelsize=13)

ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
ax.grid(axis='y', which='major', color='#cccccc', lw=0.8, alpha=0.5)
ax.grid(axis='y', which='minor', color='#dddddd', lw=0.5, alpha=0.3)
ax.set_axisbelow(True)

annotations = {
    'Chronos-Small': r'$\sim$40$\times$',
    'TimesFM-2.5':   r'$\sim$99$\times$',
    'Moirai-2.0':    r'$\sim$99$\times$',
}
for model, label in annotations.items():
    idx = MODEL_ORDER.index(model)
    val = summary.loc[model, 'pi_raw']
    bar_top = min(val, 0.48)
    ax.text(idx - w / 2 - 0.02, bar_top + 0.01, label,
            ha='center', va='bottom', fontsize=10, color='black')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
          fontsize=12, frameon=False, ncol=3)

FIG_DIR.mkdir(exist_ok=True)
for ext in ['pdf', 'png']:
    fig.savefig(OUT / f'fig_violation_rates.{ext}',
                dpi=200, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(FIG_DIR / f'fig_violation_rates.{ext}',
                dpi=200, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(SLIDE_DIR / f'fig_violation_rates.{ext}',
                dpi=200, bbox_inches='tight', pad_inches=0.05)

plt.close(fig)
print("\nSaved: fig_violation_rates.pdf/.png")
