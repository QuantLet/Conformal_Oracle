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
# Moirai-1.1 used to live in a separate file and was concatenated here. Since
# 2026-08-17 all_results.csv is regenerated over every model in cfp_config and
# carries it directly, so concatenating both entered the model twice -- and the
# separate file is a pre-correction vintage, so the duplicate rows disagreed
# with the live ones. See analysis/moirai11_reconciliation/.
dup = df.duplicated(subset=['model', 'symbol', 'alpha']).sum()
assert dup == 0, f'{dup} duplicated (model, symbol, alpha) rows in all_results.csv'
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

# Broken y-axis: the raw rates span two orders of magnitude
# (0.4% for GJR-GARCH up to ~99% for TimesFM 2.5 / Moirai 2.0).
# A single linear axis truncated at 0.5 hides the true height of the
# ~0.99 bars, so we split into an upper panel (0.93-1.02, the near-1.0
# replacement-regime bars) and a lower panel (0-0.5, everything else).
fig, (ax_top, ax) = plt.subplots(
    2, 1, sharex=True, figsize=(12, 5.6),
    gridspec_kw={'height_ratios': [1, 2.8], 'hspace': 0.05})

# bars on both panels; label only once (on the lower panel) for the legend
ax.bar(x - w / 2 - 0.02, summary['pi_raw'], w,
       color=IDA_RED, edgecolor='black', linewidth=0.5, label='Raw')
ax.bar(x + w / 2 + 0.02, summary['pi_cp'], w,
       color=FOREST, edgecolor='black', linewidth=0.5, label='Corrected')
ax_top.bar(x - w / 2 - 0.02, summary['pi_raw'], w,
           color=IDA_RED, edgecolor='black', linewidth=0.5)
ax_top.bar(x + w / 2 + 0.02, summary['pi_cp'], w,
           color=FOREST, edgecolor='black', linewidth=0.5)

# 1% target line lives on the lower (detail) panel
ax.axhline(0.01, color='red', ls='--', lw=1.5,
           label=r'1% target ($\alpha=0.01$)')

# --- broken-axis limits ---
ax.set_ylim(0, 0.5)          # lower panel: detail up to 0.5
ax_top.set_ylim(0.93, 1.02)  # upper panel: the ~0.99 bars, true height

# hide the facing spines and draw the diagonal break marks
ax.spines['top'].set_visible(False)
ax_top.spines['bottom'].set_visible(False)
ax_top.tick_params(axis='x', which='both', bottom=False)

dsz = .6  # slant marker size ratio
kwargs = dict(marker=[(-1, -dsz), (1, dsz)], markersize=17,
              linestyle='none', color='k', mec='k', mew=2.2, clip_on=False)
ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
ax.plot([0, 1], [1, 1], transform=ax.transAxes, **kwargs)
# explicit "axis break" label so it reads at projector distance
ax.annotate('axis break', xy=(0.012, 1.0), xycoords='axes fraction',
            xytext=(0.012, 1.0), ha='left', va='center', fontsize=11,
            style='italic', color='#444444')

ax.set_xticks(x)
ax.set_xticklabels(MODEL_SHORT, fontsize=11, rotation=35, ha='right')
ax.set_ylabel(r'Mean violation rate $\hat{\pi}$', fontsize=14)
ax.yaxis.set_label_coords(-0.055, 0.62)
ax.tick_params(axis='y', labelsize=13)
ax_top.tick_params(axis='y', labelsize=13)

ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
ax_top.yaxis.set_major_locator(plt.FixedLocator([0.95, 1.00]))
for a in (ax, ax_top):
    a.grid(axis='y', which='major', color='#cccccc', lw=0.8, alpha=0.5)
    a.set_axisbelow(True)
ax.grid(axis='y', which='minor', color='#dddddd', lw=0.5, alpha=0.3)

# annotations: Chronos-Small (~40x) sits in the lower panel;
# TimesFM 2.5 / Moirai 2.0 (~99x) reach their true tops in the upper panel
lower_ann = {'Chronos-Small': r'$\sim$40$\times$'}
upper_ann = {'TimesFM-2.5': r'$\sim$99$\times$', 'Moirai-2.0': r'$\sim$99$\times$'}
for model, label in lower_ann.items():
    idx = MODEL_ORDER.index(model)
    val = summary.loc[model, 'pi_raw']
    ax.text(idx - w / 2 - 0.02, val + 0.012, label,
            ha='center', va='bottom', fontsize=10, color='black')
for model, label in upper_ann.items():
    idx = MODEL_ORDER.index(model)
    val = summary.loc[model, 'pi_raw']
    ax_top.text(idx - w / 2 - 0.02, min(val + 0.004, 1.015), label,
                ha='center', va='bottom', fontsize=10, color='black')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
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
