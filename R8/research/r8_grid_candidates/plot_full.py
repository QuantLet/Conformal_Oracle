"""Transparent candidate-comparison figure from the saved result matrices."""
import os
os.environ['MPLCONFIGDIR'] = '/private/tmp/irfa-grid-matplotlib'
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]/'artifacts/r8_grid_candidates/common_evaluation'
summary = pd.read_csv(ROOT/'summary.csv').set_index(['model', 'method'])
intervals = pd.read_csv(ROOT/'intervals.csv').set_index(['comparison', 'block_length'])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
    'axes.labelcolor': '#182334', 'text.color': '#182334', 'xtick.color': '#334155',
    'ytick.color': '#182334', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.4, 7.6), gridspec_kw={'width_ratios': [1, 1.16]})
fig.subplots_adjust(left=.12, right=.97, top=.81, bottom=.24, wspace=.66)
fig.suptitle('Native-tail forecasts on a common evaluation sample', x=.12, y=.97,
             ha='left', fontsize=18, fontweight='bold')
fig.text(.12, .909, '24 assets  ·  37,483 test observations per model  ·  1% lower quantile', fontsize=11.5)
models = ['Chronos-2', 'PatchTST-FM', 'Moirai-1.1', 'Lag-Llama', 'GJR-GARCH',
          'GJR-GARCH-t', 'GARCH-N', 'Hist-Sim', 'EWMA']
colours = {'Raw': '#F47716', 'Static': '#8538E6', 'Rolling250': '#008BCB'}
labels = {'Raw': 'Raw', 'Static': 'Static shift', 'Rolling250': 'Rolling 250'}
for i, model in enumerate(models):
    values = [summary.loc[(model, m), 'QS_x10000'] for m in colours]
    ax.plot([min(values), max(values)], [i, i], color='#CBD5E1', linewidth=1.5, zorder=1)
for method, colour in colours.items():
    values = [summary.loc[(m, method), 'QS_x10000'] for m in models]
    ax.scatter(values, np.arange(len(models)), s=49, color=colour, label=labels[method], zorder=3,
               marker={'Raw': 'o', 'Static': 'D', 'Rolling250': 's'}[method])
ax.set_yticks(np.arange(len(models)), models); ax.invert_yaxis()
ax.tick_params(axis='y', length=0, pad=9)
ax.set_xlim(4.6, 6.35); ax.set_xlabel('Mean Quantile Score × 10,000\nLower is better', labelpad=12)
ax.set_title('Forecast losses', loc='left', fontsize=13, fontweight='bold', pad=19)
ax.grid(axis='x', color='#CBD5E1', alpha=.55, linewidth=.65); ax.set_axisbelow(True)
ax.legend(loc='upper center', bbox_to_anchor=(.48, -.20), ncol=3, frameon=False,
          columnspacing=1, handletextpad=.35, fontsize=9.5)
names = list(dict.fromkeys(intervals.index.get_level_values(0)))
display = ['PatchTST\nStatic − Raw', 'PatchTST\nRolling 250 − Raw',
           'Chronos-2\nStatic − Raw', 'Chronos-2\nRolling 250 − Raw',
           'Raw\nPatchTST − Chronos-2', 'Static\nPatchTST − Chronos-2']
for length, offset, colour, marker in [(20, -.12, '#E64A19', 'o'), (60, .12, '#6D28D9', 's')]:
    rows = intervals.xs(length, level=1).loc[names]
    point = rows.estimate_x10000.to_numpy()
    error = np.vstack([point-rows.simultaneous_lo_x10000.to_numpy(),
                       rows.simultaneous_hi_x10000.to_numpy()-point])
    bx.errorbar(point, np.arange(6)+offset, xerr=error, fmt=marker, color=colour,
                markersize=4.7, capsize=3, linewidth=1.6, label=f'{length}-day blocks')
bx.set_yticks(np.arange(6), display); bx.invert_yaxis()
bx.tick_params(axis='y', length=0, pad=9)
bx.axvline(0, color='#64748B', linestyle=(0, (3, 3)), linewidth=1)
bx.grid(axis='x', color='#CBD5E1', alpha=.55, linewidth=.65); bx.set_axisbelow(True)
bx.set_xlim(-.57, .7)
bx.set_xlabel('Difference in Quantile Score × 10,000\nNegative favours the first method', labelpad=12)
bx.set_title('Paired differences and uncertainty', loc='left', fontsize=13, fontweight='bold', pad=19)
bx.legend(loc='upper center', bbox_to_anchor=(.5, -.20), ncol=2, frameon=False,
          columnspacing=1.3, handletextpad=.6, fontsize=9.5)
fig.text(.12, .043, 'Equal asset weights. Simultaneous 95% bootstrap bands for the six declared contrasts;\n'
         '999 circular calendar-block resamples per block length. Reference-model rankings are descriptive.', fontsize=9.5,
         linespacing=1.5, color='#475569')
folder = ROOT/'figures'; folder.mkdir(exist_ok=True)
for extension in ['png', 'pdf', 'svg']:
    fig.savefig(folder/f'native_common_comparison.{extension}', dpi=200, transparent=True)
plt.close(fig)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
paths = [Path(__file__), ROOT/'summary.csv', ROOT/'intervals.csv']
(folder/'receipt.json').write_text(json.dumps({'inputs': {str(p): sha(p) for p in paths},
    'outputs': {p.name: sha(p) for p in folder.glob('native_common_comparison.*')},
    'transparent': True, 'legends': 'outside, below each plot'}, indent=2)+'\n')
print(folder/'native_common_comparison.png')
