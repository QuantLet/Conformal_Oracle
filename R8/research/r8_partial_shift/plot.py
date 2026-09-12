"""Transparent figures from the complete fixed-design partial-shift results."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/irfa-partial-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/irfa-partial-cache')
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/r8_partial_shift'
COLOURS = ['#006BFF', '#FF8C00', '#00A878', '#E6007E']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary = pd.read_csv(OUT / 'run/summary.csv')
    decomposition = pd.read_csv(OUT / 'run/decomposition.csv')
    outputs = []
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'svg.hashsalt':'irfa-partial-shift'}):
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8), sharey=True)
        components = [('removable_loss', -1, 'Available loss reduction'),
                      ('inner_estimation_cost', 1, 'Shift estimation cost'),
                      ('oracle_shrinkage_gain', -1, 'Available shrinkage gain'),
                      ('selection_regret', 1, 'Cost of choosing the fraction')]
        for ax, alpha in zip(axes, [.01, .05]):
            d = decomposition[decomposition.alpha == alpha].groupby('n_cal').mean(numeric_only=True)
            x = np.arange(len(d)); above = np.zeros(len(d)); below = np.zeros(len(d))
            for (name, sign, label), colour in zip(components, COLOURS):
                height = sign * d[name].to_numpy() * 1e4
                bottom = above if sign > 0 else below
                ax.bar(x, height, bottom=bottom.copy(), width=.62, color=colour, label=label)
                bottom += height
            ax.scatter(x, d.selected_change * 1e4, color='#1C2636', marker='D', s=38, zorder=4)
            ax.axhline(0, color='#1C2636', lw=.8)
            ax.set_xticks(x, d.index.astype(str)); ax.set_xlabel('Calibration observations')
            ax.set_title(f'{int(alpha * 100)}% tail', loc='left', weight='bold')
            ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
        axes[0].set_ylabel('Expected pinball-loss contribution (×10⁴)')
        fig.suptitle('Choosing the correction has its own estimation cost', x=.075, ha='left', weight='bold', fontsize=16)
        handles = [Patch(color=c, label=v[2]) for v, c in zip(components, COLOURS)]
        handles.append(Line2D([], [], color='#1C2636', marker='D', lw=0, label='Net selected-minus-raw loss'))
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.51,.035), ncol=2, frameon=False)
        fig.text(.51,.012,'Equal configuration weights; 500 shared histories per cell. Oracles are infeasible references.', ha='center', fontsize=9)
        fig.subplots_adjust(left=.075, right=.985, top=.87, bottom=.29, wspace=.15)
        for suffix in ['png','svg']:
            p = OUT / ('selection_cost.' + suffix)
            fig.savefig(p, dpi=190, transparent=True, bbox_inches='tight', metadata={'Date':None} if suffix == 'svg' else {})
            outputs.append(p)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.5))
        names = [('Raw','Raw','#58657A','o'), ('Full-CP','Full correction','#006BFF','s'),
                 ('Half-Full','Fixed half correction','#00A878','^'),
                 ('Selected-Inner','Past-selected fraction','#E6007E','D'),
                 ('Oracle-Grid','Grid oracle (infeasible)','#FF8C00','*')]
        for ax, alpha in zip(axes, [.01,.05]):
            d = summary[summary.alpha == alpha].groupby('method').mean(numeric_only=True)
            for method, label, colour, marker in names:
                ax.scatter(100*d.loc[method,'expected_violation'],1e4*d.loc[method,'expected_loss'],
                           color=colour,marker=marker,s=90 if marker!='*' else 160,zorder=3)
            ax.axvline(100*alpha,color='#58657A',ls='--',lw=1)
            ax.set_xlabel('Expected violation rate (%)')
            ax.set_title(f'{int(alpha*100)}% tail',loc='left',weight='bold')
            ax.grid(alpha=.15);ax.set_axisbelow(True)
            ax.margins(x=.14,y=.15)
        axes[0].set_ylabel('Expected pinball loss (×10⁴; lower is better)')
        fig.suptitle('Loss and coverage assess different costs of correction',x=.075,ha='left',weight='bold',fontsize=16)
        handles=[Line2D([],[],color=colour,marker=marker,lw=0,label=label,markersize=9) for _,label,colour,marker in names]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.51,.04),ncol=3,frameon=False)
        fig.text(.51,.012,'72 equally weighted configurations per panel. Dashed line: nominal rate. No financial panel is re-estimated.',ha='center',fontsize=9)
        fig.subplots_adjust(left=.075,right=.985,top=.86,bottom=.29,wspace=.26)
        for suffix in ['png','svg']:
            p=OUT/('partial_frontier.'+suffix)
            fig.savefig(p,dpi=190,transparent=True,bbox_inches='tight',metadata={'Date':None} if suffix=='svg' else {})
            outputs.append(p)
        plt.close(fig)
    receipt=dict(producer_sha256=sha(__file__),inputs={str(p.relative_to(ROOT)):sha(p) for p in
        [OUT/'run/summary.csv',OUT/'run/decomposition.csv']},outputs={p.name:sha(p) for p in outputs})
    (OUT/'figures.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Created four transparent figure files')


if __name__ == '__main__':
    main()
