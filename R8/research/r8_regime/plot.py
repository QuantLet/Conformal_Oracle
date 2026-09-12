"""All regimes and both levels; transparent exports with external bottom legends."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/irfa-regime-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/irfa-regime-cache')
import json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import engine as e
_spec = importlib.util.spec_from_file_location('regime_runner', Path(__file__).with_name('run.py'))
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)

STYLES = {'Raw': ('#556274', '--'), 'Static-CP': ('#AA43E8', '-.'),
          'Rolling250': ('#0075FF', '-'), 'Rolling1000': ('#F47700', '-'),
          'VolRolling250': ('#00A878', '-'), 'DtACI500': ('#E6007E', '-'),
          'InitialKupiecGate': ('#009CAB', ':')}
LABELS = {'Raw': 'Raw', 'Static-CP': 'Static correction', 'Rolling250': 'Rolling 250',
          'Rolling1000': 'Rolling 1,000', 'VolRolling250': 'Scaled rolling 250',
          'DtACI500': 'Projected DtACI', 'InitialKupiecGate': 'Initial Kupiec gate'}
TITLES = {'correct': 'Correct throughout', 'biased': 'Biased throughout',
          'appears': 'Bias appears', 'disappears': 'Bias disappears', 'reverses': 'Bias reverses',
          'jump_oracle': 'Scale doubles · oracle forecast',
          'steady_ewma': 'Steady scale · EWMA forecast', 'jump_ewma': 'Scale doubles · EWMA forecast'}


def make_curves(df, kind, alpha, scenarios):
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(10, 2.3*len(scenarios)+1.2), sharex=True)
    for i, scenario in enumerate(scenarios):
        for j, metric in enumerate(('excess_loss', 'hit_probability')):
            ax = axes[i, j]
            for method, (colour, ls) in STYLES.items():
                g = df[(df.scenario == scenario)&(df.innovation == kind)&(df.alpha == alpha)&
                       (df.metric == metric)&(df.method == method)].sort_values('day_from_break')
                # Fixed, nonoverlapping 20-day bin averages; actual breakpoint
                # is a bin boundary. No result-dependent smoothing or truncation.
                day = g.day_from_break.to_numpy()
                binid = np.floor((day-1)/20).astype(int)
                means = g.assign(bin=binid).groupby('bin')[['day_from_break', 'mean']].mean()
                factor = 1/e.V0 if metric == 'excess_loss' else 100
                ax.plot(means.day_from_break, means['mean']*factor, color=colour, ls=ls, lw=1.55, label=LABELS[method])
            ax.axvline(0, color='#95A1B1', lw=.8)
            if j == 1:
                ax.axhline(100*alpha, color='#95A1B1', lw=.8, ls='--')
            ax.set_title(TITLES[scenario], loc='left', weight='bold')
            ax.set_ylabel('Excess QS / baseline σ' if j == 0 else 'Expected violations (%)')
            ax.set_xlim(-250, 1000)
            ax.grid(alpha=.15)
            ax.patch.set_alpha(0)
            if i == len(scenarios)-1:
                ax.set_xlabel('Trading days relative to the break')
    handles = [Line2D([], [], color=c, ls=ls, lw=2, label=LABELS[m]) for m, (c, ls) in STYLES.items()]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .01), ncol=4,
               frameon=False, columnspacing=1.6, handlelength=2.5, fontsize=9)
    fig.suptitle(f'{"Normal" if kind == "normal" else "Student t(5)"} innovations · {alpha:.0%} tail',
                 x=.075, y=.99, ha='left', fontsize=13, weight='bold')
    fig.subplots_adjust(left=.08, right=.98, top=.92,
                        bottom=.105 if len(scenarios) == 5 else .15, hspace=.48, wspace=.25)
    fig.patch.set_alpha(0)
    return fig


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 10.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#95A1B1',
        'text.color': '#17243B', 'axes.labelcolor': '#17243B', 'svg.fonttype': 'none', 'svg.hashsalt': 'irfa-regime'})
    p = run.OUT/'results/curves.parquet'
    df = pd.read_parquet(p)
    out = run.OUT/'figures'
    out.mkdir(exist_ok=True)
    files = []
    # All eight scenarios appear: no figure selection from observed rankings.
    for kind in ('normal', 't5'):
        for alpha in e.ALPHAS:
            for group, scenarios in [('bias', e.SCENARIOS[:5]), ('scale', e.SCENARIOS[5:])]:
                fig = make_curves(df, kind, alpha, scenarios)
                for ext in ('png', 'svg'):
                    dest = out/f'{kind}_{alpha:g}_{group}.{ext}'
                    meta = {'metadata': {'Date': None}} if ext == 'svg' else {}
                    fig.savefig(dest, dpi=190, transparent=True, bbox_inches='tight', pad_inches=.08, **meta)
                    files.append(dest)
                plt.close(fig)
    (run.OUT/'figures.json').write_text(json.dumps({'producer_sha256': e.sha(__file__),
        'input_sha256': e.sha(p), 'outputs': {str(p.relative_to(run.OUT)): e.sha(p) for p in files}}, indent=2)+'\n')
    print('Rendered', len(files), 'transparent figure exports.')


if __name__ == '__main__':
    main()
