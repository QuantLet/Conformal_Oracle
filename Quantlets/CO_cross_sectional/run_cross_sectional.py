"""
CO_cross_sectional — Cross-sectional correlation: qV vs asset characteristics.
Produces tab_cross_sectional.tex  (Table 3 in the paper).
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis

# ── Paths ──────────────────────────────────────────────────────────
DATA_DIR  = Path(__file__).resolve().parent.parent.parent / 'cfp_ijf_data'
RES_DIR   = DATA_DIR / 'paper_outputs' / 'tables'
RET_DIR   = DATA_DIR / 'returns'
OUT_DIR   = Path(__file__).resolve().parent

# Ordered TSFMs first, then the classical benchmarks. The two Chronos
# configurations sit adjacent so the reader can see what the sampling default
# does to the cross-sectional association, which is the one thing this table is
# for.
MODEL_ORDER = ['Chronos-Small', 'Chronos-Small-A', 'Chronos-Mini', 'Chronos-Mini-A',
               'TimesFM-2.5', 'Moirai-1.1', 'Moirai-2.0', 'Lag-Llama',
               'GJR-GARCH', 'GJR-GARCH-t', 'GARCH-N', 'Hist-Sim', 'EWMA']

TSFM_ALL = ['Chronos-Small', 'Chronos-Small-A', 'Chronos-Mini', 'Chronos-Mini-A',
            'TimesFM-2.5', 'Moirai-1.1', 'Moirai-2.0', 'Lag-Llama']

# The 'replacement-regime' summary row is gone with the taxonomy it belonged to.
# It averaged the four series that turned out to be defective and reported the
# average as a property of foundation models.

# ── Load results (alpha = 0.01) ───────────────────────────────────
# Moirai-1.1 is in all_results.csv since the 17 August rebuild; the legacy file
# it used to be concatenated from would now duplicate its rows.
df = pd.read_csv(RES_DIR / 'all_results.csv')
dup = df.duplicated(subset=['model', 'symbol', 'alpha']).sum()
assert dup == 0, f'{dup} duplicated (model, symbol, alpha) rows in all_results.csv'
missing = [m for m in MODEL_ORDER if m not in set(df['model'])]
assert not missing, f'not in all_results.csv: {missing}'
d01 = df[df['alpha'] == 0.01].copy()
assets = sorted(d01['symbol'].unique())
print(f'Loaded {len(d01)} rows at alpha=0.01  '
      f'({d01["model"].nunique()} models, {len(assets)} assets)')

# ── Asset-level characteristics ───────────────────────────────────
chars = {}
for sym in assets:
    r = pd.read_csv(RET_DIR / f'{sym}.csv')['log_return'].values
    mu    = np.mean(r)
    sigma = np.std(r)
    chars[sym] = {
        'ann_vol':   sigma * np.sqrt(252),
        'ex_kurt':   kurtosis(r, fisher=True),
        'tail_freq': np.mean(np.abs(r - mu) > 3 * sigma),
    }

vol_arr  = np.array([chars[a]['ann_vol']   for a in assets])
kurt_arr = np.array([chars[a]['ex_kurt']   for a in assets])
tail_arr = np.array([chars[a]['tail_freq'] for a in assets])

# ── 9 × 3 Pearson correlation matrix ─────────────────────────────
rows = []
for model in MODEL_ORDER:
    qv = (d01[d01['model'] == model]
          .set_index('symbol')
          .reindex(assets)['qV']
          .values)
    rows.append({
        'model':      model,
        'Ann.Vol':    np.corrcoef(qv, vol_arr)[0, 1],
        'Kurt.':      np.corrcoef(qv, kurt_arr)[0, 1],
        'Tail Freq.': np.corrcoef(qv, tail_arr)[0, 1],
    })

result = pd.DataFrame(rows).set_index('model')
print('\nCorrelation matrix:')
print(result.to_string(float_format=lambda x: f'{x:.4f}'))

# ── Prose summary ─────────────────────────────────────────────────
mean_vol_all  = result.loc[TSFM_ALL, 'Ann.Vol'].mean()
mean_tail_all = result.loc[TSFM_ALL, 'Tail Freq.'].mean()
print(f'\nTSFM (all {len(TSFM_ALL)}) mean vol corr:  {mean_vol_all:.4f}')
print(f'TSFM (all {len(TSFM_ALL)}) mean tail corr: {mean_tail_all:.4f}')
for m in MODEL_ORDER:
    print(f'  {m:16s} vol {result.loc[m, "Ann.Vol"]:+.4f}   '
          f'tail {result.loc[m, "Tail Freq."]:+.4f}')

# ── Save CSV ──────────────────────────────────────────────────────
result.to_csv(OUT_DIR / 'tab_cross_sectional.csv')
result.to_csv(RES_DIR / 'cross_sectional_reproduced.csv')

# ── Save LaTeX ────────────────────────────────────────────────────
def fmt(x):
    s = f'{x:.3f}'
    return f'$-${s[1:]}' if x < 0 else s

LABELS = {
    'Chronos-Small':   'Chronos-Small (default)',
    'Chronos-Small-A': 'Chronos-Small (analytic)',
    'Chronos-Mini':    'Chronos-Mini (default)',
    'Chronos-Mini-A':  'Chronos-Mini (analytic)',
    'TimesFM-2.5':     'TimesFM 2.5',
    'Moirai-1.1':      'Moirai 1.1',
    'Moirai-2.0':      'Moirai 2.0',
    'Lag-Llama':       'Lag-Llama',
    'GJR-GARCH':       'GJR-GARCH',
    'GJR-GARCH-t':     r'GJR-GARCH-$t$',
    'GARCH-N':         'GARCH-N',
    'Hist-Sim':        r'Hist.\ Sim.',
    'EWMA':            'EWMA',
}

lines = [
    r'\begin{tabular}{@{}l rrr@{}}',
    r'\hline\hline',
    r'Model & Ann.\ Volatility & Excess Kurtosis & Tail Frequency \\',
    r'\hline',
]
for i, model in enumerate(MODEL_ORDER):
    r = result.loc[model]
    label = LABELS[model]
    line = f'{label} & {fmt(r["Ann.Vol"])} & {fmt(r["Kurt."])} & {fmt(r["Tail Freq."])} \\\\'
    lines.append(line)
    if model == TSFM_ALL[-1]:      # rule between TSFMs and classical benchmarks
        lines.append(r'\hline')

lines.append(r'\hline')
lines.append(r'\multicolumn{4}{@{}l}{\itshape Summary mean} \\[2pt]')
lines.append(f'TSFM series ({len(TSFM_ALL)}) & {fmt(mean_vol_all)} & '
             f'{fmt(result.loc[TSFM_ALL, "Kurt."].mean())} & '
             f'{fmt(mean_tail_all)} \\\\')
lines.append(r'\hline\hline')
lines.append(r'\end{tabular}')

tex = '\n'.join(lines) + '\n'
tex_path = OUT_DIR / 'tab_cross_sectional.tex'
tex_path.write_text(tex)
print(f'\nSaved {tex_path.name}')
print(tex)
