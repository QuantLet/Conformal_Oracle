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

MODEL_ORDER = ['Chronos-Small', 'Chronos-Mini', 'TimesFM-2.5',
               'Moirai-2.0', 'Lag-Llama',
               'GJR-GARCH', 'GARCH-N', 'Hist-Sim', 'EWMA']

# ── Load results (alpha = 0.01) ───────────────────────────────────
df  = pd.read_csv(RES_DIR / 'all_results.csv')
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
tsfm = ['Chronos-Small', 'Chronos-Mini', 'TimesFM-2.5',
        'Moirai-2.0', 'Lag-Llama']
mean_vol  = result.loc[tsfm, 'Ann.Vol'].mean()
mean_tail = result.loc[tsfm, 'Tail Freq.'].mean()
print(f'\nTSFM mean vol corr:  {mean_vol:.2f}')
print(f'TSFM mean tail corr: {mean_tail:.2f}')
print(f'GJR-GARCH vol corr:  {result.loc["GJR-GARCH", "Ann.Vol"]:.3f}')

# ── Save CSV ──────────────────────────────────────────────────────
result.to_csv(OUT_DIR / 'tab_cross_sectional.csv')
result.to_csv(RES_DIR / 'cross_sectional_reproduced.csv')

# ── Save LaTeX ────────────────────────────────────────────────────
def fmt(x):
    s = f'{x:.3f}'
    return f'$-${s[1:]}' if x < 0 else s

lines = [
    r'\begin{tabular}{@{}l rrr@{}}',
    r'\toprule',
    r'Model & Ann.\ Volatility & Excess Kurtosis & Tail Frequency \\',
    r'\midrule',
]
for i, model in enumerate(MODEL_ORDER):
    r = result.loc[model]
    label = model.replace('-', '-').replace('2.0', '2.0').replace('2.5', '2.5')
    if model == 'Hist-Sim':
        label = r'Hist.\ Sim.'
    elif model == 'TimesFM-2.5':
        label = 'TimesFM 2.5'
    elif model == 'Moirai-2.0':
        label = 'Moirai 2.0'
    line = f'{label} & {fmt(r["Ann.Vol"])} & {fmt(r["Kurt."])} & {fmt(r["Tail Freq."])} \\\\'
    lines.append(line)
    if i == 4:
        lines.append(r'\midrule')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')

tex = '\n'.join(lines) + '\n'
tex_path = OUT_DIR / 'tab_cross_sectional.tex'
tex_path.write_text(tex)
print(f'\nSaved {tex_path.name}')
print(tex)
