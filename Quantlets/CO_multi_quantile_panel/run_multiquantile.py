"""
CO_multi_quantile_panel — Multi-quantile evaluation for TSFMs.
Produces tab_multiquantile.tex  (Table 5 in the paper).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

# ── Paths ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'cfp_ijf_data'
RES_DIR  = DATA_DIR / 'paper_outputs' / 'tables'
OUT_DIR  = Path(__file__).resolve().parent

TSFM_ORDER = ['Chronos-Small', 'Chronos-Mini', 'TimesFM-2.5',
              'Moirai-1.1', 'Moirai-2.0', 'Lag-Llama']
ALPHAS = [0.01, 0.025, 0.05, 0.10]

# ── Load ───────────────────────────────────────────────────────────
df = pd.read_csv(RES_DIR / 'all_results.csv')
moirai11 = pd.read_csv(RES_DIR / 'moirai11_full_results.csv')
df = pd.concat([df, moirai11], ignore_index=True)
df = df[df['model'].isin(TSFM_ORDER)].copy()
print(f'Loaded {len(df)} TSFM rows '
      f'({df["model"].nunique()} models, '
      f'{df["symbol"].nunique()} assets, '
      f'{df["alpha"].nunique()} alphas)')

# ── Compute 5 × 4 × 2 cells ──────────────────────────────────────
rows = []
for model in TSFM_ORDER:
    for alpha in ALPHAS:
        sub = df[(df['model'] == model) & (df['alpha'] == alpha)]
        pi_hat = sub['pihat_cp'].mean()
        rej = int((sub['p_kup_cp'] < 0.05).sum())
        n_assets = len(sub)
        rows.append({
            'model': model, 'alpha': alpha,
            'pi_hat': pi_hat, 'rej': rej, 'n_assets': n_assets,
        })

result = pd.DataFrame(rows)
print('\nMulti-quantile table:')
for _, r in result.iterrows():
    print(f'  {r["model"]:16s}  α={r["alpha"]:.3f}  '
          f'π̂={r["pi_hat"]:.3f}  Rej={r["rej"]:2d}/{r["n_assets"]}')

# ── Save CSV ──────────────────────────────────────────────────────
result.to_csv(OUT_DIR / 'tab_multiquantile.csv', index=False)

# ── Save LaTeX ────────────────────────────────────────────────────
lines = [
    r'\begin{tabular}{@{}l rr rr rr rr@{}}',
    r'\toprule',
    r'& \multicolumn{2}{c}{$\alpha=0.01$}',
    r'& \multicolumn{2}{c}{$\alpha=0.025$}',
    r'& \multicolumn{2}{c}{$\alpha=0.05$}',
    r'& \multicolumn{2}{c}{$\alpha=0.10$} \\',
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
    r'\cmidrule(lr){6-7}\cmidrule(l){8-9}',
    r'Model & $\hat\pi$ & Rej.',
    r'& $\hat\pi$ & Rej.',
    r'& $\hat\pi$ & Rej.',
    r'& $\hat\pi$ & Rej. \\',
    r'\midrule',
]

for model in TSFM_ORDER:
    label = model
    if model == 'TimesFM-2.5':
        label = 'TimesFM 2.5'
    elif model == 'Moirai-1.1':
        label = 'Moirai 1.1'
    elif model == 'Moirai-2.0':
        label = 'Moirai 2.0'

    cells = []
    for alpha in ALPHAS:
        r = result[(result['model'] == model) & (result['alpha'] == alpha)].iloc[0]
        pi_dec = Decimal(str(r['pi_hat'])).quantize(
            Decimal('0.001'), rounding=ROUND_HALF_UP)
        pi_str = f'.{int(pi_dec * 1000):03d}'
        rej_str = f'{r["rej"]:d}/{r["n_assets"]}'
        cells.append(f'{pi_str} & {rej_str}')

    line = f'{label} & ' + '\n& '.join(cells) + r' \\'
    lines.append(line)

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')

tex = '\n'.join(lines) + '\n'
tex_path = OUT_DIR / 'tab_multiquantile.tex'
tex_path.write_text(tex)
print(f'\nSaved {tex_path.name}')
print(tex)
