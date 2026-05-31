"""
CO_garch_conformal — Static vs rolling conformal correction (Table 9).
Produces tab_rolling_vs_static.tex and tab_rolling_vs_static.csv.

Reads pre-computed static and rolling conformal results from
rolling_vs_static.csv (9 models × 24 assets). For each model
reports: corrected π̂ (equal-weighted mean), Basel Green count,
and Christoffersen conditional coverage pass count (5% level)
under both static (70/30 split) and rolling (250-day window)
conformal correction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

# ── Paths ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'cfp_ijf_data'
RES_DIR  = DATA_DIR / 'paper_outputs' / 'tables'
OUT_DIR  = Path(__file__).resolve().parent

MODEL_ORDER = ['Chronos-Small', 'Chronos-Mini', 'TimesFM-2.5',
               'Moirai-2.0', 'Lag-Llama',
               'GJR-GARCH', 'GARCH-N', 'Hist-Sim', 'EWMA']

MODEL_LABELS = {
    'TimesFM-2.5': 'TimesFM 2.5',
    'Moirai-2.0':  'Moirai 2.0',
    'Hist-Sim':    r'Hist.\ Sim.',
}

# ── Load data ─────────────────────────────────────────────────────
rv = pd.read_csv(RES_DIR / 'rolling_vs_static.csv')
print(f'Loaded rolling_vs_static.csv: {rv.shape[0]} rows, '
      f'{rv["model"].nunique()} models, '
      f'{rv["asset"].nunique()} assets')

# ── Format helpers (round-half-up) ────────────────────────────────
def fmt_pi(x):
    s3 = Decimal(str(x)).quantize(Decimal('0.001'),
                                   rounding=ROUND_HALF_UP)
    return '.' + format(s3, '.3f')[2:]

# ── Compute per-model aggregates ─────────────────────────────────
rows = []
for model in MODEL_ORDER:
    sub = rv[rv['model'] == model]
    assert len(sub) == 24, f'{model}: expected 24 assets, got {len(sub)}'

    s_pi  = sub['static_pihat'].mean()
    s_grn = int((sub['static_tl'] == 'Green').sum())
    s_cc  = int(sub['static_cc_pass'].sum())

    r_pi  = sub['rolling_pihat'].mean()
    r_grn = int((sub['rolling_tl'] == 'Green').sum())
    r_cc  = int(sub['rolling_cc_pass'].sum())

    rows.append({
        'model': model,
        's_pi': s_pi, 's_grn': s_grn, 's_cc': s_cc,
        'r_pi': r_pi, 'r_grn': r_grn, 'r_cc': r_cc,
    })

    print(f'  {model:16s}  S: pi={fmt_pi(s_pi)} G={s_grn}/24 CC={s_cc}/24  '
          f'R: pi={fmt_pi(r_pi)} G={r_grn}/24 CC={r_cc}/24')

# ── Build LaTeX ──────────────────────────────────────────────────
lines = [
    r'\begin{tabular}{@{}lcccccc@{}}',
    r'\toprule',
    r'& \multicolumn{3}{c}{\textit{Static}}',
    r'& \multicolumn{3}{c}{\textit{Rolling}} \\',
    r'\cmidrule(lr){2-4}\cmidrule(l){5-7}',
    r'Model & $\hat\pi$ & Green & CC pass',
    r'& $\hat\pi$ & Green & CC pass \\',
    r'\midrule',
]

for i, r in enumerate(rows):
    label = MODEL_LABELS.get(r['model'], r['model'])
    line = (f'{label} & {fmt_pi(r["s_pi"])} & {r["s_grn"]}/24 & '
            f'{r["s_cc"]:2d}/24\n'
            f'& {fmt_pi(r["r_pi"])} & {r["r_grn"]}/24 & '
            f'{r["r_cc"]:2d}/24 \\\\')
    lines.append(line)
    if i == 4:
        lines.append(r'\midrule')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')

tex = '\n'.join(lines) + '\n'
tex_path = OUT_DIR / 'tab_rolling_vs_static.tex'
tex_path.write_text(tex)
print(f'\nSaved {tex_path.name}')
print(tex)

# ── Save CSV ─────────────────────────────────────────────────────
result = pd.DataFrame(rows).set_index('model')
result.to_csv(OUT_DIR / 'tab_rolling_vs_static.csv')
