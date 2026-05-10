"""
CO_full_evaluation — Master results table (Table 4 in the paper).
Produces tab_master_results.tex.

Nine models in two panels based on the |qV|/|VaR_raw| < 1 criterion:
  Panel A (genuine recalibration): 7 models
  Panel B (effective replacement):  2 Chronos models

Columns: Raw/Corrected pi-hat, Raw/Corrected Kupiec pass count,
Raw/Corrected QS (×10^4), corrected VaR width, Width/GJR ratio,
Green count after correction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

# ── Paths ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'cfp_ijf_data'
RES_DIR  = DATA_DIR / 'paper_outputs' / 'tables'
OUT_DIR  = Path(__file__).resolve().parent

ALPHA = 0.01

PANEL_A = ['TimesFM-2.5', 'Moirai-2.0', 'Lag-Llama',
           'GJR-GARCH', 'GARCH-N', 'Hist-Sim', 'EWMA']
PANEL_B = ['Chronos-Small', 'Chronos-Mini']
MODEL_ORDER = PANEL_A + PANEL_B

LABELS = {
    'TimesFM-2.5': 'TimesFM 2.5', 'Moirai-2.0': 'Moirai 2.0',
    'Hist-Sim': r'Hist.\ Sim.',
}

# ── Load ───────────────────────────────────────────────────────────
df = pd.read_csv(RES_DIR / 'all_results.csv')
d01 = df[df['alpha'] == ALPHA].copy()
print(f'Loaded {len(d01)} rows at alpha={ALPHA}')

# ── Compute per-model statistics ──────────────────────────────────
gjr_width = d01[d01['model'] == 'GJR-GARCH']['VaR_width'].mean()

rows = []
for model in MODEL_ORDER:
    sub = d01[d01['model'] == model]
    n = len(sub)
    rows.append({
        'model': model,
        'raw_pi': sub['pihat_raw'].mean(),
        'cor_pi': sub['pihat_cp'].mean(),
        'raw_kup': int((sub['p_kup_raw'] >= 0.05).sum()),
        'cor_kup': int((sub['p_kup_cp'] >= 0.05).sum()),
        'raw_qs': sub['QS_raw'].mean() * 1e4,
        'cor_qs': sub['QS_cp'].mean() * 1e4,
        'width': sub['VaR_width'].mean(),
        'w_gjr': sub['VaR_width'].mean() / gjr_width,
        'green': int((sub['TL_cp'] == 'Green').sum()),
        'n': n,
    })

result = pd.DataFrame(rows)

for _, r in result.iterrows():
    print(f'  {r["model"]:16s}  pi={r["raw_pi"]:.3f}->{r["cor_pi"]:.3f}  '
          f'Kup={r["raw_kup"]}/{r["n"]}->{r["cor_kup"]}/{r["n"]}  '
          f'QS={r["raw_qs"]:.1f}->{r["cor_qs"]:.1f}  '
          f'W={r["width"]:.3f}  W/GJR={r["w_gjr"]:.2f}  '
          f'Green={r["green"]}/{r["n"]}')

# ── Save CSV ──────────────────────────────────────────────────────
result.to_csv(OUT_DIR / 'tab_master_results.csv', index=False)

# ── Format helpers (round-half-up) ────────────────────────────────
def rhu(x, dp):
    d = Decimal(str(x)).quantize(Decimal(10) ** -dp, rounding=ROUND_HALF_UP)
    return format(d, f'.{dp}f')

def fmt_pi(x):
    return '.' + rhu(x, 3)[2:]

def fmt_qs(x):
    return rhu(x, 1)

def fmt_w(x):
    return '.' + rhu(x, 3)[2:]

def fmt_ratio(x):
    return rhu(x, 2)

# ── Build LaTeX ───────────────────────────────────────────────────
lines = [
    r'\begin{tabular}{@{}l rr rr rr rr r@{}}',
    r'\toprule',
    r'& \multicolumn{2}{c}{$\hat\pi$}',
    r'& \multicolumn{2}{c}{Kupiec pass}',
    r'& \multicolumn{2}{c}{QS} \\',
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
    r'\cmidrule(lr){6-7}',
    r'Model & Raw & Corr. & Raw & Corr.',
    r'& Raw & Corr. & VaR width & Width/GJR & Green \\',
    r'\midrule',
    r'\multicolumn{10}{@{}l}{\textit{Panel~A: Signal-preserving recalibration}',
    r'	($|\qV|/|\VaR_{\mathrm{raw}}| < 1$)} \\[2pt]',
]

for i, model in enumerate(MODEL_ORDER):
    r = result[result['model'] == model].iloc[0]
    label = LABELS.get(model, model)
    n = int(r['n'])

    line = (f'{label} & {fmt_pi(r["raw_pi"])} & {fmt_pi(r["cor_pi"])}'
            f' & {int(r["raw_kup"])}/{n} & {int(r["cor_kup"])}/{n}\n'
            f'& {fmt_qs(r["raw_qs"])} & {fmt_qs(r["cor_qs"])}'
            f' & {fmt_w(r["width"])} & {fmt_ratio(r["w_gjr"])}'
            f' & {int(r["green"])}/{n} \\\\')
    lines.append(line)

    if i == len(PANEL_A) - 1:
        lines.append(r'\midrule')
        lines.append(r'\multicolumn{10}{@{}l}{\textit{Panel~B: Effective replacement}')
        lines.append(r'	($|\qV|/|\VaR_{\mathrm{raw}}| > 1$)} \\[2pt]')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')

tex = '\n'.join(lines) + '\n'
tex_path = OUT_DIR / 'tab_master_results.tex'
tex_path.write_text(tex)
print(f'\nSaved {tex_path.name}')
print(tex)
