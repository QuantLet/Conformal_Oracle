"""
CO_asset_overview — Asset universe table (Table 1).
Produces tab_assets.tex from assets.csv.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'cfp_ijf_data'
RES_DIR  = DATA_DIR / 'paper_outputs' / 'tables'
OUT_DIR  = Path(__file__).resolve().parent

df = pd.read_csv(RES_DIR / 'assets.csv')
print(f'Loaded assets.csv: {len(df)} assets, '
      f'{df["group"].nunique()} groups')

NAME_ESC = {
    'S&P 500':     r"S\&P 500",
    'S&P/ASX 200': r"S\&P/ASX 200",
}

lines = [
    r'\begin{tabular}{@{}lllrll@{}}',
    r'\toprule',
    r'Symbol & Name & Class & $T$ & Start & End \\',
    r'\midrule',
]

groups = df['group'].unique()
for gi, group in enumerate(groups):
    if gi > 0:
        lines.append(r'\midrule')
    lines.append(
        r'\multicolumn{6}{@{}l}{\textit{' + group + r'}} \\')
    sub = df[df['group'] == group]
    for _, row in sub.iterrows():
        name = NAME_ESC.get(row['name'], row['name'])
        line = (f'{row["symbol"]:7s} & {name:30s} & {row["class"]:9s}'
                f' & {row["T"]} & {row["start"]} & {row["end"]} \\\\')
        lines.append(line)

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')

tex = '\n'.join(lines) + '\n'
tex_path = OUT_DIR / 'tab_assets.tex'
tex_path.write_text(tex)
print(f'Saved {tex_path.name}')
print(tex)
