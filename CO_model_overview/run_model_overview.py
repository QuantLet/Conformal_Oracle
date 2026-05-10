"""
CO_model_overview — Model overview table (Table 2).
Produces tab_models.tex from models.csv.
Two-panel structure: Panel A (TSFMs), Panel B (Classical Benchmarks).
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'cfp_ijf_data'
RES_DIR  = DATA_DIR / 'paper_outputs' / 'tables'
OUT_DIR  = Path(__file__).resolve().parent

df = pd.read_csv(RES_DIR / 'models.csv')
panel_a = df[df['panel'] == 'A']
panel_b = df[df['panel'] == 'B']
print(f'Loaded models.csv: {len(panel_a)} TSFMs, {len(panel_b)} benchmarks')

# ── Panel A ─────────────────────────────────────────────────────
lines_a = [
    r'\begin{tabular}{@{}llrlll@{}}',
    r'\toprule',
    r'Model & Architecture & Parameters'
    r' & Distribution & Forecast output & Context \\',
    r'\midrule',
]

for _, row in panel_a.iterrows():
    params = row['parameters']
    dist = row['distribution']
    fo = row['forecast_output']
    ctx = row['context']
    if dist == 'Student-t':
        dist = r'Student-$t$'
    fo_tex = fo.replace(',', '{,}')
    line = (f'{row["model"]} & {row["architecture"]} & {params}\n'
            f'& {dist} & {fo_tex}\n'
            f'& {ctx} \\\\')
    lines_a.append(line)

lines_a.append(r'\bottomrule')
lines_a.append(r'\end{tabular}')

# ── Panel B ─────────────────────────────────────────────────────
lines_b = [
    r'\begin{tabular}{@{}lllll@{}}',
    r'\toprule',
    'Model & Type & Innovation dist.\\\n'
    r'& Est.\ parameters & Est.\ window \\',
    r'\midrule',
]

MODEL_B_ESC = {
    'GJR-GARCH(1.1)': 'GJR-GARCH(1,1)',
    'GARCH(1.1)-N':   'GARCH(1,1)-N',
    'Hist. Sim.':     r'Hist.\ Sim.',
}

PARAM_ESC = {
    'omega.alpha_1.beta_1.gamma.nu.xi':
        r'$\omega,\alpha_1,\beta_1,\gamma,\nu,\xi$',
    'omega.alpha_1.beta_1':
        r'$\omega,\alpha_1,\beta_1$',
    'lambda = 0.94':
        r'$\lambda = 0.94$',
}

DIST_ESC = {
    'Skewed-t': r'Skewed-$t$',
}

for _, row in panel_b.iterrows():
    model = MODEL_B_ESC.get(row['model'], row['model'])
    typ = row['type']
    innov = DIST_ESC.get(row['innovation_dist'], row['innovation_dist'])
    est_p = PARAM_ESC.get(str(row['est_parameters']),
                          str(row['est_parameters']))
    if est_p == 'nan':
        est_p = ''
    est_w = str(row['est_window'])
    if est_w == 'nan':
        est_w = ''
    line = (f'{model} & {typ} & {innov}\n'
            f'& {est_p}\n'
            f'& {est_w} \\\\')
    lines_b.append(line)

lines_b.append(r'\bottomrule')
lines_b.append(r'\end{tabular}')

# ── Combined output ─────────────────────────────────────────────
combined = []
combined.append(r'\par\smallskip')
combined.append(r'\textit{Panel~A: Time Series Foundation Models}')
combined.append(r'\smallskip')
combined.append('')
combined.extend(lines_a)
combined.append('')
combined.append(r'\bigskip')
combined.append(r'\textit{Panel~B: Classical Benchmarks}')
combined.append(r'\smallskip')
combined.append('')
combined.extend(lines_b)

tex = '\n'.join(combined) + '\n'
tex_path = OUT_DIR / 'tab_models.tex'
tex_path.write_text(tex)
print(f'Saved {tex_path.name}')
print(tex)
