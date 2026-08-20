"""
CO_model_overview — Model overview table (Table 2).
Produces tab_models.tex from models.csv.
Two-panel structure: Panel A (TSFMs), Panel B (Classical Benchmarks).

models.csv was rewritten on 2026-08-20. What it had said about Moirai 2.0 --
"Masked encoder", mixture distribution, 1,000 samples -- is the description of
Moirai 1.1, and the primary source for 2.0 (arXiv:2511.11698) describes a
decoder-only model with a single patch and a quantile loss. It also listed the
GJR-GARCH innovation as skewed-t, which is neither what the manuscript describes
nor what the corrected series computes.

Every architectural cell now carries the arXiv identifier it comes from, in the
`source` column, and parameter counts that could not be verified against a
primary source are left as `--` rather than guessed. That rule is the standing
requirement from analysis/provenance/VERIFICATION_INVENTORY.md.
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
# Fixed-width paragraph columns: the architecture and forecast-output cells now
# carry full descriptions and an @{}llrlll@{} run ran 278pt past the margin.
lines_a = [
    r'\footnotesize',
    r'\begin{tabular}{@{}p{2.45cm}p{2.9cm}r p{2.5cm}p{3.2cm}l@{}}',
    r'\toprule',
    r'Model & Architecture & Param. '
    r'& Distribution & Forecast output & Context \\',
    r'\midrule',
]

for _, row in panel_a.iterrows():
    params = row['parameters']
    dist = row['distribution']
    fo = row['forecast_output']
    ctx = row['context']
    if dist == 'Student-t':
        dist = r'Student-$t$'
    params = '--' if pd.isna(params) or str(params) == '--' else params
    fo_tex = fo.replace(',', '{,}')
    line = (f'{row["model"]} & {row["architecture"]} & {params}\n'
            f'& {dist} & {fo_tex}\n'
            f'& {ctx} \\\\')
    lines_a.append(line)

lines_a.append(r'\bottomrule')
lines_a.append(r'\end{tabular}')
lines_a.append(r'\normalsize')

# ── Panel B ─────────────────────────────────────────────────────
lines_b = [
    r'\footnotesize',
    r'\begin{tabular}{@{}p{3.0cm}p{2.2cm}p{2.2cm}p{4.0cm}l@{}}',
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

# Parameter lists are written in models.csv as dot-separated plain names
# ('omega.alpha_1.beta_1.gamma') and rendered as maths here. This used to be a
# lookup table of the exact strings then in the file, so adding a row -- as the
# GJR-GARCH-t row did -- emitted raw underscores and broke the build. Escaping
# the general form removes that failure mode.
GREEK = ('omega', 'alpha', 'beta', 'gamma', 'nu', 'xi', 'lambda', 'sigma')


def esc_params(raw: str) -> str:
    raw = str(raw).strip()
    if raw in ('', 'nan'):
        return ''
    if '=' in raw:                       # e.g. 'lambda = 0.94'
        name, _, val = raw.partition('=')
        name = name.strip()
        head = rf'\{name}' if name in GREEK else name
        return rf'${head} = {val.strip()}$'
    parts = []
    for tok in raw.split('.'):
        base, _, sub = tok.partition('_')
        head = rf'\{base}' if base in GREEK else base
        parts.append(rf'{head}_{sub}' if sub else head)
    return '$' + ','.join(parts) + '$'

DIST_ESC = {
    'Skewed-t': r'Skewed-$t$',
}

for _, row in panel_b.iterrows():
    model = MODEL_B_ESC.get(row['model'], row['model'])
    typ = row['type']
    innov = DIST_ESC.get(row['innovation_dist'], row['innovation_dist'])
    est_p = esc_params(row['est_parameters'])
    est_w = str(row['est_window'])
    if est_w == 'nan':
        est_w = ''
    line = (f'{model} & {typ} & {innov}\n'
            f'& {est_p}\n'
            f'& {est_w} \\\\')
    lines_b.append(line)

lines_b.append(r'\bottomrule')
lines_b.append(r'\end{tabular}')
lines_b.append(r'\normalsize')

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
