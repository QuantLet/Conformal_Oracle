"""
CO_bound_validation — Empirical bound validation (Table 11).

Evaluates the coverage bound from Theorem 3.5 on six representative
(model, asset) pairs spanning a range of dependence structures.

Methodology:
  - Score series: s_t = VaR_t - r_t on the calibration window
    (first f_c = 0.70 of post-warmup observations).
  - rho_hat: first-order autocorrelation of |s_t| (score magnitude
    persistence, a proxy for mixing speed).
  - Delta_n = C * sqrt(log(n_cal) / n_cal) with C = 5/2 (conservative
    plug-in for the unspecified constant in Theorem 3.5).
  - Guaranteed coverage = 1 - alpha - Delta_n.
  - Empirical coverage = 1 - pihat_cp from all_results.csv.

Output: tab_bound_validation.tex, tab_bound_validation.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent.parent
DATA_DIR = BASE / 'cfp_ijf_data'

ALPHA = 0.01
F_CAL = 0.70
C_BOUND = 2.5   # 5/2

MODEL_DIRS = {
    'Chronos-Small': ('chronos_small', '{asset}.parquet'),
    'TimesFM-2.5':   ('timesfm25',     '{asset}.parquet'),
    'Moirai-2.0':    ('moirai2',       '{asset}.parquet'),
    'Lag-Llama':     ('lagllama',      '{asset}.parquet'),
    'GJR-GARCH':     ('benchmarks',    '{asset}_gjr_garch.parquet'),
}

PAIRS = [
    ('Chronos-Small', 'SP500',   'S\\&P~500'),
    ('Chronos-Small', 'BTC',     'BTC'),
    ('TimesFM-2.5',   'SP500',   'S\\&P~500'),
    ('Moirai-2.0',    'GDAXI',   'DAX'),
    ('Lag-Llama',     'FTSE100', 'FTSE'),
    ('GJR-GARCH',     'SP500',   'S\\&P~500'),
]

ar = pd.read_csv(DATA_DIR / 'paper_outputs' / 'tables' / 'all_results.csv')

rows = []
for model, asset, display_asset in PAIRS:
    subdir, pattern = MODEL_DIRS[model]
    fname = pattern.format(asset=asset)

    ret = pd.read_csv(DATA_DIR / 'returns' / f'{asset}.csv',
                      index_col=0, parse_dates=True)
    fcast = pd.read_parquet(DATA_DIR / subdir / fname)

    col = f'VaR_{ALPHA}'
    common = ret.index.intersection(fcast.index).sort_values()
    r = ret.loc[common, 'log_return'].values
    v = fcast.loc[common, col].values

    n = len(r)
    n_cal = int(n * F_CAL)

    scores_cal = v[:n_cal] - r[:n_cal]
    sc_clean = scores_cal[~np.isnan(scores_cal)]

    rho_hat = pd.Series(np.abs(sc_clean)).autocorr(lag=1)

    delta_n = C_BOUND * np.sqrt(np.log(n_cal) / n_cal)
    guaranteed = (1 - ALPHA - delta_n) * 100

    mask = ((ar['model'] == model) & (ar['symbol'] == asset)
            & (ar['alpha'] == ALPHA))
    pihat_cp = ar.loc[mask, 'pihat_cp'].values[0]
    empirical = (1 - pihat_cp) * 100

    row = {
        'model': model, 'asset': asset, 'display_asset': display_asset,
        'n_cal': n_cal, 'rho_hat': rho_hat, 'delta_n': delta_n,
        'guaranteed': guaranteed, 'empirical': empirical,
    }
    rows.append(row)
    print(f'{model:16s} {asset:8s}  ncal={n_cal}  rho={rho_hat:.2f}'
          f'  Delta={delta_n:.3f}  Guar={guaranteed:.1f}%  Emp={empirical:.1f}%')

result = pd.DataFrame(rows)
result.to_csv(SCRIPT_DIR / 'tab_bound_validation.csv', index=False)

# ── Sanity check: all guaranteed < empirical ──
assert all(result['guaranteed'] < result['empirical']), \
    'Bound violated: guaranteed exceeds empirical for some pair'

# ── Generate LaTeX ───────────────────────────────────────────────

def rhu(x, dp):
    d = Decimal(str(x)).quantize(Decimal(10) ** -dp, rounding=ROUND_HALF_UP)
    return format(d, f'.{dp}f')

lines = [
    r'\begin{tabular}{@{}llccccc@{}}',
    r'\toprule',
    (r'Model & Asset & $n_{\mathrm{cal}}$'
     r' & $\hat\rho$ & $\hat\Delta_n$'
     r' & Guaranteed & Empirical \\'),
    r'\midrule',
]

for row in rows:
    model_tex = row['model']
    if model_tex == 'TimesFM-2.5':
        model_tex = 'TimesFM'
    elif model_tex == 'Moirai-2.0':
        model_tex = 'Moirai'
    elif model_tex == 'Lag-Llama':
        model_tex = 'Lag-Llama'

    ncal_str = f'{row["n_cal"]:,d}'.replace(',', '{,}')
    rho_str = rhu(row['rho_hat'], 2)
    delta_str = rhu(row['delta_n'], 3)
    guar_str = rhu(row['guaranteed'], 1) + r'\%'
    emp_str = rhu(row['empirical'], 1) + r'\%'

    line = (f'\t\t{model_tex} & {row["display_asset"]} & {ncal_str}'
            f' & {rho_str} & {delta_str} & {guar_str} & {emp_str} \\\\')
    lines.append(line)

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')

tex = '\n'.join(lines) + '\n'
tex_path = SCRIPT_DIR / 'tab_bound_validation.tex'
tex_path.write_text(tex)
print(f'\nSaved {tex_path.name}')
print(tex)
