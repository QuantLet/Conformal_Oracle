"""
CO_fz_scores — Fissler-Ziegel FZ_0 joint VaR-ES scores (Table 13).
Produces tab_fz_scores.tex and tab_fz_scores.csv.

FZ_0 specification (Fissler & Ziegel, 2016):
  S(v, e, r) = (1/(alpha*e)) * 1(r<v) * (r-v) + v/e + log(-e) - 1

Raw ES via Gaussian tail-mean: ES_t = mu_t - sigma_t * phi(z_alpha)/alpha.
Conformal ES correction: median shift (Appendix C).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from math import ceil
from decimal import Decimal, ROUND_HALF_UP

# ── Paths ──────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent.parent.parent
DATA     = BASE / 'cfp_ijf_data'
OUT_DIR  = Path(__file__).resolve().parent

ALPHA = 0.01
F_CAL = 0.70

SYMBOLS = ['SP500', 'STOXX', 'GDAXI', 'FCHI', 'FTSE100', 'ICLN',
           'NIKKEI', 'HSI', 'BOVESPA', 'NIFTY', 'ASX200', 'CBU0',
           'TLT', 'IBGL', 'DJCI', 'GOLD', 'WTI', 'NATGAS',
           'BTC', 'ETH', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']

MODELS = {
    'Chronos-Small': ('chronos_small', None),
    'Chronos-Mini':  ('chronos_mini',  None),
    'TimesFM-2.5':   ('timesfm25',     None),
    'Moirai-2.0':    ('moirai2',       None),
    'Lag-Llama':     ('lagllama',      None),
    'GJR-GARCH':     ('benchmarks',    'gjr_garch'),
    'GARCH-N':       ('benchmarks',    'garch_n'),
    'Hist-Sim':      ('benchmarks',    'hs'),
    'EWMA':          ('benchmarks',    'ewma'),
}

MODEL_ORDER = list(MODELS.keys())

MODEL_LABELS = {
    'TimesFM-2.5': 'TimesFM 2.5',
    'Moirai-2.0':  'Moirai 2.0',
    'Hist-Sim':    r'Hist.\ Sim.',
}

z_alpha = norm.ppf(ALPHA)
ES_MULT = norm.pdf(z_alpha) / ALPHA

# ── Data loading ─────────────────────────────────────────────────
def load_data(model_key, symbol):
    subdir, suffix = MODELS[model_key]
    ret = pd.read_csv(DATA / 'returns' / f'{symbol}.csv',
                      index_col=0, parse_dates=True)
    ret.columns = ['r']
    if suffix is None:
        fc = pd.read_parquet(DATA / subdir / f'{symbol}.parquet')
    else:
        fc = pd.read_parquet(DATA / subdir / f'{symbol}_{suffix}.parquet')
    common = ret.index.intersection(fc.index)
    ret, fc = ret.loc[common], fc.loc[common]
    var_col = f'VaR_{ALPHA}'
    mask = fc[var_col].notna()
    ret, fc = ret.loc[mask], fc.loc[mask]

    r      = ret['r'].values
    var_t  = fc[var_col].values
    mean_t = fc['mean'].values
    std_t  = fc['std'].values
    es_t   = mean_t - std_t * ES_MULT

    return r, var_t, es_t

# ── Conformal correction ────────────────────────────────────────
def conformal_correction(r, var_t, es_t):
    T     = len(r)
    n_cal = int(T * F_CAL)
    r_cal, r_test = r[:n_cal], r[n_cal:]
    v_cal, v_test = var_t[:n_cal], var_t[n_cal:]
    e_cal, e_test = es_t[:n_cal], es_t[n_cal:]

    s_V       = v_cal - r_cal
    q_hat_V   = np.quantile(s_V,
                    ceil((n_cal + 1) * (1 - ALPHA)) / n_cal)
    v_corr    = v_test - q_hat_V

    viol_cal  = r_cal < v_cal
    if np.sum(viol_cal) > 0:
        s_E    = r_cal[viol_cal] - e_cal[viol_cal]
        q_hat_E = np.median(s_E)
        e_corr = e_test + q_hat_E
    else:
        e_corr = e_test.copy()

    e_corr = np.minimum(e_corr, v_corr)
    return r_test, v_test, e_test, v_corr, e_corr

# ── FZ_0 score ──────────────────────────────────────────────────
def fz0_score(r, v, e):
    e_safe    = np.where(e < -1e-10, e, -1e-10)
    indicator = (r < v).astype(float)
    term1     = (1.0 / (ALPHA * e_safe)) * indicator * (r - v)
    term2     = v / e_safe
    term3     = np.log(-e_safe)
    return np.mean(term1 + term2 + term3 - 1.0)

# ── Compute ─────────────────────────────────────────────────────
rows = []
for model in MODEL_ORDER:
    raw_list, corr_list = [], []
    for sym in SYMBOLS:
        try:
            r, var_t, es_t = load_data(model, sym)
        except Exception:
            continue
        T = len(r)
        n_cal = int(T * F_CAL)
        if n_cal < 100 or T - n_cal < 50:
            continue
        r_test, v_raw, e_raw, v_corr, e_corr = \
            conformal_correction(r, var_t, es_t)
        raw_list.append(fz0_score(r_test, v_raw, e_raw))
        corr_list.append(fz0_score(r_test, v_corr, e_corr))

    n = len(raw_list)
    assert n == 24, f'{model}: expected 24 assets, got {n}'
    mean_raw  = np.mean(raw_list)
    mean_corr = np.mean(corr_list)
    if abs(mean_raw) > 0:
        improv = (mean_raw - mean_corr) / abs(mean_raw) * 100
    else:
        improv = 0.0

    rows.append({'model': model, 'raw_fz': mean_raw,
                 'corr_fz': mean_corr, 'improvement_pct': improv})

    replacement = abs(mean_raw) >= 1000 or mean_raw > 0
    r_tag = '(undef)' if replacement else f'{mean_raw:.4f}'
    print(f'  {model:16s}  raw={r_tag:>12s}  '
          f'corr={mean_corr:.4f}  imp={improv:.1f}%')

# ── Format helpers (round-half-up) ──────────────────────────────
def rhu(x, places):
    fmt = '0.' + '0' * places
    return Decimal(str(x)).quantize(Decimal(fmt), rounding=ROUND_HALF_UP)

# ── Build LaTeX (tabular only) ──────────────────────────────────
lines = [
    r'\begin{tabular}{@{}lrrr@{}}',
    r'\toprule',
    r'Model & Raw FZ$_0$ & Corrected FZ$_0$'
    r' & Improvement\,\% \\',
    r'\midrule',
]
for i, row in enumerate(rows):
    label = MODEL_LABELS.get(row['model'], row['model'])
    replacement = abs(row['raw_fz']) >= 1000 or row['raw_fz'] > 0
    if replacement:
        line = (f'{label} & --- & $-${rhu(-row["corr_fz"], 2)}'
                f' & --- \\\\')
    else:
        line = (f'{label} & $-${rhu(-row["raw_fz"], 2)}'
                f' & $-${rhu(-row["corr_fz"], 2)}'
                f' & {rhu(row["improvement_pct"], 1)} \\\\')
    lines.append(line)
    if i == 4:
        lines.append(r'\midrule')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')

tex = '\n'.join(lines) + '\n'
tex_path = OUT_DIR / 'tab_fz_scores.tex'
tex_path.write_text(tex)
print(f'\nSaved {tex_path.name}')
print(tex)

# ── Save CSV ────────────────────────────────────────────────────
pd.DataFrame(rows).set_index('model').to_csv(
    OUT_DIR / 'tab_fz_scores.csv')
print(f'Saved tab_fz_scores.csv')
