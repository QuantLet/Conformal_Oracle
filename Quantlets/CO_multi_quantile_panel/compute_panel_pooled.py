"""
CO_multi_quantile_panel — Independent reconstruction of the violation
indicators, as a CHECK on the committed sequences (alpha = 0.01, Table 6).

This script used to WRITE violation_sequences/. It no longer does.
scripts/build_qs_sequences.py is the single producer of those files, and it
verifies them against all_results.csv cell by cell. What is worth keeping here
is the second, independent path: violations rebuilt from the forecast parquets,
the returns and qV, by code that shares nothing with the producer. If the two
disagree, one of them is wrong, and a single producer that verifies only against
its own summary table could not tell us that.

N_panel, total violations, pi_pooled, p_Kupiec, and cluster-robust p-values
are recomputed here and compared against the committed sequences.

HAC SE uses Driscoll-Kraay: Newey-West (Bartlett kernel, Andrews 1991 AR(1)
plug-in bandwidth) applied to the cross-sectional sum of violations S_t,
scaled by T/N.  This may differ from the original computation whose code
was lost from version control.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2, norm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

# ── Paths ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'cfp_ijf_data'
RES_DIR  = DATA_DIR / 'paper_outputs' / 'tables'
RET_DIR  = DATA_DIR / 'returns'
OUT_DIR  = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfp_config import MODELS as CFG_MODELS  # noqa: E402

MODEL_ORDER = list(CFG_MODELS)
ALPHA = 0.01

VIOL_DIR = DATA_DIR / 'paper_outputs' / 'violation_sequences'


def parquet_path(model, symbol):
    subdir, suffix = CFG_MODELS[model]
    fname = f'{symbol}_{suffix}.parquet' if suffix else f'{symbol}.parquet'
    return DATA_DIR / subdir / fname


def viol_key(model):
    subdir, suffix = CFG_MODELS[model]
    return suffix if suffix else subdir

# ── Load all_results ─────────────────────────────────────────────
ar = pd.read_csv(RES_DIR / 'all_results.csv')
d01 = ar[ar['alpha'] == ALPHA].copy()

# ── Reconstruct violations ───────────────────────────────────────
print('Reconstructing violation indicators from parquets + returns + qV ...')
all_violations = {}

for model in MODEL_ORDER:
    sub = d01[d01['model'] == model]
    assets = sorted(sub['symbol'].unique())
    viols = {}
    for sym in assets:
        row = sub[sub['symbol'] == sym].iloc[0]
        pq = pd.read_parquet(parquet_path(model, sym))
        ret = pd.read_csv(RET_DIR / f'{sym}.csv')
        ret['date'] = pd.to_datetime(ret['date'])
        pq.index = pd.to_datetime(pq.index)
        merged = pq[['VaR_0.01']].join(
            ret.set_index('date')['log_return'], how='inner')
        n_cal = int(row['n_cal'])
        n_test = int(row['n_test'])
        test = merged.iloc[n_cal:n_cal + n_test]
        v = (test['log_return'] < (test['VaR_0.01'] - row['qV'])).astype(int)
        viols[sym] = v
    all_violations[model] = viols
    total_v = sum(v.sum() for v in viols.values())
    total_n = sum(len(v) for v in viols.values())
    print(f'  {model:16s}  {len(assets)} assets  '
          f'{total_v:4d}/{total_n} violations')

# ── Compute panel-pooled statistics ──────────────────────────────
print('\nComputing panel-pooled statistics ...')
rows = []

for model in MODEL_ORDER:
    viols = all_violations[model]
    sub = d01[d01['model'] == model]
    assets = sorted(sub['symbol'].unique())
    J = len(assets)

    n_panel    = sum(len(v) for v in viols.values())
    total_viol = sum(v.sum() for v in viols.values())
    pi_pooled  = total_viol / n_panel

    # Kupiec LR
    x, n = int(total_viol), n_panel
    lr = -2 * (x * np.log(ALPHA / (x / n))
               + (n - x) * np.log((1 - ALPHA) / (1 - x / n)))
    p_kupiec = 1 - chi2.cdf(lr, 1)

    # Cluster-robust z and p
    pihat_assets = np.array([
        sub[sub['symbol'] == sym].iloc[0]['pihat_cp'] for sym in assets])
    cluster_se = np.sqrt(np.var(pihat_assets, ddof=1) / J)
    z_cluster = (pi_pooled - ALPHA) / cluster_se
    p_cluster = 2 * (1 - norm.cdf(abs(z_cluster)))

    # Driscoll-Kraay HAC SE
    all_dates = sorted(set().union(*(v.index for v in viols.values())))
    panel_df = pd.DataFrame(index=all_dates)
    for sym, v in viols.items():
        panel_df[sym] = v
    S_t = panel_df.sum(axis=1).values.astype(float)
    T = len(S_t)
    ols = OLS(S_t, np.ones((T, 1))).fit()
    cov_dk = cov_hac(ols)
    hac_se = np.sqrt(cov_dk[0, 0]) * T / n_panel

    rows.append({
        'model':      model,
        'N_panel':    n_panel,
        'total_viol': int(total_viol),
        'pi_pooled':  pi_pooled,
        'HAC_SE':     hac_se,
        'p_kupiec':   p_kupiec,
        'z_cluster':  z_cluster,
        'p_cluster':  p_cluster,
    })

result = pd.DataFrame(rows)

# ── Compare against the committed sequences ──────────────────────
# The producer verifies its output against all_results.csv, which it also
# derives; this check is against a reconstruction that shares no code with it.
print(f'\n{"Model":18s}  {"N ok":>5s}  {"V(here)":>8s}  {"V(committed)":>12s}  '
      f'{"max cell diff":>13s}')
print('-' * 66)

ok = True
for model in MODEL_ORDER:
    r = result[result['model'] == model].iloc[0]
    seq = pd.read_parquet(VIOL_DIR / f'{viol_key(model)}_violations.parquet')
    v_committed = int(np.nansum(seq.values))
    here = all_violations[model]
    worst = 0
    for sym, v in here.items():
        a = v.astype(float)
        b = seq[sym].dropna().astype(float)
        if len(a) != len(b):
            worst = max(worst, 1)
            continue
        worst = max(worst, float(np.max(np.abs(a.values - b.values))))
    n_ok = int(r['total_viol']) == v_committed and worst == 0
    ok = ok and n_ok
    print(f'{model:18s}  {"OK" if n_ok else "!":>5s}  {int(r["total_viol"]):>8d}  '
          f'{v_committed:>12d}  {worst:>13.0f}')

print('\n' + ('The independent reconstruction agrees with the committed '
              'sequences on every cell.' if ok else
              'DISAGREEMENT: one of the two paths is wrong. Do not use either '
              'until it is resolved.'))

# ── Save ─────────────────────────────────────────────────────────
out_path = OUT_DIR / 'panel_pooled_reproduced.csv'
result.to_csv(out_path, index=False)
print(f'Saved {out_path.name}')

raise SystemExit(0 if ok else 1)
