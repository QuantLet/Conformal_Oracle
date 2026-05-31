"""
CO_multi_quantile_panel — Reconstruct violation indicators and recompute
panel-pooled backtest statistics at alpha=0.01 (Table 6).

Violation indicators are reconstructed from forecast parquets + returns + qV.
N_panel, total violations, pi_pooled, p_Kupiec, and cluster-robust p-values
are independently verifiable.

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

MODEL_ORDER = ['Chronos-Small', 'Chronos-Mini', 'TimesFM-2.5',
               'Moirai-2.0', 'Lag-Llama',
               'GJR-GARCH', 'GARCH-N', 'Hist-Sim', 'EWMA']
ALPHA = 0.01

MODEL_SUBDIR = {
    'Chronos-Small': 'chronos_small', 'Chronos-Mini': 'chronos_mini',
    'TimesFM-2.5': 'timesfm25', 'Moirai-2.0': 'moirai2',
    'Lag-Llama': 'lagllama',
}
BENCH_SUFFIX = {
    'GJR-GARCH': 'gjr_garch', 'GARCH-N': 'garch_n',
    'Hist-Sim': 'hs', 'EWMA': 'ewma',
}

def parquet_path(model, symbol):
    if model in MODEL_SUBDIR:
        return DATA_DIR / MODEL_SUBDIR[model] / f'{symbol}.parquet'
    return DATA_DIR / 'benchmarks' / f'{symbol}_{BENCH_SUFFIX[model]}.parquet'

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

# ── Compare with pre-computed values ─────────────────────────────
original = pd.read_csv(RES_DIR / 'panel_pooled.csv')
original = original.set_index('model').reindex(MODEL_ORDER)

print(f'\n{"Model":16s}  {"N":>5s}  {"Viol":>5s}  '
      f'{"HAC_SE(orig)":>12s}  {"HAC_SE(DK)":>10s}  {"Diff%":>6s}  '
      f'{"p_cl(orig)":>10s}  {"p_cl(DK)":>10s}  {"Match4dp":>8s}')
print('-' * 95)

for model in MODEL_ORDER:
    r = result[result['model'] == model].iloc[0]
    o = original.loc[model]
    n_ok = r['N_panel'] == int(o['N_panel'])
    v_ok = r['total_viol'] == int(o['total_viol'])
    hac_diff = (r['HAC_SE'] - o['HAC_SE']) / o['HAC_SE'] * 100
    pcl_match = abs(r['p_cluster'] - o['p_cluster']) < 5e-4
    print(f'{model:16s}  '
          f'{"OK" if n_ok else "!":>5s}  {"OK" if v_ok else "!":>5s}  '
          f'{o["HAC_SE"]:12.6f}  {r["HAC_SE"]:10.6f}  {hac_diff:+5.1f}%  '
          f'{o["p_cluster"]:10.4f}  {r["p_cluster"]:10.4f}  '
          f'{"YES" if pcl_match else "NO":>8s}')

# ── Save ─────────────────────────────────────────────────────────
out_path = OUT_DIR / 'panel_pooled_reproduced.csv'
result.to_csv(out_path, index=False)
print(f'\nSaved {out_path.name}')

# ── Save violation sequences ─────────────────────────────────────
viol_dir = DATA_DIR / 'paper_outputs' / 'violation_sequences'
viol_dir.mkdir(parents=True, exist_ok=True)

for model in MODEL_ORDER:
    viols = all_violations[model]
    df_v = pd.DataFrame(viols)
    df_v.index.name = 'date'
    suffix = MODEL_SUBDIR.get(model, BENCH_SUFFIX.get(model, model.lower()))
    df_v.to_parquet(viol_dir / f'{suffix}_violations.parquet')

print(f'Saved {len(MODEL_ORDER)} violation parquets to '
      f'{viol_dir.relative_to(DATA_DIR.parent)}/')
