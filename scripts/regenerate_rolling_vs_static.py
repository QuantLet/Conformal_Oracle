"""
Regenerate rolling_vs_static.csv from fresh forecast data.
Computes both static (70/30) and rolling (250-day) conformal correction
for all 9 models × 24 assets at alpha=0.01.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from math import ceil
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'cfp_ijf_data'

ALPHA = 0.01
F_CAL = 0.70
W_ROLL = 250

# Model and asset sets come from cfp_config. This file used to carry its own
# nine-model dictionary, which is why the static-versus-rolling table stayed a
# nine-model table after Moirai-1.1, GJR-GARCH-t and the two analytic Chronos
# series entered the panel.
import sys
sys.path.insert(0, str(BASE / 'Quantlets'))
from cfp_config import MODELS, SYMBOLS as _SYMBOLS  # noqa: E402

SYMBOLS = sorted(_SYMBOLS)


def load_pair(model_key, symbol):
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
    return ret['r'].values[mask], fc[var_col].values[mask]


def qhat_ceil(scores, alpha):
    n = len(scores)
    k = int(ceil((n + 1) * (1 - alpha))) - 1
    k = min(k, n - 1)
    return np.sort(scores)[k]


def traffic_light(x, n):
    if n == 0:
        return 'Green'
    annual = x * (250.0 / n)
    if annual <= 4:
        return 'Green'
    if annual <= 9:
        return 'Yellow'
    return 'Red'


def cc_pval(violations_bool):
    v = violations_bool.astype(int)
    n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
    n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
    n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
    n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))
    if (n00 + n01) == 0 or (n10 + n11) == 0 or (n01 + n11) == 0:
        return np.nan
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    if pi01 in (0, 1) or pi11 in (0, 1) or pi in (0, 1):
        return np.nan
    lr_ind = -2.0 * (
        (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
        - n00 * np.log(1 - pi01) - n01 * np.log(pi01)
        - n10 * np.log(1 - pi11) - n11 * np.log(pi11)
    )
    return 1.0 - stats.chi2.cdf(abs(lr_ind), 1)


def static_conformal(r, q, alpha=ALPHA, f_cal=F_CAL):
    n = len(r)
    n_cal = int(n * f_cal)
    r_cal, r_test = r[:n_cal], r[n_cal:]
    q_cal, q_test = q[:n_cal], q[n_cal:]

    scores = q_cal - r_cal
    sorted_s = np.sort(scores)
    k = int(ceil((n_cal + 1) * (1 - alpha))) - 1
    k = min(k, n_cal - 1)
    qV = float(sorted_s[k])

    var_cp = q_test - qV
    viol = (r_test < var_cp).astype(int)
    x = int(viol.sum())
    n_test = len(r_test)
    pihat = x / n_test
    tl = traffic_light(x, n_test)
    p_cc = cc_pval(r_test < var_cp)
    cc_pass = int(p_cc > 0.05) if not np.isnan(p_cc) else 0
    return n_test, pihat, tl, p_cc, cc_pass


def kupiec_p(x, n, alpha=ALPHA):
    if n == 0:
        return 1.0
    pi_hat = x / n
    if pi_hat == 0:
        lr = 2 * n * np.log(1 / (1 - alpha))
    elif pi_hat == 1:
        lr = 2 * n * np.log(alpha)
    else:
        lr = 2 * (x * np.log(pi_hat / alpha) +
                  (n - x) * np.log((1 - pi_hat) / (1 - alpha)))
    return 1 - stats.chi2.cdf(lr, 1)


def quantile_score(r, v, alpha=ALPHA):
    diff = r - v
    return np.mean(np.where(diff < 0, (alpha - 1) * diff, alpha * diff))


def rolling_conformal(r, q, w=W_ROLL, alpha=ALPHA, f_cal=F_CAL):
    n = len(r)
    n_cal = int(n * f_cal)
    if n_cal < w or n - n_cal < 50:
        return None
    r_cal, r_test = r[:n_cal], r[n_cal:]
    q_cal, q_test = q[:n_cal], q[n_cal:]

    scores_cal = q_cal - r_cal
    history = list(scores_cal[-w:])

    n_test = len(r_test)
    corrected = np.full(n_test, np.nan)
    for t in range(n_test):
        window = np.array(history[-w:])
        q_hat = qhat_ceil(window, alpha)
        corrected[t] = q_test[t] - q_hat
        history.append(q_test[t] - r_test[t])

    viol = (r_test < corrected).astype(int)
    x = int(viol.sum())
    pihat = x / n_test
    tl = traffic_light(x, n_test)
    p_cc = cc_pval(r_test < corrected)
    cc_pass = int(p_cc > 0.05) if not np.isnan(p_cc) else 0
    kup = kupiec_p(x, n_test, alpha)
    qs = quantile_score(r_test, corrected, alpha)
    width = float(np.mean(np.abs(corrected)))
    return n_test, pihat, tl, p_cc, cc_pass, kup, qs, width


rows = []
rolling_rows = []
for model in MODELS:
    for sym in SYMBOLS:
        try:
            r, q = load_pair(model, sym)
        except Exception as e:
            print(f'  SKIP {model}/{sym}: {e}')
            continue

        n_test_s, s_pi, s_tl, s_cc_p, s_cc_pass = \
            static_conformal(r, q)

        roll = rolling_conformal(r, q)
        if roll is None:
            print(f'  SKIP rolling {model}/{sym}: too short')
            continue
        n_test_r, r_pi, r_tl, r_cc_p, r_cc_pass, r_kup, r_qs, r_w = roll

        rows.append({
            'model': model, 'asset': sym,
            'n_test': n_test_s,
            'static_pihat': s_pi, 'static_tl': s_tl,
            'static_cc_p': s_cc_p, 'static_cc_pass': s_cc_pass,
            'rolling_pihat': r_pi, 'rolling_tl': r_tl,
            'rolling_cc_p': r_cc_p, 'rolling_cc_pass': r_cc_pass,
        })
        rolling_rows.append({
            'model': model, 'symbol': sym,
            'n_test': n_test_r, 'viol': int(r_pi * n_test_r),
            'pi_hat': r_pi, 'kupiec_p': r_kup,
            'TL': r_tl, 'QS': r_qs, 'width': r_w, 'w': W_ROLL,
        })

        print(f'  {model:16s} {sym:8s}  S: pi={s_pi:.4f} TL={s_tl:6s}  '
              f'R: pi={r_pi:.4f} TL={r_tl:6s}')

df = pd.DataFrame(rows)
expected = len(MODELS) * len(SYMBOLS)
assert len(df) == expected, f'Expected {expected} rows, got {len(df)}'

out_path = DATA / 'paper_outputs' / 'tables' / 'rolling_vs_static.csv'
df.to_csv(out_path, index=False)
print(f'\nSaved {out_path} ({len(df)} rows)')

# Verify static Green counts match all_results.csv
ar = pd.read_csv(DATA / 'paper_outputs' / 'tables' / 'all_results.csv')
ar1 = ar[ar['alpha'] == 0.01][['model', 'symbol', 'TL_cp']].rename(
    columns={'symbol': 'asset'})
merged = df.merge(ar1, on=['model', 'asset'], how='left')
mismatches = merged[merged['static_tl'] != merged['TL_cp']]
if len(mismatches) == 0:
    print('Static Green counts MATCH all_results.csv exactly.')
else:
    print(f'WARNING: {len(mismatches)} static TL mismatches vs all_results.csv:')
    print(mismatches[['model', 'asset', 'static_tl', 'TL_cp']].to_string(index=False))

# Summary
for m in MODELS:
    sub = df[df['model'] == m]
    sg = (sub['static_tl'] == 'Green').sum()
    rg = (sub['rolling_tl'] == 'Green').sum()
    print(f'  {m:16s}  Static={sg}/24  Rolling={rg}/24')

total_sg = (df['static_tl'] == 'Green').sum()
total_rg = (df['rolling_tl'] == 'Green').sum()
print(f'  TOTAL            Static={total_sg}/{len(SYMBOLS) * len(MODELS)}  Rolling={total_rg}/{len(SYMBOLS) * len(MODELS)}')

# Save rolling_w250_pooled.csv (for compile_tab_baselines.py)
rdf = pd.DataFrame(rolling_rows)
legacy_dir = BASE / 'legacy' / 'results'
legacy_dir.mkdir(parents=True, exist_ok=True)
rdf.to_csv(legacy_dir / 'rolling_w250_pooled.csv', index=False)
print(f'Saved rolling_w250_pooled.csv ({len(rdf)} rows)')
