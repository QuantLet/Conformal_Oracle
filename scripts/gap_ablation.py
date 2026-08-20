"""
Empirical ablation for the gap parameter g_n in Theorem 3.5.

For 4 representative (model, asset) pairs, compare conformal
coverage with g_n = 0 (adjacent splits) vs g_n = c log n with
c = 1 / |log rho_hat|.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2

BASE = Path(__file__).resolve().parent.parent / 'cfp_ijf_data'
RET_DIR = BASE / 'returns'
OUT_DIR = Path(__file__).resolve().parent.parent / 'Quantlets' / 'CO_robustness'

# Chronos enters as the analytic series: the shipped one is sampled at the
# checkpoint default top_k = 50, and an ablation of the gap parameter on a
# truncated predictive distribution measures the truncation, not the gap.
PAIRS = [
    ('Chronos-Small-A', 'chronos_small_analytic', 'SP500'),
    ('Lag-Llama', 'lagllama', 'BTC'),
    ('GJR-GARCH', 'benchmarks', 'WTI'),
    ('Moirai-2.0', 'moirai2', 'NATGAS'),
]

BENCH_SUFFIX = {'GJR-GARCH': 'gjr_garch'}

ALPHA = 0.01
FC = 0.70

PERIODS = {
    'Full': (None, None),
    'GFC': ('2008-09-01', '2009-03-31'),
    'COVID': ('2020-02-01', '2020-06-30'),
    'SVB': ('2023-03-01', '2023-06-30'),
}


def load_scores(model_name, subdir, symbol):
    if subdir == 'benchmarks':
        suffix = BENCH_SUFFIX.get(model_name, model_name.lower().replace('-', '_'))
        pq = pd.read_parquet(BASE / subdir / f'{symbol}_{suffix}.parquet')
    else:
        pq = pd.read_parquet(BASE / subdir / f'{symbol}.parquet')
    pq.index = pd.to_datetime(pq.index)

    ret = pd.read_csv(RET_DIR / f'{symbol}.csv', parse_dates=['date'])
    ret = ret.set_index('date')

    merged = pq[['VaR_0.01']].join(ret['log_return'], how='inner').dropna()
    scores = merged['VaR_0.01'] - merged['log_return']
    return scores, merged['log_return'], merged.index


def conformal_with_gap(scores_arr, gap, alpha=ALPHA, fc=FC):
    n = len(scores_arr)
    n_cal = int(fc * n)
    cal_scores = scores_arr[:n_cal]
    qV = np.quantile(cal_scores, 1 - alpha)

    test_start = n_cal + gap
    if test_start >= n:
        return None
    test_scores = scores_arr[test_start:]
    violations = (test_scores > qV).sum()
    n_test = len(test_scores)
    pi_hat = violations / n_test

    if pi_hat == 0 or pi_hat == 1:
        p_kup = 0.0
    else:
        lr = -2 * (violations * np.log(alpha / pi_hat) +
                    (n_test - violations) * np.log((1 - alpha) / (1 - pi_hat)))
        p_kup = 1 - chi2.cdf(abs(lr), 1)

    annual_rate = violations / (n_test / 250)
    tl = 'Green' if annual_rate < 5 else ('Yellow' if annual_rate < 10 else 'Red')

    return {
        'n_test': n_test, 'qV': qV, 'pi_hat': pi_hat,
        'violations': int(violations), 'p_kupiec': p_kup, 'TL': tl,
    }


def estimate_rho(scores_arr):
    s = pd.Series(scores_arr)
    return s.autocorr(lag=1)


results = []
for model_name, subdir, symbol in PAIRS:
    print(f'\n--- {model_name} / {symbol} ---')
    scores, returns, dates = load_scores(model_name, subdir, symbol)
    scores_arr = scores.values
    returns_arr = returns.values
    dates_arr = dates

    for period_name, (start, end) in PERIODS.items():
        if start is not None:
            mask = (dates_arr >= pd.Timestamp(start)) & (dates_arr <= pd.Timestamp(end))
            if mask.sum() < 100:
                print(f'  {period_name}: skipped (only {mask.sum()} obs)')
                continue
            s_p = scores_arr[mask]
        else:
            s_p = scores_arr

        n = len(s_p)
        rho = estimate_rho(s_p)

        if rho is None or np.isnan(rho) or rho <= 0 or rho >= 0.999:
            g_log = max(5, int(np.log(n)))
        else:
            c = 1.0 / abs(np.log(rho))
            g_log = max(5, int(c * np.log(n)))

        for gap_val, gap_label in [(0, 'g=0'), (g_log, f'g=c*log(n)')]:
            res = conformal_with_gap(s_p, gap_val)
            if res is None:
                continue
            res.update({
                'model': model_name, 'asset': symbol,
                'period': period_name, 'gap_label': gap_label,
                'gap_value': gap_val, 'rho_hat': rho, 'n_total': n,
            })
            results.append(res)
            print(f'  {period_name:6s} {gap_label:15s}  '
                  f'pi={res["pi_hat"]:.4f}  n_test={res["n_test"]}  '
                  f'TL={res["TL"]}  rho={rho:.3f}')

df_out = pd.DataFrame(results)
df_out.to_csv(OUT_DIR / 'gap_ablation.csv', index=False)
print(f'\nSaved {OUT_DIR / "gap_ablation.csv"}')

# Summary
print('\n=== Summary: |Δπ̂| across all pairs and periods ===')
for period in PERIODS:
    sub = df_out[df_out['period'] == period]
    if len(sub) == 0:
        continue
    g0 = sub[sub['gap_label'] == 'g=0']
    gl = sub[sub['gap_label'] == 'g=c*log(n)']
    if len(g0) == 0 or len(gl) == 0:
        continue
    merged = g0.merge(gl, on=['model', 'asset', 'period'], suffixes=('_g0', '_gl'))
    merged['delta_pi'] = abs(merged['pi_hat_g0'] - merged['pi_hat_gl'])
    max_delta = merged['delta_pi'].max()
    print(f'  {period:6s}: max |Δπ̂| = {max_delta:.4f} ({max_delta*100:.2f} pp)')
