"""
CO_robustness — Robustness summary table (Table D.15).

Eight rows: weighted conformal prediction (3 lambdas), calibration
fraction sensitivity (4 f_c values), and rolling 250-day conformal
correction.  All rows use Chronos-Small across 24 assets at alpha = 0.01.

Weighted CP implements exponentially decaying observation weights
following Barber et al. (2023, §3, Theorem 1):
  w_i = exp(-lambda * lag_i),   lag_i = n_cal - 1 - i
The test-point weight is w_{n+1} = 1.  The weighted conformal quantile
is the smallest q such that
  sum_{s_i <= q} w_i / (sum_j w_j + 1) >= 1 - alpha.
Effective sample size: ESS = (sum w)^2 / sum(w^2).

Output: tab_robustness_summary.tex, tab_robustness_summary.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent.parent
DATA_DIR = BASE / 'cfp_ijf_data'

ALPHA = 0.01
MODEL = 'Chronos-Small'
SUBDIR = 'chronos_small'

ASSETS = [
    'ASX200', 'AUDUSD', 'BOVESPA', 'BTC', 'FCHI', 'CBU0', 'DJCI', 'ETH',
    'EURUSD', 'FTSE100', 'GBPUSD', 'GDAXI', 'GOLD', 'HSI', 'IBGL', 'ICLN',
    'NATGAS', 'NIFTY', 'NIKKEI', 'SP500', 'STOXX', 'TLT', 'USDJPY', 'WTI',
]


def load_pair(asset):
    ret = pd.read_csv(DATA_DIR / 'returns' / f'{asset}.csv',
                      index_col=0, parse_dates=True)
    fcast = pd.read_parquet(DATA_DIR / SUBDIR / f'{asset}.parquet')
    col = f'VaR_{ALPHA}'
    common = ret.index.intersection(fcast.index).sort_values()
    return ret.loc[common, 'log_return'].values, fcast.loc[common, col].values


def qhat_ceil(scores, alpha):
    """Conformal ceiling quantile (Vovk et al. 2005, Prop. 2.2)."""
    ss = np.sort(scores)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    k = min(k, n - 1)
    return float(ss[k])


def weighted_qhat(scores, alpha, lam):
    """Weighted conformal quantile (Barber et al. 2023, Theorem 1).

    Per-observation weights w_i = exp(-lam * lag_i) where
    lag_i = (n-1-i) counts days between score i and the most recent
    calibration point.  The test point receives weight w_{n+1} = 1.

    Returns (qhat, ess) where ess = (sum w)^2 / sum(w^2).
    """
    n = len(scores)
    lags = np.arange(n - 1, -1, -1, dtype=float)
    weights = np.exp(-lam * lags)

    ess = float(weights.sum() ** 2 / np.sum(weights ** 2))

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_weights = weights[order]
    total = sorted_weights.sum() + 1.0
    cumw = np.cumsum(sorted_weights)
    idx = np.searchsorted(cumw / total, 1 - alpha, side='left')
    idx = min(idx, n - 1)
    return float(sorted_scores[idx]), ess


def basel_tl(n_viol, n_days):
    scaled = n_viol * 250.0 / n_days
    if scaled <= 4:
        return 'Green'
    elif scaled <= 9:
        return 'Yellow'
    return 'Red'


# ── Section 1: Weighted CP ───────────────────────────────────────

LAMBDAS = [0.0005, 0.002, 0.01]

print('=== Weighted CP ===')
wcp_rows = []
for lam in LAMBDAS:
    qvs, pihats, greens, ess_vals = [], [], [], []
    for asset in ASSETS:
        r, v = load_pair(asset)
        n = len(r)
        n_cal = int(n * 0.70)
        r_test, v_test = r[n_cal:], v[n_cal:]
        n_test = len(r_test)

        scores = v[:n_cal] - r[:n_cal]
        qV, ess = weighted_qhat(scores, ALPHA, lam)
        var_cp = v_test - qV
        viol = int(np.sum(r_test < var_cp))
        pihat = viol / n_test

        qvs.append(qV)
        pihats.append(pihat)
        greens.append(1 if basel_tl(viol, n_test) == 'Green' else 0)
        ess_vals.append(ess)

    mean_qV = np.mean(qvs)
    n_green = sum(greens)
    mean_pi = np.mean(pihats)
    mean_ess = np.mean(ess_vals)
    wcp_rows.append({
        'check': 'Weighted CP', 'variant': f'lambda={lam}',
        'mean_qV': mean_qV, 'green': n_green, 'pihat': mean_pi,
        'ess': mean_ess,
    })
    print(f'  lam={lam:.4f}  ESS={mean_ess:.0f}  qV={mean_qV:.4f}'
          f'  Green={n_green}/24  pi={mean_pi:.4f}')

# ── Sanity check: lambda=0.0005 vs unweighted ──
ar = pd.read_csv(DATA_DIR / 'paper_outputs' / 'tables' / 'all_results.csv')
cs01 = ar[(ar['model'] == MODEL) & (ar['alpha'] == ALPHA)]
ref_qV = cs01['qV'].mean()
ref_pi = cs01['pihat_cp'].mean()
ref_green = int((cs01['TL_cp'] == 'Green').sum())
lam_mild = wcp_rows[0]
print(f'\n  Sanity (lam=0.0005 vs unweighted):')
print(f'    lam=0.0005:  qV={lam_mild["mean_qV"]:.4f}  Green={lam_mild["green"]}  pi={lam_mild["pihat"]:.4f}')
print(f'    unweighted:  qV={ref_qV:.4f}  Green={ref_green}  pi={ref_pi:.4f}')
diff_qV = abs(lam_mild['mean_qV'] - ref_qV)
print(f'    qV diff: {diff_qV:.6f}  (expect < 0.002)')

# ── Section 2: f_c sensitivity ───────────────────────────────────

FC_VALUES = [0.50, 0.60, 0.70, 0.80]

print('\n=== Cal. fraction sensitivity ===')
fc_rows = []
for fc in FC_VALUES:
    qvs, pihats, greens = [], [], []
    for asset in ASSETS:
        r, v = load_pair(asset)
        n = len(r)
        n_cal = int(n * fc)
        r_test, v_test = r[n_cal:], v[n_cal:]
        n_test = len(r_test)

        scores = v[:n_cal] - r[:n_cal]
        qV = qhat_ceil(scores, ALPHA)
        var_cp = v_test - qV
        viol = int(np.sum(r_test < var_cp))
        pihat = viol / n_test

        qvs.append(qV)
        pihats.append(pihat)
        greens.append(1 if basel_tl(viol, n_test) == 'Green' else 0)

    mean_qV = np.mean(qvs)
    n_green = sum(greens)
    mean_pi = np.mean(pihats)
    fc_rows.append({
        'check': 'Cal. fraction', 'variant': f'f_c={fc:.2f}',
        'mean_qV': mean_qV, 'green': n_green, 'pihat': mean_pi,
    })
    print(f'  f_c={fc:.2f}  qV={mean_qV:.4f}  Green={n_green}/24  pi={mean_pi:.4f}')

# ── Sanity check: f_c=0.70 vs all_results.csv ──
fc70 = fc_rows[2]
print(f'\n  Sanity (f_c=0.70 vs all_results.csv):')
print(f'    f_c=0.70:    qV={fc70["mean_qV"]:.6f}  Green={fc70["green"]}  pi={fc70["pihat"]:.6f}')
print(f'    all_results: qV={ref_qV:.6f}  Green={ref_green}  pi={ref_pi:.6f}')
assert fc70['green'] == ref_green, f'Green mismatch: {fc70["green"]} vs {ref_green}'
assert abs(fc70['mean_qV'] - ref_qV) < 1e-8, f'qV mismatch: {fc70["mean_qV"]} vs {ref_qV}'

# ── Section 3: Rolling 250-day ───────────────────────────────────

WINDOW = 250

print('\n=== Rolling (250d) ===')
rolling_qvs, rolling_pihats = [], []
for asset in ASSETS:
    r, v = load_pair(asset)
    n = len(r)
    n_cal = int(n * 0.70)
    test_start = max(n_cal, WINDOW)
    n_test = n - test_start

    if n_test < 50:
        print(f'  {asset}: SKIP (only {n_test} test days)')
        continue

    qv_t = np.zeros(n_test)
    for i in range(n_test):
        t = test_start + i
        window_scores = v[t - WINDOW:t] - r[t - WINDOW:t]
        qv_t[i] = qhat_ceil(window_scores, ALPHA)

    var_cp = v[test_start:] - qv_t
    viol = r[test_start:] < var_cp
    pihat = viol.mean()
    mean_qv = qv_t.mean()

    rolling_qvs.append(mean_qv)
    rolling_pihats.append(pihat)

mean_rolling_qV = np.mean(rolling_qvs)
mean_rolling_pi = np.mean(rolling_pihats)
rolling_row = {
    'check': 'Rolling (250d)', 'variant': '',
    'mean_qV': mean_rolling_qV, 'green': np.nan, 'pihat': mean_rolling_pi,
}
print(f'  qV={mean_rolling_qV:.4f}  pi={mean_rolling_pi:.4f}')

# ── Sanity check: rolling pihat vs rolling_vs_static.csv ──
rvs = pd.read_csv(DATA_DIR / 'paper_outputs' / 'tables' / 'rolling_vs_static.csv')
rvs_cs = rvs[rvs['model'] == MODEL]
ref_rolling_pi = rvs_cs['rolling_pihat'].mean()
print(f'\n  Sanity (rolling pihat):')
print(f'    computed:           {mean_rolling_pi:.6f}')
print(f'    rolling_vs_static:  {ref_rolling_pi:.6f}')

# ── Combine all rows ─────────────────────────────────────────────

all_rows = wcp_rows + fc_rows + [rolling_row]
result = pd.DataFrame(all_rows)
result.to_csv(SCRIPT_DIR / 'tab_robustness_summary.csv', index=False)

# ── Generate LaTeX ───────────────────────────────────────────────

def rhu(x, dp):
    d = Decimal(str(x)).quantize(Decimal(10) ** -dp, rounding=ROUND_HALF_UP)
    return format(d, f'.{dp}f')

lines = [
    r'\begin{tabular}{@{}llccc@{}}',
    r'\toprule',
    r'Check & Variant & Mean $\qV$ & Green & $\hat\pi$ \\',
    r'\midrule',
]

for i, row in enumerate(all_rows):
    check = row['check'] if (i == 0 or all_rows[i - 1]['check'] != row['check']) else ''
    variant = row['variant']

    if 'lambda' in variant:
        variant = variant.replace('lambda=', r'$\lambda=') + '$'
    elif 'f_c' in variant:
        variant = variant.replace('f_c=', r'$f_c=') + '$'

    qv_str = rhu(row['mean_qV'], 3)
    pi_str = rhu(row['pihat'], 3)

    if np.isnan(row.get('green', np.nan)):
        green_str = ''
    else:
        green_str = f'{int(row["green"])}/24'

    line = f'\t\t{check}\n\t\t& {variant} & {qv_str} & {green_str} & {pi_str} \\\\'
    lines.append(line)

    if i == 2 or i == 6:
        lines.append(r'\midrule')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')

tex = '\n'.join(lines) + '\n'
tex_path = SCRIPT_DIR / 'tab_robustness_summary.tex'
tex_path.write_text(tex)
print(f'\nSaved {tex_path.name}')
print(tex)
