"""
Tuned GBM-QR ablation on S&P 500 (TODO[EMPIRICAL] E1).

Runs an 8-config grid over {n_estimators, max_depth, lr} for the
LightGBM quantile-regression baseline, restricted to SP500 and the
9 base forecasters.  Reports per-config pooled metrics and selects
the best config by QS.

Output:
  - tuned_gbm_qr_grid.csv        full 8×9 = 72-row results
  - tuned_gbm_qr_summary.csv     8-row summary (one per config)
  - tab_baselines_tuned_row.tex   single best-config LaTeX row
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from math import ceil
from scipy.stats import chi2
from decimal import Decimal, ROUND_HALF_UP
from itertools import product

import lightgbm as lgb

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / 'cfp_ijf_data'
OUT  = Path(__file__).resolve().parent

ALPHA = 0.01
F_CAL = 0.70
VAL_FRAC = 0.20
EARLY_STOP_ROUNDS = 50
SYMBOL = 'SP500'

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

GRID = list(product(
    [100, 500],          # n_estimators (num_boost_round)
    [3, 5],              # max_depth → num_leaves = 2^d - 1
    [0.01, 0.05],        # learning_rate
))


def load_pair(model_key):
    subdir, suffix = MODELS[model_key]
    ret = pd.read_csv(DATA / 'returns' / f'{SYMBOL}.csv',
                      index_col=0, parse_dates=True)
    ret.columns = ['r']
    fname = (f'{SYMBOL}.parquet' if suffix is None
             else f'{SYMBOL}_{suffix}.parquet')
    fc = pd.read_parquet(DATA / subdir / fname)
    common = ret.index.intersection(fc.index)
    ret = ret.loc[common]
    fc = fc.loc[common]
    var_col = f'VaR_{ALPHA}'
    mask = fc[var_col].notna()
    return ret['r'].values[mask], fc[var_col].values[mask]


def make_features(r, q_lo):
    vol5 = pd.Series(r).rolling(5, min_periods=1).std().fillna(0.0).values
    vol20 = pd.Series(r).rolling(20, min_periods=1).std().fillna(0.0).values
    vol5_lag = np.concatenate([[0.0], vol5[:-1]])
    vol20_lag = np.concatenate([[0.0], vol20[:-1]])
    return np.column_stack([q_lo, vol5_lag, vol20_lag])


def kupiec_p(x, n, alpha=ALPHA):
    if n == 0:
        return 1.0
    pi = x / n
    if pi == 0:
        lr = 2 * n * np.log(1 / (1 - alpha))
    elif pi == 1:
        lr = 2 * n * np.log(alpha)
    else:
        lr = 2 * (x * np.log(pi / alpha) +
                   (n - x) * np.log((1 - pi) / (1 - alpha)))
    return 1 - chi2.cdf(lr, 1)


def traffic_light(x, n):
    if n == 0:
        return 'Green'
    annual = x * (250.0 / n)
    if annual <= 4:
        return 'Green'
    if annual <= 9:
        return 'Yellow'
    return 'Red'


def pinball(r, var_neg, alpha=ALPHA):
    var_pos = -var_neg
    violations = (r < -var_pos).astype(float)
    return float(np.mean((alpha - violations) * (r + var_pos)))


def eval_config(model_key, n_est, max_depth, lr):
    try:
        r, q_lo = load_pair(model_key)
    except Exception:
        return None
    T = len(r)
    n_cal = int(T * F_CAL)
    if n_cal < 200 or T - n_cal < 50:
        return None

    X = make_features(r, q_lo)
    X_cal, X_test = X[:n_cal], X[n_cal:]
    y_cal, y_test = r[:n_cal], r[n_cal:]

    n_val = max(int(n_cal * VAL_FRAC), 30)
    X_tr, y_tr = X_cal[:-n_val], y_cal[:-n_val]
    X_vl, y_vl = X_cal[-n_val:], y_cal[-n_val:]

    num_leaves = 2 ** max_depth - 1
    params = dict(
        objective='quantile',
        alpha=ALPHA,
        learning_rate=lr,
        num_leaves=num_leaves,
        min_data_in_leaf=20,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
    )
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_vl, label=y_vl, reference=dtrain)
    try:
        model = lgb.train(
            params, dtrain,
            num_boost_round=n_est,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False)],
        )
    except Exception:
        return None

    pred = model.predict(X_test, num_iteration=model.best_iteration)
    n_test = len(y_test)
    violations = (y_test < pred).astype(int)
    x = int(violations.sum())
    return {
        'model': model_key,
        'n_est': n_est,
        'max_depth': max_depth,
        'lr': lr,
        'n_test': n_test,
        'viol': x,
        'pi_hat': x / n_test,
        'kupiec_p': kupiec_p(x, n_test),
        'TL': traffic_light(x, n_test),
        'QS': pinball(y_test, pred),
        'width': float(np.mean(np.abs(pred))),
        'best_iter': model.best_iteration,
    }


def rhup(val, ndigits):
    return float(Decimal(str(val)).quantize(
        Decimal(10) ** -ndigits, rounding=ROUND_HALF_UP))


def main():
    rows = []
    for n_est, max_depth, lr in GRID:
        tag = f"n={n_est} d={max_depth} lr={lr}"
        for mname in MODELS:
            res = eval_config(mname, n_est, max_depth, lr)
            if res is None:
                continue
            rows.append(res)
            print(f"  {tag:20s} {mname:16s} pi={res['pi_hat']:.3f} "
                  f"QS={res['QS']*1e4:.2f} {res['TL']} iter={res['best_iter']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'tuned_gbm_qr_grid.csv', index=False)
    print(f"\nSaved grid ({len(df)} rows) to tuned_gbm_qr_grid.csv")

    summary_rows = []
    for n_est, max_depth, lr in GRID:
        sub = df[(df['n_est'] == n_est) &
                 (df['max_depth'] == max_depth) &
                 (df['lr'] == lr)]
        if len(sub) == 0:
            continue
        n = len(sub)
        pi = sub['pi_hat'].mean()
        kup_rej = int((sub['kupiec_p'] < 0.05).sum())
        qs_mean = sub['QS'].mean() * 1e4
        width = sub['width'].mean()
        green = int((sub['TL'] == 'Green').sum())
        summary_rows.append({
            'n_est': n_est, 'max_depth': max_depth, 'lr': lr,
            'n_pairs': n,
            'pi_hat': pi, 'kup_rej': kup_rej,
            'QS_x1e4': qs_mean, 'width': width,
            'green': green, 'green_pct': 100.0 * green / n,
        })

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(OUT / 'tuned_gbm_qr_summary.csv', index=False)
    print("\nGrid summary:")
    print(sdf.to_string(index=False))

    best = sdf.loc[sdf['QS_x1e4'].idxmin()]
    pi_s = f".{int(rhup(best['pi_hat'], 3) * 1000):03d}"
    qs_s = f"{rhup(best['QS_x1e4'], 2):.2f}"
    w_s = f".{int(rhup(best['width'], 3) * 1000):03d}"
    gr_s = f"{rhup(best['green_pct'], 1):.1f}"
    kup_s = f"{int(best['kup_rej'])}/{int(best['n_pairs'])}"
    tag = (f"$n$={int(best['n_est'])}, $d$={int(best['max_depth'])}, "
           f"$\\eta$={best['lr']}")

    tex = (f"GBM-QR (tuned, {tag}) & {pi_s} & {kup_s} & {qs_s} "
           f"& {w_s} & {gr_s} \\\\")
    (OUT / 'tab_baselines_tuned_row.tex').write_text(tex + '\n')
    print(f"\nBest config LaTeX row:\n{tex}")
    print(f"\nSaved to tab_baselines_tuned_row.tex")


if __name__ == '__main__':
    main()
