"""
Gradient-Boosted Quantile Regression (GBM-QR) baseline.

For each of the 216 model-asset pairs:
  1. Load returns and base-VaR forecasts.
  2. Construct features: q_lo (base forecast), lagged 5-day and 20-day
     realised volatilities.
  3. Fit LightGBM quantile regression at alpha=0.01 on the calibration
     sub-sample, with early stopping on a validation fold.
  4. Predict corrected VaR on the test set.
  5. Compute pooled Kupiec, QS, Width, Green Zone.

Run from project root:  python3 scripts/baseline_gbm_qr.py
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from math import ceil
from scipy.stats import chi2

import lightgbm as lgb

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / 'cfp_ijf_data'
OUT  = Path(__file__).resolve().parent


ALPHA = 0.01
F_CAL = 0.70
VAL_FRAC = 0.20        # fraction of calibration sample held out for early stopping
EARLY_STOP_ROUNDS = 50
NUM_BOOST = 500

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


def load_pair(model_key, symbol):
    subdir, suffix = MODELS[model_key]
    ret = pd.read_csv(DATA / 'returns' / f'{symbol}.csv',
                      index_col=0, parse_dates=True)
    ret.columns = ['r']
    fname = f'{symbol}.parquet' if suffix is None else f'{symbol}_{suffix}.parquet'
    fc = pd.read_parquet(DATA / subdir / fname)
    common = ret.index.intersection(fc.index)
    ret = ret.loc[common]
    fc = fc.loc[common]
    var_col = f'VaR_{ALPHA}'
    mask = fc[var_col].notna()
    r = ret['r'].values[mask]
    q = fc[var_col].values[mask]           # negative (lower quantile)
    return r, q


def make_features(r, q_lo):
    """Build features: q_lo, 5-day vol, 20-day vol of returns."""
    T = len(r)
    vol5 = pd.Series(r).rolling(5, min_periods=1).std().fillna(0.0).values
    vol20 = pd.Series(r).rolling(20, min_periods=1).std().fillna(0.0).values
    # include a lagged version so no look-ahead in vol features
    vol5_lag = np.concatenate([[0.0], vol5[:-1]])
    vol20_lag = np.concatenate([[0.0], vol20[:-1]])
    X = np.column_stack([q_lo, vol5_lag, vol20_lag])
    return X


def kupiec_p(x, n, alpha=ALPHA):
    if n == 0:
        return 1.0
    pi = x / n
    if pi == 0:
        lr = 2 * n * np.log(1 / (1 - alpha))
    elif pi == 1:
        lr = 2 * n * np.log(alpha)
    else:
        lr = 2 * (x * np.log(pi / alpha) + (n - x) * np.log((1 - pi) / (1 - alpha)))
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


def fit_gbm_qr(X_train, y_train, X_val, y_val):
    params = dict(
        objective='quantile',
        alpha=ALPHA,
        learning_rate=0.05,
        num_leaves=15,
        min_data_in_leaf=20,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
    )
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False)],
    )
    return model


def eval_pair(model_key, symbol):
    try:
        r, q_lo = load_pair(model_key, symbol)
    except Exception as e:
        return None
    T = len(r)
    n_cal = int(T * F_CAL)
    if n_cal < 200 or T - n_cal < 50:
        return None

    X = make_features(r, q_lo)
    X_cal, X_test = X[:n_cal], X[n_cal:]
    y_cal, y_test = r[:n_cal], r[n_cal:]

    # Train / validation split within calibration
    n_val = max(int(n_cal * VAL_FRAC), 30)
    X_tr, y_tr = X_cal[:-n_val], y_cal[:-n_val]
    X_vl, y_vl = X_cal[-n_val:], y_cal[-n_val:]

    try:
        model = fit_gbm_qr(X_tr, y_tr, X_vl, y_vl)
    except Exception:
        return None

    # Predict on test set; prediction is the alpha-quantile (negative for lower tail)
    pred = model.predict(X_test, num_iteration=model.best_iteration)
    # Convention: VaR_neg is the predicted lower quantile (negative number)
    var_neg = pred

    n_test = len(y_test)
    violations = (y_test < var_neg).astype(int)
    x = int(violations.sum())
    pi_hat = x / n_test
    kup = kupiec_p(x, n_test)
    tl = traffic_light(x, n_test)
    qs = pinball(y_test, var_neg)
    width = float(np.mean(np.abs(var_neg)))
    return {
        'model': model_key, 'symbol': symbol,
        'n_test': n_test, 'viol': x, 'pi_hat': pi_hat,
        'kupiec_p': kup, 'TL': tl, 'QS': qs, 'width': width,
    }


def main():
    rows = []
    for mname in MODELS:
        for sym in SYMBOLS:
            res = eval_pair(mname, sym)
            if res is None:
                continue
            rows.append(res)
            print(f"  {mname:16s} {sym:8s}: pi={res['pi_hat']:.3f} "
                  f"QS={res['QS']*1e4:.2f} W={res['width']:.3f} {res['TL']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'gbm_qr_results.csv', index=False)

    n = len(df)
    pi = df['pi_hat'].mean()
    kup_rej = int((df['kupiec_p'] < 0.05).sum())
    qs_mean = df['QS'].mean() * 1e4
    width = df['width'].mean()
    green = int((df['TL'] == 'Green').sum())

    print()
    print("=" * 60)
    print(f"GBM-QR baseline ({n} model-asset pairs)")
    print("=" * 60)
    print(f"pi_hat:      {pi:.3f}")
    print(f"Kupiec rej:  {kup_rej}/{n}")
    print(f"QS (x1e-4):  {qs_mean:.2f}")
    print(f"Width:       {width:.3f}")
    print(f"Green:       {green}/{n} ({100*green/n:.1f}%)")
    print()
    print("LaTeX row:")
    print(f"GBM-QR & .{int(pi*1000):03d} & {kup_rej}/{n} & {qs_mean:.2f} "
          f"& .{int(width*1000):03d} & {100*green/n:.1f} \\\\")


if __name__ == '__main__':
    main()
