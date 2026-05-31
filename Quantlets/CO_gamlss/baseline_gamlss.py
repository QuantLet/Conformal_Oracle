"""
GAMLSS baseline with skewed-t (Fernandez-Steel) innovations.

Static calibration split (matching GBM-QR protocol):
  - 70% calibration, 30% test
  - Fit GAMLSS once on calibration, predict on test

Model:
  location:  mu_i    = beta0 + beta1 * q_lo_i
  scale:     sigma_i = exp(gamma0 + gamma1 * vol5_lag_i + gamma2 * vol20_lag_i)
  shape:     df      = exp(delta) + 2
  skewness:  xi      = exp(eta)
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import t as t_dist, chi2, norm
from numpy.linalg import lstsq

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / 'cfp_ijf_data'
OUT  = Path(__file__).resolve().parent


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
    q = fc[var_col].values[mask]
    return r, q


def make_features(r, q_lo):
    vol5 = pd.Series(r).rolling(5, min_periods=1).std().fillna(0.0).values
    vol20 = pd.Series(r).rolling(20, min_periods=1).std().fillna(0.0).values
    vol5_lag = np.concatenate([[0.0], vol5[:-1]])
    vol20_lag = np.concatenate([[0.0], vol20[:-1]])
    return q_lo, vol5_lag, vol20_lag


def sst_logpdf(y, mu, sigma, df, xi):
    z = (y - mu) / sigma
    z_adj = np.where(z >= 0, z / xi, z * xi)
    log_c = np.log(2.0 * xi / (xi**2 + 1.0))
    return log_c - np.log(sigma) + t_dist.logpdf(z_adj, df)


def sst_quantile(alpha, mu, sigma, df, xi):
    p_split = 1.0 / (xi**2 + 1.0)
    if isinstance(mu, np.ndarray):
        out = np.empty_like(mu)
        for i in range(len(mu)):
            out[i] = _sst_quantile_scalar(alpha, mu[i], sigma[i], df, xi)
        return out
    return _sst_quantile_scalar(alpha, mu, sigma, df, xi)


def _sst_quantile_scalar(alpha, mu, sigma, df, xi):
    p_split = 1.0 / (xi**2 + 1.0)
    if alpha < p_split:
        q_std = t_dist.ppf(alpha * (xi**2 + 1.0) / 2.0, df)
        return mu + sigma * q_std / xi
    else:
        q_std = t_dist.ppf(1.0 - (1.0 - alpha) * (xi**2 + 1.0) / 2.0, df)
        return mu + sigma * q_std * xi


def neg_loglik(params, y, q_lo, vol5_lag, vol20_lag):
    beta0, beta1, gamma0, gamma1, gamma2, delta, eta = params
    mu = beta0 + beta1 * q_lo
    sigma = np.exp(np.clip(gamma0 + gamma1 * vol5_lag + gamma2 * vol20_lag, -10, 10))
    df = np.exp(np.clip(delta, -5, 5)) + 2.0
    xi = np.exp(np.clip(eta, -3, 3))
    ll = sst_logpdf(y, mu, sigma, df, xi)
    return -np.sum(ll)


def fit_gamlss(y, q_lo, vol5_lag, vol20_lag):
    x0 = np.array([0.0, 1.0, np.log(np.std(y) + 1e-6), 0.0, 0.0, np.log(3.0), 0.0])
    best = None
    for method in ['L-BFGS-B', 'Nelder-Mead']:
        try:
            opts = {'maxiter': 2000} if method == 'L-BFGS-B' else {'maxiter': 5000}
            res = minimize(neg_loglik, x0, args=(y, q_lo, vol5_lag, vol20_lag),
                           method=method, options=opts)
            if np.isfinite(res.fun) and res.fun < 1e10:
                if best is None or res.fun < best.fun:
                    best = res
                if method == 'L-BFGS-B':
                    break
        except Exception:
            continue
    return best.x if best is not None else None


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


def eval_pair(model_key, symbol):
    try:
        r, q_lo = load_pair(model_key, symbol)
    except Exception:
        return None
    T = len(r)
    n_cal = int(T * F_CAL)
    if n_cal < 200 or T - n_cal < 50:
        return None

    q_feat, v5_lag, v20_lag = make_features(r, q_lo)

    # Calibration / test split
    y_cal, y_test = r[:n_cal], r[n_cal:]
    q_cal, q_test = q_feat[:n_cal], q_feat[n_cal:]
    v5_cal, v5_test = v5_lag[:n_cal], v5_lag[n_cal:]
    v20_cal, v20_test = v20_lag[:n_cal], v20_lag[n_cal:]

    params = fit_gamlss(y_cal, q_cal, v5_cal, v20_cal)
    converged = params is not None

    if converged:
        beta0, beta1, gamma0, gamma1, gamma2, delta, eta = params
        mu = beta0 + beta1 * q_test
        sigma = np.exp(np.clip(gamma0 + gamma1 * v5_test + gamma2 * v20_test, -10, 10))
        df = np.exp(np.clip(delta, -5, 5)) + 2.0
        xi = np.exp(np.clip(eta, -3, 3))
        var_pred = sst_quantile(ALPHA, mu, sigma, df, xi)
    else:
        X_cal = np.column_stack([np.ones(n_cal), q_cal])
        beta, _, _, _ = lstsq(X_cal, y_cal, rcond=None)
        X_test = np.column_stack([np.ones(len(q_test)), q_test])
        mu = X_test @ beta
        sig = np.std(y_cal - X_cal @ beta)
        var_pred = mu + sig * norm.ppf(ALPHA)

    n_test = len(y_test)
    violations = (y_test < var_pred).astype(int)
    x = int(violations.sum())
    pi_hat = x / n_test
    kup = kupiec_p(x, n_test)
    tl = traffic_light(x, n_test)
    qs = pinball(y_test, var_pred)
    width = float(np.mean(np.abs(var_pred)))

    return {
        'model': model_key, 'symbol': symbol,
        'n_test': n_test, 'viol': x, 'pi_hat': pi_hat,
        'kupiec_p': kup, 'TL': tl, 'QS': qs, 'width': width,
        'converged': int(converged),
    }


def main():
    import time
    rows = []
    total_pairs = len(MODELS) * len(SYMBOLS)
    done = 0
    t_start = time.time()
    for mname in MODELS:
        for sym in SYMBOLS:
            done += 1
            t0 = time.time()
            res = eval_pair(mname, sym)
            dt = time.time() - t0
            if res is None:
                print(f"  [{done}/{total_pairs}] {mname:16s} {sym:8s}: SKIP ({dt:.1f}s)")
                continue
            rows.append(res)
            print(f"  [{done}/{total_pairs}] {mname:16s} {sym:8s}: "
                  f"pi={res['pi_hat']:.3f} QS={res['QS']*1e4:.2f} "
                  f"W={res['width']:.3f} {res['TL']} "
                  f"conv={res['converged']} [{dt:.1f}s]", flush=True)

    elapsed = time.time() - t_start
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'gamlss_results.csv', index=False)

    n = len(df)
    pi = df['pi_hat'].mean()
    kup_rej = int((df['kupiec_p'] < 0.05).sum())
    qs_mean = df['QS'].mean() * 1e4
    width = df['width'].mean()
    green = int((df['TL'] == 'Green').sum())
    conv = df['converged'].sum()

    print()
    print("=" * 60)
    print(f"GAMLSS-SST baseline ({n} model-asset pairs, {elapsed:.0f}s)")
    print("=" * 60)
    print(f"pi_hat:          {pi:.3f}")
    print(f"Kupiec rej:      {kup_rej}/{n}")
    print(f"QS (x1e-4):      {qs_mean:.2f}")
    print(f"Width:           {width:.3f}")
    print(f"Green:           {green}/{n} ({100*green/n:.1f}%)")
    print(f"Convergence:     {conv}/{n} ({100*conv/n:.1f}%)")
    print()
    print("LaTeX row:")
    print(f"GAMLSS-SST & .{int(pi*1000):03d} & {kup_rej}/{n} & {qs_mean:.2f} "
          f"& .{int(width*1000):03d} & {100*green/n:.1f} \\\\")


if __name__ == '__main__':
    main()
