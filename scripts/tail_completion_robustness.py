"""
Tail-completion robustness check for TimesFM 2.5 and Moirai 2.0.

For each asset in {SP500, BTC, NATGAS}, reconstruct the native 9-decile
quantile grid from stored Student-t parameters, then apply three closures
(Student-t, Gaussian, linear extrapolation) to obtain alternative 1% VaR
series. Run the static conformal pipeline and compare q̂_V, violation
rate, and Basel classification.
"""

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist, norm
from scipy.optimize import minimize
import os
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "Quantlets"))
from cfp_config import split_indices  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'cfp_ijf_data')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'Quantlets', 'CO_robustness')
os.makedirs(OUT_DIR, exist_ok=True)

ASSETS = ['SP500', 'BTC', 'NATGAS']
MODELS = {
    'TimesFM-2.5': 'timesfm25',
    'Moirai-2.0': 'moirai2',
}
QUANTILE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ALPHA = 0.01
F_C = 0.70


def reconstruct_deciles(nu, mu, sigma):
    return t_dist.ppf(QUANTILE_LEVELS, df=nu, loc=mu, scale=sigma)


def student_t_closure(deciles, target_alpha=0.01):
    def objective(params):
        nu, mu, sigma = params
        if nu <= 2 or sigma <= 0:
            return 1e10
        predicted = t_dist.ppf(QUANTILE_LEVELS, df=nu, loc=mu, scale=sigma)
        return np.sum((predicted - deciles) ** 2)

    x0 = [5.0, np.median(deciles), max(np.std(deciles), 1e-6)]
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': 1000, 'xatol': 1e-6})
    nu, mu, sigma = res.x
    nu = max(nu, 2.01)
    sigma = max(sigma, 1e-8)
    return t_dist.ppf(target_alpha, df=nu, loc=mu, scale=sigma)


def gaussian_closure(deciles, target_alpha=0.01):
    def objective(params):
        mu, sigma = params
        if sigma <= 0:
            return 1e10
        predicted = norm.ppf(QUANTILE_LEVELS, loc=mu, scale=sigma)
        return np.sum((predicted - deciles) ** 2)

    x0 = [np.median(deciles), max(np.std(deciles), 1e-6)]
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': 1000, 'xatol': 1e-6})
    mu, sigma = res.x
    sigma = max(sigma, 1e-8)
    return norm.ppf(target_alpha, loc=mu, scale=sigma)


def linear_closure(deciles, target_alpha=0.01):
    q10 = deciles[0]  # u = 0.1
    q20 = deciles[1]  # u = 0.2
    slope = (q20 - q10) / (0.2 - 0.1)
    return q10 + slope * (target_alpha - 0.1)


def conformal_pipeline(var_raw, returns):
    common = var_raw.index.intersection(returns.index)
    var_raw = var_raw.loc[common]
    ret = returns.loc[common, 'log_return']

    n = len(common)
    _cal, _test, _g = split_indices(n, (var_raw - ret).values, f_cal=F_C)
    n_cal, t0 = len(_cal), int(_test[0])

    scores = var_raw - ret
    cal_scores = scores.iloc[:n_cal].values
    test_scores = scores.iloc[t0:].values
    test_ret = ret.iloc[t0:].values
    test_var = var_raw.iloc[t0:].values

    ss = np.sort(cal_scores)
    n_c = len(cal_scores)
    k = int(np.ceil((n_c + 1) * (1 - ALPHA))) - 1
    k = min(k, n_c - 1)
    qV = float(ss[k])

    var_cp = test_var - qV
    violations = (test_ret < var_cp)
    pi_hat = violations.mean()
    n_test = len(test_ret)

    raw_violations = (test_ret < test_var)
    pi_raw = raw_violations.mean()

    mean_var_raw = np.mean(np.abs(test_var))
    R = abs(qV) / mean_var_raw if mean_var_raw > 0 else np.inf

    n_viol = violations.sum()
    if n_viol == 0 or n_viol == n_test:
        tl = 'Green' if n_viol / n_test <= 0.015 else 'Red'
    else:
        from scipy.stats import binom
        p_binom = 1 - binom.cdf(n_viol - 1, n_test, ALPHA)
        if p_binom >= 0.9999:
            tl = 'Red'
        elif p_binom >= 0.95:
            tl = 'Yellow'
        else:
            tl = 'Green'

    return {
        'qV': qV,
        'pi_raw': pi_raw,
        'pi_corr': pi_hat,
        'R': R,
        'basel': tl,
        'n_cal': n_cal,
        'n_test': n_test,
    }


def run():
    results = []
    closures = {
        'Student-$t$': student_t_closure,
        'Gaussian': gaussian_closure,
        'Linear': linear_closure,
    }

    for asset in ASSETS:
        returns = pd.read_csv(
            os.path.join(DATA_DIR, 'returns', f'{asset}.csv'),
            parse_dates=['date'], index_col='date')

        for model_name, model_dir in MODELS.items():
            fc = pd.read_parquet(
                os.path.join(DATA_DIR, model_dir, f'{asset}.parquet'))

            for closure_name, closure_fn in closures.items():
                var_series = pd.Series(index=fc.index, dtype=float)

                for t in range(len(fc)):
                    row = fc.iloc[t]
                    nu = row['df_student']
                    mu = row['mean']
                    sigma = row['std']
                    deciles = reconstruct_deciles(nu, mu, sigma)
                    q_alpha = closure_fn(deciles)
                    # The alpha-quantile IS the Value-at-Risk threshold and is
                    # already negative for a lower tail. This line read
                    # `-q_alpha` until 2026-08-20, which is the same sign
                    # inversion that corrupted the stored TimesFM and Moirai
                    # series -- reproduced here, in the robustness script the
                    # manuscript cites for closure-rule invariance. Verified
                    # against the promoted series: q_alpha reproduces the stored
                    # VaR_0.01 to floating point; -q_alpha is its negation.
                    var_series.iloc[t] = q_alpha

                result = conformal_pipeline(var_series, returns)
                result['asset'] = asset
                result['model'] = model_name
                result['closure'] = closure_name
                results.append(result)
                print(f"{asset:8s} {model_name:14s} {closure_name:12s} "
                      f"qV={result['qV']:.4f} R={result['R']:.2f} "
                      f"pi={result['pi_corr']:.3f} {result['basel']}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, 'tail_closure_robustness.csv'), index=False)
    print(f"\nSaved to {os.path.join(OUT_DIR, 'tail_closure_robustness.csv')}")
    return df


if __name__ == '__main__':
    run()
