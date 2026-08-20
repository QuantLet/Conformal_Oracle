"""
Stationary block bootstrap (Politis-Romano) for the conformal
threshold q̂_V on representative model-asset pairs.
"""

import hashlib

import numpy as np
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'cfp_ijf_data')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'Quantlets', 'CO_robustness')
os.makedirs(OUT_DIR, exist_ok=True)

ASSETS = ['SP500', 'BTC', 'GOLD', 'EURUSD', 'WTI', 'NATGAS']
# Chronos enters as the analytic series. The shipped one is sampled at the
# checkpoint default top_k = 50, and a block bootstrap of its q_V would report a
# confidence interval for a sampling parameter. GJR-GARCH-t is added because the
# corrected Gaussian-innovation GJR under-covers, which makes the fat-tailed
# variant the obvious parametric comparison.
MODELS_TSFM = {
    'Chronos-Small-A': 'chronos_small_analytic',
    'TimesFM-2.5': 'timesfm25',
    'Moirai-2.0': 'moirai2',
    'Moirai-1.1': 'moirai',
    'Lag-Llama': 'lagllama',
}
MODELS_BENCH = {
    'GJR-GARCH': 'benchmarks/gjr_garch',
    'GJR-GARCH-t': 'benchmarks/gjr_t',
    'GARCH-N': 'benchmarks/garch_n',
}

ALPHA = 0.01
F_C = 0.70
N_BOOTSTRAP = 1000
BLOCK_LENGTH = 50
# Per-(asset, model) seeding. A single shared stream made every interval depend
# on how many models happened to precede it in the loop: adding one model moved
# 30 of 36 published intervals. Keying the seed on the cell identity makes each
# one independent of what else is in the dictionary, so the table is stable
# under future additions.
SEED = 42


def rng_for(*keys):
    h = hashlib.sha256("|".join(str(k) for k in keys).encode()).digest()
    return np.random.default_rng(SEED + int.from_bytes(h[:8], "big"))


def load_var_and_returns(model_name, model_dir, asset):
    returns = pd.read_csv(
        os.path.join(DATA_DIR, 'returns', f'{asset}.csv'),
        parse_dates=['date'], index_col='date')

    if model_dir.startswith('benchmarks/'):
        suffix = model_dir.split('/')[-1]
        fpath = os.path.join(DATA_DIR, 'benchmarks', f'{asset}_{suffix}.parquet')
        fc = pd.read_parquet(fpath)
        var_raw = fc['VaR_0.01']
    else:
        fpath = os.path.join(DATA_DIR, model_dir, f'{asset}.parquet')
        fc = pd.read_parquet(fpath)
        var_raw = fc['VaR_0.01']

    return var_raw, returns


def compute_scores(var_raw, returns):
    common = var_raw.index.intersection(returns.index)
    var_raw = var_raw.loc[common]
    ret = returns.loc[common, 'log_return']
    scores = var_raw - ret
    n = len(common)
    n_cal = int(F_C * n)
    return scores.iloc[:n_cal].values, np.mean(np.abs(var_raw.iloc[n_cal:].values))


def stationary_bootstrap(scores, block_length, n_replications, rng):
    n = len(scores)
    p = 1.0 / block_length
    qV_estimates = np.empty(n_replications)
    for b in range(n_replications):
        boot_indices = np.empty(n, dtype=int)
        idx = rng.integers(0, n)
        for j in range(n):
            boot_indices[j] = idx
            if rng.uniform() < p:
                idx = rng.integers(0, n)
            else:
                idx = (idx + 1) % n
        boot_scores = scores[boot_indices]
        ss = np.sort(boot_scores)
        nn = len(ss)
        kk = int(np.ceil((nn + 1) * (1 - ALPHA))) - 1
        kk = min(kk, nn - 1)
        qV_estimates[b] = float(ss[kk])
    return qV_estimates


def qhat_ceil(scores, alpha):
    ss = np.sort(scores)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    k = min(k, n - 1)
    return float(ss[k])


def run():
    all_models = {}
    all_models.update(MODELS_TSFM)
    all_models.update(MODELS_BENCH)

    results = []
    for asset in ASSETS:
        for model_name, model_dir in all_models.items():
            var_raw, returns = load_var_and_returns(model_name, model_dir, asset)
            if var_raw is None:
                print(f"SKIP {asset} {model_name}: data not found")
                continue

            cal_scores, mean_abs_var = compute_scores(var_raw, returns)
            point = qhat_ceil(cal_scores, ALPHA)
            boot = stationary_bootstrap(cal_scores, BLOCK_LENGTH,
                                        N_BOOTSTRAP, rng_for(asset, model_name))
            lo, hi = np.quantile(boot, [0.025, 0.975])
            se = np.std(boot)
            R = abs(point) / mean_abs_var if mean_abs_var > 0 else np.inf
            replacement = 'Yes' if R > 1.5 else 'No'

            row = {
                'asset': asset,
                'model': model_name,
                'qV_point': point,
                'qV_lo': lo,
                'qV_hi': hi,
                'qV_se': se,
                'R': R,
                'replacement': replacement,
            }
            results.append(row)
            print(f"{asset:8s} {model_name:16s} qV={point:.4f} "
                  f"[{lo:.4f}, {hi:.4f}] SE={se:.4f} R={R:.2f} {replacement}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, 'qV_bootstrap_ci.csv'), index=False)
    print(f"\nSaved to {os.path.join(OUT_DIR, 'qV_bootstrap_ci.csv')}")
    return df


if __name__ == '__main__':
    run()
