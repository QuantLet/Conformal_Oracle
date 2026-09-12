#!/usr/bin/env python3
"""Re-estimate the five R7 classical benchmarks on the frozen August data.

Separate output vintage; stores all fitted parameters and rejected attempts.
Gaussian GARCH/GJR retain their existing exception-only fallback. GJR-t keeps
the existing convergence/plausibility checks and last-valid-nu rule. These
different specifications are recorded, not silently harmonised.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
ALPHAS = np.array([.01, .025, .05, .10])
WINDOW = 250


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_one(root, asset, model):
    warnings.filterwarnings('ignore')
    from arch import arch_model
    from arch.univariate import StudentsT
    t0 = time.monotonic()
    root = Path(root)
    inp = root / 'data/returns' / f'{asset}.csv'
    output = root / 'data/benchmarks' / f'{asset}_{model}.parquet'
    fitted = root / 'parameters' / model / f'{asset}.parquet'
    done = root / 'provenance' / model / f'{asset}.json'
    for p in (output, fitted, done):
        p.parent.mkdir(parents=True, exist_ok=True)
    binding = dict(input_sha256=sha(inp), producer_sha256=sha(__file__), model=model, asset=asset,
                   packages={k: importlib.metadata.version(k) for k in ['numpy', 'pandas', 'scipy', 'arch', 'pyarrow']},
                   python=platform.python_version(), window=WINDOW, alphas=ALPHAS.tolist())
    if done.exists():
        old = json.loads(done.read_text())
        if old['binding'] != binding or sha(output) != old['forecast_sha256'] or sha(fitted) != old['parameters_sha256']:
            raise ValueError(f'Existing output binding mismatch: {asset}/{model}')
        return f'{asset} {model}: already complete, verified'
    ret = pd.read_csv(inp, index_col='date', parse_dates=True)['log_return']
    vals, dates = ret.to_numpy(), ret.index
    normal_q = stats.norm.ppf(ALPHAS)
    rows, params, last_nu = [], [], np.nan
    variance = np.empty(len(ret))
    if model == 'ewma':
        variance[0] = vals[:WINDOW].var(ddof=1)
        for t in range(1, len(ret)):
            variance[t] = .94 * variance[t - 1] + .06 * vals[t - 1] ** 2
    for t in range(WINDOW, len(ret)):
        r_win = ret.iloc[t-WINDOW:t] * 100
        sd_win = float(r_win.std()) / 100
        p = {'date': dates[t], 'context_start': dates[t-WINDOW], 'context_end': dates[t-1],
             'context_sha256': hashlib.sha256(vals[t-WINDOW:t].astype('<f8').tobytes()).hexdigest()}
        z = normal_q
        if model == 'hs':
            w = vals[t-WINDOW:t]
            mu, sd = w.mean(), w.std()
            quantiles = np.percentile(w, ALPHAS*100)
            p.update(method='empirical_linear_percentile', ddof=0)
        elif model == 'ewma':
            mu, sd = 0., np.sqrt(variance[t])
            p.update(method='full_history_recursion', seed_variance=variance[0], conditional_variance=variance[t], decay=.94)
            quantiles = mu + sd*z
        else:
            def fit(distname, prefix, guarded):
                try:
                    am = arch_model(r_win, vol='GARCH', p=1, o=0 if model == 'garch_n' else 1, q=1, dist=distname)
                    res = am.fit(disp='off', show_warning=False)
                    p.update({prefix+k: float(v) for k, v in res.params.items()})
                    p[prefix+'convergence_flag'] = int(res.convergence_flag)
                    p[prefix+'loglikelihood'] = float(res.loglikelihood)
                    fc = res.forecast(horizon=1, reindex=False)
                    m = float(fc.mean.iloc[-1, 0])/100
                    s = float(np.sqrt(fc.variance.iloc[-1, 0]))/100
                    nu = float(res.params['nu']) if distname == 't' else np.nan
                    valid = np.isfinite(m) and np.isfinite(s) and s > 0 and sd_win > 0 and s <= 10*sd_win and abs(m) <= 10*sd_win
                    p[prefix+'plausible'] = bool(valid)
                    if guarded and (res.convergence_flag != 0 or not valid):
                        p[prefix+'rejected'] = True
                        return None
                    p[prefix+'rejected'] = False
                    return m, s, nu
                except Exception as e:
                    p[prefix+'error'] = type(e).__name__ + ': ' + str(e)
                    return None
            if model == 'gjr_t':
                got = fit('t', 't_', True)
                p['normal_fallback'] = got is None
                if got is None:
                    got = fit('normal', 'normal_', True)
            else:
                got = fit('normal', 'normal_', False)
            p['window_sd_fallback'] = got is None
            mu, sd, nu = got if got is not None else (0., sd_win, np.nan)
            if model == 'gjr_t':
                deg = not np.isfinite(nu) or nu <= 2.10
                if not deg:
                    last_nu = nu
                z = StudentsT().ppf(ALPHAS, np.array([last_nu])) if np.isfinite(last_nu) else normal_q
                p.update(nu_used=last_nu, degenerate=deg)
            quantiles = mu + sd*z
        rows.append({'date': dates[t], 'mean': mu, 'std': sd,
                     **{f'VaR_{a:g}': float(q) for a, q in zip(ALPHAS, quantiles)}})
        p.update(forecast_mean=mu, forecast_std=sd)
        params.append(p)
        if (t-WINDOW+1) % 1000 == 0:
            print(f'{asset} {model}: {t-WINDOW+1}/{len(ret)-WINDOW}, {time.monotonic()-t0:.0f}s', flush=True)
    df = pd.DataFrame(rows).set_index('date')
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(f'{asset}/{model}: nonfinite forecast; do not publish')
    df.to_parquet(output)
    pd.DataFrame(params).set_index('date').to_parquet(fitted)
    done.write_text(json.dumps(dict(binding=binding, rows=len(df), first_date=str(df.index[0].date()),
                                   last_date=str(df.index[-1].date()), forecast_sha256=sha(output),
                                   parameters_sha256=sha(fitted), elapsed_seconds=time.monotonic()-t0), indent=2)+'\n')
    return f'{asset} {model}: complete, {len(df)} dates, {time.monotonic()-t0:.0f}s'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT / 'artifacts/extension_20260831')
    ap.add_argument('--assets', nargs='+')
    ap.add_argument('--models', nargs='+', choices=['hs','ewma','garch_n','gjr_garch','gjr_t'], default=['hs','ewma','garch_n','gjr_garch','gjr_t'])
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    assets = a.assets or sorted(p.stem for p in (a.root / 'data/returns').glob('*.csv'))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(run_one, str(a.root), asset, model) for model in a.models for asset in assets]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == '__main__':
    main()
