#!/usr/bin/env python3
"""Refit CAViaR/GAS with saved parameters and same-parameter GAS comparison.

The existing GAS likelihood uses t.logpdf(y/sigma,nu)-log(sigma), so sigma is
the Student-t scale, not its standard deviation. Its matching quantile is
sigma*t.ppf(alpha,nu). Archive the old extra standardisation separately.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
from numba import njit

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/analysis/phase3_dynamic'),str(ROOT/'source/Quantlets')]
import run_dynamic_var as original

# Preserve arithmetic order and optimisation specifications. JIT only the
# deterministic recursions; no fast-math or parallel reductions.
original.caviar_path=njit(original.caviar_path,cache=False)
original.gas_sigma=njit(original.gas_sigma,cache=False)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def work(asset):
    out=ROOT/'artifacts/r8_commodity_etp'
    inp=out/'data/returns'/f'{asset}.csv'
    ret=pd.read_csv(inp,index_col='date',parse_dates=True).log_return
    y=ret.to_numpy();n_cal=int(.7*len(y));results=[]
    for name in ['CAViaR-SAV','CAViaR-AS','GAS-t']:
        target=out/'data/dynamic'/f'{asset}_{name}.parquet'
        fitfile=out/'parameters/dynamic'/f'{asset}_{name}.json'
        for p in [target,fitfile]:p.parent.mkdir(parents=True,exist_ok=True)
        binding=dict(input_sha256=sha(inp),producer_sha256=sha(__file__),original_producer_sha256=sha(original.__file__),
                     alpha=.01,n_cal=n_cal,packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','numba']})
        if fitfile.exists():
            saved=json.loads(fitfile.read_text())
            assert saved['binding']==binding and saved['forecast_sha256']==sha(target)
            results.append(saved['metrics']);continue
        attempts=[];minimizer=original.optimize.minimize
        def audit(*a,**kw):
            r=minimizer(*a,**kw)
            attempts.append(dict(start=np.asarray(a[1]).tolist(),x=r.x.tolist(),success=bool(r.success),
                                 status=int(r.status),fun=float(r.fun),nit=int(r.nit),nfev=int(r.nfev),message=str(r.message)))
            return r
        original.optimize.minimize=audit
        started=time.monotonic()
        try:
            if name.startswith('CAViaR'):
                spec=name.split('-')[1];theta,q0,ok=original.fit_caviar(y[:n_cal],.01,spec)
                if not ok:raise ValueError(f'{asset}/{name}: no fit')
                q=original.caviar_path(theta,y,spec,q0)
                frame=pd.DataFrame({'VaR_0.01':q},index=ret.index)
                extra=dict(initial_quantile=q0)
            else:
                theta,ok=original.fit_gas(y[:n_cal])
                if not ok:raise ValueError(f'{asset}/{name}: no fit')
                omega,a,b,nu=theta
                if not (0<a<1 and 0<b<1 and 2.05<nu<60):raise ValueError('Invalid GAS fitted parameters')
                sig=original.gas_sigma(theta,y)
                q=sig*stats.t.ppf(.01,nu)
                old=q/np.sqrt(nu/(nu-2))
                frame=pd.DataFrame({'VaR_0.01':q,'student_t_scale':sig,'old_extra_standardisation_quantile':old},index=ret.index)
                extra=dict(old_mapping_metrics=original.evaluate(y[n_cal:],old[n_cal:],y[:n_cal],old[:n_cal],.01))
        finally:original.optimize.minimize=minimizer
        if not np.isfinite(q).all():raise ValueError('Nonfinite dynamic quantile')
        frame.to_parquet(target)
        metric=original.evaluate(y[n_cal:],q[n_cal:],y[:n_cal],q[:n_cal],.01)
        best=min(attempts,key=lambda a:a['fun'])
        metric.update(model=name,asset=asset,converged=best['success'],n_cal=n_cal)
        fitfile.write_text(json.dumps(dict(binding=binding,theta=theta.tolist(),attempts=attempts,
                                           forecast_sha256=sha(target),metrics=metric,**extra),indent=2)+'\n')
        results.append(metric)
        print(asset,name,'complete',round(time.monotonic()-started,1),'s',flush=True)
    return results


if __name__=='__main__':
    out=ROOT/'artifacts/r8_commodity_etp'
    assets=sorted(p.stem for p in (out/'data/returns').glob('*.csv'))
    records=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(work,a) for a in assets]):records.extend(f.result())
    (out/'results').mkdir(exist_ok=True)
    pd.DataFrame(records).sort_values(['model','asset']).to_csv(out/'results/dynamic_var.csv',index=False)
