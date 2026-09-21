#!/usr/bin/env python3
"""Build lower-quantile forecasts from frozen native TSFM outputs.

Student-t fitting retains each R7 grid model's documented notebook algorithm,
including its different parameterisation and fallback. All final VaR columns
store lower RETURN quantiles (negative for a usual loss), as the corrected
R7 input tree does. Price-loss sign conversion occurs only at presentation.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[3]
ALPHAS=np.array([.01,.025,.05,.1]);PROBS=np.arange(1,10)/10


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_grid(grid,model):
    loc0=float(np.median(grid));scale0=max(float(np.std(grid)),1e-6)
    if model=='timesfm25':
        def objective(theta):
            nu,loc,scale=theta
            if nu<=2 or scale<=0:return 1e10
            return np.sum((student_t.ppf(PROBS,nu,loc=loc,scale=scale)-grid)**2)
        x0=np.array([5.,loc0,scale0])
        res=minimize(objective,x0,method='Nelder-Mead',options={'maxiter':1000,'xatol':1e-6})
        nu,loc,scale=res.x
        nu_used=max(nu,2.01);scale_used=max(scale,1e-8)
        q=student_t.ppf(ALPHAS,nu_used,loc=loc,scale=scale_used)
        q[-1]=grid[0]  # native 0.10 quantile in the existing TimesFM producer
    else:
        def unpack(theta):return 2.01+np.exp(theta[2]),theta[0],np.exp(theta[1])
        def objective(theta):
            nu,loc,scale=unpack(theta)
            return np.mean((student_t.ppf(PROBS,nu,loc=loc,scale=scale)-grid)**2)
        x0=np.array([loc0,np.log(scale0),np.log(8.-2.01)])
        res=minimize(objective,x0,method='Nelder-Mead',options={'maxiter':1000,'xatol':1e-6})
        nu,loc,scale=unpack(res.x)
        nu_used=nu if res.success and nu>=2 else 2.01
        scale_used=scale
        q=student_t.ppf(ALPHAS,nu_used,loc=loc,scale=scale_used)
    p=dict(nu_fitted=float(nu),mu=float(loc),sigma_fitted=float(scale),nu_used=float(nu_used),
           sigma_used=float(scale_used),success=bool(res.success),status=int(res.status),nit=int(res.nit),
           nfev=int(res.nfev),objective=float(res.fun),message=str(res.message),
           native_crossings=int((np.diff(grid)<0).sum()))
    p.update({f'theta_{i}':float(v) for i,v in enumerate(res.x)})
    return q,p


def work(root,model,asset):
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    root=Path(root);out=root/'data'/model/f'{asset}.parquet';pf=root/'parameters'/model/f'{asset}.parquet'
    done=root/'provenance'/model/f'{asset}.json';native=root/'native'/model/asset
    for p in [out,pf,done]:p.parent.mkdir(parents=True,exist_ok=True)
    complete=json.loads((native/'complete.json').read_text())
    binding=dict(native_binding_sha256=sha(native/'binding.json'),producer_sha256=sha(__file__),
                 packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','pyarrow']})
    if binding['native_binding_sha256']!=complete['binding_sha256']:raise ValueError('Native binding mismatch')
    if done.exists():
        old=json.loads(done.read_text())
        if old['binding']!=binding or sha(out)!=old['forecast_sha256'] or sha(pf)!=old['parameters_sha256']:
            raise ValueError('Existing reduction mismatch')
        return f'{model} {asset}: already verified'
    blocks=[];dates=[];positions=[];input_hashes={}
    for chunk in sorted(native.glob('*.npz')):
        meta=json.loads(chunk.with_suffix('.json').read_text());digest=sha(chunk)
        if digest!=meta['sha256']:raise ValueError('Native chunk changed')
        input_hashes[chunk.name]=digest
        with np.load(chunk) as z:
            blocks.append(z['native']);dates.extend(z['date']);positions.extend(z['positions'])
    raw=np.concatenate(blocks);dates=pd.DatetimeIndex(dates,name='date')
    ret=pd.read_csv(root/'data/returns'/f'{asset}.csv',index_col='date',parse_dates=True)
    if len(raw)!=complete['rows'] or not dates.equals(ret.index[512:]) or not np.array_equal(positions,np.arange(512,len(ret))):
        raise ValueError('Native forecast coverage incomplete')
    started=time.monotonic();rows=[];params=[]
    for i,(date,values) in enumerate(zip(dates,raw)):
        if model in ('moirai','lagllama'):
            q=np.percentile(values,ALPHAS*100);mean=float(values.mean());std=float(values.std())
            params.append(dict(date=date,n_draws=len(values),native_min=float(values.min()),native_max=float(values.max())))
        else:
            grid=values[1:] if model=='timesfm25' else values
            q,p=fit_grid(grid.astype(float),model);mean=p['mu'];std=p['sigma_used']
            params.append(dict(date=date,**p,**{f'q_{u:g}':float(v) for u,v in zip(PROBS,grid)}))
        if not np.isfinite(q).all():raise ValueError(f'Nonfinite quantile {model}/{asset}/{date}')
        rows.append(dict(date=date,mean=mean,std=std,**{f'VaR_{a:g}':float(v) for a,v in zip(ALPHAS,q)}))
        if (i+1)%1000==0:print(f'{model} {asset} reduction {i+1}/{len(raw)} {time.monotonic()-started:.0f}s',flush=True)
    pd.DataFrame(rows).set_index('date').to_parquet(out)
    pd.DataFrame(params).set_index('date').to_parquet(pf)
    done.write_text(json.dumps(dict(binding=binding,native_chunks=input_hashes,rows=len(rows),
                                    forecast_sha256=sha(out),parameters_sha256=sha(pf),
                                    elapsed_seconds=time.monotonic()-started),indent=2)+'\n')
    return f'{model} {asset}: reduced {len(rows)} forecasts'


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT/'artifacts/extension_20260831')
    ap.add_argument('--models',nargs='+',default=['moirai','moirai2','timesfm25','lagllama'])
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--ready-only',action='store_true')
    a=ap.parse_args();tasks=[]
    for m in a.models:
        for ret in sorted((a.root/'data/returns').glob('*.csv')):
            ready=(a.root/'native'/m/ret.stem/'complete.json').exists()
            if not ready and not a.ready_only:raise FileNotFoundError(f'{m}/{ret.stem}: inference incomplete')
            if ready:tasks.append((str(a.root),m,ret.stem))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(work,*t) for t in tasks]
        for f in as_completed(futures):print(f.result(),flush=True)
