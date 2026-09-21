"""Annual past-only CAViaR-AS/GAS-t fits using the existing optimisers."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata as md
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
from numba import njit
from prepare import PROJECT, OUT, ASSETS, sha

sys.path[:0]=[str(PROJECT/'source/analysis/phase3_dynamic'),str(PROJECT/'source/Quantlets')]
import run_dynamic_var as original
original.caviar_path=njit(original.caviar_path,cache=False)
original.gas_sigma=njit(original.gas_sigma,cache=False)


def forecast(theta,y,name,q0=None):
    if name=='CAViaR-AS':
        return original.caviar_path(theta,y,'AS',q0),None
    scale=original.gas_sigma(theta,y)
    return scale*stats.t.ppf(.01,theta[3]),scale


def fit_context(y,name):
    attempts=[];minimizer=original.optimize.minimize
    def audit(*args,**kwargs):
        try:
            r=minimizer(*args,**kwargs)
            attempts.append(dict(start=np.asarray(args[1]).tolist(),parameters=r.x.tolist(),
                                 success=bool(r.success),status=int(r.status),objective=float(r.fun),
                                 iterations=int(r.nit),evaluations=int(r.nfev),message=str(r.message)))
            return r
        except Exception as e:
            attempts.append(dict(start=np.asarray(args[1]).tolist(),error=type(e).__name__+': '+str(e)))
            raise
    original.optimize.minimize=audit
    try:
        if name=='CAViaR-AS':theta,q0,ok=original.fit_caviar(y,.01,'AS')
        else:theta,ok=original.fit_gas(y);q0=None
    finally:original.optimize.minimize=minimizer
    if not ok or not np.isfinite(theta).all():raise ValueError('No finite dynamic fit')
    if name=='CAViaR-AS':assert abs(theta[1])<.999
    else:assert 0<theta[1]<1 and 0<theta[2]<1 and 2.05<theta[3]<60
    valid=[a for a in attempts if 'objective' in a]
    best=min(valid,key=lambda a:a['objective'])
    np.testing.assert_array_equal(theta,best['parameters'])
    q,scale=forecast(theta,y,name,q0)
    assert np.isfinite(q).all()
    objective=original.tick(y,q,.01) if name=='CAViaR-AS' else -(stats.t.logpdf(y/scale,df=theta[3])-np.log(scale)).sum()
    np.testing.assert_allclose(objective,best['objective'],rtol=1e-12,atol=1e-9)
    return theta,q0,dict(attempts=attempts,selected_success=best['success'],objective=float(objective))


def work(asset,name,replay=False,years=None):
    start_time=time.monotonic();rp=OUT/'data/returns'/f'{asset}.csv'
    ret=pd.read_csv(rp,index_col='date',parse_dates=True).log_return;y=ret.to_numpy()
    binding=dict(input_sha256=sha(rp),producer_sha256=sha(__file__),
                 protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')),
                 original_sha256=sha(original.__file__),context=1250,alpha=.01,
                 packages={n:md.version(n) for n in ['numpy','scipy','pandas','numba','pyarrow']})
    base=OUT/('dynamic_replay' if replay else 'dynamic')/f'{name}__{asset}'
    base.mkdir(parents=True,exist_ok=True)
    frames=[]
    for year in (years or range(2000,2027)):
        positions=np.flatnonzero(ret.index.year==year);assert len(positions)>0
        first,last=int(positions[0]),int(positions[-1])+1;assert first>=1250
        receipt=base/f'{year}.json';target=base/f'{year}.parquet'
        if receipt.exists():
            saved=json.loads(receipt.read_text());assert saved['binding']==binding and sha(target)==saved['forecast_sha256']
            frames.append(pd.read_parquet(target));continue
        context=y[first-1250:first]
        theta,q0,details=fit_context(context,name)
        whole=y[first-1250:last];pred,scale=forecast(theta,whole,name,q0)
        assert np.isfinite(pred).all()
        frame=pd.DataFrame({'VaR_0.01':pred[1250:]},index=ret.index[first:last])
        if scale is not None:frame['student_t_scale']=scale[1250:]
        frame.index.name='date';frame.to_parquet(target)
        saved=dict(binding=binding,year=year,theta=theta.tolist(),initial_quantile=q0,
                   context_start=str(ret.index[first-1250].date()),context_end=str(ret.index[first-1].date()),
                   context_sha256=__import__('hashlib').sha256(context.astype('<f8').tobytes()).hexdigest(),
                   first_forecast=str(frame.index[0].date()),last_forecast=str(frame.index[-1].date()),
                   forecast_sha256=sha(target),**details)
        receipt.write_text(json.dumps(saved,indent=2,allow_nan=False)+'\n')
        frames.append(frame)
        print(name,asset,year,'saved',round(time.monotonic()-start_time,1),'s',flush=True)
    if years is None:
        all_f=pd.concat(frames);assert all_f.index.equals(ret.index[ret.index.year>=2000])
        target=base/'forecasts.parquet';all_f.to_parquet(target)
        (base/'complete.json').write_text(json.dumps(dict(binding=binding,forecast_sha256=sha(target),
            years={str(y):sha(base/f'{y}.json') for y in range(2000,2027)},rows=len(all_f)),indent=2)+'\n')
    return name,asset,round(time.monotonic()-start_time,1)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--assets',nargs='+',default=ASSETS)
    ap.add_argument('--models',nargs='+',default=['CAViaR-AS','GAS-t']);ap.add_argument('--years',nargs='+',type=int)
    ap.add_argument('--workers',type=int,default=3);ap.add_argument('--replay',action='store_true');a=ap.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(work,asset,name,a.replay,a.years) for name in a.models for asset in a.assets]
        for j in as_completed(jobs):print('complete',*j.result(),flush=True)
