#!/usr/bin/env python3
"""Reconstruct every classical forecast from frozen returns/fit parameters."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import StudentsT
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[3]/'artifacts/extension_20260831'
ALPHAS=np.array([.01,.025,.05,.1])


def work(asset,model):
    warnings.filterwarnings('ignore')
    inp=ROOT/'data/returns'/f'{asset}.csv';f=ROOT/'data/benchmarks'/f'{asset}_{model}.parquet'
    p=ROOT/'parameters'/model/f'{asset}.parquet';meta=json.loads((ROOT/'provenance'/model/f'{asset}.json').read_text())
    for path,digest in [(inp,meta['binding']['input_sha256']),(f,meta['forecast_sha256']),(p,meta['parameters_sha256'])]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    r=pd.read_csv(inp,index_col='date',parse_dates=True).log_return;fc=pd.read_parquet(f);fit=pd.read_parquet(p)
    assert fc.index.equals(r.index[250:]) and fit.index.equals(fc.index)
    rows=[];v=r.to_numpy();last_nu=np.nan
    variance=v[:250].var(ddof=1)
    # EWMA replay starts at t=0 and advances before each one-step forecast.
    ewma=np.empty(len(v));ewma[0]=variance
    for t in range(1,len(v)):ewma[t]=.94*ewma[t-1]+.06*v[t-1]**2
    max_sigma_difference=0.
    for j,(date,p_row) in enumerate(fit.iterrows()):
        t=j+250;window=r.iloc[t-250:t]*100;params=p_row.to_dict()
        assert params['context_start']==r.index[t-250] and params['context_end']==r.index[t-1]
        assert params['context_sha256']==hashlib.sha256(v[t-250:t].astype('<f8').tobytes()).hexdigest()
        z=norm.ppf(ALPHAS)
        if model=='hs':q=np.percentile(v[t-250:t],ALPHAS*100)
        elif model=='ewma':q=np.sqrt(ewma[t])*z
        else:
            if params['window_sd_fallback']:
                mu,sd,nu=0.,float(window.std())/100,np.nan
            else:
                dist='t' if model=='gjr_t' and not params['normal_fallback'] else 'normal'
                prefix='t_' if dist=='t' else 'normal_'
                names=['mu','omega','alpha[1]']+(['gamma[1]'] if model!='garch_n' else [])+['beta[1]']+(['nu'] if dist=='t' else [])
                theta=np.array([params[prefix+k] for k in names])
                am=arch_model(window,vol='GARCH',p=1,o=0 if model=='garch_n' else 1,q=1,dist=dist)
                predicted=am.fix(theta).forecast(horizon=1,reindex=False)
                mu=float(predicted.mean.iloc[-1,0])/100
                sd=float(np.sqrt(predicted.variance.iloc[-1,0]))/100
                nu=params['t_nu'] if dist=='t' else np.nan
            if model=='gjr_t':
                if np.isfinite(nu) and nu>2.10:last_nu=nu
                assert (np.isnan(last_nu) and np.isnan(params['nu_used'])) or last_nu==params['nu_used']
                z=StudentsT().ppf(ALPHAS,np.array([last_nu])) if np.isfinite(last_nu) else z
            max_sigma_difference=max(max_sigma_difference,abs(sd-fc.loc[date,'std']))
            q=mu+sd*z
        rows.append(q)
    replay=np.asarray(rows);actual=fc[[f'VaR_{a:g}' for a in ALPHAS]].to_numpy()
    difference=float(np.max(np.abs(replay-actual)))
    assert difference<=1e-12,(asset,model,difference)
    return dict(asset=asset,model=model,dates=len(fc),quantile_cells=int(actual.size),
                max_abs_quantile_difference=difference,max_abs_sigma_difference=max_sigma_difference,passed=True)


if __name__=='__main__':
    assets=sorted(p.stem for p in (ROOT/'data/returns').glob('*.csv'));rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(work,a,m) for m in ['hs','ewma','garch_n','gjr_garch','gjr_t'] for a in assets]
        for f in as_completed(futures):
            row=f.result();rows.append(row);print(row['asset'],row['model'],'PASS',row['max_abs_quantile_difference'],flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'quality/classical_parameter_replay.csv',index=False)
    print('PASS',sum(r['quantile_cells'] for r in rows),'quantile cells')
