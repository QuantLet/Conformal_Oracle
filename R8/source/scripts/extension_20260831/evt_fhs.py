#!/usr/bin/env python3
"""Rolling GARCH-filtered POT/FHS, with complete daily fit provenance.

Corrects the ndarray/.values incompatibility in the original pre-filter,
handles xi=0 by the GPD inverse survival function, and retains its fallback
thresholds. Shared panel_statistics supplies actual Kupiec LR boundary cases.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import genpareto
from panel_statistics import ROOT,scores


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def work(asset):
    warnings.filterwarnings('ignore');started=time.monotonic()
    inp=ROOT/'data/returns'/f'{asset}.csv';out=ROOT/'evt_fhs';out.mkdir(exist_ok=True)
    dailyfile=out/f'{asset}.parquet';fitfile=out/f'{asset}_parameters.parquet';done=out/f'{asset}.json'
    binding=dict(input_sha256=sha(inp),producer_sha256=sha(__file__),
                 statistics_sha256=sha(Path(__file__).with_name('panel_statistics.py')),
                 packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','arch']})
    if done.exists():
        old=json.loads(done.read_text());assert old['binding']==binding
        assert old['daily_sha256']==sha(dailyfile) and old['fit_sha256']==sha(fitfile)
        return old['metrics']
    ret=pd.read_csv(inp,index_col='date',parse_dates=True).log_return;r=ret.to_numpy();start=int(.7*len(r))
    rows=[];params=[];previous_sigma=None
    weights=.06*.94**np.arange(249,-1,-1);weights/=weights.sum()
    for t in range(start,len(r)):
        window=r[t-250:t];p=dict(date=ret.index[t],context_start=ret.index[t-250],context_end=ret.index[t-1])
        try:
            model=arch_model(window*100,vol='GARCH',p=1,q=1,mean='Zero',dist='normal',rescale=False)
            fit=model.fit(disp='off',show_warning=False)
            p.update({k:float(v) for k,v in fit.params.items()});p['convergence_flag']=int(fit.convergence_flag)
            sigma_window=np.asarray(fit.conditional_volatility)/100
            sigma_next=float(np.sqrt(fit.forecast(horizon=1,reindex=False).variance.values[-1,0]))/100
            if not np.isfinite(sigma_next) or sigma_next<=0:raise ValueError('Nonpositive/nonfinite forecast variance')
            z=window/np.where(sigma_window>0,sigma_window,np.nan);p['garch_fallback']=False
        except Exception as e:
            p.update(garch_fallback=True,garch_error=type(e).__name__+': '+str(e))
            sigma_next=float(np.sqrt(np.sum(weights*window**2)))
            if not np.isfinite(sigma_next) or sigma_next<=0:
                if previous_sigma is None:raise ValueError(f'{asset}: no defined volatility fallback')
                sigma_next=previous_sigma;p['carried_previous_sigma']=True
            z=window/sigma_next
        previous_sigma=sigma_next;z=z[np.isfinite(z)]
        if len(z)<50:raise ValueError(f'{asset}/{ret.index[t]} insufficient residuals')
        q_fhs=sigma_next*float(np.quantile(z,.01));losses=-z;u=float(np.quantile(losses,.95));exc=losses[losses>u]-u
        p.update(sigma_next=sigma_next,tail_threshold=u,n_residuals=len(z),n_exceedances=len(exc),gpd_fallback=False)
        q_evt=q_fhs
        try:
            if len(exc)<10:raise ValueError('Fewer than ten threshold exceedances')
            xi,loc,beta=genpareto.fit(exc,floc=0)
            p.update(xi=float(xi),beta=float(beta))
            if not (-.5<=xi<=.5 and beta>0):raise ValueError('GPD parameter outside retained admissible range')
            q_evt=-sigma_next*float(genpareto.isf(len(z)/len(exc)*.01,xi,loc=u,scale=beta))
            if not np.isfinite(q_evt) or q_evt>=0 or q_evt < -1:raise ValueError('POT quantile outside retained admissible range')
        except Exception as e:
            p.update(gpd_fallback=True,gpd_error=type(e).__name__+': '+str(e));q_evt=q_fhs
        rows.append(dict(date=ret.index[t],r=r[t],FHS=q_fhs,EVT_POT=q_evt));params.append(p)
    daily=pd.DataFrame(rows).set_index('date');daily.to_parquet(dailyfile)
    fits=pd.DataFrame(params).set_index('date');fits.to_parquet(fitfile)
    result=[dict(asset=asset,method=label,**scores(daily.r,daily[col])) for col,label in [('FHS','FHS'),('EVT_POT','EVT-POT')]]
    done.write_text(json.dumps(dict(binding=binding,daily_sha256=sha(dailyfile),fit_sha256=sha(fitfile),
                                    metrics=result,garch_fallbacks=int(fits.garch_fallback.sum()),
                                    gpd_fallbacks=int(fits.gpd_fallback.sum()),n_test=len(daily)),indent=2)+'\n')
    print(asset,'EVT/FHS complete',round(time.monotonic()-started,1),'s',flush=True)
    return result


if __name__=='__main__':
    assets=sorted(p.stem for p in (ROOT/'data/returns').glob('*.csv'));records=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(work,a) for a in assets]):records.extend(f.result())
    pd.DataFrame(records).sort_values(['method','asset']).to_csv(ROOT/'results/evt_fhs.csv',index=False)
