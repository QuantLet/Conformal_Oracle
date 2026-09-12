#!/usr/bin/env python3
"""Reconstruct every POT/FHS forecast from its saved daily fit, without refitting."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
import warnings
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import genpareto
from panel_statistics import ROOT,scores


def check(asset):
    warnings.filterwarnings('ignore')
    ret=pd.read_csv(ROOT/'data/returns'/f'{asset}.csv',index_col='date',parse_dates=True).log_return
    p=ROOT/'evt_fhs';meta=json.loads((p/f'{asset}.json').read_text())
    for name,key in [(f'{asset}.parquet','daily_sha256'),(f'{asset}_parameters.parquet','fit_sha256')]:
        assert hashlib.sha256((p/name).read_bytes()).hexdigest()==meta[key]
    d=pd.read_parquet(p/f'{asset}.parquet');fits=pd.read_parquet(p/f'{asset}_parameters.parquet')
    assert d.index.equals(fits.index) and d.index.equals(ret.index[int(.7*len(ret)):])
    weights=.06*.94**np.arange(249,-1,-1);weights/=weights.sum()
    result=[];previous=None;maximum=0.
    for date,fit in fits.iterrows():
        t=ret.index.get_loc(date);window=ret.iloc[t-250:t].to_numpy()
        assert fit.context_start==ret.index[t-250] and fit.context_end==ret.index[t-1]
        if not fit.garch_fallback:
            model=arch_model(window*100,vol='GARCH',p=1,q=1,mean='Zero',dist='normal',rescale=False)
            fixed=model.fix(fit[['omega','alpha[1]','beta[1]']].to_numpy(dtype=float))
            sigma=np.sqrt(fixed.forecast(horizon=1,reindex=False).variance.values[-1,0])/100
            sd=np.asarray(fixed.conditional_volatility)/100
            z=window/np.where(sd>0,sd,np.nan)
        else:
            sigma=np.sqrt(np.sum(weights*window**2))
            if not np.isfinite(sigma) or sigma<=0:sigma=previous
            z=window/sigma
        previous=sigma;z=z[np.isfinite(z)]
        u=float(np.quantile(-z,.95));exc=-z[-z>u]-u
        assert len(z)==fit.n_residuals and len(exc)==fit.n_exceedances
        assert np.isclose(sigma,fit.sigma_next,rtol=1e-12,atol=1e-14)
        assert np.isclose(u,fit.tail_threshold,rtol=1e-12,atol=1e-14)
        fhs=sigma*np.quantile(z,.01)
        evt=fhs if fit.gpd_fallback else -sigma*genpareto.isf(len(z)/len(exc)*.01,fit.xi,loc=u,scale=fit.beta)
        expected=np.array([fhs,evt]);stored=d.loc[date,['FHS','EVT_POT']].to_numpy(dtype=float)
        maximum=max(maximum,float(np.max(np.abs(expected-stored))))
        assert np.allclose(expected,stored,rtol=1e-12,atol=1e-13),(asset,date,expected,stored)
        assert d.loc[date,'r']==ret.loc[date]
    for metric in meta['metrics']:
        now=scores(d.r,d['EVT_POT' if metric['method']=='EVT-POT' else 'FHS'])
        for key,value in now.items():
            if isinstance(value,str):assert value==metric[key]
            else:assert np.isclose(value,metric[key],atol=1e-14,rtol=1e-12,equal_nan=True)
    print(asset,'PASS',len(d)*2,maximum,flush=True)
    return dict(asset=asset,cells=2*len(d),max_abs=maximum)


if __name__=='__main__':
    assets=sorted(p.stem for p in (ROOT/'data/returns').glob('*.csv'))
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(check,assets))
    pd.DataFrame(rows).to_csv(ROOT/'quality/evt_fhs_parameter_replay.csv',index=False)
    print('PASS',sum(r['cells'] for r in rows),'forecast cells',flush=True)
