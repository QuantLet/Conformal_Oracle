#!/usr/bin/env python3
"""Reconstruct post-hoc and dynamic forecasts from archived fit parameters."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm,t as student
import panel_statistics as st


def compare(a,b):
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
    err=float(np.max(np.abs(a-b)));assert np.allclose(a,b,atol=1e-13,rtol=1e-12),(err,a.shape)
    return err


def aci(y,q,start,gamma):
    state=.01;s=q-y;result=[]
    for i in range(start,len(y)):
        threshold=q[i]-np.quantile(s[i-250:i],1-state)
        result.append(threshold)
        state=min(.10,max(.001,state+gamma*(.01-(y[i]<threshold))))
    return np.array(result)


def posthoc(root,assets):
    import lightgbm as lgb
    rows=[];st.ROOT=root
    for model in st.MODELS:
        for asset in assets:
            stem=f'{model}__{asset}';meta=json.loads((root/'posthoc'/f'{stem}.json').read_text());p=meta['fits']
            path=root/'posthoc'/f'{stem}.parquet';assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['daily_sha256']
            d=pd.read_parquet(path);y,f=st.load_pair(model,asset);n=meta['n_cal'];assert len(d)==len(y)-n
            errors=[];cells=0
            for alpha in st.ALPHAS:
                q=f[f'VaR_{alpha:g}'].to_numpy();s=q-y;k=int(np.ceil((n+1)*(1-alpha)))-1
                shift=np.sort(s[:n])[k];kw=int(np.ceil(251*(1-alpha)))-1
                rolling=np.sort(np.lib.stride_tricks.sliding_window_view(s,250)[n-250:-1],axis=1)[:,kw]
                for key,got in [('raw',q[n:]),('static',q[n:]-shift),('rolling',q[n:]-rolling)]:
                    errors.append(compare(got,d[f'{key}_{alpha:g}']));cells+=len(d)
            q=f['VaR_0.01'].to_numpy();expected={
                'Raw':q[n:],'Conformal':q[n:]-p['static_shift'],
                'Scale':p['scale']['c']*q[n:],
                'Hist-Quantile':np.full(len(d),p['historical_quantile']),
                'QR-Residual':p['qr']['params'][0]+p['qr']['params'][1]*q[n:]}
            iso=p['isotonic'];ix=np.searchsorted(iso['x'],q[n:],side='right')-1
            expected['Isotonic']=np.array(iso['q'])[np.clip(ix,0,len(iso['q'])-1)]
            expected['aci']=aci(y,q,n,p['aci']['gamma'])
            v5=pd.Series(y).rolling(5,min_periods=1).std().fillna(0).shift(1).fillna(0).to_numpy()
            v20=pd.Series(y).rolling(20,min_periods=1).std().fillna(0).shift(1).fillna(0).to_numpy()
            gp=p['gamlss']
            if 'params' in gp:
                b0,b1,g0,g1,g2,delta,eta=gp['params'];mu=b0+b1*q[n:]
                sigma=np.exp(np.clip(g0+g1*v5[n:]+g2*v20[n:],-10,10));nu=2+np.exp(np.clip(delta,-5,5));xi=np.exp(np.clip(eta,-3,3))
                prob=.01*(1+xi*xi)/2 if .01<1/(1+xi*xi) else 1-.99*(1+xi*xi)/(2*xi*xi)
                factor=1/xi if .01<1/(1+xi*xi) else xi
                expected['gamlss']=mu+sigma*factor*student.ppf(prob,nu)
            else:
                expected['gamlss']=np.c_[np.ones(len(d)),q[n:]]@np.array(gp['fallback_beta'])+gp['fallback_sigma']*norm.ppf(.01)
            model_file=root/'posthoc'/f'{stem}_lightgbm.txt'
            assert hashlib.sha256(model_file.read_bytes()).hexdigest()==p['gbm']['model_sha256']
            gbm=lgb.Booster(model_file=str(model_file));expected['gbm']=gbm.predict(np.c_[q[n:],v5[n:],v20[n:]],num_iteration=p['gbm']['best_iteration'],num_threads=2)
            for method,got in expected.items():errors.append(compare(got,d[method]));cells+=len(d)
            for metric in meta['metrics']:
                current=st.scores(d.r,d[metric['method']])
                for k,v in current.items():
                    if isinstance(v,str):assert v==metric[k]
                    else:assert np.isclose(v,metric[k],rtol=1e-12,atol=1e-14,equal_nan=True),(model,asset,k)
            rows.append(dict(model=model,asset=asset,cells=cells,max_abs=max(errors)))
            print(model,asset,'PASS',max(errors),flush=True)
    return rows


def dynamic(root,assets):
    import dynamic as dy
    rows=[]
    for asset in assets:
        y=pd.read_csv(root/'data/returns'/f'{asset}.csv',index_col='date',parse_dates=True).log_return.to_numpy()
        for model in ['CAViaR-SAV','CAViaR-AS','GAS-t']:
            p=json.loads((root/'parameters/dynamic'/f'{asset}_{model}.json').read_text());theta=np.array(p['theta'])
            d=pd.read_parquet(root/'data/dynamic'/f'{asset}_{model}.parquet')
            if model=='GAS-t':got=dy.original.gas_sigma(theta,y)*student.ppf(.01,theta[-1])
            else:got=dy.original.caviar_path(theta,y,model.split('-')[1],p['initial_quantile'])
            err=compare(got,d['VaR_0.01']);rows.append(dict(model=model,asset=asset,cells=len(y),max_abs=err))
            print(model,asset,'PASS',err,flush=True)
    return rows


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=st.ROOT);ap.add_argument('--scope',choices=['posthoc','dynamic'],required=True);a=ap.parse_args()
    assets=sorted(p.stem for p in (a.root/'data/returns').glob('*.csv'))
    rows=globals()[a.scope](a.root,assets);(a.root/'quality').mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(a.root/'quality'/f'{a.scope}_parameter_replay.csv',index=False)
    print('PASS',sum(r['cells'] for r in rows),'forecast cells',flush=True)
