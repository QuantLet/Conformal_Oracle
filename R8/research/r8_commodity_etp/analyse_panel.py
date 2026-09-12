#!/usr/bin/env python3
"""Complete-vintage comparisons; never accepts a partial main panel."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from panel_scope import ROOT, MODELS
from panel_statistics import ALPHAS,load_pair,qshift,scores

from panel_scope import ROOT, MODELS, CLASSES, CLASS
import panel_statistics
panel_statistics.ROOT = ROOT

METHODS=['Raw','Conformal','Scale','QR-Residual','Isotonic','Hist-Quantile','aci','rolling','gamlss','gbm']


def loss(y,q,alpha=.01):return (alpha-(y<q))*(y-q)


def rollshift(s,w,alpha=.01):
    k=int(np.ceil((w+1)*(1-alpha)))-1;out=np.full(len(s),np.nan)
    out[w:]=np.partition(np.lib.stride_tricks.sliding_window_view(s,w)[:-1],k,axis=1)[:,k]
    return out


def bootstrap(frames):
    observed=pd.DatetimeIndex(sorted(set().union(*(set(d.index) for d in frames))))
    dates=pd.date_range(observed.min(),observed.max(),freq='D')
    names=METHODS+['Gate-static','Gate-rolling'];P=len(frames);T=len(dates)
    losses=np.zeros((T,P,len(names)));available=np.zeros((T,P))
    for j,d in enumerate(frames):
        idx=dates.get_indexer(d.index);available[idx,j]=1
        for k,m in enumerate(names):losses[idx,j,k]=loss(d.r.to_numpy(),d[m].to_numpy())
    point=(losses.sum(axis=0)/available.sum(axis=0)[:,None]).mean(axis=0)
    rows=[];rng=np.random.default_rng(20260908);B=999
    for L in [20,60]:
        draws=[]
        for first in range(0,B,50):
            n=min(50,B-first);counts=np.empty((n,T))
            for b in range(n):
                starts=rng.integers(0,T,size=int(np.ceil(T/L)))
                ix=((starts[:,None]+np.arange(L))%T).ravel()[:T]
                counts[b]=np.bincount(ix,minlength=T)
            den=counts@available;assert (den>0).all()
            num=(counts@losses.reshape(T,-1)).reshape(n,P,len(names))
            draws.append((num/den[:,:,None]).mean(axis=1))
        draws=np.concatenate(draws)
        for m in names:
            if m=='Conformal':continue
            k=names.index(m);delta=draws[:,k]-draws[:,names.index('Conformal')]
            rows.append(dict(comparator=m,reference='Conformal',block_calendar_days=L,B=B,
                             difference=(point[k]-point[names.index('Conformal')])*1e4,
                             lower=np.quantile(delta,.025)*1e4,upper=np.quantile(delta,.975)*1e4))
        for m,reference in [('Gate-static','Raw'),('Gate-rolling','Raw'),('Gate-static','Conformal'),('Gate-rolling','rolling')]:
            k=names.index(m);j=names.index(reference);delta=draws[:,k]-draws[:,j]
            rows.append(dict(comparator=m,reference=reference,block_calendar_days=L,B=B,
                             difference=(point[k]-point[j])*1e4,lower=np.quantile(delta,.025)*1e4,upper=np.quantile(delta,.975)*1e4))
        np.savez_compressed(ROOT/'results'/f'bootstrap_L{L}.npz',means=draws,methods=names)
    pd.DataFrame(rows).to_csv(ROOT/'results/paired_loss_intervals.csv',index=False)


def main():
    assets=sorted(CLASS);ledger=pd.read_csv(ROOT/'results/indication.csv');base=pd.read_csv(ROOT/'results/posthoc.csv')
    assert len(ledger)==len(MODELS)*len(assets)*4 and len(base)==len(MODELS)*len(assets)*10
    assert set(zip(ledger.model,ledger.asset))=={(m,a) for m in MODELS for a in assets}
    windows=[];gaps=[];master=[];policy=[];strata=[];conventions=[];frames=[];common=[];fitqc=[]
    for asset in assets:
        paths={};data={}
        for model in MODELS:
            y,pred=load_pair(model,asset);q=pred['VaR_0.01'].to_numpy();nc=int(.7*len(y));s=q-y
            meta=json.loads((ROOT/'posthoc'/f'{model}__{asset}.json').read_text())
            d=pd.read_parquet(ROOT/'posthoc'/f'{model}__{asset}.parquet');assert d.index.equals(pred.index[nc:])
            row=ledger[(ledger.model==model)&(ledger.asset==asset)&(ledger.alpha==.01)].iloc[0]
            gate=bool(row.p_kup_cal<.05 or row.TL_cal!='Green')
            d['Gate-static']=d['Conformal'] if gate else d.Raw;d['Gate-rolling']=d['rolling'] if gate else d.Raw
            frames.append(d);paths[model]=d;data[model]=(y,pred,nc)
            sd=float(np.std(y[:nc],ddof=1));assert sd>0
            for method in METHODS+['Gate-static','Gate-rolling']:
                z=scores(d.r,d[method]);policy.append(dict(model=model,asset=asset,asset_class=CLASS[asset],method=method,applied=gate,**z))
                strata.append(dict(model=model,asset=asset,asset_class=CLASS[asset],method=method,QS=z['QS'],normalised_QS=z['QS']/sd))
            sh=qshift(s[:nc]);emp=float(np.quantile(s[:nc],.99));plain=scores(y[nc:],q[nc:]-emp)
            conventions.append(dict(model=model,asset=asset,conformal_shift=sh,empirical_shift=emp,shift_difference=sh-emp,
                                    n_cal=nc,effective_rank=int(np.ceil((nc+1)*.99))/nc,
                                    violations_conformal=int(row.viol_static),violations_empirical=plain['viol']))
            rho=float(pd.Series(s[:nc]).autocorr());assert np.isfinite(rho) and abs(rho)<1
            g=5 if abs(rho)<=1e-12 else int(np.ceil(1.1*np.log(nc)/abs(np.log(abs(rho)))))
            memory=600 if model=='EWMA' else (250 if MODELS[model][0]=='benchmarks' else 512)
            for tag,gap in [('Contiguous',0),('Logarithmic',g),('Full proxy',memory+g)]:
                z=scores(y[nc+gap:],q[nc+gap:]-sh)
                gaps.append(dict(model=model,asset=asset,variant=tag,n_cal=nc,n_original_test=len(y)-nc,
                                 gap=gap,memory_proxy=memory,rho=rho,qV=sh,**z))
            for w in [125,250,500]:
                shift=rollshift(s,w)
                windows.append(dict(model=model,asset=asset,w=w,k=int(np.ceil((w+1)*.99)),effective_rank=np.ceil((w+1)*.99)/w,
                                    **scores(y[nc:],q[nc:]-shift[nc:])))
            raw=scores(y[nc:],q[nc:]);cor=scores(y[nc:],q[nc:]-sh)
            master.append(dict(model=model,asset=asset,n_cal=nc,qV=sh,R=abs(sh)/raw['width'],
                               **{k+'_raw':v for k,v in raw.items()},**{k+'_static':v for k,v in cor.items()}))
            if MODELS[model][0] in ['timesfm25','moirai2']:
                par=pd.read_parquet(ROOT/'parameters'/MODELS[model][0]/f'{asset}.parquet')
                fitqc.append(dict(model=model,asset=asset,n=len(par),nonconverged=int((~par.success).sum()),
                                  native_crossings=int(par.native_crossings.sum()),nu_min=par.nu_used.min(),nu_max=par.nu_used.max(),
                                  max_absolute_quantile=float(np.max(np.abs(q))),sigma_min=par.sigma_used.min()))
        ret=pd.read_csv(ROOT/'data/returns'/f'{asset}.csv',index_col='date',parse_dates=True).log_return
        for model in ['CAViaR-SAV','CAViaR-AS','GAS-t']:
            f=pd.read_parquet(ROOT/'data/dynamic'/f'{asset}_{model}.parquet');nc=int(.7*len(ret));q=f['VaR_0.01'].to_numpy();y=ret.to_numpy()
            sh=qshift(q[:nc]-y[:nc]);d=pd.DataFrame({'r':y[nc:],'Raw':q[nc:],'Conformal':q[nc:]-sh},index=ret.index[nc:]);paths[model]=d
            raw=scores(d.r,d.Raw);cor=scores(d.r,d.Conformal)
            master.append(dict(model=model,asset=asset,n_cal=nc,qV=sh,R=abs(sh)/raw['width'],
                               **{k+'_raw':v for k,v in raw.items()},**{k+'_static':v for k,v in cor.items()}))
        dates=paths[next(iter(paths))].index
        for d in paths.values():dates=dates.intersection(d.index)
        for model,d in paths.items():
            for method in ['Raw','Conformal']:
                common.append(dict(model=model,asset=asset,method=method,first=str(dates[0].date()),last=str(dates[-1].date()),**scores(d.loc[dates,'r'],d.loc[dates,method])))
        # Dedicated VaR models enter once per asset on this common support.
        ev=pd.read_parquet(ROOT/'evt_fhs'/f'{asset}.parquet')
        for method in ['EVT-POT','FHS']:
            col={'EVT-POT':'EVT_POT','FHS':'FHS'}[method]
            common.append(dict(model=method,asset=asset,method='Raw',first=str(dates[0].date()),last=str(dates[-1].date()),**scores(ret.loc[dates],ev.loc[dates,col])))
    for name,rows in [('windows',windows),('gaps',gaps),('master',master),('policy',policy),('strata',strata),('order_convention',conventions),('common_support',common),('grid_fit_quality',fitqc)]:
        pd.DataFrame(rows).to_csv(ROOT/'results'/f'{name}.csv',index=False)
    strata=pd.DataFrame(strata);sensitivity=[]
    for excluded in [None]+list(CLASSES):
        d=strata if excluded is None else strata[strata.asset_class!=excluded]
        for m,g in d.groupby('method'):
            sensitivity.append(dict(excluded_class=excluded or 'None',method=m,n_pairs=len(g),QS=g.QS.mean(),normalised_QS=g.normalised_QS.mean()))
    pd.DataFrame(sensitivity).to_csv(ROOT/'results/class_sensitivity.csv',index=False)
    bootstrap(frames)
    print('Complete replacement panel:', len(MODELS)*len(assets), 'pairs; sensitivity and paired intervals saved', flush=True)


if __name__=='__main__':main()
