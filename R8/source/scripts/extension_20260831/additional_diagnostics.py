#!/usr/bin/env python3
"""Exploratory quotation and 2026-period sensitivities; additional DQ diagnostics."""
import numpy as np
import pandas as pd
from scipy.stats import chi2
from panel_statistics import ROOT,MODELS,scores
from analyse_panel import METHODS


def dq(y,q,alpha=.01,lags=4):
    y=np.asarray(y);q=np.asarray(q);hit=(y<q).astype(float)-alpha
    X=np.c_[np.ones(len(y)-lags),*[hit[lags-j:-j] for j in range(1,lags+1)],q[lags:]]
    # Column scaling changes neither the projection nor the statistic.
    norms=np.linalg.norm(X,axis=0);X=X/np.where(norms>0,norms,1)
    beta,_,rank,_=np.linalg.lstsq(X,hit[lags:],rcond=None)
    if rank<X.shape[1]:return dict(DQ=np.nan,p_DQ=np.nan,DQ_rank=int(rank))
    fitted=X@beta;stat=float(fitted@fitted/(alpha*(1-alpha)))
    return dict(DQ=stat,p_DQ=float(chi2.sf(stat,rank)),DQ_rank=int(rank))


def main():
    rows=[];diag=[]
    for model in MODELS:
        for ret in sorted((ROOT/'data/returns').glob('*.csv')):
            asset=ret.stem;d=pd.read_parquet(ROOT/'posthoc'/f'{model}__{asset}.parquet')
            for method in METHODS:
                diag.append(dict(model=model,asset=asset,method=method,**dq(d.r,d[method])))
                masks={'All':np.ones(len(d),bool),'2026 only':d.index>=pd.Timestamp('2026-01-01')}
                if asset!='NATGAS':masks['Exclude NATGAS']=np.ones(len(d),bool)
                if asset not in ['GOLD','NATGAS','WTI']:masks['Exclude individual futures']=np.ones(len(d),bool)
                masks['Omit NATGAS roll-date loss']=(d.index!=pd.Timestamp('2026-01-29')) if asset=='NATGAS' else np.ones(len(d),bool)
                for variant,mask in masks.items():
                    rows.append(dict(model=model,asset=asset,method=method,variant=variant,**scores(d.loc[mask,'r'],d.loc[mask,method])))
    df=pd.DataFrame(rows);df.to_csv(ROOT/'results/quotation_period_sensitivity.csv',index=False)
    pd.DataFrame(diag).to_csv(ROOT/'results/dq_diagnostics.csv',index=False)
    summary=df.groupby(['variant','method']).agg(n=('QS','size'),QS=('QS','mean'),pi=('pihat','mean')).reset_index()
    summary.to_csv(ROOT/'results/quotation_period_summary.csv',index=False)
    print(summary.pivot(index='method',columns='variant',values='QS').mul(1e4).round(3).to_string(),flush=True)


if __name__=='__main__':main()
