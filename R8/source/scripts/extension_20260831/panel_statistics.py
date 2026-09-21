"""Shared evaluation and correction definitions for the August panel."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2

ROOT=Path(__file__).resolve().parents[3]/'artifacts/extension_20260831'
MODELS={'TimesFM-2.5':('timesfm25',None),'Moirai-2.0':('moirai2',None),
        'Moirai-1.1':('moirai',None),'Lag-Llama':('lagllama',None),
        'GJR-GARCH':('benchmarks','gjr_garch'),'GJR-GARCH-t':('benchmarks','gjr_t'),
        'GARCH-N':('benchmarks','garch_n'),'Hist-Sim':('benchmarks','hs'),'EWMA':('benchmarks','ewma')}
ALPHAS=[.01,.025,.05,.1]


def load_pair(model,asset):
    d,s=MODELS[model];f=ROOT/'data'/d/(f'{asset}_{s}.parquet' if s else f'{asset}.parquet')
    ret=pd.read_csv(ROOT/'data/returns'/f'{asset}.csv',index_col='date',parse_dates=True).log_return
    pred=pd.read_parquet(f)
    if pred.index.has_duplicates or not pred.index.is_monotonic_increasing or not pred.index.isin(ret.index).all():
        raise ValueError(f'Invalid forecast calendar: {model}/{asset}')
    if not np.isfinite(pred.filter(like='VaR_').to_numpy()).all():raise ValueError('Nonfinite forecasts')
    if pred.index[-1]!=ret.index[-1]:raise ValueError('Truncated forecast endpoint')
    return ret.loc[pred.index].to_numpy(),pred


def qshift(scores,alpha=.01):
    x=np.asarray(scores,dtype=float);k=int(np.ceil((len(x)+1)*(1-alpha)))
    if not 1<=k<=len(x):raise ValueError('No finite conformal order statistic at this sample size')
    return float(np.partition(x,k-1)[k-1])


def scores(y,q,alpha=.01):
    y=np.asarray(y);q=np.asarray(q)
    if len(y)<1 or not np.isfinite(y).all() or not np.isfinite(q).all():raise ValueError('Invalid scoring sample')
    h=y<q;n=len(y);v=int(h.sum());p=v/n
    lr_uc=2*(xlogy(v,p/alpha)+xlogy(n-v,(1-p)/(1-alpha)))
    pairs=2*h[:-1].astype(int)+h[1:].astype(int)
    n00,n01,n10,n11=np.bincount(pairs,minlength=4)
    lr_ind=np.nan
    if n00+n01>0 and n10+n11>0:
        p0=n01/(n00+n01);p1=n11/(n10+n11);pall=(n01+n11)/(n-1)
        ll_ind=xlogy(n00,1-p0)+xlogy(n01,p0)+xlogy(n10,1-p1)+xlogy(n11,p1)
        ll_null=xlogy(n00+n10,1-pall)+xlogy(n01+n11,pall)
        lr_ind=max(0.,2*(ll_ind-ll_null))
    scaled=v*250/n
    return dict(n_test=n,viol=v,pihat=p,p_kup=float(chi2.sf(lr_uc,1)),
                p_ind=float(chi2.sf(lr_ind,1)),p_cc=float(chi2.sf(lr_uc+lr_ind,2)),
                lr_uc=float(lr_uc),lr_ind=float(lr_ind),QS=float(np.mean((alpha-h)*(y-q))),
                width=float(np.mean(np.abs(q))),TL='Green' if scaled<=4 else ('Yellow' if scaled<=9 else 'Red'))


def fit_scale(y,q,alpha=.01):
    """Exact finite-breakpoint search of the supplement's coverage objective.

    Includes positive breakpoints, intervening intervals, and c=1. Ties are
    resolved by minimum absolute log(c), retaining the smallest change from 1.
    Direct evaluation verifies the chosen strict-inequality violation count.
    """
    y=np.asarray(y);q=np.asarray(q);nonzero=q!=0
    ratio=y[nonzero]/q[nonzero]
    events=np.unique(ratio[np.isfinite(ratio)&(ratio>0)])
    if not len(events):return 1.,float(np.mean(y<q))
    intervals=np.r_[events[0]/2,events[:-1]+np.diff(events)/2,events[-1]*2]
    candidates=np.unique(np.r_[1.,events,intervals])
    candidates=candidates[np.isfinite(candidates)&(candidates>0)]
    pos=np.sort(y[q>0]/q[q>0]);neg=np.sort(y[q<0]/q[q<0])
    counts=(int(((q==0)&(y<0)).sum())+np.searchsorted(pos,candidates,side='left')
            +len(neg)-np.searchsorted(neg,candidates,side='right'))
    order=np.lexsort((candidates,np.abs(np.log(candidates)),np.abs(counts-len(y)*alpha)))
    best=None
    # Floating arithmetic at exact breakpoints can change strict equality by
    # one ulp. Validate every candidate tied at the optimal theoretical count,
    # plus the next event-count distance, then select using actual violations.
    bound=np.abs(counts[order[0]]-len(y)*alpha)+1
    relevant=order[np.abs(counts[order]-len(y)*alpha)<=bound]
    for i in relevant:
        c=float(candidates[i]);p=float(np.mean(y<c*q));key=(abs(p-alpha),abs(np.log(c)),c)
        if best is None or key<best[0]:best=(key,c,p)
    return best[1],best[2]


def isotonic_quantile(x,y,alpha=.01):
    """Generalised PAV for a nondecreasing pinball-loss fit, with tied x pooled.

    Predictions are right-continuous steps, clipped outside calibration support.
    Use the lower empirical quantile as the deterministic minimiser of a block.
    """
    x=np.asarray(x);y=np.asarray(y);order=np.argsort(x,kind='stable');sx=x[order];sy=y[order]
    levels,starts=np.unique(sx,return_index=True);ends=np.r_[starts[1:],len(sx)]
    blocks=[]
    def quantile(vals):return float(np.quantile(vals,alpha,method='inverted_cdf'))
    for i,(a,b) in enumerate(zip(starts,ends)):
        v=sy[a:b].copy();blocks.append([i,i,v,quantile(v)])
        while len(blocks)>1 and blocks[-2][3]>blocks[-1][3]:
            right=blocks.pop();left=blocks.pop();v=np.r_[left[2],right[2]]
            blocks.append([left[0],right[1],v,quantile(v)])
    fitted=np.empty(len(levels))
    for a,b,vals,q in blocks:fitted[a:b+1]=q
    return levels,fitted


def predict_isotonic(levels,values,x):
    return values[np.clip(np.searchsorted(levels,x,side='right')-1,0,len(values)-1)]
