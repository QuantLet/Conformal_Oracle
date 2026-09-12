#!/usr/bin/env python3
"""Seeded GARCH experiment with iid unit-variance innovations and 2000 burn-in."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import hashlib
import json
import zlib
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma
from panel_statistics import ROOT

DGPS=['normal','t5','t3','skewt3','mixnormal'];GRID=[500,1000,2000,5000,10000]
ALPHA=.01;FC=.7;REPS=500;BURN=2000;OMEGA=1e-5;A=.1;B=.85


def innovation(kind,shape,rng):
    if kind=='normal':return rng.standard_normal(shape)
    if kind in ['t5','t3']:
        df=int(kind[1:]);return rng.standard_t(df,size=shape)/np.sqrt(df/(df-2))
    if kind=='mixnormal':
        # Population variance: .95*1 + .05*25 = 2.2. No sample demeaning.
        u=rng.uniform(size=shape)
        return np.where(u<.95,rng.normal(0,1,shape),rng.normal(0,5,shape))/np.sqrt(2.2)
    df=3;eta=-.5;c=gamma((df+1)/2)/(np.sqrt(np.pi*(df-2))*gamma(df/2))
    a=4*eta*c*(df-2)/(df-1);b=np.sqrt(1+3*eta**2-a*a);sc=np.sqrt((df-2)/df)
    u=rng.uniform(size=shape);left=u<(1-eta)/2;out=np.empty(shape)
    out[left]=((1-eta)*stats.t.ppf(u[left]/(1-eta),df)*sc-a)/b
    out[~left]=((1+eta)*stats.t.ppf((u[~left]+eta)/(1+eta),df)*sc-a)/b
    return out


def run(kind,T):
    seed=zlib.crc32(f'20260908|{kind}|{T}|iid'.encode())&0xffffffff
    rng=np.random.default_rng(seed);eps=innovation(kind,(REPS,T+BURN),rng)
    r=np.zeros((REPS,T));q=np.zeros_like(r);s2=np.full(REPS,OMEGA/(1-A-B));last=np.zeros(REPS)
    z=stats.norm.ppf(ALPHA)
    for t in range(T+BURN):
        s2=OMEGA+A*last**2+B*s2;last=np.sqrt(s2)*eps[:,t]
        if t>=BURN:r[:,t-BURN]=last;q[:,t-BURN]=np.sqrt(s2)*z
    nc=int(FC*T);k=int(np.ceil((nc+1)*(1-ALPHA)));shift=np.partition(q[:,:nc]-r[:,:nc],k-1,axis=1)[:,k-1]
    y=r[:,nc:];raw=q[:,nc:];cp=raw-shift[:,None]
    raw_hit=y<raw;cp_hit=y<cp
    def qs(v):return np.mean((ALPHA-(y<v))*(y-v),axis=1)
    d=pd.DataFrame(dict(dgp=kind,T=T,rep=np.arange(REPS),seed=seed,qV=shift,
        raw_pi=raw_hit.mean(axis=1),corr_pi=cp_hit.mean(axis=1),raw_QS=qs(raw),corr_QS=qs(cp)))
    d['raw_green']=d.raw_pi<=4/250;d['corr_green']=d.corr_pi<=4/250
    file=ROOT/'results/monte_carlo'/f'{kind}_{T}.csv';d.to_csv(file,index=False)
    print(kind,T,'complete',flush=True)
    return d


def main():
    out=ROOT/'results/monte_carlo';out.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        frames=[f.result() for f in as_completed([pool.submit(run,d,t) for d in DGPS for t in GRID])]
    d=pd.concat(frames).sort_values(['dgp','T','rep']);d.to_csv(out/'replications.csv',index=False)
    rows=[]
    for (kind,T),g in d.groupby(['dgp','T']):
        rows.append(dict(dgp=kind,T=T,n_test=T-int(FC*T),Mean_qV=g.qV.mean(),Std_qV=g.qV.std(ddof=1),
                         Raw_pi=g.raw_pi.mean(),Corr_pi=g.corr_pi.mean(),RawGreen=100*g.raw_green.mean(),
                         CorrGreen=100*g.corr_green.mean(),Raw_QS=g.raw_QS.mean(),Corr_QS=g.corr_QS.mean(),
                         DQS_mean=(g.raw_QS-g.corr_QS).mean(),DQS_se=(g.raw_QS-g.corr_QS).std(ddof=1)/np.sqrt(REPS),
                         share_worse=float((g.corr_QS>g.raw_QS).mean()),qV_mean_se=g.qV.std(ddof=1)/np.sqrt(REPS)))
    pd.DataFrame(rows).to_csv(out/'grid.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        n_reps=REPS,burn_in=BURN,omega=OMEGA,alpha1=A,beta1=B,alpha=ALPHA,calibration_fraction=FC,
        seed_rule='crc32(20260908|dgp|T|iid)',initial_variance=OMEGA/(1-A-B),
        mixture='0.95 N(0,1)+0.05 N(0,25), divided by sqrt(2.2)',
        legacy_boundary='Old path-standardised mixture results are archived and not pooled with this experiment'),indent=2)+'\n')


if __name__=='__main__':main()
