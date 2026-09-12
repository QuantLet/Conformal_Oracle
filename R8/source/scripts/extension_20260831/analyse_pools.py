#!/usr/bin/env python3
"""Conditional empirical-pool bootstrap and signed forecast perturbation sensitivity.

The pool is a finite reference, not the unknown true predictive quantile.
Perturbations are independent across days and added to an already sampled
forecast. Their dispersion is a sensitivity measure, not an upper bound.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from panel_statistics import ROOT,load_pair,qshift


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    rows=[];prop=[];seeds=[];inputs={};R=200
    out=ROOT/'results/predictive_sampling';out.mkdir(exist_ok=True)
    for model,name in [('moirai','Moirai-1.1'),('lagllama','Lag-Llama')]:
        for asset in ['BTC','SP500']:
            f=next((ROOT/'draws'/model).glob(asset+'*_samples.npy'));inputs[str(f.relative_to(ROOT))]=sha(f)
            pool=np.load(f).astype(float);days,N=pool.shape
            pq=pd.read_parquet(f.with_name(f.name.replace('_samples.npy','.parquet')))
            y,fc=load_pair(name,asset);v=fc['VaR_0.01'].to_numpy();nc=int(.7*len(y));shift=qshift(v[:nc]-y[:nc])
            returns=pd.read_csv(ROOT/'data/returns'/f'{asset}.csv',index_col='date',parse_dates=True).log_return.loc[pq.index].to_numpy()
            ref=np.percentile(pool,1,axis=1);assert (np.abs(ref)>0).all()
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'42:{model}:{asset}'.encode()).digest()[:8],'little'))
            for n in [1000,5000]:
                if n>=N:continue
                qq=np.empty((days,R))
                for d in range(days):
                    ids=rng.integers(0,N,size=(R,n));qq[d]=np.percentile(pool[d,ids],1,axis=1)
                noise=((qq-ref[:,None])/np.abs(ref[:,None])).ravel()
                sd=qq.std(axis=1,ddof=1);bias=qq.mean(axis=1)-ref
                row=dict(model=name,asset=asset,n=n,N_ref=N,days=days,n_cal=nc,qV=shift,
                         sd_over_VaR_median=float(np.median(sd/np.abs(ref))),
                         sd_over_qV_median=float(np.median(sd/abs(shift))),
                         bias_over_VaR_median=float(np.median(bias/np.abs(ref))),
                         bias_over_absqV_median=float(np.median(bias/abs(shift))),
                         qV_over_VaR=abs(shift)/np.median(np.abs(ref)),
                         pool_violations=int((returns<ref).sum()),expected_redrawn_violations=float((returns[:,None]<qq).sum()/R))
                rows.append(row);draws=[];legacy=[]
                for rep in range(R):
                    e=rng.choice(noise,size=nc,replace=True)
                    draws.append(qshift(v[:nc]+np.abs(v[:nc])*e-y[:nc]))
                    legacy.append(qshift(v[:nc]*(1+e)-y[:nc]))
                draws=np.asarray(draws)
                prop.append(dict(model=name,asset=asset,n=n,n_cal=nc,qV=shift,
                                 noise_mean=float(noise.mean()),noise_sd=float(noise.std(ddof=1)),
                                 sd_over_qV=float(draws.std(ddof=1)/abs(shift)),
                                 p95_absdev_over_qV=float(np.percentile(np.abs(draws-shift),95)/abs(shift)),
                                 mean_displacement=float(draws.mean()-shift),
                                 legacy_sign_sd_over_qV=float(np.std(legacy,ddof=1)/abs(shift))))
                np.savez_compressed(out/f'{model}_{asset}_{n}.npz',pool_quantile=ref,bootstrap_quantiles=qq,
                                    propagated_shifts=draws,legacy_sign_shifts=legacy)
            if model=='lagllama':
                qs=np.stack([np.percentile(pool[:,b:b+1000],1,axis=1) for b in range(0,N,1000)],axis=1)
                seeds.append(dict(asset=asset,days=days,seeds=N//1000,
                                  sd_over_VaR_median=float(np.median(qs.std(axis=1,ddof=1)/np.abs(ref))),
                                  sd_over_qV_median=float(np.median(qs.std(axis=1,ddof=1)/abs(shift)))))
            print(name,asset,'complete',flush=True)
    pd.DataFrame(rows).to_csv(out/'draws.csv',index=False);pd.DataFrame(prop).to_csv(out/'propagation.csv',index=False)
    pd.DataFrame(seeds).to_csv(out/'seeds.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(producer_sha256=sha(__import__('pathlib').Path(__file__)),
        R=R,bootstrap='with replacement within each fixed-date empirical pool',
        propagation='q + abs(q)*e; independent across days; sensitivity, not upper bound',inputs=inputs),indent=2)+'\n')


if __name__=='__main__':main()
