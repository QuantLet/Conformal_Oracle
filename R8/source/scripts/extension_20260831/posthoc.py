#!/usr/bin/env python3
"""Recompute post-hoc methods on identical per-pair test dates; archive fits."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from panel_statistics import ROOT,MODELS,ALPHAS,load_pair,qshift,scores,fit_scale,isotonic_quantile,predict_isotonic

SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'Quantlets/CO_gamlss'))
import baseline_gamlss as gm


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rolling(s,w=250,alpha=.01):
    result=np.full(len(s),np.nan);k=min(int(np.ceil((w+1)*(1-alpha))),w)-1
    windows=np.lib.stride_tricks.sliding_window_view(s,w)[:-1]
    result[w:]=np.partition(windows,k,axis=1)[:,k]
    return result


def aci_path(y,q,start,gamma,alpha=.01,w=250):
    out=np.full(len(y),np.nan);state=alpha;s=q-y
    for t in range(start,len(y)):
        shift=np.quantile(s[max(0,t-w):t],1-np.clip(state,.001,.1))
        out[t]=q[t]-shift
        state=float(np.clip(state+gamma*(alpha-float(y[t]<out[t])),.001,.1))
    return out


def work(model,asset):
    warnings.filterwarnings('ignore')
    import lightgbm as lgb
    started=time.monotonic();d,suffix=MODELS[model]
    input_forecast=ROOT/'data'/d/(f'{asset}_{suffix}.parquet' if suffix else f'{asset}.parquet')
    input_return=ROOT/'data/returns'/f'{asset}.csv'
    key=model+'__'+asset;out=ROOT/'posthoc';out.mkdir(exist_ok=True)
    fitfile=out/(key+'.json');dailyfile=out/(key+'.parquet')
    binding=dict(return_sha256=sha(input_return),forecast_sha256=sha(input_forecast),producer_sha256=sha(__file__),
                 statistics_sha256=sha(Path(__file__).with_name('panel_statistics.py')),gamlss_sha256=sha(gm.__file__),
                 packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','lightgbm']})
    if fitfile.exists():
        saved=json.loads(fitfile.read_text());assert saved['binding']==binding and sha(dailyfile)==saved['daily_sha256']
        return saved['metrics'],saved['indication']
    y,forecast=load_pair(model,asset);n=len(y);nc=int(.7*n);test=np.arange(nc,n)
    fits={};metrics=[];indication=[];daily=pd.DataFrame({'r':y[test]},index=forecast.index[test])
    for alpha in ALPHAS:
        q=forecast[f'VaR_{alpha:g}'].to_numpy();shift=qshift(q[:nc]-y[:nc],alpha)
        static=q[test]-shift;roll=q[test]-rolling(q-y,alpha=alpha)[test]
        raw=scores(y[test],q[test],alpha);st=scores(y[test],static,alpha);ro=scores(y[test],roll,alpha)
        cal=scores(y[:nc],q[:nc],alpha)
        row=dict(model=model,asset=asset,alpha=alpha,n_cal=nc,n_test=n-nc,qV=shift,
                 p_kup_cal=cal['p_kup'],TL_cal=cal['TL'],
                 dQS_static=raw['QS']-st['QS'],dQS_roll=raw['QS']-ro['QS'])
        for name,value in [('raw',raw),('static',st),('roll',ro)]:
            row.update({k+'_'+name:v for k,v in value.items()})
        indication.append(row)
        daily[f'raw_{alpha:g}']=q[test];daily[f'static_{alpha:g}']=static;daily[f'rolling_{alpha:g}']=roll
        if alpha!=.01:continue
        methods={'Raw':q[test],'Conformal':static,'rolling':roll}
        fits['static_shift']=shift
        scale,cal_rate=fit_scale(y[:nc],q[:nc]);methods['Scale']=scale*q[test]
        fits['scale']=dict(c=scale,calibration_violation_rate=cal_rate,objective='absolute deviation from alpha',tie_break='smallest absolute log(c)')
        hq=float(np.quantile(y[:nc],alpha));methods['Hist-Quantile']=np.full(len(test),hq);fits['historical_quantile']=hq
        def objective(theta):
            e=y[:nc]-(theta[0]+theta[1]*q[:nc]);return float(np.sum((alpha-(e<0))*e))
        qr=minimize(objective,[0.,1.],method='Nelder-Mead',options={'maxiter':5000,'xatol':1e-8})
        methods['QR-Residual']=qr.x[0]+qr.x[1]*q[test]
        fits['qr']=dict(params=qr.x.tolist(),success=bool(qr.success),loss=float(qr.fun),nit=int(qr.nit))
        levels,values=isotonic_quantile(q[:nc],y[:nc],alpha)
        methods['Isotonic']=predict_isotonic(levels,values,q[test]);fits['isotonic']=dict(x=levels.tolist(),q=values.tolist())
        trials=[]
        for gamma in [.001,.005,.01]:
            pred=aci_path(y[:nc],q[:nc],250,gamma)
            trials.append(dict(gamma=gamma,**scores(y[250:nc],pred[250:nc])))
        gamma=min(trials,key=lambda z:abs(z['pihat']-alpha))['gamma']
        methods['aci']=aci_path(y,q,nc,gamma)[test]
        fits['aci']=dict(gamma=gamma,calibration_trials=trials,window=250,clipping=[.001,.1],selection='prequential past-only calibration coverage')
        qf,v5,v20=gm.make_features(y,q)
        attempts=[];minimizer=gm.minimize
        def audit(*args,**kwargs):
            r=minimizer(*args,**kwargs)
            attempts.append(dict(method=kwargs['method'],x0=np.asarray(args[1]).tolist(),x=r.x.tolist(),fun=float(r.fun),
                                 success=bool(r.success),status=int(r.status),nit=int(r.nit),nfev=int(r.nfev)))
            return r
        gm.minimize=audit
        try:params=gm.fit_gamlss(y[:nc],qf[:nc],v5[:nc],v20[:nc])
        finally:gm.minimize=minimizer
        if params is not None:
            b0,b1,g0,g1,g2,delta,eta=params
            mu=b0+b1*q[test];sigma=np.exp(np.clip(g0+g1*v5[test]+g2*v20[test],-10,10))
            df=np.exp(np.clip(delta,-5,5))+2;xi=np.exp(np.clip(eta,-3,3))
            methods['gamlss']=gm.sst_quantile(alpha,mu,sigma,df,xi)
            fits['gamlss']=dict(params=params.tolist(),df=float(df),xi=float(xi),right_branch=bool(alpha>=1/(1+xi**2)),attempts=attempts)
        else:
            X=np.c_[np.ones(nc),q[:nc]];beta=np.linalg.lstsq(X,y[:nc],rcond=None)[0]
            sd=float(np.std(y[:nc]-X@beta));methods['gamlss']=np.c_[np.ones(len(test)),q[test]]@beta+sd*norm.ppf(alpha)
            fits['gamlss']=dict(fallback_beta=beta.tolist(),fallback_sigma=sd,attempts=attempts)
        X=np.c_[q,v5,v20];nv=max(int(nc*.2),30)
        params=dict(objective='quantile',alpha=alpha,learning_rate=.05,num_leaves=15,min_data_in_leaf=20,
                    feature_fraction=.9,bagging_fraction=.8,bagging_freq=5,seed=20260827,bagging_seed=20260827,
                    feature_fraction_seed=20260827,data_random_seed=20260827,deterministic=True,force_row_wise=True,
                    verbose=-1,num_threads=2)
        train=lgb.Dataset(X[:nc-nv],label=y[:nc-nv]);validation=lgb.Dataset(X[nc-nv:nc],label=y[nc-nv:nc],reference=train)
        gbm=lgb.train(params,train,num_boost_round=500,valid_sets=[validation],callbacks=[lgb.early_stopping(50,verbose=False)])
        methods['gbm']=gbm.predict(X[test],num_iteration=gbm.best_iteration)
        gbmfile=out/(key+'_lightgbm.txt');gbm.save_model(str(gbmfile))
        fits['gbm']=dict(params=params,best_iteration=gbm.best_iteration,model_sha256=sha(gbmfile),n_validation=nv)
        for method,pred in methods.items():
            metrics.append(dict(model=model,asset=asset,method=method,**scores(y[test],pred)))
            daily[method]=pred
        # The historical scale expression is stored only for attribution of
        # the bug fix, not presented as a valid comparator in the new table.
        old_rate=float(np.mean(y[:nc]<q[:nc]));old_c=alpha/old_rate if old_rate>0 else 1.
        fits['legacy_scale_same_inputs']=dict(c=old_c,metrics=scores(y[test],old_c*q[test]))
    daily.to_parquet(dailyfile)
    fitfile.write_text(json.dumps(dict(binding=binding,n_cal=nc,test_first=str(forecast.index[nc].date()),
                                       test_last=str(forecast.index[-1].date()),fits=fits,metrics=metrics,
                                       indication=indication,daily_sha256=sha(dailyfile),
                                       elapsed_seconds=time.monotonic()-started),indent=2)+'\n')
    print(key,'complete',round(time.monotonic()-started,1),'s',flush=True)
    return metrics,indication


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--ready-only',action='store_true');ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args();tasks=[]
    for model,(d,s) in MODELS.items():
        for ret in sorted((ROOT/'data/returns').glob('*.csv')):
            f=ROOT/'data'/d/(f'{ret.stem}_{s}.parquet' if s else f'{ret.stem}.parquet')
            if not f.exists() and not a.ready_only:raise FileNotFoundError(f)
            if f.exists():tasks.append((model,ret.stem))
    results=[];ledger=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(work,*t) for t in tasks]):
            rows,ind=f.result();results.extend(rows);ledger.extend(ind)
    (ROOT/'results').mkdir(exist_ok=True)
    pd.DataFrame(results).sort_values(['model','asset','method']).to_csv(ROOT/'results/posthoc.csv',index=False)
    pd.DataFrame(ledger).sort_values(['model','asset','alpha']).to_csv(ROOT/'results/indication.csv',index=False)
