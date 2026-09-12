"""Matched correction families on the original complete calibration split."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from controlled_comparisons import PROJECT,ROOT,MODELS,ALPHAS,OUT,candidates,past_volatility,sha,scores


def work(model,asset):
    started=time.monotonic()
    directory,suffix=MODELS[model]
    path=ROOT/'data'/directory/(f'{asset}_{suffix}.parquet' if suffix else f'{asset}.parquet')
    if model=='Lag-Llama':path=ROOT/'data/lagllama'/f'{asset}.parquet'
    retpath=ROOT/'data/returns'/f'{asset}.csv'
    folder=OUT/f'{model}__{asset}'/'full';folder.mkdir(parents=True,exist_ok=True)
    binding={'producer_sha256':sha(__file__),'candidate_sha256':sha(Path(__file__).with_name('controlled_comparisons.py')),
             'forecast_sha256':sha(path),'return_sha256':sha(retpath)}
    done=folder/'complete.json'
    if done.exists():
        record=json.loads(done.read_text());assert record['binding']==binding
        assert all(sha(folder/k)==v for k,v in record['outputs'].items())
        return model,asset,'verified'
    ret=pd.read_csv(retpath,index_col='date',parse_dates=True).log_return
    forecast=pd.read_parquet(path);y=ret.loc[forecast.index].to_numpy()
    sigma=past_volatility(ret,forecast.index);nc=int(.7*len(y))
    high=sigma[nc:]>np.quantile(sigma[:nc],.9)
    daily=pd.DataFrame({'r':y[nc:],'sigma':sigma[nc:],'high_volatility':high},index=forecast.index[nc:])
    rows=[];fits={}
    for alpha in ALPHAS:
        q=forecast[f'VaR_{alpha:g}'].to_numpy()
        predictions,params=candidates(y[:nc],q[:nc],sigma[:nc],q[nc:],sigma[nc:],alpha)
        fits[f'{alpha:g}']=params
        for method,pred in predictions.items():
            daily[f'{alpha:g}/{method}']=pred
            for state,mask in [('All',np.ones(len(y)-nc,dtype=bool)),('High',high),('Other',~high)]:
                if mask.sum():rows.append(dict(model=model,asset=asset,alpha=alpha,n_cal=nc,method=method,state=state,**scores(y[nc:][mask],pred[mask],alpha)))
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    daily.to_parquet(folder/'daily.parquet')
    (folder/'parameters.json').write_text(json.dumps(fits,indent=2)+'\n')
    done.write_text(json.dumps({'binding':binding,'outputs':{n:sha(folder/n) for n in ['metrics.csv','daily.parquet','parameters.json']},
                               'elapsed_seconds':time.monotonic()-started},indent=2)+'\n')
    return model,asset,round(time.monotonic()-started,1)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+');a=ap.parse_args()
    models=a.models or list(MODELS)
    assets=sorted(p.stem for p in (ROOT/'data/returns').glob('*.csv'))
    with ProcessPoolExecutor(max_workers=3) as pool:
        tasks=[pool.submit(work,m,s) for m in models for s in assets]
        for task in as_completed(tasks):print(*task.result(),flush=True)
