"""Execute the declared 216-pair research extension without changing R8."""
import os
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import methods as m

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'research/r8_review'))
from controlled_comparisons import ROOT,MODELS,OUT as REFERENCES,past_volatility,scores
from posthoc import rolling
OUT=PROJECT/'artifacts/r8_decision'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(model,asset):
    directory,suffix=MODELS[model]
    fp=ROOT/'data'/directory/(f'{asset}_{suffix}.parquet' if suffix else f'{asset}.parquet')
    if model=='Lag-Llama':fp=PROJECT/'artifacts/review_20260909/calendar/data/lagllama'/f'{asset}.parquet'
    rp=ROOT/'data/returns'/f'{asset}.csv'
    ret=pd.read_csv(rp,index_col='date',parse_dates=True).log_return
    forecast=pd.read_parquet(fp)
    assert forecast.index.is_unique and forecast.index.is_monotonic_increasing
    assert forecast.index[-1]==ret.index[-1]
    y=ret.loc[forecast.index].to_numpy();q=forecast['VaR_0.01'].to_numpy()
    sigma=past_volatility(ret,forecast.index)
    assert np.isfinite(np.c_[y,q,sigma]).all()
    ref=REFERENCES/f'{model}__{asset}'
    source_files=[fp,rp,ref/'full/daily.parquet',ref/'policy_daily.parquet',
                  ROOT/'posthoc'/f'{model}__{asset}.parquet',
                  Path(__file__),Path(m.__file__),Path(__file__).with_name('PROTOCOL.md'),
                  PROJECT/'research/r8_review/controlled_comparisons.py',
                  PROJECT/'source/scripts/extension_20260831/panel_statistics.py',
                  PROJECT/'source/scripts/extension_20260831/posthoc.py']
    binding={str(f.relative_to(PROJECT)):sha(f) for f in source_files}
    return y,q,sigma,forecast.index,ref,binding


def compute(y,q,sigma,nc,key):
    v=max(1000,int(.7*nc));assert nc-v>=250
    fitted,selection=m.select_state(y,q,sigma,v,nc)
    preds={'State-L1':m.predict_state(q[nc:],sigma[nc:],fitted),
           'State-L1-clipped':m.predict_state(q[nc:],sigma[nc:],fitted,clip=True)}
    params={'inner_split':v,'calibration_stop':nc,'state':{'fit':fitted,**selection}}
    for normalised,name in [(False,'POT-Shift'),(True,'POT-Vol')]:
        fitted,selection=m.select_pot(y,q,sigma,v,nc,normalised)
        preds[name]=q[nc:]-fitted['quantile']*(sigma[nc:] if normalised else 1.)
        params[name]={'fit':fitted,**selection}
    # The gate holds its fitted static functions fixed after inner validation.
    shift=m.cp(q[:v]-y[:v]);vol=m.weighted_quantile((q[:v]-y[:v])/sigma[:v],sigma[:v],.99)
    rolls=rolling(q-y,w=500)
    paths={'Raw':q,'Inner-Shift':q-shift,'Inner-Vol':q-vol*sigma,'Rolling500':q-rolls}
    validation={name:m.loss(y[v:nc],path[v:nc]) for name,path in paths.items()}
    gate=m.loss_gate(validation,key)
    preds['Loss-gate']=paths[gate['selected']][nc:]
    preds['Past-minimum']=paths[gate['past_minimum_selected']][nc:]
    params['gate']={**gate,'shift':shift,'vol_coefficient':vol,'static_refit_after_validation':False}
    projected=m.dtaci(q-y,q,projected=True)
    original=m.dtaci(q-y,q,projected=False)
    for name,result in [('projected',projected),('unprojected',original)]:
        params['dtaci_'+name]=result['meta']
    seeded,choices=m.mixture_path(projected,key)
    preds['DtACI-projected-seed']=seeded[nc:]
    mixloss=np.sum(projected['probabilities'][nc:]*m.loss(y[nc:,None],projected['predictions'][nc:]),axis=1)
    mixhits=np.sum(projected['probabilities'][nc:]*(y[nc:,None]<projected['predictions'][nc:]),axis=1)
    mixwidth=np.sum(projected['probabilities'][nc:]*np.abs(projected['predictions'][nc:]),axis=1)
    mixthreshold=np.sum(projected['probabilities'][nc:]*projected['predictions'][nc:],axis=1)
    seed_rows=[]
    for i in range(8):
        path,_=m.mixture_path(projected,key,i)
        seed_rows.append({'replicate':i,'seed':m.seed_for(key,i),**scores(y[nc:],path[nc:])})
    original_path,_=m.mixture_path(original,key)
    infmask=~np.isfinite(original['predictions'][nc:])
    params['dtaci_unprojected_audit']={
        'test_days_any_nonfinite_expert':int(infmask.any(axis=1).sum()),
        'mean_nonfinite_output_probability':float(np.mean(np.sum(original['probabilities'][nc:]*infmask,axis=1))),
        'primary_path_nonfinite_forecasts':int((~np.isfinite(original_path[nc:])).sum()),
        'expected_pinball_loss':'infinite' if infmask.any() else 'finite'}
    mixtures={'loss':mixloss,'hits':mixhits,'width':mixwidth,'mean_threshold':mixthreshold}
    return preds,params,projected,original,mixtures,seed_rows


def work(model,asset,replay=False):
    start=time.monotonic();key=f'{model}__{asset}'
    y,q,sigma,index,ref,binding=load(model,asset);nc=int(.7*len(y))
    folder=OUT/('replay' if replay else 'pairs')/key;folder.mkdir(parents=True,exist_ok=True)
    done=folder/'complete.json'
    if done.exists():
        old=json.loads(done.read_text());assert old['binding']==binding
        assert all(sha(folder/f)==h for f,h in old['outputs'].items())
        return key,'verified'
    pred,params,proj,orig,mix,seeds=compute(y,q,sigma,nc,key)
    full=pd.read_parquet(ref/'full/daily.parquet')
    policy=pd.read_parquet(ref/'policy_daily.parquet');policy=policy[policy.origin=='Original']
    legacy=pd.read_parquet(ROOT/'posthoc'/f'{key}.parquet')
    for frame in (full,policy,legacy):
        assert frame.index.equals(index[nc:])
        assert np.array_equal(frame.r.to_numpy(),y[nc:])
    assert np.array_equal(full['0.01/Raw'].to_numpy(),q[nc:])
    assert np.array_equal(full.sigma.to_numpy(),sigma[nc:])
    for name in ['Raw','Shift-CP','Shift-ERM','Vol-CP','Vol-ERM','State2-ERM','State4-ERM']:
        pred[name]=full[f'0.01/{name}'].to_numpy()
    for name in ['Rolling250','Rolling500','Selected-rolling','Gate-selected-rolling']:
        pred[name]=policy[name].to_numpy()
    pred['ACI-existing']=legacy.aci.to_numpy()
    daily=pd.DataFrame({'r':y[nc:],'sigma':sigma[nc:]},index=index[nc:]);daily.index.name='date'
    rows=[]
    for name,path in pred.items():
        assert np.isfinite(path).all(),(key,name)
        daily[name]=path
        rows.append({'model':model,'asset':asset,'method':name,'calibration_scale':float(np.std(y[:nc],ddof=1)),**scores(y[nc:],path)})
    for name,value in mix.items():daily[f'DtACI-expected/{name}']=value
    rows.append({'model':model,'asset':asset,'method':'DtACI-projected-expected',
                 'calibration_scale':float(np.std(y[:nc],ddof=1)),
                 'n_test':len(y)-nc,'QS':float(mix['loss'].mean()),'pihat':float(mix['hits'].mean()),
                 'width':float(mix['width'].mean()),'viol':float(mix['hits'].sum()),
                 'p_kup':np.nan,'p_ind':np.nan,'p_cc':np.nan,'TL':None})
    daily.to_parquet(folder/'daily.parquet')
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    pd.DataFrame(seeds).to_csv(folder/'dtaci_seed_metrics.csv',index=False)
    params['dates']={'first':str(index[0].date()),'fit_last':str(index[params['inner_split']-1].date()),
                     'calibration_last':str(index[nc-1].date()),'test_first':str(index[nc].date()),'test_last':str(index[-1].date())}
    (folder/'fits.json').write_text(json.dumps(params,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(folder/'dtaci_experts.npz',date=index.to_numpy(),
        projected_q=proj['predictions'],projected_p=proj['probabilities'],projected_levels=proj['states'],
        unprojected_q=orig['predictions'],unprojected_p=orig['probabilities'],unprojected_levels=orig['states'])
    files=['daily.parquet','metrics.csv','fits.json','dtaci_experts.npz','dtaci_seed_metrics.csv']
    done.write_text(json.dumps({'binding':binding,'outputs':{name:sha(folder/name) for name in files},
        'n':len(y),'n_cal':nc,'inner_fit':params['inner_split'],'elapsed_seconds':time.monotonic()-start,
        'packages':{p:md.version(p) for p in ['numpy','scipy','pandas','pyarrow']}},indent=2)+'\n')
    return key,round(time.monotonic()-start,2)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+');ap.add_argument('--assets',nargs='+')
    ap.add_argument('--workers',type=int,default=3);ap.add_argument('--replay',action='store_true');args=ap.parse_args()
    before=json.loads((OUT/'before.json').read_text())
    assert before['protocol_sha256']==sha(Path(__file__).with_name('PROTOCOL.md'))
    assets=args.assets or sorted(p.stem for p in (ROOT/'data/returns').glob('*.csv'))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending=[pool.submit(work,model,asset,args.replay) for model in (args.models or MODELS) for asset in assets]
        for result in as_completed(pending):print(*result.result(),flush=True)
