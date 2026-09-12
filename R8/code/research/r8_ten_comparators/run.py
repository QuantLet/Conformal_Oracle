"""Existing twenty-method study on the validated ten-model common support."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import importlib.metadata as md
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'research/r8_model_extension'))
from evaluate_ten import ROOT as NATIVE, EXT, MODELS, forecast, sha, scores
sys.path.insert(0,str(PROJECT/'research/r8_decision'))
import methods as m


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj


original=module('original_strong_compute',PROJECT/'research/r8_decision/run.py')
controlled=module('common_controlled',PROJECT/'research/r8_commodity_etp/controlled_comparisons.py')
from posthoc import rolling,aci_path
OUT=PROJECT/'artifacts/r8_ten_comparators'
RESULTS=OUT/'results'
METHODS=['Raw','Shift-CP','Shift-ERM','Vol-CP','Vol-ERM','State2-ERM','State4-ERM',
         'State-L1','State-L1-clipped','POT-Shift','POT-Vol','ACI-existing',
         'DtACI-projected-seed','DtACI-projected-expected','Rolling250','Rolling500',
         'Selected-rolling','Gate-selected-rolling','Loss-gate','Past-minimum']
ASSETS=pd.read_csv(NATIVE/'full_preflight/support.csv').asset.tolist()


def load(model,asset):
    rp=EXT/'data/returns'/f'{asset}.csv'
    ret=pd.read_csv(rp,index_col='date',parse_dates=True).log_return
    index=ret.index[512:];y=ret.iloc[512:].to_numpy();q,fp,_=forecast(model,asset,index)
    sigma=controlled.past_volatility(ret,index)
    paths=[Path(__file__),Path(__file__).with_name('PROTOCOL.md'),rp,fp,Path(original.__file__),
           Path(controlled.__file__),Path(m.__file__),
           PROJECT/'source/scripts/extension_20260831/posthoc.py',
           PROJECT/'source/scripts/extension_20260831/panel_statistics.py',
           PROJECT/'research/r8_model_extension/evaluate_ten.py',
           NATIVE/'ten_common_evaluation/complete.json',NATIVE/'ten_common_evaluation/validation.json',
           NATIVE/'ten_policy_evaluation/complete.json',NATIVE/'ten_policy_evaluation/validation.json',
           NATIVE/'ten_common_evaluation/daily'/f'{asset}.parquet',
           NATIVE/'ten_policy_evaluation/daily'/f'{asset}.parquet']
    binding={str(p.relative_to(PROJECT)):sha(p) for p in paths}
    assert np.isfinite(np.c_[y,q,sigma]).all()
    return y,q,sigma,index,binding


def compute(y,q,sigma,nc,key):
    pred,params,proj,orig,mix,seeds=original.compute(y,q,sigma,nc,key)
    full,fullfit=controlled.candidates(y[:nc],q[:nc],sigma[:nc],q[nc:],sigma[nc:],.01)
    pred.update(full);params['controlled_full']=fullfit
    rolls={w:rolling(q-y,w) for w in controlled.WINDOWS}
    gate,window,choice=controlled.gate_and_window(y,q,nc,rolls)
    params['window']={**choice,'gate':gate,'selected_window':window}
    pred['Rolling250']=q[nc:]-rolls[250][nc:]
    pred['Rolling500']=q[nc:]-rolls[500][nc:]
    pred['Selected-rolling']=q[nc:]-rolls[window][nc:]
    pred['Gate-selected-rolling']=pred['Selected-rolling'] if gate else q[nc:]
    trials=[]
    for gamma in [.001,.005,.01]:
        path=aci_path(y[:nc],q[:nc],250,gamma)
        trials.append(dict(gamma=gamma,**scores(y[250:nc],path[250:nc])))
    gamma=min(trials,key=lambda z:abs(z['pihat']-.01))['gamma']
    pred['ACI-existing']=aci_path(y,q,nc,gamma)[nc:]
    params['aci']=dict(gamma=gamma,calibration_trials=trials,window=250,clipping=[.001,.1])
    return pred,params,proj,orig,mix,seeds


def work(model,asset,replay=False):
    begin=time.monotonic();key=f'{model}__{asset}'
    y,q,sigma,index,binding=load(model,asset);nc=int(.7*len(y))
    folder=OUT/('replay' if replay else 'pairs')/key;folder.mkdir(parents=True,exist_ok=True)
    done=folder/'complete.json'
    if done.exists():
        old=json.loads(done.read_text());assert old['binding']==binding
        assert all(sha(folder/p)==h for p,h in old['outputs'].items())
        return key,'verified'
    pred,params,proj,orig,mix,seeds=compute(y,q,sigma,nc,key)
    for source,mapping in [('ten_common_evaluation',{'Raw':'Raw','Shift-CP':'Static','Rolling250':'Rolling250'}),
                           ('ten_policy_evaluation',{'Vol-ERM':'Vol-ERM','Rolling500':'Rolling500','Loss-gate':'Loss-gate','Past-minimum':'Past-minimum'})]:
        reference=pd.read_parquet(NATIVE/source/'daily'/f'{asset}.parquet')
        assert reference.index.equals(index[nc:]);np.testing.assert_array_equal(reference.r,y[nc:])
        for name,ref in mapping.items():np.testing.assert_array_equal(pred[name],reference[f'{model}/{ref}'])
    daily=pd.DataFrame({'r':y[nc:],'sigma':sigma[nc:]},index=index[nc:]);daily.index.name='date'
    rows=[];scale=float(np.std(y[:nc],ddof=1))
    for name in METHODS:
        if name=='DtACI-projected-expected':continue
        path=pred[name];assert np.isfinite(path).all(),(key,name)
        daily[name]=path
        rows.append(dict(model=model,asset=asset,method=name,calibration_scale=scale,**scores(y[nc:],path)))
    for name,value in mix.items():daily[f'DtACI-expected/{name}']=value
    rows.append(dict(model=model,asset=asset,method='DtACI-projected-expected',calibration_scale=scale,
        n_test=len(y)-nc,QS=float(mix['loss'].mean()),pihat=float(mix['hits'].mean()),
        width=float(mix['width'].mean()),viol=float(mix['hits'].sum()),p_kup=np.nan,p_ind=np.nan,p_cc=np.nan,TL=None))
    params['dates']=dict(first=str(index[0].date()),fit_last=str(index[params['inner_split']-1].date()),
        calibration_last=str(index[nc-1].date()),test_first=str(index[nc].date()),test_last=str(index[-1].date()))
    daily.to_parquet(folder/'daily.parquet');pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    pd.DataFrame(seeds).to_csv(folder/'dtaci_seed_metrics.csv',index=False)
    (folder/'fits.json').write_text(json.dumps(params,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(folder/'dtaci_experts.npz',date=index.to_numpy(),projected_q=proj['predictions'],
        projected_p=proj['probabilities'],projected_levels=proj['states'],unprojected_q=orig['predictions'],
        unprojected_p=orig['probabilities'],unprojected_levels=orig['states'])
    files=['daily.parquet','metrics.csv','fits.json','dtaci_experts.npz','dtaci_seed_metrics.csv']
    for path,expected in binding.items():assert sha(PROJECT/path)==expected,path
    done.write_text(json.dumps(dict(binding=binding,outputs={p:sha(folder/p) for p in files},n=len(y),n_cal=nc,
        inner_fit=params['inner_split'],elapsed_seconds=time.monotonic()-begin,
        packages={p:md.version(p) for p in ['numpy','scipy','pandas','pyarrow']}),indent=2)+'\n')
    return key,round(time.monotonic()-begin,2)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true')
    parser.add_argument('--models',nargs='+');parser.add_argument('--assets',nargs='+');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks=[pool.submit(work,model,asset,args.replay) for model in (args.models or MODELS) for asset in (args.assets or ASSETS)]
        for task in as_completed(tasks):print(*task.result(),flush=True)
