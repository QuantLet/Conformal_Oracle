"""Fixed external correction policies; calibration 2000–2014, test 2015–July 2026."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from prepare import PROJECT,OUT,ASSETS,sha
sys.path.insert(0,str(PROJECT/'research/r8_decision'))
import methods as m


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


original=module('external_original_compute',PROJECT/'research/r8_decision/run.py')
controlled=module('external_controlled',PROJECT/'research/r8_commodity_etp/controlled_comparisons.py')
from posthoc import rolling
from panel_statistics import scores
MODELS=['HS','GJR-GARCH-t','CAViaR-AS','GAS-t']
METHODS=['Raw','Shift-CP','Vol-ERM','State-L1','POT-Shift','POT-Vol','DtACI-projected-expected',
         'Rolling500','Selected-rolling','Gate-selected-rolling','Loss-gate','Past-minimum',
         'State-L1-clipped','DtACI-projected-seed']


def forecast_path(model,asset,replay=False):
    if model in ['HS','GJR-GARCH-t']:
        base=OUT/'classical_replay' if replay else OUT
        return base/'data/benchmarks'/f'{asset}_{"hs" if model=="HS" else "gjr_t"}.parquet'
    return OUT/('dynamic_replay' if replay else 'dynamic')/f'{model}__{asset}'/'forecasts.parquet'


def load(model,asset):
    rp=OUT/'data/returns'/f'{asset}.csv';fp=forecast_path(model,asset)
    ret=pd.read_csv(rp,index_col='date',parse_dates=True).log_return
    index=ret.index[ret.index.year>=2000];base=pd.read_parquet(fp)
    assert base.index.is_unique and base.index.is_monotonic_increasing and base.index[-1]==index[-1]
    q=base.loc[index,'VaR_0.01'].to_numpy();y=ret.loc[index].to_numpy()
    sigma=controlled.past_volatility(ret,index);nc=int((index.year<=2014).sum())
    assert index[nc]==pd.Timestamp('2015-01-02') and index[-1]==pd.Timestamp('2026-07-31')
    assert np.isfinite(np.c_[y,q,sigma]).all()
    paths=[Path(__file__),Path(__file__).with_name('PROTOCOL.md'),rp,fp,Path(original.__file__),Path(controlled.__file__),Path(m.__file__),
           OUT/'admission.json',PROJECT/'source/scripts/extension_20260831/panel_statistics.py',
           PROJECT/'source/scripts/extension_20260831/posthoc.py']
    binding={str(p.relative_to(PROJECT)):sha(p) for p in paths}
    return y,q,sigma,index,nc,binding


def compute(y,q,sigma,nc,key):
    pred,fit,proj,unproj,mix,seeds=original.compute(y,q,sigma,nc,key)
    shift=m.cp(q[:nc]-y[:nc]);vol=m.weighted_quantile((q[:nc]-y[:nc])/sigma[:nc],sigma[:nc],.99)
    pred.update({'Raw':q[nc:],'Shift-CP':q[nc:]-shift,'Vol-ERM':q[nc:]-vol*sigma[nc:]})
    fit['full_static']=dict(shift=shift,vol_coefficient=vol)
    rolls={w:rolling(q-y,w) for w in [125,250,500]}
    gate,window,choice=controlled.gate_and_window(y,q,nc,rolls)
    fit['window']={**choice,'gate':gate,'selected_window':window}
    pred['Rolling500']=q[nc:]-rolls[500][nc:]
    pred['Selected-rolling']=q[nc:]-rolls[window][nc:]
    pred['Gate-selected-rolling']=pred['Selected-rolling'] if gate else q[nc:]
    assert set(pred)==set(METHODS)-{'DtACI-projected-expected'}
    return pred,fit,proj,unproj,mix,seeds


def work(model,asset,replay=False):
    y,q,sigma,index,nc,binding=load(model,asset);key=f'{model}__{asset}'
    folder=OUT/('correction_replay' if replay else 'pairs')/key;folder.mkdir(parents=True,exist_ok=True)
    done=folder/'complete.json'
    if done.exists():
        old=json.loads(done.read_text());assert old['binding']==binding
        for p,h in old['outputs'].items():assert sha(folder/p)==h
        return key,'verified'
    pred,fit,proj,unproj,mix,seeds=compute(y,q,sigma,nc,key)
    f=pd.DataFrame({'r':y[nc:],'sigma':sigma[nc:]},index=index[nc:]);f.index.name='date'
    rows=[];scale=float(np.std(y[:nc],ddof=1))
    for name,path in pred.items():
        assert np.isfinite(path).all();f[name]=path
        rows.append(dict(model=model,asset=asset,method=name,calibration_scale=scale,**scores(y[nc:],path)))
    for name,array in mix.items():f['DtACI-expected/'+name]=array
    rows.append(dict(model=model,asset=asset,method='DtACI-projected-expected',calibration_scale=scale,n_test=len(y)-nc,
                     QS=float(mix['loss'].mean()),pihat=float(mix['hits'].mean()),width=float(mix['width'].mean()),
                     viol=float(mix['hits'].sum()),p_kup=np.nan,p_ind=np.nan,p_cc=np.nan,TL=None))
    fit['dates']=dict(first=str(index[0].date()),inner_fit_last=str(index[fit['inner_split']-1].date()),
                      calibration_last=str(index[nc-1].date()),test_first=str(index[nc].date()),test_last=str(index[-1].date()))
    f.to_parquet(folder/'daily.parquet');pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    pd.DataFrame(seeds).to_csv(folder/'dtaci_seed_metrics.csv',index=False)
    (folder/'fits.json').write_text(json.dumps(fit,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(folder/'dtaci_experts.npz',date=index.to_numpy(),projected_q=proj['predictions'],projected_p=proj['probabilities'],
                        projected_levels=proj['states'],unprojected_q=unproj['predictions'],unprojected_p=unproj['probabilities'])
    files=['daily.parquet','metrics.csv','dtaci_seed_metrics.csv','fits.json','dtaci_experts.npz']
    done.write_text(json.dumps(dict(binding=binding,n_cal=nc,n_test=len(y)-nc,outputs={p:sha(folder/p) for p in files}),indent=2)+'\n')
    return key,'complete'


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);ap.add_argument('--replay',action='store_true')
    ap.add_argument('--models',nargs='+',default=MODELS);ap.add_argument('--assets',nargs='+',default=ASSETS);a=ap.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(work,model,asset,a.replay) for model in a.models for asset in a.assets]
        for f in as_completed(futures):print(*f.result(),flush=True)
