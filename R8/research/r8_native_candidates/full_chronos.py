"""Complete fixed-protocol native Chronos-2 replay on the 24 archived assets."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

PROJECT=Path(__file__).resolve().parents[2]
ROOT=PROJECT/'artifacts/r8_native_candidates'
DATA=PROJECT/'artifacts/extension_20260831/data/returns'
OUT=ROOT/'chronos-2-full'
sys.path.insert(0,str(PROJECT/'source/scripts/extension_20260831'))
from panel_statistics import qshift,scores
from analyse_panel import rollshift


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    assert torch.backends.mps.is_available();torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(20260910)
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'models/manifest.json').read_text())
    for name,h in manifest['files'].items():
        if name.startswith('chronos-2/'):assert sha(ROOT/'models'/name)==h
    model=Chronos2Pipeline.from_pretrained(str(ROOT/'models/chronos-2'),device_map='mps',torch_dtype=torch.float32,local_files_only=True)
    levels=list(model.quantiles);q_index=levels.index(.01)
    assets=sorted(DATA.glob('*.csv'));assert len(assets)==24
    records=[];receipts=[]
    for path in assets:
        asset=path.stem;series=pd.read_csv(path,index_col='date',parse_dates=True).log_return
        values=series.to_numpy(dtype=np.float32);positions=np.arange(512,len(values));begin=time.perf_counter();chunks=[]
        for start in range(0,len(positions),512):
            pos=positions[start:start+512]
            context=np.stack([values[p-512:p] for p in pos])[:,None,:]
            with torch.inference_mode():
                predictions=model.predict(torch.from_numpy(context),prediction_length=1,context_length=512,batch_size=16,cross_learning=False)
            chunks.append(torch.stack([x[0,:,0] for x in predictions]).cpu().numpy())
        native=np.concatenate(chunks);torch.mps.synchronize();elapsed=time.perf_counter()-begin
        assert native.shape==(len(positions),len(levels)) and np.isfinite(native).all()
        crossing=int((np.diff(native,axis=1)<-1e-7).any(axis=1).sum())
        p=OUT/f'{asset}.npz'
        if args.replay:
            previous=np.load(p);assert np.array_equal(previous['native'],native),(asset,'native replay')
            assert np.array_equal(previous['positions'],positions)
        else:np.savez_compressed(p,native=native,levels=np.array(levels),positions=positions,dates=series.index[positions].to_numpy())
        # Score in the original double-precision observed return units.
        q=native[:,q_index].astype(float);y=series.iloc[positions].to_numpy();n_cal=int(.7*len(y));s=q-y
        shift=qshift(s[:n_cal]);rolling=rollshift(s,250)
        for method,quantile in [('Raw',q),('Static',q-shift),('Rolling250',q-rolling)]:
            r=scores(y[n_cal:],quantile[n_cal:]);records.append({'asset':asset,'method':method,'n_cal':n_cal,
                'qV':shift,'first_test':str(series.index[positions[n_cal]].date()),'last_test':str(series.index[-1].date()),**r})
        receipts.append({'asset':asset,'rows':len(positions),'native_crossings':crossing,'seconds':elapsed,
                         'input_sha256':sha(path),'output_sha256':sha(p)})
        print(asset,len(positions),'rows',round(elapsed,2),'seconds','crossings',crossing,flush=True)
    metrics=pd.DataFrame(records);summary=metrics.groupby('method').agg(assets=('asset','size'),QS=('QS','mean'),pi=('pihat','mean'),
        observations=('n_test','sum'),violations=('viol','sum'),kupiec_rejections=('p_kup',lambda s:int((s<.05).sum())))
    summary['QS_x10000']=summary.QS*1e4
    if args.replay:
        old=pd.read_csv(OUT/'metrics.csv');pd.testing.assert_frame_equal(old,metrics,check_exact=False,rtol=1e-12,atol=1e-15)
    else:
        metrics.to_csv(OUT/'metrics.csv',index=False);summary.to_csv(OUT/'summary.csv')
    receipt={'model':'chronos-2','rows':sum(r['rows'] for r in receipts),'assets':receipts,'native_levels':levels,'device':'mps',
             'exact_fresh_replay':args.replay,'protocol_sha256':sha(Path(__file__).with_name('PROTOCOL.md')),
             'producer_sha256':sha(__file__),'model_manifest_sha256':sha(ROOT/'models/manifest.json'),
             'versions':{n:importlib.metadata.version(n) for n in ['torch','chronos-forecasting','transformers','numpy','pandas']},
             'scoring_producers':{n:sha(PROJECT/'source/scripts/extension_20260831'/n) for n in ['panel_statistics.py','analyse_panel.py']}}
    (OUT/('replay.json' if args.replay else 'complete.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(summary.to_string(),flush=True)


if __name__=='__main__':main()
