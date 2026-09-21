"""Actual-calendar Lag-Llama pools, with a separate-process exact replay mode."""
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
import lag_calendar as cal


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);assert torch.backends.mps.is_available()
    interface=cal.Interface('mps');out=cal.OUT/'draws/lagllama';out.mkdir(parents=True,exist_ok=True)
    records=[]
    for asset in ['SP500','BTC']:
        start=time.monotonic();retfile=cal.OLD/'data/returns'/f'{asset}.csv'
        ret=pd.read_csv(retfile,index_col='date',parse_dates=True).log_return
        vals=ret.to_numpy(np.float32);positions=np.arange(len(ret)-150,len(ret));parts=[];seeds=[]
        for j in range(0,150,16):
            pos=positions[j:j+16];contexts=np.stack([vals[t-512:t] for t in pos])
            dates=[ret.index[t-512:t+1] for t in pos];repeats=[]
            for repeat in range(5):
                seed=int.from_bytes(hashlib.sha256(f'pool:42:lagllama:{asset}:{int(pos[0])}:{repeat}'.encode()).digest()[:4],'little')
                seeds.append(seed);samples,_=interface.predict(contexts,dates,'actual',seed);repeats.append(samples)
            parts.append(np.concatenate(repeats,axis=1))
        raw=np.concatenate(parts);assert raw.shape==(150,5000) and np.isfinite(raw).all()
        f=out/f'{asset}_N5000_samples.npy';pq=out/f'{asset}_N5000.parquet';meta=out/f'{asset}.json'
        binding=dict(input_sha256=cal.sha(retfile),producer_sha256=cal.sha(__file__),interface_sha256=cal.sha(cal.__file__),
                     checkpoint_sha256=cal.sha(cal.OLD/'models/lagllama/lag-llama.ckpt'),
                     model_manifest_sha256=cal.sha(cal.OLD/'models/manifest.json'),
                     calendar='actual observed timestamps',device='mps',draws_per_pass=1000,repeats=5,days=150,batch=16)
        if a.replay:
            record=json.loads(meta.read_text());assert record['binding']==binding and record['seeds']==seeds
            assert record['samples_sha256']==cal.sha(f) and np.array_equal(raw,np.load(f))
            records.append(dict(model='lagllama',asset=asset,draws=raw.size,exact=True,
                                input_sha256=cal.sha(retfile),samples_sha256=cal.sha(f)))
        else:
            if meta.exists():raise FileExistsError('Use --replay for an existing pool')
            np.save(f,raw)
            pd.DataFrame({f'VaR_{alpha:g}':np.percentile(raw,alpha*100,axis=1) for alpha in [.01,.025,.05,.1]},index=ret.index[positions]).to_parquet(pq)
            legacy=cal.OLD/'draws/lagllama'/f.name
            if asset=='BTC':assert np.array_equal(raw,np.load(legacy)),'Regular-calendar negative control failed'
            meta.write_text(json.dumps(dict(binding=binding,seeds=seeds,samples_sha256=cal.sha(f),quantiles_sha256=cal.sha(pq),
                legacy_samples_sha256=cal.sha(legacy),legacy_exact=bool(np.array_equal(raw,np.load(legacy)))),indent=2)+'\n')
        print(asset,'replayed' if a.replay else 'generated',raw.size,'draws',round(time.monotonic()-start,1),'s',flush=True)
    if a.replay:
        (cal.OUT/'pools_replay.json').write_text(json.dumps(dict(exact=True,checker_sha256=cal.sha(__file__),rows=records),indent=2)+'\n')


if __name__=='__main__':main()
