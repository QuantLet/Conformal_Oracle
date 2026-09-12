#!/usr/bin/env python3
"""Freeze larger predictive pools on SP500/BTC for the N=1000 sensitivity."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
import infer_tsfm as inference

ROOT=inference.ROOT/'artifacts/extension_20260831'


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['moirai','lagllama'],required=True)
    a=ap.parse_args();model=a.model;device='cpu' if model=='moirai' else 'mps'
    torch.set_num_threads(2)
    inference.N_SAMPLES=10000 if model=='moirai' else 1000
    repeats=1 if model=='moirai' else 5;days=300 if model=='moirai' else 150;pool_size=inference.N_SAMPLES*repeats
    predict=inference.predictor(ROOT,model,device,16)
    out=ROOT/'draws'/model;out.mkdir(parents=True,exist_ok=True)
    for asset in ['SP500','BTC']:
        inp=ROOT/'data/returns'/f'{asset}.csv'
        ret=pd.read_csv(inp,index_col='date',parse_dates=True).log_return
        vals=ret.to_numpy(dtype=np.float32);positions=np.arange(len(ret)-days,len(ret));arrays=[];seeds=[]
        f=out/f'{asset}_N{pool_size}_samples.npy';pq=out/f'{asset}_N{pool_size}.parquet';meta=out/f'{asset}.json'
        binding=dict(input_sha256=inference.sha(inp),producer_sha256=inference.sha(__file__),
                     inference_sha256=inference.sha(inference.__file__),model_manifest_sha256=inference.sha(ROOT/'models/manifest.json'),
                     device=device,draws_per_pass=inference.N_SAMPLES,repeats=repeats,days=days,batch=16)
        if meta.exists():
            record=json.loads(meta.read_text());assert record['binding']==binding and record['samples_sha256']==inference.sha(f)
            continue
        started=time.monotonic()
        for j in range(0,days,16):
            ii=positions[j:j+16];contexts=np.stack([vals[t-512:t] for t in ii]);starts=ret.index[ii-512];reps=[]
            for repeat in range(repeats):
                seed=int.from_bytes(hashlib.sha256(f'pool:42:{model}:{asset}:{int(ii[0])}:{repeat}'.encode()).digest()[:4],'little')
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);seeds.append(seed)
                with torch.no_grad():reps.append(predict(contexts,starts))
            arrays.append(np.concatenate(reps,axis=1))
        raw=np.concatenate(arrays).astype(np.float32)
        assert raw.shape==(days,pool_size) and np.isfinite(raw).all()
        np.save(f,raw)
        pd.DataFrame({f'VaR_{alpha:g}':np.percentile(raw,alpha*100,axis=1) for alpha in [.01,.025,.05,.1]},
                     index=ret.index[positions]).to_parquet(pq)
        meta.write_text(json.dumps(dict(binding=binding,seeds=seeds,samples_sha256=inference.sha(f),
                                       quantiles_sha256=inference.sha(pq),elapsed_seconds=time.monotonic()-started),indent=2)+'\n')
        print(model,asset,raw.shape,'saved',round(time.monotonic()-started,1),'s',flush=True)
