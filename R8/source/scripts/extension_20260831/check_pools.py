#!/usr/bin/env python3
"""Replay every larger-pool batch used in the predictive-sampling sensitivity."""
import infer_tsfm as inf
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['moirai','lagllama'],required=True);a=ap.parse_args()
    model=a.model;root=inf.ROOT/'artifacts/extension_20260831';device='cpu' if model=='moirai' else 'mps'
    if model=='lagllama' and (root/'calendar_primary.json').exists():
        import subprocess,sys,shutil
        subprocess.run([sys.executable,str(inf.ROOT/'research/r8_review/calendar_pools.py'),'--replay'],check=True)
        corrected=inf.ROOT/'artifacts/review_20260909/calendar'
        for asset in ['SP500','BTC']:
            name=f'{asset}_N5000_samples.npy'
            assert inf.sha(root/'draws/lagllama'/name)==inf.sha(corrected/'draws/lagllama'/name)
        shutil.copy2(corrected/'pools_replay.json',root/'quality/pools_replay_lagllama.json')
        return
    inf.N_SAMPLES=10000 if model=='moirai' else 1000;torch.set_num_threads(2)
    predict=inf.predictor(root,model,device,16);rows=[]
    for asset in ['SP500','BTC']:
        meta=json.loads((root/'draws'/model/f'{asset}.json').read_text());b=meta['binding']
        retfile=root/'data/returns'/f'{asset}.csv';assert inf.sha(retfile)==b['input_sha256']
        ret=pd.read_csv(retfile,index_col='date',parse_dates=True).log_return;vals=ret.to_numpy(np.float32)
        pool=root/'draws'/model/f'{asset}_N{b["draws_per_pass"]*b["repeats"]}_samples.npy'
        assert inf.sha(pool)==meta['samples_sha256'];want=np.load(pool);parts=[];seeds=[]
        positions=np.arange(len(ret)-b['days'],len(ret))
        for j in range(0,len(positions),16):
            ii=positions[j:j+16];contexts=np.stack([vals[t-512:t] for t in ii]);samples=[]
            for repeat in range(b['repeats']):
                seed=int.from_bytes(hashlib.sha256(f'pool:42:{model}:{asset}:{int(ii[0])}:{repeat}'.encode()).digest()[:4],'little')
                seeds.append(seed);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                with torch.no_grad():samples.append(predict(contexts,ret.index[ii-512]))
            parts.append(np.concatenate(samples,axis=1))
        got=np.concatenate(parts).astype(np.float32)
        assert seeds==meta['seeds'] and np.array_equal(want,got),(model,asset,float(np.max(abs(want-got))))
        rows.append(dict(model=model,asset=asset,draws=got.size,exact=True,input_sha256=inf.sha(retfile),samples_sha256=inf.sha(pool)))
        print(model,asset,'PASS',got.size,'predictive draws',flush=True)
    (root/'quality'/f'pools_replay_{model}.json').write_text(json.dumps(dict(exact=True,rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
