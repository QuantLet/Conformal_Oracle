#!/usr/bin/env python3
"""Verify all native hashes/contexts and replay three original batches per asset."""
import infer_tsfm as inf  # sets runtime flags before importing torch
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True)
    ap.add_argument('--device',default='mps');ap.add_argument('--all-batches',action='store_true');a=ap.parse_args()
    root=inf.ROOT/'artifacts/extension_20260831/unfiltered_repair';(root/'quality').mkdir(exist_ok=True);torch.set_num_threads(2)
    predict=inf.predictor(root,a.model,a.device,16);rows=[]
    for retfile in sorted((root/'data/returns').glob('*.csv')):
        asset=retfile.stem;native=root/'native'/a.model/asset
        done=json.loads((native/'complete.json').read_text());binding=json.loads((native/'binding.json').read_text())
        assert binding['input_sha256']==inf.sha(retfile)
        assert binding['producer_sha256']==inf.sha(inf.__file__)
        assert done['binding_sha256']==inf.sha(native/'binding.json')
        assert binding['device']==a.device and binding['batch']==16
        ret=pd.read_csv(retfile,index_col='date',parse_dates=True)['log_return'];vals=ret.to_numpy(np.float32)
        n=len(ret)-512;assert done['rows']==n
        offsets=list(range(0,n,16)) if a.all_batches else [0,((n//2)//16)*16,((n-1)//16)*16]
        count=0;replays=[]
        for path in sorted(native.glob('*.npz')):
            meta=json.loads(path.with_suffix('.json').read_text());assert inf.sha(path)==meta['sha256']
            with np.load(path) as z:
                positions=z['positions'];start=int(path.stem)
                assert np.array_equal(positions,np.arange(512+start,512+start+len(positions)))
                assert np.array_equal(z['date'],ret.index[positions].values.astype('datetime64[D]'))
                contexts=np.stack([vals[t-512:t] for t in positions])
                assert hashlib.sha256(contexts.astype('<f4').tobytes()).hexdigest()==meta['context_sha256']
                count+=len(positions)
                for offset in offsets:
                    if not start<=offset<start+len(positions):continue
                    b=offset-start;pos=positions[b:b+16]
                    seed=int.from_bytes(hashlib.sha256(f'42:{a.model}:{asset}:{int(pos[0])}'.encode()).digest()[:4],'little')
                    assert seed==z['seeds'][b//16]
                    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                    with torch.no_grad():got=np.asarray(predict(contexts[b:b+16],ret.index[pos-512]),dtype=np.float32)
                    expected=z['native'][b:b+16]
                    delta=float(np.max(np.abs(got-expected)))
                    replays.append(dict(offset=offset,rows=len(pos),exact=bool(np.array_equal(got,expected)),max_abs=delta))
        assert count==n and len(replays)==len(offsets)
        row=dict(model=a.model,asset=asset,hashed_rows=n,batches=replays,exact=all(r['exact'] for r in replays))
        rows.append(row);print(a.model,asset,row['exact'],max(r['max_abs'] for r in replays),flush=True)
    target=root/'quality'/f'native_replay_{a.model}{"_full" if a.all_batches else ""}.json'
    target.write_text(json.dumps(dict(producer_sha256=inf.sha(__file__),scope='All original batches recomputed' if a.all_batches else 'All hashes and context arrays; first/middle/last original batches independently recomputed',rows=rows),indent=2)+'\n')
    assert all(r['exact'] for r in rows),'Native replay was not bit-for-bit identical; inspect detailed report'


if __name__=='__main__':main()
