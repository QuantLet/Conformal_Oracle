"""Past-only technical pilot of pinned native-tail checkpoints; no calibration."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_MODULES_CACHE'] = '/private/tmp/irfa-native-hf-modules'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '2'
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import time
import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT/'artifacts/r8_native_candidates'
DATA = PROJECT/'artifacts/extension_20260831/data/returns'
ASSETS = ['SP500','BTC']
LEVELS = [.01,.025,.05,.1,.5,.9,.99]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.backends.mps.is_available(): torch.mps.manual_seed(value)


def contexts(asset):
    path = DATA/f'{asset}.csv'
    series = pd.read_csv(path,index_col='date',parse_dates=True).log_return
    assert series.index[-1] == pd.Timestamp('2026-08-31')
    positions = np.arange(len(series)-32,len(series))
    arrays = np.stack([series.iloc[p-512:p].to_numpy(dtype=np.float32) for p in positions])
    assert arrays.shape == (32,512) and np.isfinite(arrays).all()
    # Outcome perturbation must leave the evaluated date's context identical.
    for p, array in zip(positions,arrays):
        changed = series.to_numpy(copy=True); changed[p:] = 123456.
        assert np.array_equal(changed[p-512:p].astype(np.float32),array)
        assert series.index[p-1] < series.index[p]
    return series.index[positions], arrays, {'input_sha256':sha(path),'positions':positions.tolist(),
           'context_sha256':hashlib.sha256(arrays.tobytes()).hexdigest()}


def synchronize(device):
    if device == 'mps': torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--model',choices=['chronos-2','sundial-base-128m'],required=True)
    parser.add_argument('--device',choices=['cpu','mps'],default='mps')
    parser.add_argument('--replay',action='store_true'); args = parser.parse_args()
    if args.device == 'mps': assert torch.backends.mps.is_available(), 'MPS unavailable'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    model_path = ROOT/'models'/args.model; out = ROOT/args.model; out.mkdir(exist_ok=True)
    manifest = json.loads((ROOT/'models/manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if name.startswith(args.model+'/'): assert sha(ROOT/'models'/name) == digest
    seed(20260910); begin = time.perf_counter()
    if args.model == 'chronos-2':
        from chronos import Chronos2Pipeline
        model = Chronos2Pipeline.from_pretrained(str(model_path),device_map=args.device,torch_dtype=torch.float32,local_files_only=True)
        levels = list(model.quantiles); assert .01 in levels
        def predict(array, n=1000):
            tensor = torch.as_tensor(array[:,None,:],dtype=torch.float32)
            result = model.predict(tensor,prediction_length=1,context_length=512,batch_size=16,cross_learning=False)
            return torch.stack([x[0,:,0] for x in result]).cpu().numpy()
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(str(model_path),trust_remote_code=True,local_files_only=True).to(args.device).eval()
        levels = LEVELS
        def predict(array, n=1000):
            tensor = torch.as_tensor(array,dtype=torch.float32,device=args.device)
            with torch.inference_mode(): result = model.generate(tensor,max_new_tokens=1,num_samples=n)
            assert result.shape == (len(array),n,1), tuple(result.shape)
            return result[:,:,0].cpu().numpy()
    synchronize(args.device); load_seconds = time.perf_counter()-begin
    records = []; bindings = {}; timings = []; replay = []
    for asset_i, asset in enumerate(ASSETS):
        dates, arrays, binding = contexts(asset); bindings[asset] = binding
        if args.model == 'chronos-2':
            begin = time.perf_counter(); native = predict(arrays); synchronize(args.device)
            timings.append({'asset':asset,'dates':32,'seconds':time.perf_counter()-begin})
            assert native.shape == (32,len(levels)) and np.isfinite(native).all()
            assert np.all(np.diff(native,axis=1) >= -1e-7), 'Native quantile crossing'
            path = out/f'{asset}_native.npz'
            if args.replay:
                old = np.load(path)['native']; assert np.array_equal(native,old), 'Fresh-process replay differs'
            else: np.savez_compressed(path,native=native,levels=np.array(levels),dates=dates.to_numpy())
            single = predict(arrays[[0,31]])
            np.testing.assert_allclose(single,native[[0,31]],rtol=1e-5,atol=1e-7)
            replay.append({'asset':asset,'batch_independence_max_absolute':float(np.max(np.abs(single-native[[0,31]])))})
            for j,date in enumerate(dates): records.append({'asset':asset,'date':str(date.date()),'q01':float(native[j,levels.index(.01)])})
        else:
            for j,date in enumerate(dates):
                if args.replay and j not in [0,31]: continue
                current_seed = 20260910+asset_i*10000+j; seed(current_seed)
                begin = time.perf_counter(); samples = predict(arrays[j:j+1])[0]; synchronize(args.device)
                seconds = time.perf_counter()-begin
                assert samples.shape == (1000,) and np.isfinite(samples).all()
                quantiles = np.quantile(samples,LEVELS,method='linear')
                path = out/f'{asset}_{date.date()}.npz'
                if args.replay:
                    old = np.load(path)['samples']; assert np.array_equal(samples,old), 'Fresh-process replay differs'
                    replay.append({'asset':asset,'date':str(date.date()),'exact':True})
                    continue
                np.savez_compressed(path,samples=samples,levels=np.array(LEVELS),quantiles=quantiles,seed=current_seed)
                records.append({'asset':asset,'date':str(date.date()),'q01':float(quantiles[0]),'seed':current_seed})
                timings.append({'asset':asset,'date':str(date.date()),'draws':1000,'seconds':seconds})
                print(args.model,asset,str(date.date()),'seconds',round(seconds,3),flush=True)
                if j in [30,31]:
                    pool_seed = current_seed+1000000; seed(pool_seed)
                    begin = time.perf_counter(); pool = predict(arrays[j:j+1],10000)[0]; synchronize(args.device)
                    assert pool.shape == (10000,) and np.isfinite(pool).all()
                    np.savez_compressed(out/f'{asset}_{date.date()}_pool.npz',samples=pool,seed=pool_seed)
                    timings.append({'asset':asset,'date':str(date.date()),'draws':10000,'seconds':time.perf_counter()-begin})
    receipt = {'model':args.model,'device':args.device,'load_seconds':load_seconds,'bindings':bindings,
               'protocol_sha256':sha(Path(__file__).with_name('PROTOCOL.md')),'producer_sha256':sha(__file__),
               'model_manifest_sha256':sha(ROOT/'models/manifest.json'),'replay':replay,'timings':timings,
               'versions':{name:importlib.metadata.version(name) for name in ['torch','transformers','numpy','pandas']},
               'platform':platform.platform(),'q01_from_native':True,'external_tail_fit':False,'half_normal_tail':False}
    if args.replay:
        (out/'replay.json').write_text(json.dumps(receipt,indent=2)+'\n')
    else:
        pd.DataFrame(records).to_csv(out/'quantiles.csv',index=False)
        receipt['outputs']={p.name:sha(p) for p in out.iterdir() if p.suffix in ['.npz','.csv']}
        (out/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(args.model,'replay' if args.replay else 'pilot','complete',flush=True)


if __name__ == '__main__': main()
