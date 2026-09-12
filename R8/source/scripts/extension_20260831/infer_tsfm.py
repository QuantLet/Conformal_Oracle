#!/usr/bin/env python3
"""Run pinned zero-shot forecasters and save every native sample/grid.

No calibration or tail fitting occurs here. Stable per-batch seeds and frozen
chunk boundaries make interrupted runs resumable with verifiable bindings.
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
CONTEXT, N_SAMPLES, CHUNK = 512, 1000, 512


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predictor(root, model, device, batch):
    checkpoint = root / 'models' / model
    if device == 'mps':
        import importlib
        batchify = importlib.import_module('gluonts.torch.batchify')
        original_stack = batchify.stack
        def stack_float32(data, device=None):
            # GluonTS 0.14.4 creates a float64 0/1 padding indicator. MPS
            # cannot transfer float64; converting these masks is lossless.
            if isinstance(data[0], np.ndarray) and data[0].dtype == np.float64:
                data = [x.astype(np.float32) for x in data]
            return original_stack(data, device)
        batchify.stack = stack_float32
        if model in ('moirai', 'moirai2'):
            from uni2ts.module.packed_scaler import PackedScaler
            original_scaler = PackedScaler.forward
            def scaler_cpu64(self, target, observed_mask=None, sample_id=None, variate_id=None):
                # Preserve the published float64 scaling calculation on CPU;
                # return its float32 outputs to MPS for the neural network.
                args = [x.cpu() if x is not None else None for x in (target, observed_mask, sample_id, variate_id)]
                loc, scale = original_scaler(self, *args)
                return loc.to(target.device), scale.to(target.device)
            PackedScaler.forward = scaler_cpu64
    if model in ('moirai', 'moirai2'):
        from gluonts.dataset.common import ListDataset
        if model == 'moirai':
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
            module = MoiraiModule.from_pretrained(str(checkpoint), local_files_only=True)
            forecast = MoiraiForecast(module=module, prediction_length=1, context_length=CONTEXT,
                                      patch_size='auto', num_samples=N_SAMPLES, target_dim=1,
                                      feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
            from types import MethodType
            from moirai_sampling import forward as marginal_forward
            forecast.forward = MethodType(marginal_forward, forecast)
        else:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
            module = Moirai2Module.from_pretrained(str(checkpoint), local_files_only=True)
            forecast = Moirai2Forecast(module=module, prediction_length=1, context_length=CONTEXT,
                                       target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        forecast.to(device).eval()
        pred = forecast.create_predictor(batch_size=batch, device=device)
        def predict(contexts, starts):
            ds = ListDataset([{'target': c, 'start': pd.Period(s, freq='D')} for c, s in zip(contexts, starts)], freq='D')
            outs = list(pred.predict(ds))
            if model == 'moirai':
                return np.stack([f.samples.reshape(N_SAMPLES, -1)[:, 0] for f in outs])
            expected = np.arange(1, 10)/10
            if not all(np.allclose(np.array(f.forecast_keys, dtype=float), expected) for f in outs):
                raise ValueError('Moirai2 quantile levels changed')
            return np.stack([f.forecast_array[:, 0] for f in outs])
        return predict
    if model == 'lagllama':
        from gluonts.dataset.common import ListDataset
        sys.path.insert(0, str(root / 'models/lag-llama-source'))
        from lag_llama.gluon.estimator import LagLlamaEstimator
        est = LagLlamaEstimator(prediction_length=1, context_length=CONTEXT, input_size=1,
                               n_layer=8, n_embd_per_head=36, n_head=4, num_parallel_samples=N_SAMPLES,
                               batch_size=batch, device=torch.device(device),
                               rope_scaling={'type':'linear', 'factor':16.0},
                               ckpt_path=str(checkpoint/'lag-llama.ckpt'), time_feat=True)
        module = est.create_lightning_module()
        # For horizon=1 every draw conditions on exactly the same observed
        # history. Evaluate that deterministic network once, then sample the
        # identical Student-t law 1000 times (the upstream single-pass path).
        # There is no greedy-feedback approximation at a one-step horizon.
        module.use_single_pass_sampling = True
        pred = est.create_predictor(est.create_transformation(), module)
        def predict(contexts, starts):
            ds = ListDataset([{'target': c, 'start': pd.Period(s, freq='D')} for c, s in zip(contexts, starts)], freq='D')
            return np.stack([f.samples.reshape(N_SAMPLES, -1)[:, 0] for f in pred.predict(ds, num_samples=N_SAMPLES)])
        return predict
    import timesfm
    tfm = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
    tfm.load_checkpoint(str(checkpoint), torch_compile=False)
    # TimesFM's default CUDA/CPU selection omits MPS. Only move the tensors;
    # the checkpoint, forward pass, and forecast flags remain unchanged.
    tfm.model.device = torch.device(device)
    tfm.model.to(device).eval()
    tfm.compile(timesfm.ForecastConfig(max_horizon=1, max_context=CONTEXT, per_core_batch_size=batch))
    def predict(contexts, starts):
        _, q = tfm.forecast(horizon=1, inputs=list(contexts))
        if q.shape != (len(contexts), 1, 10):
            raise ValueError(f'TimesFM native shape changed: {q.shape}')
        return q[:, 0, :]  # point + nine deciles; keep all ten
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT/'artifacts/extension_20260831')
    ap.add_argument('--model', choices=['moirai','moirai2','lagllama','timesfm25'], required=True)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--assets', nargs='+')
    ap.add_argument('--limit', type=int, default=0, help='Smoke test: first N forecast dates, separate output')
    a = ap.parse_args()
    torch.set_num_threads(2)
    if a.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; run with GPU permission or explicitly select CPU')
    models = json.loads((a.root/'models/manifest.json').read_text())
    for name, digest in models[a.model]['files'].items():
        if sha(a.root/'models'/a.model/name) != digest:
            raise ValueError('Checkpoint hash mismatch')
    packages = {p: importlib.metadata.version(p) for p in ['torch','numpy','pandas','scipy','uni2ts','gluonts','timesfm','huggingface-hub']}
    output_root = (a.root/'smoke'/sha(__file__)[:12] if a.limit else a.root/'native') / a.model
    output_root.mkdir(parents=True, exist_ok=True)
    predict = predictor(a.root, a.model, a.device, a.batch)
    assets = a.assets or sorted(p.stem for p in (a.root/'data/returns').glob('*.csv'))
    for asset in assets:
        started = time.monotonic()
        inp = a.root/'data/returns'/f'{asset}.csv'
        ret = pd.read_csv(inp, index_col='date', parse_dates=True)['log_return']
        vals = ret.to_numpy(dtype=np.float32)
        positions = np.arange(CONTEXT, len(ret))
        if a.limit:
            positions = positions[:a.limit]
        out = output_root/asset
        out.mkdir(parents=True, exist_ok=True)
        binding = dict(model=models[a.model], input_sha256=sha(inp), producer_sha256=sha(__file__),
                       moirai_sampler_sha256=sha(Path(__file__).with_name('moirai_sampling.py')) if a.model=='moirai' else None,
                       context=CONTEXT, samples=N_SAMPLES if a.model in ('moirai','lagllama') else None,
                       batch=a.batch, chunk=CHUNK, base_seed=42, device=a.device, packages=packages,
                       python=platform.python_version(), mps_fallback=True, smoke_limit=a.limit)
        bindfile = out/'binding.json'
        if bindfile.exists() and json.loads(bindfile.read_text()) != binding and list(out.glob('*.npz')):
            raise ValueError(f'Binding mismatch: {out}')
        bindfile.write_text(json.dumps(binding, indent=2)+'\n')
        for offset in range(0, len(positions), CHUNK):
            chunk = positions[offset:offset+CHUNK]
            target = out/f'{offset:06d}.npz'
            meta = target.with_suffix('.json')
            if target.exists() and meta.exists():
                if sha(target) != json.loads(meta.read_text())['sha256']:
                    raise ValueError('Native chunk hash mismatch')
                continue
            arrays, seeds = [], []
            contexts = np.stack([vals[t-CONTEXT:t] for t in chunk])
            starts = ret.index[chunk-CONTEXT]
            for b in range(0, len(chunk), a.batch):
                seed = int.from_bytes(hashlib.sha256(f'42:{a.model}:{asset}:{int(chunk[b])}'.encode()).digest()[:4], 'little')
                seeds.append(seed)
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                with torch.no_grad():
                    arrays.append(np.asarray(predict(contexts[b:b+a.batch], starts[b:b+a.batch]), dtype=np.float32))
            native = np.concatenate(arrays)
            expected = N_SAMPLES if a.model in ('moirai','lagllama') else (10 if a.model == 'timesfm25' else 9)
            if native.shape != (len(chunk), expected) or not np.isfinite(native).all():
                raise ValueError(f'Invalid native output: {native.shape}')
            np.savez_compressed(target, native=native, date=ret.index[chunk].values.astype('datetime64[D]'),
                                context_start=starts.values.astype('datetime64[D]'), positions=chunk,
                                seeds=np.array(seeds,dtype=np.uint32))
            meta.write_text(json.dumps(dict(sha256=sha(target), rows=len(chunk),
                                            context_sha256=hashlib.sha256(contexts.astype('<f4').tobytes()).hexdigest(),
                                            first_date=str(ret.index[chunk[0]].date()), last_date=str(ret.index[chunk[-1]].date())), indent=2)+'\n')
            print(f'{a.model} {asset}: {min(offset+CHUNK,len(positions))}/{len(positions)}, {time.monotonic()-started:.1f}s', flush=True)
        (out/'complete.json').write_text(json.dumps(dict(rows=len(positions), binding_sha256=sha(bindfile),
                                                         chunks=(len(positions)+CHUNK-1)//CHUNK), indent=2)+'\n')


if __name__ == '__main__':
    main()
