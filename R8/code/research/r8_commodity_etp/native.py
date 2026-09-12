#!/usr/bin/env python3
"""Reuse admitted native-tail interfaces on the three replacement exposures."""
import os
for key in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'PYTORCH_ENABLE_MPS_FALLBACK']:
    os.environ[key] = '1'
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '2'
import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata as md
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT/'artifacts/r8_commodity_etp'
OLD = PROJECT/'artifacts/extension_20260831'
sys.path.insert(0, str(PROJECT/'source/scripts/extension_20260831'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n')


def interface(model):
    bound = {}
    def bind(path):
        bound[str(path.relative_to(PROJECT))] = sha(path)
    if model in ['moirai', 'lagllama']:
        manifest = OLD/'models/manifest.json'; bind(manifest)
        for name, expected in json.loads(manifest.read_text())[model]['files'].items():
            path = OLD/'models'/model/name; assert sha(path) == expected; bind(path)
        if model == 'moirai':
            import infer_tsfm
            bind(Path(infer_tsfm.__file__))
            bind(Path(infer_tsfm.__file__).with_name('moirai_sampling.py'))
            predictor = infer_tsfm.predictor(OLD, model, 'cpu', 16)
            def predict(contexts, dates, seed):
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                with torch.no_grad():
                    return np.asarray(predictor(contexts, [d[0] for d in dates]), dtype=np.float32), None
        else:
            sys.path.insert(0, str(PROJECT/'research/r8_review'))
            import lag_calendar
            bind(Path(lag_calendar.__file__))
            for path in sorted((OLD/'models/lag-llama-source/lag_llama').rglob('*.py')): bind(path)
            adapter = lag_calendar.Interface('mps')
            def predict(contexts, dates, seed):
                return adapter.predict(contexts, dates, 'actual', seed)
        return predict, None, bound
    if model == 'chronos2':
        from chronos import Chronos2Pipeline
        folder = PROJECT/'artifacts/r8_native_candidates/models'
        manifest = folder/'manifest.json'; bind(manifest)
        for name, expected in json.loads(manifest.read_text())['files'].items():
            if name.startswith('chronos-2/'):
                path = folder/name; assert sha(path) == expected; bind(path)
        model_obj = Chronos2Pipeline.from_pretrained(str(folder/'chronos-2'),
            device_map='mps', torch_dtype=torch.float32, local_files_only=True)
        levels = np.asarray(model_obj.quantiles)
        def predict(contexts, dates, seed):
            with torch.inference_mode():
                predictions = model_obj.predict(torch.from_numpy(contexts[:, None, :]),
                    prediction_length=1, context_length=512, batch_size=16, cross_learning=False)
            return torch.stack([x[0, :, 0] for x in predictions]).cpu().numpy(), None
        bind(PROJECT/'research/r8_native_candidates/full_chronos.py')
        return predict, levels, bound
    sys.path.insert(0, str(PROJECT/'research/r8_grid_candidates'))
    import full_patchtst
    _, source = full_patchtst.bindings()
    bind(Path(full_patchtst.__file__))
    bind(PROJECT/'research/r8_grid_candidates/pilot.py')
    for path in source.rglob('*.py'): bind(path)
    folder = PROJECT/'artifacts/r8_grid_candidates'
    for name in ['source_manifest.json', 'weights_manifest.json']: bind(folder/name)
    for path in (folder/'models/patchtst').rglob('*'):
        if path.is_file(): bind(path)
    sys.path.insert(0, str(source))
    module = importlib.import_module('tsfm_public.models.patchtst_fm.modeling_patchtst_fm')
    module.get_autocast_context = lambda device: contextlib.nullcontext()
    adapter = full_patchtst.PatchAdapter('mps')
    def predict(contexts, dates, seed):
        return adapter.predict(contexts), None
    return predict, full_patchtst.LEVELS, bound


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['moirai', 'lagllama', 'chronos2', 'patchtst'], required=True)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    device = 'cpu' if args.model == 'moirai' else 'mps'
    if device == 'mps':
        assert torch.backends.mps.is_available(), 'MPS access required for unchanged configuration'
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.manual_seed(20260910)
    if device == 'mps': torch.mps.manual_seed(20260910)
    admission = json.loads((ROOT/'data_admission.json').read_text())
    assert admission['status'] == 'admitted_for_forecast_recomputation'
    predict, levels, sources = interface(args.model)
    folder = ROOT/'native'/args.model; folder.mkdir(parents=True, exist_ok=True)
    binding = dict(producer_sha256=sha(__file__), protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')),
        admission_sha256=sha(ROOT/'data_admission.json'), sources=sources, model=args.model,
        batch=16, chunk=512, context=512, device=device, dtype='float32',
        seed_rule='sha256(42:model:asset:first_position), first four bytes little endian for sampling',
        native_levels=None if levels is None else levels.tolist(),
        packages={d.metadata['Name']: d.version for d in md.distributions()}, python=sys.version)
    bfile = folder/'binding.json'
    if bfile.exists(): assert json.loads(bfile.read_text()) == binding
    else:
        assert not args.replay
        write(bfile, binding)
    all_records = []
    for item in admission['assets']:
        asset = item['asset']; begin = time.monotonic()
        path = ROOT/'data/returns'/f'{asset}.csv'; assert sha(path) == item['input_sha256']
        series = pd.read_csv(path, index_col='date', parse_dates=True).log_return
        vals = series.to_numpy(np.float32); positions = np.arange(512, len(vals))
        out = folder/asset; out.mkdir(exist_ok=True)
        records = []
        for start in range(0, len(positions), 512):
            pos = positions[start:start+512]; target = out/f'{start:06d}.npz'
            note = target.with_suffix('.json')
            if target.exists() and not args.replay:
                saved = json.loads(note.read_text())
                assert sha(target) == saved['sha256'] and saved['input_sha256'] == sha(path)
                records.append(saved); continue
            arrays, params, seeds = [], [], []
            contexts = np.stack([vals[t-512:t] for t in pos])
            for offset in range(0, len(pos), 16):
                batch_pos = pos[offset:offset+16]
                seed = int.from_bytes(hashlib.sha256(f'42:{args.model}:{asset}:{int(batch_pos[0])}'.encode()).digest()[:4], 'little')
                calendars = [series.index[t-512:t+1] for t in batch_pos]
                native, parameters = predict(contexts[offset:offset+16], calendars, seed)
                arrays.append(native); seeds.append(seed)
                if parameters is not None: params.append(parameters)
            native = np.concatenate(arrays)
            assert native.shape == (len(pos), 1000 if levels is None else len(levels))
            assert np.isfinite(native).all()
            values = dict(native=native, positions=pos, dates=series.index[pos].to_numpy(),
                          seeds=np.asarray(seeds, dtype=np.uint32))
            if levels is not None: values['levels'] = levels
            if params: values['parameters'] = np.concatenate(params)
            context_hash = hashlib.sha256(contexts.astype('<f4').tobytes()).hexdigest()
            if args.replay:
                with np.load(target, allow_pickle=False) as old:
                    assert set(old.files) == set(values)
                    for name, value in values.items(): np.testing.assert_array_equal(old[name], value)
                saved = json.loads(note.read_text())
                assert sha(target) == saved['sha256'] and context_hash == saved['contexts_sha256']
                records.append(saved)
            else:
                np.savez_compressed(target, **values)
                record = dict(file=target.name, sha256=sha(target), rows=len(pos),
                              contexts_sha256=context_hash, input_sha256=sha(path))
                write(note, record); records.append(record)
            print(args.model, asset, start+len(pos), '/', len(positions),
                  round(time.monotonic()-begin, 1), 'seconds', 'replay' if args.replay else 'production', flush=True)
        assert sum(r['rows'] for r in records) == item['eligible_forecasts']
        record = dict(asset=asset, rows=len(positions), chunks=records, input_sha256=sha(path),
                      seconds=time.monotonic()-begin, exact_fresh_replay=args.replay)
        write(out/('replay.json' if args.replay else 'complete.json'), record)
        all_records.append(record)
    write(folder/('replay.json' if args.replay else 'complete.json'), dict(status='complete',
        binding_sha256=sha(bfile), assets=all_records, exact_fresh_replay=args.replay))


if __name__ == '__main__':
    main()
