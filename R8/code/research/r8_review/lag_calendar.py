"""Controlled correction of Lag-Llama calendar inputs; pinned weights unchanged."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '2'

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[2]
OLD = PROJECT/'artifacts/extension_20260831'
OUT = PROJECT/'artifacts/review_20260909/calendar'
sys.path.insert(0, str(OLD/'models/lag-llama-source'))
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.transform import AsNumpyArray, Chain

CONTEXT, SAMPLES, BATCH = 512, 1000, 16
FEATURES = time_features_from_frequency_str('S')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_for(asset, position):
    return int.from_bytes(hashlib.sha256(f'42:lagllama:{asset}:{position}'.encode()).digest()[:4], 'little')


def reseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def features(dates):
    index = pd.PeriodIndex(dates, freq='D')
    assert len(index) == CONTEXT+1 and index.is_unique and index.is_monotonic_increasing
    return np.vstack([f(index) for f in FEATURES]).astype(np.float32)


class Interface:
    def __init__(self, device):
        if device == 'mps':
            batchify = importlib.import_module('gluonts.torch.batchify')
            original_stack = batchify.stack
            def stack_float32(data, device=None):
                if isinstance(data[0], np.ndarray) and data[0].dtype == np.float64:
                    data = [x.astype(np.float32) for x in data]
                return original_stack(data, device)
            batchify.stack = stack_float32
        self.est = LagLlamaEstimator(
            prediction_length=1, context_length=CONTEXT, input_size=1,
            n_layer=8, n_embd_per_head=36, n_head=4, num_parallel_samples=SAMPLES,
            batch_size=BATCH, device=torch.device(device),
            rope_scaling={'type':'linear', 'factor':16.0},
            ckpt_path=str(OLD/'models/lagllama/lag-llama.ckpt'), time_feat=True)
        self.module = self.est.create_lightning_module()
        self.module.use_single_pass_sampling = True
        original = self.est.create_transformation()
        assert type(original.transformations[0]).__name__ == 'AddTimeFeatures'
        corrected = Chain([AsNumpyArray(field=FieldName.FEAT_TIME, expected_ndim=2)]
                          + original.transformations[1:])
        self.predictors = {
            'legacy': self.est.create_predictor(original, self.module),
            'actual': self.est.create_predictor(corrected, self.module),
        }
        self.parameters = []
        def capture(_module, _inputs, outputs):
            params, loc, scale = outputs
            assert len(params) == 3
            df, mu, sigma = [p[:, -1:].detach() for p in params]
            effective_mu = loc + scale*mu
            effective_sigma = scale*sigma
            matrix = torch.cat([df, effective_mu, effective_sigma], dim=1).cpu().numpy()
            assert matrix.shape[1] == 3 and np.isfinite(matrix).all()
            assert (matrix[:, 0] > 2).all() and (matrix[:, 2] > 0).all()
            self.parameters.append(matrix)
        self.hook = self.module.model.register_forward_hook(capture)

    def predict(self, contexts, dates, mode, seed):
        assert len(contexts) == len(dates) and len(contexts) <= BATCH
        entries = []
        for context, calendar in zip(contexts, dates):
            assert len(context) == CONTEXT and len(calendar) == CONTEXT+1
            entry = {'target': context, 'start': pd.Period(calendar[0], freq='D')}
            if mode == 'actual':
                entry[FieldName.FEAT_TIME] = features(calendar)
            entries.append(entry)
        dataset = ListDataset(entries, freq='D')
        self.parameters = []
        reseed(seed)
        with torch.no_grad():
            samples = np.stack([f.samples.reshape(SAMPLES, -1)[:, 0]
                                for f in self.predictors[mode].predict(dataset, num_samples=SAMPLES)])
        parameters = np.concatenate(self.parameters)
        assert samples.shape == (len(contexts), SAMPLES)
        assert parameters.shape == (len(contexts), 3) and np.isfinite(samples).all()
        return samples.astype(np.float32), parameters.astype(np.float32)


def validate(interface):
    rng = np.random.default_rng(914)
    contexts = rng.normal(0, .01, (BATCH, CONTEXT)).astype(np.float32)
    regular = [pd.date_range('2020-01-01', periods=CONTEXT+1)]*BATCH
    a, ap = interface.predict(contexts, regular, 'legacy', 83)
    b, bp = interface.predict(contexts, regular, 'actual', 83)
    assert np.array_equal(a, b) and np.array_equal(ap, bp), 'Regular-calendar equivalence failed'
    rows = []
    for asset in ['SP500', 'BTC', 'WTI']:
        ret = pd.read_csv(OLD/'data/returns'/f'{asset}.csv', index_col='date', parse_dates=True).log_return
        vals = ret.to_numpy(np.float32)
        paths = sorted((OLD/'native/lagllama'/asset).glob('*.npz'))
        for path in [paths[0], paths[len(paths)//2], paths[-1]]:
            with np.load(path) as z:
                pos = z['positions'][:BATCH]
                ctx = np.stack([vals[t-CONTEXT:t] for t in pos])
                calendars = [ret.index[t-CONTEXT:t+1] for t in pos]
                samples, _ = interface.predict(ctx, calendars, 'legacy', seed_for(asset, int(pos[0])))
                assert np.array_equal(samples, z['native'][:len(pos)]), (asset, path, 'legacy replay')
                # Test features against the upstream transform on an explicitly
                # regular calendar and the independent date attributes on the
                # real calendar. No target return is an input to either call.
                ft = features(calendars[0])
                independent = np.array([
                    [-.5, -.5, -.5, d.dayofweek/6-.5, (d.day-1)/30-.5,
                     (d.dayofyear-1)/365-.5] for d in calendars[0]], dtype=np.float32).T
                assert np.array_equal(ft, independent)
                rows.append({'asset':asset, 'chunk':path.name, 'rows':len(pos), 'legacy_exact':True})
    report = {'producer_sha256':sha(__file__), 'regular_calendar_samples_exact':True,
              'regular_calendar_parameters_exact':True, 'date_features_independently_checked':True,
              'legacy_replay':rows}
    (OUT/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Calendar input validation passed:', len(rows), 'legacy batches', flush=True)


def run_asset(interface, asset, device):
    started = time.monotonic()
    retpath = OLD/'data/returns'/f'{asset}.csv'
    ret = pd.read_csv(retpath, index_col='date', parse_dates=True).log_return
    vals = ret.to_numpy(np.float32)
    folder = OUT/'native'/asset
    folder.mkdir(parents=True, exist_ok=True)
    binding = dict(producer_sha256=sha(__file__), input_sha256=sha(retpath),
                   checkpoint_sha256=sha(OLD/'models/lagllama/lag-llama.ckpt'),
                   upstream_estimator_sha256=sha(OLD/'models/lag-llama-source/lag_llama/gluon/estimator.py'),
                   original_binding_sha256=sha(OLD/'native/lagllama'/asset/'binding.json'),
                   context=CONTEXT, samples=SAMPLES, batch=BATCH, device=device,
                   feature_names=[f.__name__ for f in FEATURES],
                   packages={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','gluonts']},
                   parameters=['df', 'effective_location', 'effective_scale'])
    bindfile = folder/'binding.json'
    if bindfile.exists():
        assert json.loads(bindfile.read_text()) == binding, 'Changed binding; do not overwrite a partial run'
    bindfile.write_text(json.dumps(binding, indent=2)+'\n')
    n = len(ret)-CONTEXT
    for offset in range(0, n, 512):
        target = folder/f'{offset:06d}.npz'
        if target.exists():
            assert sha(target) == json.loads(target.with_suffix('.json').read_text())['sha256']
            continue
        positions = np.arange(CONTEXT+offset, min(CONTEXT+offset+512, len(ret)))
        chunks = {key:[] for key in ['legacy_parameters','actual_parameters','actual_samples']}
        with np.load(OLD/'native/lagllama'/asset/target.name) as old:
            assert np.array_equal(positions, old['positions'])
            for start in range(0, len(positions), BATCH):
                pos = positions[start:start+BATCH]
                contexts = np.stack([vals[t-CONTEXT:t] for t in pos])
                dates = [ret.index[t-CONTEXT:t+1] for t in pos]
                seed = seed_for(asset, int(pos[0]))
                legacy, lp = interface.predict(contexts, dates, 'legacy', seed)
                assert np.array_equal(legacy, old['native'][start:start+len(pos)]), (asset, start, 'legacy mismatch')
                actual, cp = interface.predict(contexts, dates, 'actual', seed)
                chunks['legacy_parameters'].append(lp)
                chunks['actual_parameters'].append(cp)
                chunks['actual_samples'].append(actual)
        arrays = {key:np.concatenate(value) for key,value in chunks.items()}
        np.savez_compressed(target, positions=positions, date=ret.index[positions].values.astype('datetime64[D]'), **arrays)
        target.with_suffix('.json').write_text(json.dumps({'sha256':sha(target), 'rows':len(positions),
            'legacy_samples_exact':True}, indent=2)+'\n')
        print(asset, min(offset+512, n), '/', n, round(time.monotonic()-started, 1), 's', flush=True)
    (folder/'complete.json').write_text(json.dumps({'rows':n, 'binding_sha256':sha(bindfile)}, indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='mps')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--assets', nargs='+')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if args.device == 'mps':
        assert torch.backends.mps.is_available(), 'MPS unavailable'
    manifest = json.loads((OLD/'models/manifest.json').read_text())['lagllama']
    for name, digest in manifest['files'].items():
        assert sha(OLD/'models/lagllama'/name) == digest
    interface = Interface(args.device)
    validate(interface)
    if args.validate_only:
        return
    for asset in args.assets or sorted(p.stem for p in (OLD/'data/returns').glob('*.csv')):
        run_asset(interface, asset, args.device)


if __name__ == '__main__':
    main()
