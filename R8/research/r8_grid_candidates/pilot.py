"""Native quantile pilots, with observers that do not change model outputs."""
import os
for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
    os.environ[key] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['MPLCONFIGDIR'] = '/private/tmp/irfa-grid-matplotlib'
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '2'
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'artifacts/r8_grid_candidates'
DATA = PROJECT / 'artifacts/extension_20260831/data/returns'
LEVELS = np.arange(1, 100, dtype=float) / 100
CODE = {'patchtst': 'fe7a35697723e2a2f5246ae979474bfc554e26c0',
        'tsicl': '349f3eae4f01f78536b16a6ea53c0837760166ec'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as src:
        while block := src.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def difference(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return {'exact': bool(np.array_equal(a, b)), 'max_absolute': float(np.max(np.abs(a-b))),
            'max_relative_to_reference': float(np.max(np.abs(a-b) / np.maximum(np.abs(b), 1e-12)))}


def contexts(asset):
    path = DATA / f'{asset}.csv'
    series = pd.read_csv(path, index_col='date', parse_dates=True).log_return
    assert series.index.is_unique and series.index.is_monotonic_increasing
    assert series.index[-1] == pd.Timestamp('2026-08-31')
    positions = np.arange(len(series)-32, len(series))
    arrays = np.stack([series.iloc[p-512:p].to_numpy(dtype=np.float32) for p in positions])
    assert arrays.shape == (32, 512) and np.isfinite(arrays).all()
    for p, context in zip(positions, arrays):
        changed = series.to_numpy(copy=True)
        changed[p:] = 123456.
        assert np.array_equal(changed[p-512:p].astype(np.float32), context)
        assert series.index[p-1] < series.index[p]
    binding = {'input_sha256': sha(path), 'positions': positions.tolist(),
               'context_sha256': hashlib.sha256(arrays.tobytes()).hexdigest(),
               'future_perturbation_context_equal': True}
    return series.index[positions], arrays, binding


class PatchAdapter:
    def __init__(self, device):
        from tsfm_public.models.patchtst_fm import PatchTSTFMForPrediction
        self.device = device
        self.model, info = PatchTSTFMForPrediction.from_pretrained(
            str(ROOT / 'models/patchtst'), local_files_only=True,
            torch_dtype=torch.float32, output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), info
        self.model.to(device).eval()
        np.testing.assert_array_equal(self.model.config.quantile_levels, LEVELS)
        self.details = {'loading_info': info, 'parameters': sum(p.numel() for p in self.model.parameters()),
                        'config': self.model.config.to_dict(),
                        'precision': 'float32 weights; shipped bfloat16 autocast on MPS' if device == 'mps' else 'float32',
                        'internal_forecast_length': 128, 'observations': 512,
                        'normalization_checks': []}
        self.expected = None
        def before(module, args, kwargs):
            x = kwargs['inputs']; mask = kwargs['pred_mask'] | kwargs['pad_mask'] | kwargs['miss_mask']
            observed = x[~mask].reshape(len(x), 512).cpu().float().numpy()
            np.testing.assert_array_equal(observed, self.expected)
            assert torch.all(kwargs['pred_mask'].sum(1) == 128)
            assert torch.all(kwargs['pad_mask'].sum(1) == 8192-512-128)
            assert not kwargs['miss_mask'].any()
        def after(module, args, kwargs, output):
            mean, std = module.norm_fn.get_statistics()
            expected_mean = self.expected.astype(float).mean(1)
            expected_std = self.expected.astype(float).std(1)
            m = mean.flatten().cpu().float().numpy(); s = std.flatten().cpu().float().numpy()
            np.testing.assert_allclose(m, expected_mean, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(s, expected_std, rtol=1e-5, atol=1e-8)
            self.details['normalization_checks'].append({'count': len(m), 'past_only': True,
                'max_mean_error': float(np.max(abs(m-expected_mean))),
                'max_std_error': float(np.max(abs(s-expected_std)))})
        self.model.backbone.register_forward_pre_hook(before, with_kwargs=True)
        self.model.backbone.register_forward_hook(after, with_kwargs=True)

    def predict(self, arrays, only=False):
        results = []
        for start in range(0, len(arrays), 16):
            self.expected = arrays[start:start+16]
            tensor = torch.tensor(self.expected[:, :, None], device=self.device, dtype=torch.float32)
            with torch.inference_mode():
                result = self.model(past_values=tensor, prediction_length=1,
                                    quantile_levels=[.01] if only else None,
                                    return_dict=True).quantile_outputs
            assert result.shape == (len(tensor), 1 if only else 99, 1, 1), result.shape
            results.append(result[:, :, 0, 0].float().cpu().numpy())
        return np.concatenate(results)


class TSAdapter:
    def __init__(self, device):
        from tsicl import TSICL
        assert device == 'cpu', 'Use the unmodified official CPU path on this Mac'
        path = ROOT / 'models/tsicl/tsicl-v1.ckpt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert set(checkpoint) >= {'config', 'forecaster', 'imputer'}
        allowed = {'tsicl.TSICLNetwork', 'tsicl.model.encoder.PerceiverEncoder',
                   'tsicl.model.icl_learning.ICLearningCrossAttn'}
        def check_targets(item):
            if isinstance(item, dict):
                if '_target_' in item: assert item['_target_'] in allowed
                for value in item.values(): check_targets(value)
        check_targets(checkpoint['config'])
        self.model = TSICL(model_path=str(path), allow_auto_download=False)
        for name in ('forecaster', 'imputer'):
            state = getattr(self.model, name).state_dict()
            assert state.keys() == checkpoint[name].keys()
            assert all(torch.equal(state[k], checkpoint[name][k]) for k in state)
        self.details = {'parameters_forecaster': sum(p.numel() for p in self.model.forecaster.parameters()),
                        'checkpoint_states_exact': True, 'precision': 'float32',
                        'config': checkpoint['config'], 'normalization_checks': []}
        head = self.model.forecaster.tf_icl
        assert (head.start_quantile, head.end_quantile, head.nb_quantiles) == (.01, .99, 99)
        try:
            self.model._get_quantile_indices(LEVELS.tolist())
            self.details['full_grid_public_selector'] = 'accepted'
        except ValueError as error:
            self.details['full_grid_public_selector'] = str(error)
        self.expected = None
        self.captured = []
        rollout = self.model._rollout_f
        def observer(**kwargs):
            raw = kwargs['series_c'].squeeze(-1).cpu().numpy()
            offset = sum(len(x) for x in self.captured)
            np.testing.assert_array_equal(raw, self.expected[offset:offset+len(raw)])
            assert kwargs['covariates'] is None and not kwargs['has_covar']
            self.raw = raw
            result = rollout(**kwargs)
            assert result.shape == (len(raw), 1, 99), result.shape
            self.captured.append(result[:, 0].detach().cpu().numpy().copy())
            return result
        self.model._rollout_f = observer
        def before(module, args, kwargs):
            x = torch.tensor(self.raw[:, :, None])
            mean = x.mean(1, keepdim=True)
            std = ((x-mean).square().mean(1, keepdim=True)).sqrt()
            expected = (x-mean)/std
            np.testing.assert_allclose(kwargs['series'].cpu().numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)
            assert kwargs['series'].shape[1] == 512
            assert kwargs['target_coords'].shape[1] == 513
            assert torch.all(kwargs['target_coords'][:, -1] > kwargs['coords'][:, -1])
            self.details['normalization_checks'].append({'count': len(x), 'past_only': True,
                                                        'context_length': 512, 'future_queries': 1})
        self.model.forecaster.register_forward_pre_hook(before, with_kwargs=True)
        def no_imputation(*args, **kwargs):
            raise AssertionError('Imputer used in a forecasting-only pilot')
        self.model.imputer.register_forward_pre_hook(no_imputation)

    def predict(self, arrays, only=False):
        self.expected = arrays
        self.captured = []
        with torch.inference_mode():
            _, selected = self.model.forecast(inputs=arrays, prediction_length=1, context_length=512,
                batch_size=16, quantile_levels=[.01, .5, .99], device=torch.device('cpu'),
                denormalize=True, allow_auto_complete=False, allow_covar_forecast=False, squeeze_output=False)
        native = np.concatenate(self.captured)
        assert selected.shape == (len(arrays), 1, 1, 3), selected.shape
        np.testing.assert_array_equal(selected[:, 0, 0].numpy(), native[:, [0, 49, 98]])
        assert self.model._device.type == 'cpu'
        return native[:, :1] if only else native


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(CODE), required=True)
    parser.add_argument('--device', choices=['mps', 'cpu'], required=True)
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--cpu-check', action='store_true')
    args = parser.parse_args()
    assert not (args.replay and args.cpu_check)
    if args.cpu_check: assert args.model == 'patchtst' and args.device == 'cpu'
    out = ROOT / args.model; out.mkdir(exist_ok=True)
    phase = 'replay' if args.replay else ('cpu_check' if args.cpu_check else 'pilot')
    receipt = {'model': args.model, 'phase': phase, 'device': args.device, 'seed': 20260910,
               'protocol_sha256': sha(Path(__file__).with_name('PROTOCOL.md')),
               'producer_sha256': sha(__file__), 'python': sys.version, 'platform': platform.platform(),
               'versions': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
               'bindings': {}, 'timings': [], 'checks': [], 'native_q01': True,
               'external_tail_fit': False, 'half_normal_tail': False}
    try:
        for kind in ('source', 'weights'):
            path = ROOT / f'{kind}_manifest.json'
            receipt[kind+'_manifest_sha256'] = sha(path)
            for name, entry in json.loads(path.read_text())['files'].items():
                if f'/{args.model}/' in name or name.startswith('sources/'+args.model+'-'):
                    assert sha(ROOT / name) == entry['sha256']
        source = next((ROOT / 'sources' / f'{args.model}-{CODE[args.model]}').iterdir())
        sys.path.insert(0, str(source if args.model == 'patchtst' else source / 'src'))
        random.seed(20260910); np.random.seed(20260910); torch.manual_seed(20260910)
        torch.set_num_threads(2); torch.set_num_interop_threads(2)
        if args.device == 'mps':
            assert torch.backends.mps.is_available()
            torch.mps.manual_seed(20260910)
        start = time.perf_counter()
        adapter = PatchAdapter(args.device) if args.model == 'patchtst' else TSAdapter(args.device)
        receipt['load_seconds'] = time.perf_counter()-start
        records = []
        for asset in ('SP500', 'BTC'):
            dates, arrays, binding = contexts(asset)
            receipt['bindings'][asset] = binding
            use = [0, 31] if args.cpu_check else np.arange(32)
            start = time.perf_counter()
            native = adapter.predict(arrays[use])
            elapsed = time.perf_counter()-start
            assert native.shape == (len(use), 99) and np.isfinite(native).all()
            receipt['timings'].append({'asset': asset, 'dates': len(use), 'seconds': elapsed})
            path = out / f'{asset}_native.npz'
            checks = {'asset': asset, 'dates_with_crossing': int(np.any(np.diff(native, axis=1) < 0, axis=1).sum()),
                      'q01_crossing_dates': int(np.any(native[:, :1] > native[:, 1:], axis=1).sum()),
                      'max_adjacent_reversal': float(max(0, -np.diff(native, axis=1).min()))}
            if args.replay or args.cpu_check:
                reference = np.load(path)['native'][use]
                checks['replay' if args.replay else 'cpu_vs_mps'] = difference(native, reference)
                if args.cpu_check:
                    checks['cpu_vs_mps_q01'] = difference(native[:, 0], reference[:, 0])
                    np.savez_compressed(out / f'{asset}_cpu.npz', native=native, levels=LEVELS, dates=dates[use].to_numpy())
            else:
                assert not path.exists(), 'Refusing to overwrite existing pilot outputs'
                np.savez_compressed(path, native=native, levels=LEVELS, dates=dates.to_numpy(), contexts=arrays)
                for i, date in enumerate(dates):
                    records.append({'asset': asset, 'date': str(date.date()), 'q01': float(native[i, 0])})
            # Same four cases, evaluated individually with the identical numerical backend.
            single = np.concatenate([adapter.predict(arrays[[i]]) for i in (0, 31)])
            reference = native if args.cpu_check else native[[0, 31]]
            checks['single_vs_batch'] = difference(single, reference)
            only = adapter.predict(arrays[[0, 31]], only=True)
            paired = adapter.predict(arrays[[0, 31]])
            checks['public_selected_q01'] = difference(only[:, 0], paired[:, 0])
            assert checks['public_selected_q01']['exact']
            receipt['checks'].append(checks)
            print(args.model, asset, phase, round(elapsed, 3), 'seconds', checks, flush=True)
        receipt['model_details'] = adapter.details
        if phase == 'pilot':
            pd.DataFrame(records).to_csv(out / 'quantiles.csv', index=False)
        receipt['outputs'] = {p.name: sha(p) for p in sorted(out.glob('*.npz'))}
        if (out / 'quantiles.csv').exists(): receipt['outputs']['quantiles.csv'] = sha(out / 'quantiles.csv')
        if phase == 'replay':
            receipt['all_native_values_replay_exact'] = all(c['replay']['exact'] for c in receipt['checks'])
            assert receipt['all_native_values_replay_exact'], 'Fresh-process replay differs; inspect checks'
        receipt['status'] = 'complete'
        dump(out / f'{phase}.json', receipt)
    except Exception:
        receipt['status'] = 'failed'
        receipt['traceback'] = traceback.format_exc()
        dump(out / f'{phase}_failure_{time.time_ns()}.json', receipt)
        raise


if __name__ == '__main__':
    main()
