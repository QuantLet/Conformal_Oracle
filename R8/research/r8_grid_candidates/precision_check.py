"""Audit MPS float32 against the preserved official mixed-precision pilot."""
import contextlib
import importlib
import sys
import time
import argparse
from pathlib import Path

from pilot import ROOT, CODE, LEVELS, PatchAdapter, contexts, difference, dump, sha, np, torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.manual_seed(20260910); torch.mps.manual_seed(20260910)
    assert torch.backends.mps.is_available()
    source = next((ROOT/'sources'/f"patchtst-{CODE['patchtst']}").iterdir())
    sys.path.insert(0, str(source))
    module = importlib.import_module('tsfm_public.models.patchtst_fm.modeling_patchtst_fm')
    module.get_autocast_context = lambda device: contextlib.nullcontext()
    out = ROOT/'patchtst_float32'; out.mkdir(exist_ok=True)
    start = time.perf_counter()
    model = PatchAdapter('mps')
    model.details['precision'] = 'float32 weights and operations; autocast disabled by recorded adapter'
    receipt = {'model': 'patchtst', 'device': 'mps', 'precision': 'float32',
               'load_seconds': time.perf_counter()-start, 'checks': [], 'bindings': {},
               'producer_sha256': sha(__file__), 'pilot_producer_sha256': sha(Path(__file__).with_name('pilot.py')),
               'protocol_sha256': sha(Path(__file__).with_name('NUMERICS.md')),
               'weights_manifest_sha256': sha(ROOT/'weights_manifest.json'),
               'source_manifest_sha256': sha(ROOT/'source_manifest.json')}
    for asset in ('SP500', 'BTC'):
        dates, arrays, binding = contexts(asset); receipt['bindings'][asset] = binding
        start = time.perf_counter(); native = model.predict(arrays)
        entry = {'asset': asset, 'seconds': time.perf_counter()-start, 'dates': len(native)}
        assert np.isfinite(native).all()
        entry['dates_with_crossing'] = int(np.any(np.diff(native, axis=1) < 0, axis=1).sum())
        path = out/f'{asset}_native.npz'
        if args.replay:
            entry['replay'] = difference(native, np.load(path)['native'])
            assert entry['replay']['exact'], entry
        else:
            assert not path.exists()
            np.savez_compressed(path, native=native, levels=LEVELS, dates=dates.to_numpy(), contexts=arrays)
        cpu = np.load(ROOT/'patchtst'/f'{asset}_cpu.npz')['native']
        entry['cpu_comparison'] = difference(native[[0,31]], cpu)
        entry['cpu_comparison_q01'] = difference(native[[0,31],0], cpu[:,0])
        entry['cpu_within_tolerance'] = bool(np.allclose(native[[0,31]], cpu, rtol=1e-5, atol=1e-7))
        single = np.concatenate([model.predict(arrays[[i]]) for i in (0,31)])
        entry['single_vs_batch'] = difference(single, native[[0,31]])
        entry['batch_within_tolerance'] = bool(np.allclose(single, native[[0,31]], rtol=1e-5, atol=1e-7))
        entry['single_vs_batch_q01'] = difference(single[:,0], native[[0,31],0])
        mixed = np.load(ROOT/'patchtst'/f'{asset}_native.npz')['native']
        entry['mixed_precision_q01'] = difference(mixed[:,0], native[:,0])
        receipt['checks'].append(entry)
        print(asset, entry, flush=True)
    receipt['details'] = model.details
    receipt['outputs'] = {p.name: sha(p) for p in out.glob('*.npz')}
    receipt['all_consistency_checks_pass'] = all(x['cpu_within_tolerance'] and x['batch_within_tolerance'] for x in receipt['checks'])
    dump(out/('replay.json' if args.replay else 'pilot.json'), receipt)


if __name__ == '__main__': main()
