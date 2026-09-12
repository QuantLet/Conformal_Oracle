"""Repeat the selected configuration after recreating its dependency environment."""
import argparse
import contextlib
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

from pilot import ROOT, CODE, PatchAdapter, TSAdapter, contexts, difference, dump, sha, np, torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['patchtst', 'tsicl'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(2); torch.manual_seed(20260910)
    source = next((ROOT/'sources'/f'{args.model}-{CODE[args.model]}').iterdir())
    sys.path.insert(0, str(source if args.model == 'patchtst' else source/'src'))
    if args.model == 'patchtst':
        assert torch.backends.mps.is_available(); torch.mps.manual_seed(20260910)
        module = importlib.import_module('tsfm_public.models.patchtst_fm.modeling_patchtst_fm')
        module.get_autocast_context = lambda device: contextlib.nullcontext()
        adapter = PatchAdapter('mps'); output = ROOT/'patchtst_float32'
    else:
        adapter = TSAdapter('cpu'); output = ROOT/'tsicl'
    receipt = {'python_executable': sys.executable, 'prefix': sys.prefix,
               'bootstrap_sha256': sha(Path(__file__).with_name('bootstrap.py')),
               'producer_sha256': sha(__file__), 'adapter_sha256': sha(Path(__file__).with_name('pilot.py')),
               'source_manifest_sha256': sha(ROOT/'source_manifest.json'),
               'weights_manifest_sha256': sha(ROOT/'weights_manifest.json'),
               'lock_sha256': sha(ROOT/'source_review'/f'{args.model}_requirements.lock'),
               'versions': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}, 'checks': []}
    original = json.loads((ROOT/args.model/'pilot.json').read_text())['versions']
    ignored = {'pip', 'setuptools', 'granite-tsfm', 'tsicl'}
    for name, version in original.items():
        if name.lower() not in ignored:
            assert receipt['versions'].get(name) == version, (name, version, receipt['versions'].get(name))
    for asset in ('SP500', 'BTC'):
        _, arrays, _ = contexts(asset)
        forecast = adapter.predict(arrays)
        reference = np.load(output/f'{asset}_native.npz')['native']
        check = {'asset': asset, **difference(forecast, reference)}
        assert check['exact'], check
        receipt['checks'].append(check)
    receipt['native_values_reproduced'] = 2*32*99
    receipt['status'] = 'passed'
    dump(output/'environment_replay.json', receipt)
    print(json.dumps({'status': 'passed', 'model': args.model, 'checks': receipt['checks']}, indent=2))


if __name__ == '__main__': main()
