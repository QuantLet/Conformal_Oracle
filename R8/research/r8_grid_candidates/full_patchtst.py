"""Full native PatchTST float32 inference; immutable per-asset evidence."""
import argparse
import contextlib
import importlib
import importlib.metadata as md
import json
from pathlib import Path
import sys
import tarfile
import time
import traceback

from pilot import PROJECT, ROOT, DATA, CODE, LEVELS, PatchAdapter, sha, dump, np, pd, torch

OUT = ROOT/'patchtst_full'


def bindings():
    result = {str(p.relative_to(PROJECT)): sha(p) for p in
              [Path(__file__), Path(__file__).with_name('pilot.py'),
               Path(__file__).with_name('FULL_PROTOCOL.md'), ROOT/'source_manifest.json',
               ROOT/'weights_manifest.json', ROOT/'full_preflight/support.csv']}
    for kind in ('source', 'weights'):
        for name, item in json.loads((ROOT/f'{kind}_manifest.json').read_text())['files'].items():
            if '/patchtst/' in name or name.startswith('sources/patchtst-'):
                assert sha(ROOT/name) == item['sha256'], name
                result[str((ROOT/name).relative_to(PROJECT))] = item['sha256']
    source = next((ROOT/'sources'/f"patchtst-{CODE['patchtst']}").iterdir())
    archive = ROOT/'sources'/f"patchtst-{CODE['patchtst']}.tar.gz"
    checked = 0
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            if member.isfile() and member.name.endswith('.py'):
                assert (source.parent/member.name).read_bytes() == stream.extractfile(member).read()
                checked += 1
    result['python_source_files_verified'] = checked
    return result, source


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--replay', action='store_true')
    args = parser.parse_args(); OUT.mkdir(exist_ok=True)
    phase = 'replay' if args.replay else 'production'
    binding, source = bindings()
    support = pd.read_csv(ROOT/'full_preflight/support.csv').set_index('asset')
    assert len(support) == 24
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.manual_seed(20260910); assert torch.backends.mps.is_available(); torch.mps.manual_seed(20260910)
    sys.path.insert(0, str(source))
    module = importlib.import_module('tsfm_public.models.patchtst_fm.modeling_patchtst_fm')
    module.get_autocast_context = lambda device: contextlib.nullcontext()
    start = time.perf_counter(); adapter = PatchAdapter('mps')
    adapter.details['precision'] = 'float32; autocast disabled by recorded adapter'
    receipt = {'phase': phase, 'model': 'PatchTST-FM', 'binding': binding,
               'load_seconds': time.perf_counter()-start, 'device': 'mps', 'precision': 'float32',
               'batch_size': 16, 'outer_chunk': 512, 'context': 512, 'horizon': 1,
               'seed': 20260910, 'assets': [], 'python': sys.version,
               'packages': {d.metadata['Name']: d.version for d in md.distributions()}}
    try:
        for asset, expected in support.iterrows():
            path = DATA/f'{asset}.csv'; assert sha(path) == expected.input_sha256
            series = pd.read_csv(path, index_col='date', parse_dates=True).log_return
            assert series.index.is_unique and series.index.is_monotonic_increasing
            assert np.isfinite(series).all() and str(series.index[-1].date()) == expected.last_date
            values = series.to_numpy(dtype=np.float32); positions = np.arange(512, len(values))
            assert len(positions) == expected.eligible_forecasts
            output = OUT/f'{asset}.npz'; note = OUT/f'{asset}.json'
            if not args.replay and note.exists():
                saved = json.loads(note.read_text())
                assert saved['binding'] == binding and saved['input_sha256'] == sha(path)
                assert saved['output_sha256'] == sha(output)
                receipt['assets'].append(saved); print(asset, 'verified previous complete asset', flush=True)
                continue
            assert args.replay or not output.exists(), 'Unreceipted output exists; inspect before resuming'
            begin = time.perf_counter(); chunks = []; import hashlib
            contexts_digest = hashlib.sha256()
            for first in range(0, len(positions), 512):
                pos = positions[first:first+512]
                context = np.stack([values[p-512:p] for p in pos])
                contexts_digest.update(context.tobytes())
                prediction = adapter.predict(context)
                assert prediction.shape == (len(pos), 99) and np.isfinite(prediction).all()
                chunks.append(prediction)
                dump(OUT/f'{phase}_progress.json', {'asset': asset, 'forecasts_in_asset': first+len(pos),
                     'asset_total': len(positions), 'completed_assets': len(receipt['assets']),
                     'seconds_in_asset': time.perf_counter()-begin})
            native = np.concatenate(chunks); torch.mps.synchronize(); elapsed = time.perf_counter()-begin
            if args.replay:
                old = np.load(output, allow_pickle=False)
                np.testing.assert_array_equal(old['native'], native)
                np.testing.assert_array_equal(old['positions'], positions)
                np.testing.assert_array_equal(old['dates'], series.index[positions].to_numpy())
                np.testing.assert_array_equal(old['levels'], LEVELS)
                saved = json.loads(note.read_text())
                assert saved['binding'] == binding
                assert contexts_digest.hexdigest() == saved['contexts_sha256']
                assert sha(output) == saved['output_sha256']
            else:
                temporary = OUT/f'{asset}.partial.npz'
                np.savez_compressed(temporary, native=native, levels=LEVELS, positions=positions,
                                    dates=series.index[positions].to_numpy())
                temporary.replace(output)
            record = {'asset': asset, 'binding': binding, 'rows': len(positions), 'seconds': elapsed,
                      'input_sha256': sha(path), 'output_sha256': sha(output),
                      'contexts_sha256': contexts_digest.hexdigest(),
                      'native_crossing_dates': int((np.diff(native, axis=1)<0).any(1).sum()),
                      'q01_crossing_dates': int((native[:,:1]>native[:,1:]).any(1).sum()),
                      'max_adjacent_reversal': float(max(0., -np.diff(native, axis=1).min())),
                      'first_forecast': str(series.index[512].date()), 'last_forecast': str(series.index[-1].date()),
                      'fresh_replay_exact': args.replay}
            if not args.replay: dump(note, record)
            receipt['assets'].append(record)
            print(asset, len(native), 'forecasts', round(elapsed, 2), 'seconds',
                  record['native_crossing_dates'], 'crossings', phase, flush=True)
        receipt['rows'] = sum(x['rows'] for x in receipt['assets']); assert receipt['rows'] == 124894
        receipt['native_values'] = receipt['rows']*99
        receipt['inference_seconds'] = sum(x['seconds'] for x in receipt['assets'])
        receipt['model_details'] = adapter.details
        receipt['status'] = 'complete'; receipt['fresh_replay_exact'] = args.replay
        dump(OUT/('replay.json' if args.replay else 'complete.json'), receipt)
        print(phase, 'complete', receipt['rows'], 'forecasts', flush=True)
    except Exception:
        receipt['status'] = 'failed'; receipt['traceback'] = traceback.format_exc()
        dump(OUT/f'{phase}_failure_{time.time_ns()}.json', receipt)
        raise


if __name__ == '__main__': main()
