"""Resumable official CPU TS-ICL inference, with full fresh-process replay."""
import argparse
import hashlib
import importlib.metadata as md
import json
import sys
import tarfile
import time
from pathlib import Path
from scope import PROJECT, ROOT, GRID, EXT, sha, dump, bind

sys.path.insert(0, str(PROJECT/'research/r8_grid_candidates'))
from pilot import TSAdapter, CODE, LEVELS, np, pd, torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', required=True)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    source = next((GRID/'sources'/f"tsicl-{CODE['tsicl']}").iterdir())
    archive = GRID/'sources'/f"tsicl-{CODE['tsicl']}.tar.gz"
    binding = bind([Path(__file__), Path(__file__).with_name('scope.py'),
        Path(__file__).with_name('PROTOCOL.md'), PROJECT/'research/r8_grid_candidates/pilot.py',
        ROOT/'full_preflight/support.csv', GRID/'source_manifest.json', GRID/'weights_manifest.json'])
    for kind in ['source', 'weights']:
        for name, item in json.loads((GRID/f'{kind}_manifest.json').read_text())['files'].items():
            if '/tsicl/' in name or name.startswith('sources/tsicl-'):
                assert sha(GRID/name) == item['sha256']; binding.update(bind([GRID/name]))
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            if member.isfile() and member.name.endswith('.py'):
                path = source.parent/member.name
                assert path.read_bytes() == stream.extractfile(member).read()
                binding.update(bind([path]))
    sys.path.insert(0, str(source/'src'))
    torch.set_num_threads(2); torch.set_num_interop_threads(2); torch.manual_seed(20260910)
    row = pd.read_csv(ROOT/'full_preflight/support.csv').set_index('asset').loc[args.asset]
    rp = EXT/'data/returns'/f'{args.asset}.csv'; assert sha(rp) == row.input_sha256
    series = pd.read_csv(rp,index_col='date',parse_dates=True).log_return
    values = series.to_numpy(np.float32); positions = np.arange(512,len(series))
    folder = ROOT/'tsicl_full'/args.asset; folder.mkdir(parents=True, exist_ok=True)
    phase = 'replay' if args.replay else 'production'
    begin = time.monotonic(); adapter = TSAdapter('cpu'); records = []
    for start in range(0,len(positions),512):
        pos = positions[start:start+512]; fp = folder/f'{start:06d}.npz'; note = fp.with_suffix('.json')
        contexts = np.stack([values[t-512:t] for t in pos])
        chash = hashlib.sha256(contexts.tobytes()).hexdigest()
        if fp.exists() and not args.replay:
            old = json.loads(note.read_text())
            assert old['binding'] == binding and old['input_sha256'] == sha(rp)
            assert old['sha256'] == sha(fp) and old['contexts_sha256'] == chash
            records.append(old); continue
        native = adapter.predict(contexts)
        assert native.shape == (len(pos),99) and np.isfinite(native).all()
        arrays = dict(native=native, positions=pos, dates=series.index[pos].to_numpy(), levels=LEVELS)
        if args.replay:
            old = np.load(fp,allow_pickle=False)
            for key,value in arrays.items(): np.testing.assert_array_equal(old[key],value)
            record = json.loads(note.read_text())
            assert record['sha256'] == sha(fp) and record['binding'] == binding
            assert record['input_sha256'] == sha(rp) and record['contexts_sha256'] == chash
        else:
            temporary = fp.with_suffix('.partial.npz'); np.savez_compressed(temporary,**arrays); temporary.replace(fp)
            record = dict(binding=binding,input_sha256=sha(rp),contexts_sha256=chash,sha256=sha(fp),
                file=fp.name, rows=len(pos), crossings=int((np.diff(native,axis=1)<0).any(1).sum()),
                q01_crossings=int((native[:,:1]>native[:,1:]).any(1).sum()))
            dump(note,record)
        records.append(record)
        progress = dict(asset=args.asset,phase=phase,rows=start+len(pos),total=len(positions),seconds=time.monotonic()-begin)
        dump(folder/f'{phase}_progress.json',progress); print(progress,flush=True)
    assert sum(x['rows'] for x in records) == row.eligible_forecasts
    for name, expected in binding.items(): assert sha(PROJECT/name) == expected
    dump(folder/('replay.json' if args.replay else 'complete.json'),dict(status='complete',asset=args.asset,
        binding=binding,input_sha256=sha(rp),chunks=records,rows=len(positions),native_values=len(positions)*99,
        fresh_replay_exact=args.replay,device='cpu',dtype='float32',threads=2,batch=16,chunk=512,seed=20260910,
        seconds=time.monotonic()-begin,model_details=adapter.details,python=sys.version,
        packages={d.metadata['Name']:d.version for d in md.distributions()}))


if __name__ == '__main__': main()
