"""Bind current support and assemble exact already-replayed native grids."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scope import PROJECT, ROOT, GRID, CHRONOS, FUNDS, EXT, ASSETS, sha, dump, bind


def main():
    assert not (ROOT/'preparation.json').exists(), 'Completed preparation is immutable'
    (ROOT/'full_preflight').mkdir(parents=True, exist_ok=True)
    sources = bind([Path(__file__), Path(__file__).with_name('scope.py'),
                    Path(__file__).with_name('PROTOCOL.md')])
    support, records = [], []
    for asset in ASSETS:
        path = EXT/'data/returns'/f'{asset}.csv'
        sources.update(bind([path]))
        series = pd.read_csv(path, index_col='date', parse_dates=True).log_return
        assert series.index.is_unique and series.index.is_monotonic_increasing
        assert np.isfinite(series).all() and series.index[-1].strftime('%Y-%m') == '2026-08'
        dates = series.index[512:]; nc = int(.7*len(dates))
        support.append(dict(asset=asset, input_sha256=sha(path), eligible_forecasts=len(dates),
            n_cal=nc, n_test=len(dates)-nc, first_forecast=str(dates[0].date()),
            first_test=str(dates[nc].date()), last_date=str(dates[-1].date())))
        values = series.to_numpy(np.float32)
        contexts = np.stack([values[t-512:t] for t in range(512, len(values))])
        for model, short, old in [('PatchTST-FM','patchtst',GRID/'patchtst_full'),
                                  ('Chronos-2','chronos2',CHRONOS)]:
            destination = ROOT/(short+'_full')/f'{asset}.npz'
            destination.parent.mkdir(exist_ok=True)
            if asset in ['USO','GLD','UNG']:
                folder = FUNDS/'native'/short/asset
                complete = json.loads((folder/'complete.json').read_text())
                replay = json.loads((folder/'replay.json').read_text())
                assert replay['exact_fresh_replay'] and complete['chunks'] == replay['chunks']
                sources.update(bind([folder/'complete.json', folder/'replay.json', folder.parent/'binding.json']))
                native, positions, stored_dates, levels = [], [], [], None
                offset = 0
                for chunk in complete['chunks']:
                    fp = folder/chunk['file']; sources.update(bind([fp]))
                    assert sha(fp) == chunk['sha256'] and chunk['input_sha256'] == sha(path)
                    saved = np.load(fp, allow_pickle=False); n = len(saved['positions'])
                    assert hashlib.sha256(contexts[offset:offset+n].astype('<f4').tobytes()).hexdigest() == chunk['contexts_sha256']
                    if levels is not None: np.testing.assert_array_equal(levels, saved['levels'])
                    levels = saved['levels']; native.append(saved['native']); positions.append(saved['positions'])
                    stored_dates.append(saved['dates']); offset += n
                native = np.concatenate(native); positions = np.concatenate(positions); stored_dates = np.concatenate(stored_dates)
                assert offset == len(dates)
                np.savez_compressed(destination, native=native, positions=positions, dates=stored_dates, levels=levels)
            else:
                complete = json.loads((old/'complete.json').read_text())
                replay = json.loads((old/'replay.json').read_text())
                assert replay.get('fresh_replay_exact', replay.get('exact_fresh_replay'))
                first = next(x for x in complete['assets'] if x['asset'] == asset)
                second = next(x for x in replay['assets'] if x['asset'] == asset)
                fp = old/f'{asset}.npz'
                sources.update(bind([fp, old/'complete.json', old/'replay.json']))
                assert first['input_sha256'] == second['input_sha256'] == sha(path)
                assert first['output_sha256'] == second['output_sha256'] == sha(fp)
                if short == 'patchtst':
                    assert hashlib.sha256(contexts.tobytes()).hexdigest() == first['contexts_sha256'] == second['contexts_sha256']
                shutil.copy2(fp, destination)
            saved = np.load(destination, allow_pickle=False)
            np.testing.assert_array_equal(saved['dates'], dates.to_numpy())
            np.testing.assert_array_equal(saved['positions'], np.arange(512, len(series)))
            assert saved['native'].shape == (len(dates), len(saved['levels']))
            assert np.isfinite(saved['native']).all() and np.count_nonzero(saved['levels'] == .01) == 1
            records.append(dict(model=model, asset=asset, rows=len(dates), native_values=saved['native'].size,
                output=str(destination.relative_to(ROOT)), sha256=sha(destination),
                crossings=int((np.diff(saved['native'],axis=1)<0).any(1).sum()),
                q01_crossings=int((saved['native'][:,:1]>saved['native'][:,1:]).any(1).sum()),
                fresh_native_replay_exact=True))
    pd.DataFrame(support).to_csv(ROOT/'full_preflight/support.csv', index=False)
    protected = bind([PROJECT/'source'/f'{name}.{ext}' for name in ['main_R8','supplement_R8'] for ext in ['tex','pdf']])
    dump(ROOT/'common_evaluation_protected.json', protected)
    sources.update(bind([ROOT/'full_preflight/support.csv']))
    for name, expected in sources.items(): assert sha(PROJECT/name) == expected
    dump(ROOT/'preparation.json', dict(status='complete', binding=sources, native=records,
        assets=len(support), forecasts_per_model=sum(x['eligible_forecasts'] for x in support),
        asset_test_dates=sum(x['n_test'] for x in support)))
    print('Prepared',len(support),'assets;',sum(x['eligible_forecasts'] for x in support),'forecasts/model',flush=True)


if __name__ == '__main__': main()
