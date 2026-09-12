#!/usr/bin/env python3
"""Verify native replay and extract existing quantiles without tail completion."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT/'artifacts/r8_commodity_etp'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--models', nargs='+',
        default=['moirai', 'lagllama', 'chronos2', 'patchtst']); args = parser.parse_args()
    records = []
    for model in args.models:
        folder = ROOT/'native'/model
        complete = json.loads((folder/'complete.json').read_text())
        replay = json.loads((folder/'replay.json').read_text())
        assert replay['exact_fresh_replay'] and replay['status'] == 'complete'
        assert replay['binding_sha256'] == complete['binding_sha256'] == sha(folder/'binding.json')
        for item in complete['assets']:
            asset = item['asset']; inp = ROOT/'data/returns'/f'{asset}.csv'
            assert sha(inp) == item['input_sha256']
            series = pd.read_csv(inp, index_col='date', parse_dates=True).log_return
            frames, parameters = [], []
            for chunk in item['chunks']:
                path = folder/asset/chunk['file']; assert sha(path) == chunk['sha256']
                with np.load(path, allow_pickle=False) as data:
                    pos = data['positions']; dates = pd.DatetimeIndex(data['dates'], name='date')
                    assert dates.equals(series.index[pos])
                    observed = series.to_numpy(np.float32)
                    contexts = np.stack([observed[p-512:p] for p in pos])
                    assert hashlib.sha256(contexts.astype('<f4').tobytes()).hexdigest() == chunk['contexts_sha256']
                    native = data['native']
                    assert np.isfinite(native).all()
                    if 'levels' not in data:
                        assert native.shape[1] == 1000
                        alpha = [.01, .025, .05, .1]
                        q = np.percentile(native, np.asarray(alpha)*100, axis=1).T
                        frame = pd.DataFrame({'mean':native.mean(1), 'std':native.std(1)}, index=dates)
                    else:
                        alpha = [.01, .05, .1]
                        ix = [np.flatnonzero(data['levels'] == a).item() for a in alpha]
                        q = native[:, ix].astype(float); frame = pd.DataFrame(index=dates)
                        assert not (native[:, :1] > native[:, 1:]).any(), '1% grid crossing; inspect before admission'
                    for j, a in enumerate(alpha): frame[f'VaR_{a:g}'] = q[:, j]
                    frames.append(frame)
                    if 'parameters' in data:
                        parameters.append(pd.DataFrame(data['parameters'], index=dates,
                                                      columns=['df', 'effective_location', 'effective_scale']))
            frame = pd.concat(frames)
            assert frame.index.equals(series.index[512:]) and len(frame) == item['rows']
            out = ROOT/'data'/model/f'{asset}.parquet'; out.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(out)
            if parameters:
                target = ROOT/'parameters'/model/f'{asset}.parquet'; target.parent.mkdir(parents=True, exist_ok=True)
                pd.concat(parameters).to_parquet(target)
            records.append(dict(model=model, asset=asset, rows=len(frame), output_sha256=sha(out),
                native_binding_sha256=complete['binding_sha256'], input_sha256=sha(inp), exact_native_replay=True))
            print(model, asset, len(frame), 'forecasts reduced', flush=True)
    (ROOT/'quality'/('reduction_'+'_'.join(args.models)+'.json')).write_text(json.dumps(
        dict(producer_sha256=sha(__file__), records=records), indent=2)+'\n')


if __name__ == '__main__': main()
