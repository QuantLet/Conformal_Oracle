#!/usr/bin/env python3
"""Independent raw-price, native-quantile and daily-score reconstruction."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT/'artifacts/r8_commodity_etp'
MODELS = {'Moirai-1.1', 'Lag-Llama', 'GJR-GARCH', 'GJR-GARCH-t',
          'GARCH-N', 'Hist-Sim', 'EWMA'}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--data-only', action='store_true')
    args = parser.parse_args()
    receipt = json.loads((ROOT/'data_admission.json').read_text())
    counts = dict(returns=0, native_quantiles=0, posthoc_losses=0)
    assert {item['asset'] for item in receipt['assets']} == {'USO', 'GLD', 'UNG'}
    if not args.data_only:
        metrics = pd.read_csv(ROOT/'results/posthoc.csv')
        ledger = pd.read_csv(ROOT/'results/indication.csv')
        pairs = {(m, a) for m in MODELS for a in ['USO', 'GLD', 'UNG']}
        assert set(zip(metrics.model, metrics.asset)) == pairs
        assert len(metrics) == 210 and not metrics.duplicated(['model', 'asset', 'method']).any()
        assert metrics.groupby(['model', 'asset']).size().eq(10).all()
        assert set(zip(ledger.model, ledger.asset)) == pairs
        assert len(ledger) == 84 and not ledger.duplicated(['model', 'asset', 'alpha']).any()
    for item in receipt['assets']:
        asset = item['asset']; path = ROOT/'data/returns'/f'{asset}.csv'
        assert sha(path) == item['input_sha256']
        raw = json.loads((ROOT/'raw'/f'{asset}.json').read_text())['chart']['result'][0]
        price = raw['indicators']['adjclose'][0]['adjclose']
        returns = pd.read_csv(path, index_col='date', parse_dates=True, float_precision='round_trip').log_return
        expected = np.array([math.log(b/a) for a, b in zip(price[:-1], price[1:])])
        np.testing.assert_allclose(returns, expected, atol=2e-15, rtol=0)
        timestamps = pd.to_datetime(raw['timestamp'], unit='s', utc=True).tz_convert('America/New_York')
        dates = pd.DatetimeIndex([day.date() for day in timestamps], name='date')
        assert returns.index.equals(dates[1:]) and len(returns) == item['returns']
        assert returns.index[-1] == pd.Timestamp('2026-08-31')
        assert returns.index.is_unique and returns.index.is_monotonic_increasing
        counts['returns'] += len(returns)
        if args.data_only: continue
        for model in ['moirai', 'lagllama', 'chronos2', 'patchtst']:
            folder = ROOT/'native'/model
            replay = json.loads((folder/'replay.json').read_text())
            assert replay['exact_fresh_replay'] and replay['binding_sha256'] == sha(folder/'binding.json')
            frame = pd.read_parquet(ROOT/'data'/model/f'{asset}.parquet')
            assert frame.index.equals(returns.index[512:])
            for file in sorted((folder/asset).glob('*.npz')):
                assert sha(file) == json.loads(file.with_suffix('.json').read_text())['sha256']
                with np.load(file, allow_pickle=False) as data:
                    stored = frame.loc[data['dates']]
                    if 'levels' in data:
                        for col in stored.filter(like='VaR_'):
                            ix = list(data['levels']).index(float(col[4:]))
                            np.testing.assert_array_equal(stored[col], data['native'][:, ix])
                            counts['native_quantiles'] += len(stored)
                    else:
                        ordered = np.sort(data['native'].astype(float), axis=1)
                        for col in stored.filter(like='VaR_'):
                            rank = float(col[4:])*999; lo = math.floor(rank); hi = math.ceil(rank)
                            q = ordered[:, lo]+(rank-lo)*(ordered[:, hi]-ordered[:, lo])
                            np.testing.assert_allclose(stored[col], q, atol=2e-15, rtol=1e-13)
                            counts['native_quantiles'] += len(stored)
        files = sorted((ROOT/'posthoc').glob(f'*__{asset}.json'))
        assert {file.stem.split('__')[0] for file in files} == MODELS
        for file in files:
            meta = json.loads(file.read_text()); frame = pd.read_parquet(file.with_suffix('.parquet'))
            assert sha(file.with_suffix('.parquet')) == meta['daily_sha256']
            y = returns.reindex(frame.index).to_numpy()
            np.testing.assert_allclose(y, frame.r, atol=2e-15, rtol=0)
            for row in meta['metrics']:
                q = frame[row['method']].to_numpy()
                losses = np.where(y < q, .99*(q-y), .01*(y-q))
                assert int(np.count_nonzero(y < q)) == row['viol']
                assert abs(float(np.mean(losses))-row['QS']) < 2e-15
                model = file.stem.split('__')[0]
                saved = metrics[(metrics.model == model) & (metrics.asset == asset)
                                & (metrics.method == row['method'])].iloc[0]
                assert int(saved.viol) == row['viol']
                assert abs(float(saved.QS)-row['QS']) < 2e-15
                counts['posthoc_losses'] += len(y)
    record = dict(status='passed', scope='data_only' if args.data_only else 'data_native_quantiles_posthoc_losses',
        counts=counts, producer_sha256=sha(__file__), admission_sha256=sha(ROOT/'data_admission.json'))
    out = ROOT/'quality'/('data_validation.json' if args.data_only else 'validation.json')
    out.write_text(json.dumps(record, indent=2)+'\n'); print(json.dumps(record, indent=2))


if __name__ == '__main__': main()
