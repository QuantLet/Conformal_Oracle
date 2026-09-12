"""Independent saved-output checks for native-tail candidate experiments."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'artifacts/r8_native_candidates'
DATA = PROJECT / 'artifacts/extension_20260831/data/returns'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_bindings():
    checked = 0
    for model in ['chronos-2', 'sundial-base-128m']:
        complete = json.loads((ROOT / model / 'complete.json').read_text())
        replay = json.loads((ROOT / model / 'replay.json').read_text())
        assert complete['protocol_sha256'] == sha(Path(__file__).with_name('PROTOCOL_PILOT.md'))
        assert replay['protocol_sha256'] == sha(Path(__file__).with_name('PROTOCOL.md'))
        for receipt in [complete, replay]:
            assert receipt['producer_sha256'] == sha(Path(__file__).with_name('pilot.py'))
            assert receipt['model_manifest_sha256'] == sha(ROOT / 'models/manifest.json')
            for asset, binding in receipt['bindings'].items():
                assert binding['input_sha256'] == sha(DATA / (asset + '.csv'))
        for name, expected in complete['outputs'].items():
            assert sha(ROOT / model / name) == expected, name
            checked += 1
        if model == 'sundial-base-128m':
            assert len(replay['replay']) == 4 and all(x['exact'] for x in replay['replay'])
        else:
            assert len(replay['replay']) == 2
    return checked


def metric_values(y, q):
    residual = y - q
    hits = residual < 0
    v, n = int(hits.sum()), len(hits)
    p = v / n
    ll = 0.
    if v:
        ll += v * math.log(p / .01)
    if v < n:
        ll += (n-v) * math.log((1-p) / .99)
    return {'n_test': n, 'viol': v, 'pihat': p,
            'QS': np.where(residual >= 0, .01 * residual, -.99 * residual).mean(),
            'p_kup': chi2.sf(2 * ll, 1)}


def check_full(require_replay):
    folder = ROOT / 'chronos-2-full'
    complete = json.loads((folder / 'complete.json').read_text())
    receipts = [complete]
    if require_replay:
        replay = json.loads((folder / 'replay.json').read_text())
        assert replay['exact_fresh_replay']
        receipts.append(replay)
    for receipt in receipts:
        assert receipt['producer_sha256'] == sha(Path(__file__).with_name('full_chronos.py'))
        assert receipt['protocol_sha256'] == sha(Path(__file__).with_name('PROTOCOL.md'))
        assert receipt['model_manifest_sha256'] == sha(ROOT / 'models/manifest.json')
        for name, expected in receipt['scoring_producers'].items():
            assert sha(PROJECT / 'source/scripts/extension_20260831' / name) == expected
    metrics = pd.read_csv(folder / 'metrics.csv').set_index(['asset', 'method'])
    assert len(metrics) == 72 and metrics.index.is_unique
    rows = crossings = comparisons = 0
    crossing_assets = {}
    for asset_receipt in complete['assets']:
        asset = asset_receipt['asset']
        path = DATA / (asset + '.csv')
        saved = folder / (asset + '.npz')
        assert sha(path) == asset_receipt['input_sha256']
        assert sha(saved) == asset_receipt['output_sha256']
        series = pd.read_csv(path, index_col='date', parse_dates=True).log_return
        assert series.index.is_unique and series.index.is_monotonic_increasing
        assert series.index[-1].year == 2026 and series.index[-1].month == 8
        data = np.load(saved)
        positions = np.arange(512, len(series))
        np.testing.assert_array_equal(data['positions'], positions)
        np.testing.assert_array_equal(data['dates'], series.index[positions].to_numpy())
        assert data['native'].shape == (len(positions), 21)
        assert np.isfinite(data['native']).all()
        count = int((np.diff(data['native'], axis=1) < -1e-7).any(axis=1).sum())
        assert count == asset_receipt['native_crossings']
        crossings += count
        if count:
            crossing_assets[asset] = {'rows': count,
                'q01_above_another_quantile': int((data['native'][:, :1] > data['native'][:, 1:] + 1e-7).any(axis=1).sum()),
                'maximum_reversal': float(np.maximum(0, -np.diff(data['native'], axis=1)).max())}
        y = series.iloc[positions].to_numpy()
        q = data['native'][:, list(data['levels']).index(.01)].astype(float)
        n_cal = int(.7 * len(y))
        scores = q - y
        # Full sort and an explicit past-only loop, independent of production partition/strides.
        shift = np.sort(scores[:n_cal])[math.ceil((n_cal + 1) * .99) - 1]
        trailing = np.array([np.sort(scores[t-250:t])[248] for t in range(n_cal, len(y))])
        for method, forecast in [('Raw', q[n_cal:]), ('Static', q[n_cal:] - shift),
                                 ('Rolling250', q[n_cal:] - trailing)]:
            stored = metrics.loc[(asset, method)]
            assert stored['n_cal'] == n_cal
            np.testing.assert_allclose(stored['qV'], shift, rtol=1e-12, atol=1e-15)
            expected = metric_values(y[n_cal:], forecast)
            for name, value in expected.items():
                np.testing.assert_allclose(stored[name], value, rtol=1e-10, atol=1e-14,
                                           err_msg=f'{asset}/{method}/{name}')
                comparisons += 1
        rows += len(y)
    assert rows == complete['rows']
    summary = pd.read_csv(folder / 'summary.csv', index_col='method')
    for method, frame in metrics.reset_index().groupby('method'):
        values = {'assets': len(frame), 'QS': frame.QS.mean(), 'pi': frame.pihat.mean(),
                  'observations': frame.n_test.sum(), 'violations': frame.viol.sum(),
                  'kupiec_rejections': (frame.p_kup < .05).sum(), 'QS_x10000': frame.QS.mean() * 1e4}
        for key, value in values.items():
            np.testing.assert_allclose(summary.loc[method, key], value, rtol=1e-12, atol=1e-15)
    return {'assets': len(complete['assets']), 'forecast_rows': rows, 'native_crossings': crossings,
            'crossing_details': crossing_assets,
            'independent_metric_checks': comparisons, 'full_fresh_replay_required': require_replay}


def sundial_pools():
    rows = []
    for path in sorted((ROOT / 'sundial-base-128m').glob('*_pool.npz')):
        draws = np.load(path)['samples']
        assert len(draws) == 10000 and np.isfinite(draws).all()
        reference = np.quantile(draws, .01, method='linear')
        blocks = np.quantile(draws.reshape(10, 1000), .01, axis=1, method='linear')
        rows.append({'pool': path.name, 'q01_pool': float(reference),
                     'block_q01_sd': float(blocks.std(ddof=1)),
                     'sd_over_abs_pool_quantile': float(blocks.std(ddof=1) / abs(reference))})
    assert len(rows) == 4
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-replay', action='store_true')
    args = parser.parse_args()
    result = {'pilot_output_hashes_verified': check_bindings(),
              'chronos_full': check_full(args.require_replay), 'sundial_pool_diagnostics': sundial_pools()}
    result['producer_sha256'] = sha(Path(__file__))
    result['status'] = 'computations_verified_with_documented_native_crossings'
    (ROOT / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
