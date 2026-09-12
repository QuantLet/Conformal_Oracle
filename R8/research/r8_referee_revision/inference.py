"""Expand the reporting family without altering forecasts or earlier inference."""
import importlib.util
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'research/r8_ten_comparators'))
spec = importlib.util.spec_from_file_location('original_ten_aggregate', PROJECT / 'research/r8_ten_comparators/aggregate.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
OUT = PROJECT / 'artifacts/r8_referee_revision'
RESULTS = OUT / 'results'
FAMILY = ['Raw', 'Vol-ERM', 'State-L1', 'POT-Shift', 'POT-Vol',
          'DtACI-projected-expected', 'Loss-gate', 'Past-minimum']


def main():
    import numpy as np
    before = json.loads((OUT / 'before.json').read_text())
    assert all(original.sha(PROJECT / p) == h for p, h in before['base_result_files'].items())
    RESULTS.mkdir(exist_ok=False)
    original.FAMILY = FAMILY
    original.RESULTS = RESULTS
    frames, pairs, _, _, _, bindings = original.load_all()
    identities = list(bindings)
    scales = np.array([pairs[(pairs.model == k.split('__')[0]) & (pairs.asset == k.split('__')[1])].calibration_scale.iloc[0] for k in identities])
    noncrypto = np.array([k.split('__')[1] not in ['BTC', 'ETH'] for k in identities])
    intervals = original.bootstrap(frames, scales, noncrypto)
    intervals.to_csv(RESULTS / 'intervals.csv', index=False)
    for block in [20, 60]:
        with np.load(RESULTS / f'bootstrap_{block}.npz') as current, np.load(PROJECT / f'artifacts/r8_ten_comparators/results/bootstrap_{block}.npz') as previous:
            for key in ['draws', 'point', 'methods']:
                assert np.array_equal(current[key], previous[key]), (block, key)
    assert all(original.sha(PROJECT / p) == h for p, h in before['base_result_files'].items())
    record = dict(status='complete', pairs=240, family=FAMILY,
                  producer_sha256=original.sha(__file__),
                  original_aggregator_sha256=original.sha(PROJECT / 'research/r8_ten_comparators/aggregate.py'),
                  protocol_sha256=original.sha(Path(__file__).with_name('PROTOCOL.md')),
                  bindings=bindings, previous_draws_unchanged=True,
                  outputs={p.name:original.sha(p) for p in RESULTS.iterdir() if p.is_file()})
    (RESULTS / 'complete.json').write_text(json.dumps(record, indent=2) + '\n')
    print(intervals[intervals.method.isin(['Raw', 'Vol-ERM']) & intervals.reference.eq('Shift-CP')].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
