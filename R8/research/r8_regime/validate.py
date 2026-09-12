"""Full replay admission, independent path checks and timing counterfactuals."""
from datetime import datetime, timezone
import json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import engine as e
_spec = importlib.util.spec_from_file_location('regime_runner', Path(__file__).with_name('run.py'))
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def verify_bindings(b):
    assert b['sources'] == run.sources()
    for name, h in b['protected'].items():
        assert e.sha(e.ROOT/name) == h, name
    for name, h in b['inputs'].items():
        assert e.sha(run.OUT/name) == h, name


def main():
    binding = run.initialise()
    verify_bindings(binding)
    corrupt = json.loads(json.dumps(binding))
    corrupt['inputs'][next(iter(corrupt['inputs']))] = '0'*64
    try:
        verify_bindings(corrupt)
    except AssertionError:
        negative = True
    else:
        raise AssertionError('Input corruption was not detected')
    count, arrays, refs, decisions = 0, 0, [], []
    for kind in ('normal', 't5'):
        original_path = np.load(run.OUT/f'innovations_{kind}.npy')
        assert np.array_equal(original_path, e.innovations(kind, range(e.REPS)))
        for start in range(0, e.REPS, run.BLOCK):
            key = f'{kind}_{start:03d}_{start+run.BLOCK:03d}'
            left = run.OUT/'blocks'/key
            right = run.OUT/'replay'/key
            a = json.loads((left/'complete.json').read_text())
            b = json.loads((right/'complete.json').read_text())
            assert a['binding_sha256'] == b['binding_sha256'] == e.sha(run.OUT/'binding.json')
            assert all(b['exact_replay'].values())
            for name, h in a['outputs'].items():
                assert e.sha(left/name) == h
                assert e.sha(right/name) == b['outputs'][name]
                if name.endswith('.npz'):
                    p, q = np.load(left/name), np.load(right/name)
                    assert p.files == q.files
                    for field in p.files:
                        assert np.array_equal(p[field], q[field]), (key, name, field)
                        assert np.isfinite(p[field]).all()
                        arrays += 1
                    if 'probabilities' in p:
                        assert np.max(abs(p['probabilities'].sum(axis=2)-1)) < 2e-15
                        assert np.all(p['probabilities'] > 0)
                        assert np.min(p['states']) >= 1/501 and np.max(p['states']) <= 500/501
                else:
                    assert e.sha(left/name) == e.sha(right/name)
            for alpha in e.ALPHAS:
                groups = [('correct', 'appears', 'jump_oracle'),
                          ('biased', 'disappears', 'reverses'), ('steady_ewma', 'jump_ewma')]
                for group in groups:
                    initial = np.load(left/f'{group[0]}_{alpha:g}.npz')
                    for scenario in group[1:]:
                        other = np.load(left/f'{scenario}_{alpha:g}.npz')
                        for field in ('prediction', 'experts', 'probabilities', 'states'):
                            assert np.array_equal(initial[field][:, :250], other[field][:, :250])
                        for field in ('selected', 'gate', 'validation_loss', 'static_shift'):
                            assert np.array_equal(initial[field], other[field])
            refs.append({'key': key, 'production_receipt_sha256': e.sha(left/'complete.json'),
                         'replay_receipt_sha256': e.sha(right/'complete.json')})
            decisions.append(pd.read_csv(left/'decisions.csv'))
            count += 1
            print('Verified', key, flush=True)
    d = pd.concat(decisions)
    assert len(d) == 1000*8*2
    assert len({e.seed(kind, rep) for kind in ('normal', 't5') for rep in range(e.REPS)}) == 1000
    receipt = {'checked_utc': datetime.now(timezone.utc).isoformat(),
               'producer_sha256': e.sha(__file__), 'binding_sha256': e.sha(run.OUT/'binding.json'),
               'fresh_replay_blocks': count, 'exactly_matched_arrays': arrays,
               'regenerated_innovation_paths': 1000, 'unique_seeds': 1000,
               'prebreak_counterfactual_identity': True, 'mixture_probabilities_valid': True,
               'corruption_negative_control': negative,
               'protected_files_unchanged': len(binding['protected']), 'blocks': refs}
    (run.OUT/'validation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'blocks'}, indent=2))


if __name__ == '__main__':
    main()
