"""Bound, resumable production and exact fresh replay; no manuscript writes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import engine as e

OUT = e.ROOT/'artifacts/r8_regime'
CODE = Path(__file__).parent
BLOCK = 100


def sources():
    files = [CODE/p for p in ('PROTOCOL.md', 'engine.py', 'run.py')]
    files.append(e.ROOT/'research/r8_decision/methods.py')
    return {str(p.relative_to(e.ROOT)): e.sha(p) for p in files}


def protect():
    files = list((e.ROOT/'source/sections_r8').glob('*.tex'))
    files += [e.ROOT/p for p in ('source/main_R8.tex', 'source/supplement_R8.tex',
              'source/main_R8.pdf', 'source/supplement_R8.pdf', 'Manuscript_R8.pdf',
              'source/references.bib', 'docs/CONFERENCE_PROGRAM_ABSTRACT.md') if (e.ROOT/p).exists()]
    files += list((e.ROOT/'artifacts/r8_decision/results').glob('*.csv'))
    files += list((e.ROOT/'artifacts/r8_mechanism/results').glob('*.csv'))
    return {str(p.relative_to(e.ROOT)): e.sha(p) for p in sorted(set(files))}


def initialise():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT/'binding.json'
    if manifest.exists():
        b = json.loads(manifest.read_text())
        assert b['sources'] == sources(), 'Scientific producer changed'
        for p, h in b['protected'].items():
            assert e.sha(e.ROOT/p) == h, ('Protected document/result changed', p)
        for p, h in b['inputs'].items():
            assert e.sha(OUT/p) == h, ('Input changed', p)
        return b
    b = {'created_utc': datetime.now(timezone.utc).isoformat(), 'sources': sources(),
         'protected': protect(), 'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'pandas')},
         'independent_histories': 1000, 'replications_per_law': e.REPS, 'block_size': BLOCK,
         'methods': e.METHODS, 'scenarios': e.SCENARIOS, 'alpha': e.ALPHAS, 'inputs': {}}
    for p in b['sources']:
        dest = OUT/'code_snapshot'/p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(e.ROOT/p, dest)
    for kind in ('normal', 't5'):
        p = OUT/f'innovations_{kind}.npy'
        np.save(p, e.innovations(kind, range(e.REPS)))
        b['inputs'][p.name] = e.sha(p)
    manifest.write_text(json.dumps(b, indent=2)+'\n')
    return b


def worker(kind, start, replay=False):
    clock = time.monotonic()
    binding = json.loads((OUT/'binding.json').read_text())
    assert sources() == binding['sources']
    n = min(BLOCK, e.REPS-start)
    key = f'{kind}_{start:03d}_{start+n:03d}'
    folder = OUT/('replay' if replay else 'blocks')/key
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder/'complete.json'
    if complete.exists():
        receipt = json.loads(complete.read_text())
        assert receipt['binding_sha256'] == e.sha(OUT/'binding.json')
        for p, h in receipt['outputs'].items():
            assert e.sha(folder/p) == h, p
        return key, 'verified existing output'
    x = (e.innovations(kind, range(start, start+n)) if replay else
         np.load(OUT/f'innovations_{kind}.npy', mmap_mode='r')[start:start+n])
    rows, decisions, files = [], [], []
    for scenario in e.SCENARIOS:
        for alpha in e.ALPHAS:
            y, q, s, true_sigma = e.environment(x, scenario, kind, alpha)
            result = e.policies(y, q, s, alpha)
            metrics = e.evaluate(result, y, true_sigma, kind, alpha)
            name = f'{scenario}_{alpha:g}.npz'
            # Exact daily predictions and pre-outcome adaptive states permit
            # numerical re-evaluation without saving redundant loss tensors.
            np.savez_compressed(folder/name, **result)
            files.append(name)
            moments = {}
            for metric, values in metrics.items():
                assert np.isfinite(values).all(), (scenario, alpha, metric)
                moments[metric+'_sum'] = values.sum(axis=0)
                moments[metric+'_squared_sum'] = (values**2).sum(axis=0)
            mn = f'{scenario}_{alpha:g}_moments.npz'
            np.savez_compressed(folder/mn, **moments)
            files.append(mn)
            for i in range(n):
                decisions.append({'innovation': kind, 'scenario': scenario, 'alpha': alpha, 'replication': start+i,
                    'selected_window': e.WINDOWS[result['selected'][i]], 'apply_gate': bool(result['gate'][i]),
                    'static_shift': float(result['static_shift'][i]),
                    **{f'validation_{w}': float(result['validation_loss'][i, j]) for j, w in enumerate(e.WINDOWS)},
                    **{f'projections_{j}': int(result['projections'][i, j]) for j in range(len(e.original.GAMMAS))}})
                for period, (lo, hi) in e.PERIODS.items():
                    for j, method in enumerate(e.METHODS):
                        rows.append({'innovation': kind, 'scenario': scenario, 'alpha': alpha,
                            'replication': start+i, 'period': period, 'method': method,
                            **{metric: float(value[i, lo:hi, j].mean()) for metric, value in metrics.items()}})
            print(key, scenario, alpha, 'computed', flush=True)
    pd.DataFrame(rows).to_csv(folder/'replications.csv', index=False)
    pd.DataFrame(decisions).to_csv(folder/'decisions.csv', index=False)
    files += ['replications.csv', 'decisions.csv']
    receipt = {'binding_sha256': e.sha(OUT/'binding.json'), 'innovation': kind,
               'start': start, 'stop': start+n,
               'outputs': {p: e.sha(folder/p) for p in files}, 'elapsed_seconds': time.monotonic()-clock}
    if replay:
        original = json.loads((OUT/'blocks'/key/'complete.json').read_text())
        # Compressed NPZ bytes are deterministic in this runtime, but compare
        # every numeric array too if ZIP metadata ever differs across runtimes.
        comparisons = {}
        for name in files:
            other = OUT/'blocks'/key/name
            if name.endswith('.npz'):
                a, b = np.load(folder/name), np.load(other)
                assert a.files == b.files
                assert all(np.array_equal(a[k], b[k]) for k in a.files), name
            else:
                assert e.sha(folder/name) == original['outputs'][name], name
            comparisons[name] = True
        receipt['exact_replay'] = comparisons
    complete.write_text(json.dumps(receipt, indent=2)+'\n')
    return key, round(time.monotonic()-clock, 1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', action='store_true')
    ap.add_argument('--replay', action='store_true')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--single', nargs=2)
    args = ap.parse_args()
    initialise()
    if args.init:
        print('Protocol and inputs bound before production.')
    elif args.single:
        print(worker(args.single[0], int(args.single[1]), args.replay), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(worker, kind, start, args.replay) for kind in ('normal', 't5')
                    for start in range(0, e.REPS, BLOCK)]
            for job in as_completed(jobs):
                print('FINISHED', job.result(), flush=True)
