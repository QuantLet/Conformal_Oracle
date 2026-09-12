#!/usr/bin/env python3
"""Run unchanged post-hoc and EVT/FHS calculations in the replacement archive."""
import os
os.environ['MPLCONFIGDIR'] = '/private/tmp/irfa-commodity-mpl'
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT/'artifacts/r8_commodity_etp'
sys.path.insert(0, str(PROJECT/'source/scripts/extension_20260831'))
MODELS = ['Moirai-1.1', 'Lag-Llama', 'GJR-GARCH', 'GJR-GARCH-t', 'GARCH-N', 'Hist-Sim', 'EWMA']


def work(task):
    import panel_statistics as statistics
    import posthoc
    import evt_fhs
    import check_classical
    statistics.ROOT = ROOT
    posthoc.ROOT = ROOT
    evt_fhs.ROOT = ROOT
    check_classical.ROOT = ROOT
    kind, model, asset = task
    if kind == 'posthoc': return posthoc.work(model, asset)
    if kind == 'evt': return evt_fhs.work(asset)
    return check_classical.work(asset, model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['posthoc', 'evt', 'classical-check'], required=True)
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    models = args.models if args.kind == 'posthoc' else (
        ['hs', 'ewma', 'garch_n', 'gjr_garch', 'gjr_t'] if args.kind == 'classical-check' else [None])
    tasks = [(args.kind, m, a) for m in models for a in ['USO', 'GLD', 'UNG']]
    records, ledger = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(work, task) for task in tasks]):
            result = future.result()
            if args.kind == 'posthoc':
                rows, decisions = result; records.extend(rows); ledger.extend(decisions)
            elif args.kind == 'evt': records.extend(result)
            else: records.append(result)
    (ROOT/'results').mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(ROOT/'results'/f'{args.kind}.csv', index=False)
    if ledger: pd.DataFrame(ledger).to_csv(ROOT/'results/indication.csv', index=False)
    provenance = {name: hashlib.sha256((PROJECT/'source/scripts/extension_20260831'/name).read_bytes()).hexdigest()
                  for name in ['posthoc.py', 'evt_fhs.py', 'check_classical.py', 'panel_statistics.py']}
    (ROOT/'quality'/f'{args.kind}_runner.json').write_text(json.dumps(dict(
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        upstream_sources=provenance, tasks=tasks, complete=True), indent=2)+'\n')
    print(args.kind, 'complete', len(tasks), 'tasks', flush=True)


if __name__ == '__main__': main()
