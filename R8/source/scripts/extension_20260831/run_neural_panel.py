#!/usr/bin/env python3
"""Run four pinned TSFMs: three sequentially on MPS, Moirai 1.1 on CPU."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]/'artifacts/extension_20260831'


def run(model,device,assets=None,worker=None):
    label=model+(f'_{worker}' if worker is not None else '')
    cmd=[sys.executable,str(HERE/'infer_tsfm.py'),'--model',model,'--device',device,'--batch','16']
    if assets:cmd += ['--assets',*assets]
    with (ROOT/f'inference_{label}.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(label,'complete',flush=True)


def gpu():
    for model in ['timesfm25','moirai2','lagllama']:
        run(model,'mps')


if __name__=='__main__':
    if '--gpu-only' in sys.argv:
        gpu()
        raise SystemExit(0)
    checks=json.loads((ROOT/'quality/sampling_optimisation_checks.json').read_text())
    assert checks['passed']
    assets=sorted(p.stem for p in (ROOT/'data/returns').glob('*.csv'))
    assert len(assets)==24
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(gpu)]
        futures += [pool.submit(run,'moirai','cpu',assets[j::4],j) for j in range(4)]
        for f in futures:f.result()
