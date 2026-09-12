"""Execute the sealed 5,000-history shape-cost simulation without retuning."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import engine as e


def main():
    e.OUT.mkdir(parents=True, exist_ok=True)
    lock = json.loads((e.OUT/'lock.json').read_text())
    for name, digest in lock['files'].items():
        assert e.sha(e.ROOT/name) == digest, ('Changed after protocol lock', name)
    assert not (e.OUT/'replications.parquet').exists(), 'Preserve executed studies'
    started = time.monotonic()
    cells = e.cells()
    coef = np.empty((e.REPS, len(cells), len(e.METHODS)))
    states = np.empty((e.REPS, max(e.SIZES)), dtype=np.uint8)
    uniforms = np.empty((e.REPS, max(e.SIZES)), dtype=np.float64)
    for rep in range(e.REPS):
        state, uniform = e.history(rep)
        states[rep], uniforms[rep] = state, uniform
        innovations = {kind: e.law(kind).ppf(uniform) for kind in e.KINDS}
        for j, info in enumerate(cells):
            n = info['n']
            z = info['z_alpha'] - innovations[info['kind']][:n]
            fitted = e.fit_all(z, state[:n].astype(np.int64)+1, info['d'])
            coef[rep, j] = [fitted[m] for m in e.METHODS]
        if (rep+1) % 250 == 0:
            print('Histories', rep+1, '/', e.REPS, 'elapsed', round(time.monotonic()-started, 1), flush=True)
    np.savez_compressed(e.OUT/'histories.npz', states=states, uniforms=uniforms)
    records = []
    for j, info in enumerate(cells):
        last = states[:, info['n']-1]
        for k, method in enumerate(e.METHODS):
            metrics = e.evaluate(info, method, coef[:, j, k], last)
            assert np.all(metrics['excess_vs_conditional_oracle'] > -1e-12)
            data = dict(replication=np.arange(e.REPS), **info, method=method,
                        coefficient=coef[:,j,k], last_state=last, **metrics)
            records.append(pd.DataFrame(data))
    frame = pd.concat(records, ignore_index=True)
    assert len(frame) == e.REPS*12*6 and np.isfinite(frame.select_dtypes('number')).all().all()
    frame.to_parquet(e.OUT/'replications.parquet', index=False)
    (e.OUT/'cells.json').write_text(json.dumps(cells, indent=2)+'\n')
    files = ['histories.npz','replications.parquet','cells.json']
    record = dict(status='complete', histories=e.REPS, cells=12, methods=list(e.METHODS),
                  rows=len(frame), lock_sha256=e.sha(e.OUT/'lock.json'),
                  outputs={p:e.sha(e.OUT/p) for p in files},
                  elapsed_seconds=time.monotonic()-started)
    (e.OUT/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__ == '__main__':
    main()
