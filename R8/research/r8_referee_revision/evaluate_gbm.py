"""Common-support GBM evaluation with the fixed 22-contrast family."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from gbm import PROJECT, ROOT, sha, dump
sys.path.insert(0, str(PROJECT/'source/scripts/extension_20260831'))
from panel_statistics import scores, qshift

BASE = PROJECT/'artifacts/r8_model_extension/ten_common_evaluation'
OUT = ROOT/'evaluation'


def main():
    assert not (OUT/'complete.json').exists()
    OUT.mkdir(exist_ok=True); (OUT/'daily').mkdir(exist_ok=True)
    native = np.load(BASE/'bootstrap_inputs.npz', allow_pickle=False)
    columns = native['columns'].tolist()+['GBM/Raw', 'GBM/Static', 'GBM/Rolling250']
    family = [(f'GBM/{method}', col) for method in ['Raw', 'Static']
              for col in native['columns'].tolist() if col.endswith('/'+method)]
    family += [('GBM/Static', 'GBM/Raw'), ('GBM/Rolling250', 'GBM/Raw')]
    assert len(family) == 22
    calendar = pd.DatetimeIndex(native['dates'])
    loss = np.zeros((len(calendar), 24, 3)); metrics = []
    bindings = {str(p.relative_to(PROJECT)): sha(p) for p in
        [Path(__file__), ROOT/'run/complete.json', Path(__file__).with_name('GBM_PROTOCOL.md'),
         BASE/'complete.json', BASE/'bootstrap_inputs.npz',
         PROJECT/'source/scripts/extension_20260831/panel_statistics.py']}
    for i, asset in enumerate(native['assets']):
        path = ROOT/'run'/str(asset)/'forecast.parquet'
        bindings[str(path.relative_to(PROJECT))] = sha(path)
        f = pd.read_parquet(path); y = f.r.to_numpy(); q = f.q.to_numpy()
        nc = int(.7*len(f)); s = q-y; shift = qshift(s[:nc])
        windows = np.lib.stride_tricks.sliding_window_view(s, 250)
        rolling = np.partition(windows[nc-250:len(y)-250], 248, axis=1)[:, 248]
        targets = np.column_stack([q[nc:], q[nc:]-shift, q[nc:]-rolling])
        daily = pd.DataFrame(targets, index=f.index[nc:], columns=columns[-3:])
        daily['r'] = y[nc:]; daily.to_parquet(OUT/'daily'/f'{asset}.parquet')
        original = pd.read_parquet(BASE/'daily'/f'{asset}.parquet')
        assert daily.index.equals(original.index)
        np.testing.assert_array_equal(daily.r, original.r)
        for j, method in enumerate(['Raw', 'Static', 'Rolling250']):
            metrics.append(dict(asset=str(asset), method=method, n_cal=nc, shift=shift,
                                **scores(y[nc:], targets[:, j])))
        pos = calendar.get_indexer(daily.index)
        loss[pos, i] = (.01-(y[nc:, None] < targets))*(y[nc:, None]-targets)
    frame = pd.DataFrame(metrics); frame.to_csv(OUT/'metrics.csv', index=False)
    summary = [dict(method=method, assets=len(g), observations=int(g.n_test.sum()),
        QS_x10000=g.QS.mean()*10000, pi_mean=g.pihat.mean(),
        kupiec_rejections=int((g.p_kup < .05).sum()),
        conditional_rejections=int((g.p_cc < .05).sum()),
        worse_than_raw=int((g.set_index('asset').QS > frame[frame.method=='Raw'].set_index('asset').QS).sum()))
        for method, g in frame.groupby('method', sort=False)]
    pd.DataFrame(summary).to_csv(OUT/'summary.csv', index=False)
    point = np.r_[native['point'], (loss.sum(0)/native['valid'].sum(0)[:, None]).mean(0)]
    left = [columns.index(a) for a, b in family]; right = [columns.index(b) for a, b in family]
    estimate = point[left]-point[right]; intervals = []
    for length in (20, 60):
        path = BASE/f'bootstrap_L{length}.npz'; bindings[str(path.relative_to(PROJECT))] = sha(path)
        boot = np.load(path); counts = boot['counts']; denominator = counts@native['valid']
        new_means = ((counts@loss.reshape(len(calendar), -1)).reshape(999, 24, 3)/denominator[:, :, None]).mean(1)
        means = np.column_stack([boot['means'], new_means])
        delta = means[:, left]-means[:, right]; sd = delta.std(0, ddof=1)
        critical = float(np.quantile(np.max(np.abs((delta-estimate)/sd), axis=1), .95))
        np.savez_compressed(OUT/f'bootstrap_L{length}.npz', means=means, delta=delta,
                            estimate=estimate, sd=sd, critical=critical)
        for j, (a, b) in enumerate(family):
            lo, hi = np.quantile(delta[:, j], [.025, .975])
            intervals.append(dict(lhs=a, rhs=b, block_length=length,
                estimate=estimate[j]*10000, simultaneous_lower=(estimate[j]-critical*sd[j])*10000,
                simultaneous_upper=(estimate[j]+critical*sd[j])*10000,
                percentile_lower=lo*10000, percentile_upper=hi*10000, critical=critical))
    pd.DataFrame(intervals).to_csv(OUT/'intervals.csv', index=False)
    for p, h in bindings.items(): assert sha(PROJECT/p) == h
    outputs = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    dump(OUT/'complete.json', dict(status='complete', bindings=bindings, outputs=outputs,
        columns=columns, family=family, assets=24, asset_test_dates=int(native['valid'].sum())))
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == '__main__': main()
