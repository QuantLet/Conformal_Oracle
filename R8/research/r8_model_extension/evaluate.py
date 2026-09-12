"""Common-calendar candidate comparison, separate from the canonical paper."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
from scope import ROOT, EXT
OUT = ROOT/'common_evaluation'
CHRONOS = ROOT/'chronos2_full'
sys.path.insert(0, str(PROJECT/'source/scripts/extension_20260831'))
from panel_statistics import scores, qshift, MODELS as REFERENCE_MAP

MODELS = ['PatchTST-FM', 'Chronos-2', 'Moirai-1.1', 'Lag-Llama', 'GJR-GARCH',
          'GJR-GARCH-t', 'GARCH-N', 'Hist-Sim', 'EWMA']
METHODS = ['Raw', 'Static', 'Rolling250']
FAMILY = [
    ('PatchTST Static - Raw', 'PatchTST-FM/Static', 'PatchTST-FM/Raw'),
    ('PatchTST Rolling250 - Raw', 'PatchTST-FM/Rolling250', 'PatchTST-FM/Raw'),
    ('Chronos-2 Static - Raw', 'Chronos-2/Static', 'Chronos-2/Raw'),
    ('Chronos-2 Rolling250 - Raw', 'Chronos-2/Rolling250', 'Chronos-2/Raw'),
    ('PatchTST - Chronos-2 Raw', 'PatchTST-FM/Raw', 'Chronos-2/Raw'),
    ('PatchTST - Chronos-2 Static', 'PatchTST-FM/Static', 'Chronos-2/Static')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def forecast(model, asset, dates):
    if model in MODELS[:2]:
        folder = ROOT/'patchtst_full' if model == MODELS[0] else CHRONOS
        path = folder/f'{asset}.npz'
        saved = np.load(path, allow_pickle=False)
        assert np.array_equal(saved['dates'], dates.to_numpy())
        levels = saved['levels']
        assert np.count_nonzero(levels == .01) == 1
        q = saved['native'][:, np.flatnonzero(levels == .01)[0]].astype(float)
        original_dates = dates
    else:
        directory, suffix = REFERENCE_MAP[model]
        path = EXT/'data'/directory/(f'{asset}_{suffix}.parquet' if suffix else f'{asset}.parquet')
        frame = pd.read_parquet(path)
        assert frame.index.is_unique and frame.index.is_monotonic_increasing
        assert frame.index[-1] == dates[-1] and dates.isin(frame.index).all()
        original_dates = frame.index
        q = frame.loc[dates, 'VaR_0.01'].to_numpy(dtype=float)
    assert q.shape == (len(dates),) and np.isfinite(q).all()
    return q, path, original_dates


def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'daily').mkdir(exist_ok=True)
    assert not (OUT/'complete.json').exists(), 'Completed run is immutable; validate or choose a new output directory'
    support = pd.read_csv(ROOT/'full_preflight/support.csv').set_index('asset')
    sources = [Path(__file__), Path(__file__).with_name('PROTOCOL.md'), ROOT/'full_preflight/support.csv',
               PROJECT/'source/scripts/extension_20260831/panel_statistics.py',
               ROOT/'preparation.json', Path(__file__).with_name('scope.py')]
    binding = {str(p.relative_to(PROJECT)): sha(p) for p in sources}
    metrics, boundaries, calibration, daily = [], [], [], {}
    preparation = json.loads((ROOT/'preparation.json').read_text())
    assert preparation['status'] == 'complete'
    for item in preparation['native']:
        assert item['fresh_native_replay_exact'] and sha(ROOT/item['output']) == item['sha256']
    for asset, row in support.iterrows():
        rp = EXT/'data/returns'/f'{asset}.csv'
        assert sha(rp) == row.input_sha256
        ret = pd.read_csv(rp, index_col='date', parse_dates=True).log_return
        assert ret.index.is_unique and ret.index.is_monotonic_increasing
        dates = ret.index[512:]; y = ret.iloc[512:].to_numpy()
        nc = int(.70*len(y))
        assert nc == row.n_cal and len(y)-nc == row.n_test
        assert str(dates[nc].date()) == row.first_test
        assert str(dates[-1].date()) == row.last_date
        frame = pd.DataFrame({'r': y[nc:]}, index=dates[nc:])
        frame.index.name = 'date'
        binding[str(rp.relative_to(PROJECT))] = sha(rp)
        for model in MODELS:
            q, path, original = forecast(model, asset, dates)
            binding[str(path.relative_to(PROJECT))] = sha(path)
            original_nc = int(.70*len(original))
            boundaries.append(dict(model=model, asset=asset, original_n=len(original), common_n=len(dates),
                original_n_cal=original_nc, common_n_cal=nc,
                original_first=str(original[0].date()), common_first=str(dates[0].date()),
                original_test_first=str(original[original_nc].date()), common_test_first=str(dates[nc].date()),
                common_calibration_last=str(dates[nc-1].date()), last=str(dates[-1].date())))
            residual = q-y
            shift = qshift(residual[:nc])
            windows = np.lib.stride_tricks.sliding_window_view(residual, 250)
            rolling = np.partition(windows[nc-250:len(y)-250], 248, axis=1)[:, 248]
            assert len(rolling) == len(y)-nc
            paths = [q[nc:], q[nc:]-shift, q[nc:]-rolling]
            cal = scores(y[:nc], q[:nc])
            calibration.append(dict(model=model, asset=asset, n_cal=nc, qV=shift,
                indicated=bool(cal['p_kup'] < .05 or cal['TL'] != 'Green'), **cal))
            for method, target in zip(METHODS, paths):
                assert np.array_equal(y[nc:] < target, -y[nc:] > -target)
                metrics.append(dict(model=model, asset=asset, method=method, n_cal=nc,
                                    qV=shift, positive_threshold_days=int((target > 0).sum()),
                                    **scores(y[nc:], target)))
                frame[f'{model}/{method}'] = target
        frame.to_parquet(OUT/'daily'/f'{asset}.parquet')
        daily[asset] = frame
    table = pd.DataFrame(metrics)
    assert len(table) == 648 and table.n_test.sum() == int(support.n_test.sum())*27
    table.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(boundaries).to_csv(OUT/'boundaries.csv', index=False)
    pd.DataFrame(calibration).rename(columns={'n_test': 'cal_observations'}).to_csv(OUT/'calibration.csv', index=False)
    summaries = []
    for (model, method), group in table.groupby(['model', 'method'], sort=False):
        summaries.append(dict(model=model, method=method, assets=len(group), observations=int(group.n_test.sum()),
            violations=int(group.viol.sum()), pi_mean=group.pihat.mean(), pi_pooled=group.viol.sum()/group.n_test.sum(),
            QS=group.QS.mean(), QS_x10000=group.QS.mean()*10000, width=group.width.mean(),
            kupiec_rejections=int((group.p_kup < .05).sum()),
            independence_rejections=int((group.p_ind < .05).sum()), independence_available=int(group.p_ind.notna().sum()),
            conditional_rejections=int((group.p_cc < .05).sum()), conditional_available=int(group.p_cc.notna().sum()),
            scaled_green=int((group.TL == 'Green').sum()), scaled_yellow=int((group.TL == 'Yellow').sum()),
            scaled_red=int((group.TL == 'Red').sum())))
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv', index=False)
    columns = [f'{model}/{method}' for model in MODELS for method in METHODS]
    assets = list(support.index)
    calendar = pd.date_range(min(f.index[0] for f in daily.values()), max(f.index[-1] for f in daily.values()))
    valid = np.zeros((len(calendar), len(assets)))
    losses = np.zeros((len(calendar), len(assets), len(columns)))
    for i, asset in enumerate(assets):
        frame = daily[asset]; pos = calendar.get_indexer(frame.index)
        assert (pos >= 0).all()
        valid[pos, i] = 1
        target = frame[columns].to_numpy(); observed = frame.r.to_numpy()[:, None]
        losses[pos, i] = (.01-(observed < target))*(observed-target)
    point = (losses.sum(axis=0)/valid.sum(axis=0)[:, None]).mean(axis=0)
    lhs = np.array([columns.index(f[1]) for f in FAMILY]); rhs = np.array([columns.index(f[2]) for f in FAMILY])
    estimate = point[lhs]-point[rhs]
    np.savez_compressed(OUT/'bootstrap_inputs.npz', dates=calendar.to_numpy(), assets=np.array(assets),
                        columns=np.array(columns), valid=valid, losses=losses, point=point)
    intervals = []
    for length in (20, 60):
        rng = np.random.default_rng(20260910+length)
        counts = np.zeros((999, len(calendar)), dtype=np.int32)
        for draw in range(999):
            starts = rng.integers(len(calendar), size=int(np.ceil(len(calendar)/length)))
            selected = ((starts[:, None]+np.arange(length)) % len(calendar)).ravel()[:len(calendar)]
            counts[draw] = np.bincount(selected, minlength=len(calendar))
        denom = counts@valid
        assert (denom > 0).all()
        means = ((counts@losses.reshape(len(calendar), -1)).reshape(999, len(assets), len(columns))/denom[:, :, None]).mean(axis=1)
        delta = means[:, lhs]-means[:, rhs]
        sd = delta.std(axis=0, ddof=1); assert (sd > 0).all()
        critical = float(np.quantile(np.max(np.abs((delta-estimate)/sd), axis=1), .95))
        np.savez_compressed(OUT/f'bootstrap_L{length}.npz', counts=counts, means=means, delta=delta,
                            sd=sd, critical=critical, estimate=estimate)
        for j, (name, a, b) in enumerate(FAMILY):
            lo, hi = np.quantile(delta[:, j], [.025, .975])
            intervals.append(dict(comparison=name, lhs=a, rhs=b, block_length=length,
                estimate_x10000=estimate[j]*10000, percentile_lo_x10000=lo*10000, percentile_hi_x10000=hi*10000,
                simultaneous_lo_x10000=(estimate[j]-critical*sd[j])*10000,
                simultaneous_hi_x10000=(estimate[j]+critical*sd[j])*10000, critical=critical, draws=999))
    pd.DataFrame(intervals).to_csv(OUT/'intervals.csv', index=False)
    for path, expected in binding.items():
        assert sha(PROJECT/path) == expected, path
    outputs = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    dump(OUT/'complete.json', dict(status='complete', binding=binding, outputs=outputs, models=MODELS,
        methods=METHODS, assets=len(assets), pairs=216, asset_test_observations=int(support.n_test.sum()),
        alpha=.01, calibration_fraction=.70, common_warmup=512, family=[list(x) for x in FAMILY],
        bootstrap_draws=999, block_lengths=[20, 60], python=sys.version,
        packages={p: md.version(p) for p in ['numpy', 'pandas', 'scipy', 'pyarrow']}))
    print(pd.DataFrame(summaries)[['model', 'method', 'QS_x10000', 'pi_mean', 'kupiec_rejections']].to_string(index=False))
    print(pd.DataFrame(intervals).to_string(index=False))


if __name__ == '__main__':
    main()
