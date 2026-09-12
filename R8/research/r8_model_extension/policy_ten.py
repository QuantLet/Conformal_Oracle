"""Transfer the unchanged past-loss gate to the native-tail common sample."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import importlib.util
import importlib.metadata as md
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from evaluate_ten import PROJECT, ROOT, EXT, MODELS, forecast, sha, dump, scores

spec = importlib.util.spec_from_file_location('existing_decision_methods', PROJECT/'research/r8_decision/methods.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from posthoc import rolling
OUT = ROOT/'ten_policy_evaluation'
COMMON = ROOT/'ten_common_evaluation'
METHODS = ['Raw', 'Static', 'Vol-ERM', 'Rolling500', 'Inner-Shift', 'Inner-Vol',
           'Loss-gate', 'Past-minimum', 'Reverse-gate']
FAMILY = [('Loss-gate', b) for b in ['Raw', 'Static', 'Past-minimum', 'Rolling500', 'Vol-ERM']]


def decision(y, q, sigma, nc, key):
    # A decision receives no post-calibration observations, even when callers
    # provide longer arrays. Rolling candidates below are warmed on past data.
    y, q, sigma = y[:nc], q[:nc], sigma[:nc]
    v = max(1000, int(.7*nc)); assert nc-v >= 250
    shift = m.cp(q[:v]-y[:v])
    vol = m.weighted_quantile((q[:v]-y[:v])/sigma[:v], sigma[:v], .99)
    paths = {'Raw': q, 'Inner-Shift': q-shift, 'Inner-Vol': q-vol*sigma,
             'Rolling500': q-rolling(q-y, 500)}
    losses = {name: m.loss(y[v:nc], path[v:nc]) for name, path in paths.items()}
    gate = m.loss_gate(losses, key)
    diffs = np.column_stack([losses[name]-losses['Raw'] for name in list(paths)[1:]])
    arrays = {'validation_differences': diffs}
    reverse_bands = []
    for length in (20, 60):
        seed = m.seed_for(key, length)
        draws = m.circular_means(diffs, length, 499, np.random.default_rng(seed))
        starts = np.random.default_rng(seed).integers(0, len(diffs), size=(499, int(np.ceil(len(diffs)/length))))
        arrays[f'draws_L{length}'] = draws; arrays[f'starts_L{length}'] = starts
        sd = draws.std(axis=0, ddof=1); active = sd > 1e-15
        z = np.zeros_like(draws); z[:, active] = (diffs.mean(0)[active]-draws[:, active])/sd[active]
        critical = max(0., float(np.quantile(z.max(axis=1), .95)))
        reverse_bands.append(diffs.mean(0)+critical*sd)
    reverse_upper = np.maximum(*reverse_bands); best = int(np.argmin(reverse_upper))
    reverse_selected = list(paths)[best+1] if reverse_upper[best] < 0 else 'Raw'
    return dict(inner_fit=v, n_cal=nc, shift=shift, vol=vol, gate=gate,
        reverse_upper=reverse_upper.tolist(), reverse_selected=reverse_selected,
        validation_losses={name: float(value.mean()) for name, value in losses.items()}), arrays


def main():
    OUT.mkdir(exist_ok=True); (OUT/'pairs').mkdir(exist_ok=True); (OUT/'daily').mkdir(exist_ok=True)
    assert not (OUT/'complete.json').exists(), 'Completed policy run is immutable'
    support = pd.read_csv(ROOT/'full_preflight/support.csv').set_index('asset')
    sources = [Path(__file__), Path(__file__).with_name('PROTOCOL.md'),
        Path(__file__).with_name('evaluate_ten.py'), Path(m.__file__),
        PROJECT/'source/scripts/extension_20260831/panel_statistics.py',
        PROJECT/'source/scripts/extension_20260831/posthoc.py', COMMON/'complete.json',
        COMMON/'validation.json', ROOT/'full_preflight/support.csv']
    binding = {str(p.relative_to(PROJECT)): sha(p) for p in sources}
    common_receipt = json.loads((COMMON/'complete.json').read_text())
    for name, expected in common_receipt['binding'].items():
        assert sha(PROJECT/name) == expected
    protected_files = [PROJECT/'source'/f'{name}.{ext}' for name in ['main_R8', 'supplement_R8'] for ext in ['tex', 'pdf']]
    protected = {str(p.relative_to(PROJECT)): sha(p) for p in protected_files}
    rows, selections, all_daily = [], [], {}
    for asset in support.index:
        rp = EXT/'data/returns'/f'{asset}.csv'; binding[str(rp.relative_to(PROJECT))] = sha(rp)
        ret = pd.read_csv(rp, index_col='date', parse_dates=True).log_return
        dates = ret.index[512:]; y = ret.iloc[512:].to_numpy(); nc = int(.7*len(y))
        sigma = np.maximum(ret.shift(1).rolling(20, min_periods=20).std(ddof=1).reindex(dates).to_numpy(), 1e-8)
        assert np.isfinite(sigma).all()
        original = pd.read_parquet(COMMON/'daily'/f'{asset}.parquet')
        assert original.index.equals(dates[nc:]); np.testing.assert_array_equal(original.r, y[nc:])
        daily = pd.DataFrame({'r': y[nc:], 'sigma': sigma[nc:]}, index=dates[nc:]); daily.index.name = 'date'
        for model in MODELS:
            q, path, _ = forecast(model, asset, dates); binding[str(path.relative_to(PROJECT))] = sha(path)
            key = f'{model}__{asset}'
            fit, arrays = decision(y, q, sigma, nc, key)
            changed, _ = decision(np.r_[y[:nc], y[nc:]+123.], np.r_[q[:nc], q[nc:]-73.],
                                  np.r_[sigma[:nc], sigma[nc:]*21.], nc, key)
            assert changed == fit, key
            shift = m.cp(q[:nc]-y[:nc]); vol = m.weighted_quantile((q[:nc]-y[:nc])/sigma[:nc], sigma[:nc], .99)
            paths = {'Raw': q[nc:], 'Static': q[nc:]-shift, 'Vol-ERM': q[nc:]-vol*sigma[nc:],
                'Rolling500': q[nc:]-rolling(q-y, 500)[nc:],
                'Inner-Shift': q[nc:]-fit['shift'], 'Inner-Vol': q[nc:]-fit['vol']*sigma[nc:]}
            paths['Loss-gate'] = paths[fit['gate']['selected']]
            paths['Past-minimum'] = paths[fit['gate']['past_minimum_selected']]
            paths['Reverse-gate'] = paths[fit['reverse_selected']]
            for name in ['Raw', 'Static']:
                np.testing.assert_array_equal(paths[name], original[f'{model}/{name}'])
            for method in METHODS:
                daily[f'{model}/{method}'] = paths[method]
                rows.append(dict(model=model, asset=asset, method=method, **scores(y[nc:], paths[method])))
            fit.update(full_shift=shift, full_vol=vol, prefix_invariant=True,
                dates={'fit_last': str(dates[fit['inner_fit']-1].date()),
                       'validation_first': str(dates[fit['inner_fit']].date()),
                       'validation_last': str(dates[nc-1].date()), 'test_first': str(dates[nc].date()),
                       'test_last': str(dates[-1].date())})
            folder = OUT/'pairs'/key; folder.mkdir(exist_ok=True)
            dump(folder/'fit.json', fit); np.savez_compressed(folder/'validation.npz', **arrays)
            selections.append(dict(model=model, asset=asset, n_cal=nc, inner_fit=fit['inner_fit'],
                validation_n=nc-fit['inner_fit'], validation_expected_tail=.01*(nc-fit['inner_fit']),
                selected=fit['gate']['selected'], past_minimum=fit['gate']['past_minimum_selected'],
                reverse_selected=fit['reverse_selected'], convention_changed=fit['reverse_selected'] != fit['gate']['selected']))
        daily.to_parquet(OUT/'daily'/f'{asset}.parquet'); all_daily[asset] = daily
        print(asset, '10 policy decisions and prefix checks complete', flush=True)
    metrics = pd.DataFrame(rows); assert len(metrics) == 2160
    metrics.to_csv(OUT/'metrics.csv', index=False); pd.DataFrame(selections).to_csv(OUT/'selections.csv', index=False)
    summaries = []
    for model in ['ALL']+MODELS:
        subset = metrics if model == 'ALL' else metrics[metrics.model == model]
        for method, frame in subset.groupby('method', sort=False):
            summaries.append(dict(model=model, method=method, pairs=len(frame), QS_x10000=frame.QS.mean()*1e4,
                mean_pi=frame.pihat.mean(), violations=int(frame.viol.sum()), observations=int(frame.n_test.sum()),
                kupiec_rejections=int((frame.p_kup < .05).sum()), independence_rejections=int((frame.p_ind < .05).sum()),
                conditional_rejections=int((frame.p_cc < .05).sum()), width=frame.width.mean()))
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv', index=False)
    sample = np.load(COMMON/'bootstrap_inputs.npz'); calendar = pd.DatetimeIndex(sample['dates'])
    assets = list(support.index); columns = [f'{model}/{method}' for model in MODELS for method in METHODS]
    losses = np.zeros((len(calendar), len(assets), len(columns)))
    valid = sample['valid']; assert assets == sample['assets'].tolist()
    for i, asset in enumerate(assets):
        daily = all_daily[asset]; pos = calendar.get_indexer(daily.index)
        losses[pos, i] = m.loss(daily.r.to_numpy()[:, None], daily[columns].to_numpy())
    point = (losses.sum(0)/valid.sum(0)[:, None]).mean(0).reshape(len(MODELS), len(METHODS)).mean(0)
    intervals = []
    for length in [20, 60]:
        bp = COMMON/f'bootstrap_L{length}.npz'; binding[str(bp.relative_to(PROJECT))] = sha(bp)
        counts = np.load(bp)['counts']; denom = counts@valid; assert (denom > 0).all()
        means = ((counts@losses.reshape(len(calendar), -1)).reshape(999, len(assets), len(MODELS), len(METHODS))/denom[:, :, None, None]).mean((1, 2))
        delta = np.column_stack([means[:, METHODS.index(a)]-means[:, METHODS.index(b)] for a, b in FAMILY])
        estimate = np.array([point[METHODS.index(a)]-point[METHODS.index(b)] for a, b in FAMILY])
        sd = delta.std(0, ddof=1); assert (sd > 0).all()
        critical = float(np.quantile(np.abs((delta-estimate)/sd).max(1), .95))
        np.savez_compressed(OUT/f'bootstrap_L{length}.npz', means=means, delta=delta, estimate=estimate, sd=sd, critical=critical)
        for i, (a, b) in enumerate(FAMILY):
            lo, hi = np.quantile(delta[:, i], [.025, .975])
            intervals.append(dict(lhs=a, rhs=b, block_length=length, estimate_x10000=estimate[i]*1e4,
                percentile_lo_x10000=lo*1e4, percentile_hi_x10000=hi*1e4,
                simultaneous_lo_x10000=(estimate[i]-critical*sd[i])*1e4,
                simultaneous_hi_x10000=(estimate[i]+critical*sd[i])*1e4))
    pd.DataFrame(intervals).to_csv(OUT/'intervals.csv', index=False)
    for name, expected in {**binding, **protected}.items():
        assert sha(PROJECT/name) == expected, name
    outputs = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    dump(OUT/'complete.json', dict(status='complete', binding=binding, protected=protected, outputs=outputs,
        models=MODELS, methods=METHODS, family=FAMILY, pairs=240, rows=2160, asset_test_dates=int(support.n_test.sum()),
        prefix_checks=240, packages={p: md.version(p) for p in ['numpy', 'pandas', 'scipy', 'pyarrow']}))
    print(pd.DataFrame(summaries).query("model == 'ALL'").to_string(index=False))
    print(pd.DataFrame(intervals).to_string(index=False))


if __name__ == '__main__':
    main()
