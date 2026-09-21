"""Independent reconstruction of the transferred policy and its uncertainty."""
import os
for _key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[_key] = '1'
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from validate_ten import PROJECT, ROOT, EXT, CHRONOS, MAP, sha, metric, check_metric, check_close, BACKTEST_ROUNDOFF

OUT = ROOT/'ten_policy_evaluation'
COMMON = ROOT/'ten_common_evaluation'


def loss(y, q):
    r = y-q
    return np.maximum(.01*r, -.99*r)


def weighted(s, w):
    ordered = sorted(zip(s.tolist(), w.tolist()), key=lambda a: a[0])
    threshold = .99*sum(weight for value, weight in ordered)
    cumulative = 0.
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    raise AssertionError('Weighted quantile not found')


def main():
    complete = json.loads((OUT/'complete.json').read_text())
    for name, expected in {**complete['binding'], **complete['protected']}.items():
        assert sha(PROJECT/name) == expected, name
    for name, expected in complete['outputs'].items():
        assert sha(OUT/name) == expected, name
    assert complete['prefix_checks'] == 240
    metrics = pd.read_csv(OUT/'metrics.csv').set_index(['asset', 'model', 'method'])
    selections = pd.read_csv(OUT/'selections.csv').set_index(['asset', 'model'])
    summary = pd.read_csv(OUT/'summary.csv').set_index(['model', 'method'])
    sample = np.load(COMMON/'bootstrap_inputs.npz'); assets = sample['assets'].tolist()
    calendar = pd.DatetimeIndex(sample['dates']); models = complete['models']; methods = complete['methods']
    assert len(metrics) == 2160 and metrics.index.is_unique
    decision_checks = path_checks = metric_checks = 0
    actual_losses, actual_positions = [], []
    max_sigma_difference = 0.
    for asset in assets:
        ret = pd.read_csv(EXT/'data/returns'/f'{asset}.csv', index_col='date', parse_dates=True).log_return
        dates = ret.index[512:]; y = ret.iloc[512:].to_numpy(); nc = math.floor(.7*len(y)); v = max(1000, math.floor(.7*nc))
        sigma = np.maximum(ret.shift(1).rolling(20, min_periods=20).std(ddof=1).reindex(dates).to_numpy(), 1e-8)
        independent_sigma = np.array([max(1e-8, np.std(ret.iloc[t-20:t].to_numpy(), ddof=1)) for t in range(512, len(ret))])
        np.testing.assert_allclose(sigma, independent_sigma, rtol=1e-9, atol=1e-13)
        max_sigma_difference = max(max_sigma_difference, float(np.max(abs(sigma-independent_sigma))))
        saved = pd.read_parquet(OUT/'daily'/f'{asset}.parquet')
        assert saved.index.equals(dates[nc:]); np.testing.assert_array_equal(saved.r, y[nc:])
        columns = []
        for model in models:
            key = f'{model}__{asset}'; folder = OUT/'pairs'/key
            fit = json.loads((folder/'fit.json').read_text()); arrays = np.load(folder/'validation.npz')
            if model in ['PatchTST-FM', 'Chronos-2', 'TS-ICL']:
                native = np.load((ROOT / {'PatchTST-FM':'patchtst_full', 'Chronos-2':'chronos2_full', 'TS-ICL':'tsicl_assembled'}[model])/f'{asset}.npz')
                q = native['native'][:, np.flatnonzero(native['levels'] == .01)[0]].astype(float)
                np.testing.assert_array_equal(native['dates'], dates.to_numpy())
            else:
                directory, suffix = MAP[model]
                fp = EXT/'data'/directory/(f'{asset}_{suffix}.parquet' if suffix else f'{asset}.parquet')
                q = pd.read_parquet(fp).loc[dates, 'VaR_0.01'].to_numpy(dtype=float)
            s = q-y
            inner = sorted(s[:v])[math.ceil((v+1)*.99)-1]
            full = sorted(s[:nc])[math.ceil((nc+1)*.99)-1]
            iv = weighted(s[:v]/sigma[:v], sigma[:v]); fv = weighted(s[:nc]/sigma[:nc], sigma[:nc])
            for name, value in [('shift', inner), ('full_shift', full), ('vol', iv), ('full_vol', fv)]:
                check_close(fit[name], value)
            assert fit['inner_fit'] == v and fit['n_cal'] == nc and fit['prefix_invariant']
            assert fit['dates']['fit_last'] == str(dates[v-1].date())
            assert fit['dates']['validation_last'] == str(dates[nc-1].date())
            assert fit['dates']['test_first'] == str(dates[nc].date())
            roll = np.full(len(y), np.nan)
            for t in range(v, len(y)):
                roll[t] = sorted(s[t-500:t])[495]
            paths = {'Raw': q, 'Inner-Shift': q-inner, 'Inner-Vol': q-iv*sigma, 'Rolling500': q-roll}
            names = list(paths)
            vl = {name: loss(y[v:nc], value[v:nc]) for name, value in paths.items()}
            diffs = np.column_stack([vl[name]-vl['Raw'] for name in names[1:]])
            check_close(arrays['validation_differences'], diffs)
            means = diffs.mean(0); upper, reverse = [], []
            for length in [20, 60]:
                seed = int.from_bytes(hashlib.sha256(f'20260909/{key}/{length}'.encode()).digest()[:4], 'little')
                starts = np.random.default_rng(seed).integers(0, len(diffs), size=(499, math.ceil(len(diffs)/length)))
                np.testing.assert_array_equal(arrays[f'starts_L{length}'], starts)
                counts = np.array([np.bincount(np.concatenate([np.arange(t, t+length) % len(diffs) for t in row])[:len(diffs)],
                                              minlength=len(diffs)) for row in starts])
                draws = counts@diffs/len(diffs)
                check_close(arrays[f'draws_L{length}'], draws)
                sd = draws.std(0, ddof=1); use = sd > 1e-15; z = np.zeros_like(draws)
                z[:, use] = (draws[:, use]-means[use])/sd[use]
                critical = max(0., float(np.quantile(z.max(1), .95)))
                upper.append(means+critical*sd)
                reverse_critical = max(0., float(np.quantile((-z).max(1), .95)))
                reverse.append(means+reverse_critical*sd)
                original_band = next(b for b in fit['gate']['bands'] if b['block_observations'] == length)
                check_close(original_band['sd'], sd)
                np.testing.assert_allclose(original_band['critical'], critical, rtol=1e-8, atol=1e-10)
                check_close(original_band['upper'], upper[-1])
            bound = np.maximum(*upper); reverse_bound = np.maximum(*reverse)
            best = int(np.argmin(bound)); reverse_best = int(np.argmin(reverse_bound))
            chosen = names[best+1] if bound[best] < 0 else 'Raw'
            reverse_chosen = names[reverse_best+1] if reverse_bound[reverse_best] < 0 else 'Raw'
            plain = min(names, key=lambda name: (vl[name].mean(), names.index(name)))
            assert fit['gate']['selected'] == chosen and fit['gate']['past_minimum_selected'] == plain
            assert fit['reverse_selected'] == reverse_chosen
            row = selections.loc[(asset, model)]
            assert row.selected == chosen and row.past_minimum == plain and row.reverse_selected == reverse_chosen
            check_close(fit['gate']['upper_bounds'], bound); check_close(fit['reverse_upper'], reverse_bound)
            decision_checks += 1
            paths.update(Static=q-full, **{'Vol-ERM': q-fv*sigma, 'Loss-gate': paths[chosen],
                                          'Past-minimum': paths[plain], 'Reverse-gate': paths[reverse_chosen]})
            for method in methods:
                target = paths[method][nc:]; col = f'{model}/{method}'; columns.append(col)
                check_close(saved[col].to_numpy(), target); path_checks += len(target)
                np.testing.assert_array_equal(y[nc:] < target, -y[nc:] > -target)
                for name, value in metric(y[nc:], target).items():
                    actual = metrics.loc[(asset, model, method), name]
                    if name == 'TL':
                        assert actual == value
                    else:
                        check_metric(name, actual, value); metric_checks += 1
        actual_losses.append(loss(y[nc:, None], saved[columns].to_numpy()).reshape(len(y)-nc, len(models), len(methods)))
        actual_positions.append(calendar.get_indexer(dates[nc:]))
    assert decision_checks == 240
    for model in ['ALL']+models:
        subset = metrics.reset_index()
        if model != 'ALL':
            subset = subset[subset.model == model]
        for method, f in subset.groupby('method'):
            row = summary.loc[(model, method)]
            expected = dict(pairs=len(f), QS_x10000=f.QS.mean()*1e4, mean_pi=f.pihat.mean(),
                violations=f.viol.sum(), observations=f.n_test.sum(), width=f.width.mean(),
                kupiec_rejections=(f.p_kup < .05).sum(), independence_rejections=(f.p_ind < .05).sum(),
                conditional_rejections=(f.p_cc < .05).sum())
            for name, value in expected.items():
                check_close(row[name], value); metric_checks += 1
    point = np.array([x.mean(0) for x in actual_losses]).mean((0, 1))
    family = complete['family']; lhs = [methods.index(a) for a, b in family]; rhs = [methods.index(b) for a, b in family]
    intervals = pd.read_csv(OUT/'intervals.csv').set_index(['block_length', 'rhs'])
    for length in [20, 60]:
        counts = np.load(COMMON/f'bootstrap_L{length}.npz')['counts']
        means = np.zeros((999, len(methods)))
        for pos, array in zip(actual_positions, actual_losses):
            weighted_loss = (counts[:, pos]@array.reshape(len(pos), -1))/counts[:, pos].sum(1)[:, None]
            means += weighted_loss.reshape(999, len(models), len(methods)).mean(1)/len(assets)
        draws = np.load(OUT/f'bootstrap_L{length}.npz')
        check_close(draws['means'], means)
        delta = means[:, lhs]-means[:, rhs]; estimate = point[lhs]-point[rhs]
        sd = delta.std(0, ddof=1); critical = np.quantile(np.max(abs((delta-estimate)/sd), axis=1), .95)
        check_close(draws['delta'], delta); check_close(draws['estimate'], estimate)
        check_close(draws['sd'], sd); check_close(draws['critical'], critical)
        for j, (a, b) in enumerate(family):
            row = intervals.loc[(length, b)]; lo, hi = np.quantile(delta[:, j], [.025, .975])
            for name, value in dict(estimate_x10000=estimate[j]*1e4, percentile_lo_x10000=lo*1e4,
                percentile_hi_x10000=hi*1e4, simultaneous_lo_x10000=(estimate[j]-critical*sd[j])*1e4,
                simultaneous_hi_x10000=(estimate[j]+critical*sd[j])*1e4).items():
                check_close(row[name], value); metric_checks += 1
    result = dict(status='passed', producer_sha256=sha(__file__), complete_sha256=sha(OUT/'complete.json'),
        independent_decisions=decision_checks, reconstructed_threshold_values=path_checks,
        independent_metric_checks=metric_checks, inner_bootstrap_resamples=240*499*2,
        panel_bootstrap_resamples=1998, largest_volatility_reconstruction_difference=max_sigma_difference,
        backtest_roundoff=BACKTEST_ROUNDOFF, protected_files_unchanged=len(complete['protected']))
    (OUT/'validation.json').write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
