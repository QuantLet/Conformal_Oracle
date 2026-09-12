"""Independent full-grid, common-date, loss, backtest and bootstrap audit."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2

PROJECT = Path(__file__).resolve().parents[2]
from scope import ROOT, EXT
OUT = ROOT/'common_evaluation'
CHRONOS = ROOT/'chronos2_full'
MAP = {'Moirai-1.1': ('moirai', None), 'Lag-Llama': ('lagllama', None),
       'GJR-GARCH': ('benchmarks', 'gjr_garch'), 'GJR-GARCH-t': ('benchmarks', 'gjr_t'),
       'GARCH-N': ('benchmarks', 'garch_n'), 'Hist-Sim': ('benchmarks', 'hs'), 'EWMA': ('benchmarks', 'ewma')}
BACKTEST_ROUNDOFF = {}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as src:
        while block := src.read(8*1024*1024):
            digest.update(block)
    return digest.hexdigest()


def check_close(a, b):
    np.testing.assert_allclose(a, b, rtol=2e-10, atol=2e-14, equal_nan=True)


def check_metric(key, actual, expected):
    if key in ['lr_uc', 'lr_ind', 'p_kup', 'p_ind', 'p_cc']:
        # Independently subtracting binomial log likelihoods loses a few ulps
        # near LR=0; chi-square probabilities amplify that rounding there.
        # Keep forecast/loss tolerances stricter and require identical decisions.
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-10, equal_nan=True)
        if np.isfinite(expected):
            BACKTEST_ROUNDOFF[key] = max(BACKTEST_ROUNDOFF.get(key, 0.), abs(float(actual-expected)))
        if key.startswith('p_'):
            assert np.isfinite(actual) == np.isfinite(expected)
            assert (actual < .05) == (expected < .05)
    else:
        check_close(actual, expected)


def metric(y, q):
    residual = y-q; hits = residual < 0
    n = len(hits); v = int(hits.sum()); p = v/n
    def binomial_ll(success, failure, prob):
        total = 0.
        if success:
            total += success*math.log(prob)
        if failure:
            total += failure*math.log1p(-prob)
        return total
    uc = 2*(binomial_ll(v, n-v, p)-binomial_ll(v, n-v, .01))
    counts = np.zeros((2, 2), dtype=int)
    for a, b in zip(hits[:-1], hits[1:]):
        counts[int(a), int(b)] += 1
    ind = np.nan
    if (counts.sum(axis=1) > 0).all():
        null_p = counts[:, 1].sum()/(n-1)
        ll_null = binomial_ll(counts[:, 1].sum(), counts[:, 0].sum(), null_p)
        ll_alt = sum(binomial_ll(c[1], c[0], c[1]/c.sum()) for c in counts)
        ind = max(0., 2*(ll_alt-ll_null))
    scaled = v/n*250
    return dict(n_test=n, viol=v, pihat=p, p_kup=chi2.sf(uc, 1), p_ind=chi2.sf(ind, 1),
                p_cc=chi2.sf(uc+ind, 2), lr_uc=uc, lr_ind=ind,
                QS=np.maximum(.01*residual, -.99*residual).mean(), width=np.abs(q).mean(),
                TL='Green' if scaled <= 4 else 'Yellow' if scaled <= 9 else 'Red')


def main():
    result = {'status': 'validating', 'producer_sha256': sha(__file__)}
    completed = json.loads((OUT/'complete.json').read_text())
    for name, expected in completed['binding'].items():
        assert sha(PROJECT/name) == expected, name
    for name, expected in completed['outputs'].items():
        assert sha(OUT/name) == expected, name
    preparation = json.loads((ROOT/'preparation.json').read_text())
    assert preparation['status'] == 'complete'
    for path,expected in preparation['binding'].items():
        assert sha(PROJECT/path) == expected, path
    grid_rows = grid_crossings = contexts_checked = 0
    for item in preparation['native']:
        assert item['fresh_native_replay_exact']
        path = ROOT/item['output']; assert sha(path) == item['sha256']
        saved = np.load(path,allow_pickle=False)
        ret = pd.read_csv(EXT/'data/returns'/f"{item['asset']}.csv",index_col='date',parse_dates=True).log_return
        np.testing.assert_array_equal(saved['positions'],np.arange(512,len(ret)))
        np.testing.assert_array_equal(saved['dates'],ret.index[512:].to_numpy())
        assert np.isfinite(saved['native']).all()
        assert saved['native'].shape == (len(ret)-512,len(saved['levels']))
        assert int((np.diff(saved['native'],axis=1)<0).any(1).sum()) == item['crossings']
        if item['model'] == 'PatchTST-FM':
            np.testing.assert_array_equal(saved['levels'],np.arange(1,100)/100)
            grid_rows += len(ret)-512; grid_crossings += item['crossings']
        contexts_checked += len(ret)-512
    metrics = pd.read_csv(OUT/'metrics.csv').set_index(['asset', 'model', 'method'])
    calibration = pd.read_csv(OUT/'calibration.csv').set_index(['asset', 'model'])
    boundaries = pd.read_csv(OUT/'boundaries.csv').set_index(['asset', 'model'])
    summary = pd.read_csv(OUT/'summary.csv').set_index(['model', 'method'])
    assert len(metrics) == 648 and metrics.index.is_unique
    boot = np.load(OUT/'bootstrap_inputs.npz', allow_pickle=False)
    assert list(boot['assets']) == sorted(boot['assets']) and len(boot['assets']) == 24
    calendar = pd.DatetimeIndex(boot['dates']); columns = boot['columns'].tolist()
    loss_by_asset = []; positions_by_asset = []; checks = 0
    for ai, asset in enumerate(boot['assets']):
        ret = pd.read_csv(EXT/'data/returns'/f'{asset}.csv', index_col='date', parse_dates=True).log_return
        dates = ret.index[512:]; y = ret.iloc[512:].to_numpy(); nc = math.floor(.7*len(y))
        daily = pd.read_parquet(OUT/'daily'/f'{asset}.parquet')
        assert daily.index.equals(dates[nc:]); np.testing.assert_array_equal(daily.r, y[nc:])
        for model in completed['models']:
            if model in ['PatchTST-FM', 'Chronos-2']:
                folder = ROOT/'patchtst_full' if model == 'PatchTST-FM' else CHRONOS
                source = np.load(folder/f'{asset}.npz')
                np.testing.assert_array_equal(source['dates'], dates.to_numpy())
                q = source['native'][:, np.flatnonzero(source['levels'] == .01)[0]].astype(float)
                original = dates
            else:
                directory, suffix = MAP[model]
                path = EXT/'data'/directory/(f'{asset}_{suffix}.parquet' if suffix else f'{asset}.parquet')
                frame = pd.read_parquet(path); original = frame.index
                q = frame.loc[dates, 'VaR_0.01'].to_numpy(dtype=float)
            boundary = boundaries.loc[(asset, model)]
            assert boundary.original_test_first == str(original[int(.7*len(original))].date())
            assert boundary.common_test_first == str(dates[nc].date())
            s = q-y; shift = sorted(s[:nc])[math.ceil((nc+1)*.99)-1]
            trailing = np.array([sorted(s[t-250:t])[248] for t in range(nc, len(y))])
            cal = metric(y[:nc], q[:nc]); saved_cal = calibration.loc[(asset, model)]
            assert saved_cal.indicated == (cal['p_kup'] < .05 or cal['TL'] != 'Green')
            check_close(saved_cal.qV, shift)
            for key, value in cal.items():
                saved_key = 'cal_observations' if key == 'n_test' else key
                if key == 'TL':
                    assert saved_cal[saved_key] == value
                else:
                    check_metric(key, saved_cal[saved_key], value); checks += 1
            for method, target in [('Raw', q[nc:]), ('Static', q[nc:]-shift), ('Rolling250', q[nc:]-trailing)]:
                np.testing.assert_array_equal(daily[f'{model}/{method}'], target)
                np.testing.assert_array_equal(y[nc:] < target, -y[nc:] > -target)
                row = metrics.loc[(asset, model, method)]; check_close(row.qV, shift)
                assert row.n_cal == nc and row.positive_threshold_days == int((target > 0).sum())
                for key, value in metric(y[nc:], target).items():
                    if key == 'TL':
                        assert row[key] == value
                    else:
                        check_metric(key, row[key], value); checks += 1
        pos = calendar.get_indexer(dates[nc:]); assert (pos >= 0).all()
        residual = y[nc:, None]-daily[columns].to_numpy()
        loss = np.maximum(.01*residual, -.99*residual)
        np.testing.assert_allclose(boot['losses'][pos, ai, :], loss, rtol=1e-14, atol=1e-16)
        expected_valid = np.zeros(len(calendar)); expected_valid[pos] = 1
        np.testing.assert_array_equal(boot['valid'][:, ai], expected_valid)
        assert (boot['losses'][expected_valid == 0, ai, :] == 0).all()
        loss_by_asset.append(loss); positions_by_asset.append(pos)
    for (model, method), group in metrics.reset_index().groupby(['model', 'method']):
        row = summary.loc[(model, method)]
        values = dict(assets=24, observations=group.n_test.sum(), violations=group.viol.sum(),
            pi_mean=group.pihat.mean(), pi_pooled=group.viol.sum()/group.n_test.sum(),
            QS=group.QS.mean(), QS_x10000=group.QS.mean()*10000, width=group.width.mean(),
            kupiec_rejections=(group.p_kup < .05).sum(), independence_rejections=(group.p_ind < .05).sum(),
            independence_available=group.p_ind.notna().sum(), conditional_rejections=(group.p_cc < .05).sum(),
            conditional_available=group.p_cc.notna().sum(), scaled_green=(group.TL == 'Green').sum(),
            scaled_yellow=(group.TL == 'Yellow').sum(), scaled_red=(group.TL == 'Red').sum())
        for key, value in values.items():
            check_close(row[key], value); checks += 1
    previous = pd.read_csv(PROJECT/'artifacts/r8_native_candidates/chronos-2-full/metrics.csv').set_index(['asset', 'method'])
    for (asset, method), row in previous.iterrows():
        if asset not in set(boot['assets']): continue
        current = metrics.loc[(asset, 'Chronos-2', method)]
        for key in ['n_cal', 'n_test', 'qV', 'QS', 'pihat', 'viol', 'p_kup']:
            check_close(row[key], current[key]); checks += 1
    point = np.stack([loss.mean(axis=0) for loss in loss_by_asset]).mean(axis=0)
    check_close(boot['point'], point)
    intervals = pd.read_csv(OUT/'intervals.csv').set_index(['block_length', 'comparison'])
    family = completed['family']; lhs = [columns.index(f[1]) for f in family]; rhs = [columns.index(f[2]) for f in family]
    estimate = point[lhs]-point[rhs]
    for length in (20, 60):
        draws = np.load(OUT/f'bootstrap_L{length}.npz'); counts = draws['counts']
        assert counts.shape == (999, len(calendar)) and (counts.sum(axis=1) == len(calendar)).all()
        rng = np.random.default_rng(20260910+length)
        for i in range(999):
            starts = rng.integers(len(calendar), size=math.ceil(len(calendar)/length))
            sampled = np.concatenate([np.arange(t, t+length) % len(calendar) for t in starts])[:len(calendar)]
            np.testing.assert_array_equal(counts[i], np.bincount(sampled, minlength=len(calendar)))
        means = sum((counts[:, pos]@loss)/counts[:, pos].sum(axis=1)[:, None]
                    for pos, loss in zip(positions_by_asset, loss_by_asset))/24
        check_close(draws['means'], means)
        delta = means[:, lhs]-means[:, rhs]; check_close(draws['delta'], delta)
        sd = np.std(delta, axis=0, ddof=1)
        critical = np.quantile(np.max(np.abs((delta-estimate)/sd), axis=1), .95)
        check_close(draws['sd'], sd); check_close(draws['critical'], critical)
        for j, (name, a, b) in enumerate(family):
            row = intervals.loc[(length, name)]; lo, hi = np.quantile(delta[:, j], [.025, .975])
            assert row.lhs == a and row.rhs == b
            check_close(row[['estimate_x10000', 'percentile_lo_x10000', 'percentile_hi_x10000',
                             'simultaneous_lo_x10000', 'simultaneous_hi_x10000']].to_numpy(dtype=float),
                        np.array([estimate[j], lo, hi, estimate[j]-critical*sd[j], estimate[j]+critical*sd[j]])*10000)
    protected = json.loads((ROOT/'common_evaluation_protected.json').read_text())
    for name, expected in protected.items():
        assert sha(PROJECT/name) == expected, name
    result.update(status='passed', full_grid_rows=grid_rows, full_grid_native_values=grid_rows*99,
        patchtst_crossing_dates=grid_crossings, reconstructed_past_contexts=contexts_checked,
        fresh_full_replay_exact=True, independent_metric_checks=checks, evaluated_pairs=216,
        evaluated_pair_methods=648, asset_test_dates=preparation['asset_test_dates'], independently_verified_bootstrap_draws=1998,
        independent_bootstrap_mean_values=1998*27, protected_canonical_files_unchanged=len(protected),
        backtest_roundoff_max=BACKTEST_ROUNDOFF, backtest_absolute_tolerance=1e-10,
        backtest_rejection_decisions_exact=True,
        complete_sha256=sha(OUT/'complete.json'), preparation_sha256=sha(ROOT/'preparation.json'))
    (OUT/'validation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
