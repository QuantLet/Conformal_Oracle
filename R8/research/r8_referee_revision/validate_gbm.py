"""Independent feature, tree, rank, metric and paired-bootstrap verification."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from gbm import PROJECT, ROOT, INPUT, sha, dump

BASE = PROJECT/'artifacts/r8_model_extension/ten_common_evaluation'
OUT = ROOT/'evaluation'
spec = importlib.util.spec_from_file_location('independent_metric',
    PROJECT/'research/r8_model_extension/validate_ten.py')
audit = importlib.util.module_from_spec(spec)
import sys
sys.path.insert(0, str(PROJECT/'research/r8_model_extension'))
spec.loader.exec_module(audit)


def read_features(y):
    a = np.asarray(y); rows = []
    for t in range(250, len(a)):
        values = [a[t-j] for j in range(1, 21)]
        for w in (5, 20, 60, 250):
            sample = a[t-w:t]
            values += [float(sum(sample)/w), float(np.sqrt(sum((sample-sample.mean())**2)/(w-1)))]
        rows.append(values)
    return np.asarray(rows)


def tree_predict(node, x):
    if 'leaf_value' in node:
        return np.full(len(x), node['leaf_value'])
    assert node['decision_type'] == '<=' and node['missing_type'] in ['None', 'NaN']
    go_left = x[:, node['split_feature']] <= node['threshold']
    out = np.empty(len(x))
    out[go_left] = tree_predict(node['left_child'], x[go_left])
    out[~go_left] = tree_predict(node['right_child'], x[~go_left])
    return out


def receipt(folder):
    r = json.loads((folder/'complete.json').read_text())
    assert r['status'] == 'complete'
    for p, h in r.get('bindings', {}).items(): assert sha(PROJECT/p) == h, p
    for p, h in r['outputs'].items(): assert sha(folder/p) == h, p
    return r


def main():
    first = receipt(ROOT/'run'); replay = receipt(ROOT/'replay'); evaluation = receipt(OUT)
    assert first['outputs'] == replay['outputs'], 'Full fresh fit failed exact reproduction'
    config = json.loads((ROOT/'run/configuration.json').read_text())
    fits = pd.read_csv(ROOT/'run/fits.csv')
    native = np.load(BASE/'bootstrap_inputs.npz', allow_pickle=False)
    calendar = pd.DatetimeIndex(native['dates']); columns = evaluation['columns']
    metrics = pd.read_csv(OUT/'metrics.csv').set_index(['asset', 'method'])
    assert len(metrics) == 72 and metrics.index.is_unique
    arrays, positions, checks, forecasts = [], [], 0, 0
    for asset in native['assets']:
        y = pd.read_csv(INPUT/f'{asset}.csv', index_col='date', parse_dates=True).log_return
        x = read_features(y); records = fits[fits.asset == asset]
        expected_starts = [512]+[t for t in range(513, len(y)) if y.index[t].year != y.index[t-1].year]
        assert records.train_stop.tolist() == expected_starts
        pred = np.zeros(len(y)-512)
        for row in records.itertuples():
            assert row.train_start == max(250, row.train_stop-1250)
            assert row.train_rows == row.train_stop-row.train_start
            assert row.train_last == str(y.index[row.train_stop-1].date())
            assert row.predict_first == str(y.index[row.train_stop].date())
            assert row.train_last < row.predict_first
            model = json.loads((ROOT/'run'/asset/f'{row.tag}.json').read_text())
            assert model['feature_names'] == config['feature_names']
            assert len(model['tree_info']) == row.trees <= 200
            test = x[row.train_stop-250:row.predict_stop-250]
            manual = sum(tree_predict(t['tree_structure'], test) for t in model['tree_info'])
            pred[row.train_stop-512:row.predict_stop-512] = manual
        saved = pd.read_parquet(ROOT/'run'/asset/'forecast.parquet')
        assert saved.index.equals(y.index[512:])
        np.testing.assert_array_equal(saved.r, y.iloc[512:])
        np.testing.assert_allclose(pred, saved.q, rtol=2e-13, atol=2e-15)
        forecasts += len(pred)
        observed = y.iloc[512:].to_numpy(); q = saved.q.to_numpy()
        nc = math.floor(.7*len(q)); residuals = q-observed
        shift = sorted(residuals[:nc])[math.ceil((nc+1)*.99)-1]
        rolling = np.array([sorted(residuals[t-250:t])[248] for t in range(nc, len(q))])
        paths = np.column_stack([q[nc:], q[nc:]-shift, q[nc:]-rolling])
        daily = pd.read_parquet(OUT/'daily'/f'{asset}.parquet')
        base = pd.read_parquet(BASE/'daily'/f'{asset}.parquet')
        assert daily.index.equals(base.index)
        np.testing.assert_array_equal(daily[columns[-3:]], paths)
        for j, method in enumerate(['Raw', 'Static', 'Rolling250']):
            row = metrics.loc[(asset, method)]
            audit.check_close(row['shift'], shift)
            assert row.n_cal == nc
            for key, value in audit.metric(observed[nc:], paths[:, j]).items():
                if key == 'TL': assert row[key] == value
                else: audit.check_metric(key, row[key], value); checks += 1
        all_q = np.column_stack([base[columns[:-3]].to_numpy(), paths])
        error = observed[nc:, None]-all_q
        arrays.append(np.maximum(.01*error, -.99*error))
        positions.append(calendar.get_indexer(daily.index))
    assert forecasts == 120792 and len(fits) == first['fits']
    summary = pd.read_csv(OUT/'summary.csv').set_index('method')
    for method in ['Raw', 'Static', 'Rolling250']:
        group = metrics.xs(method, level='method')
        vals = dict(assets=24, observations=group.n_test.sum(), QS_x10000=group.QS.mean()*10000,
            pi_mean=group.pihat.mean(), kupiec_rejections=(group.p_kup < .05).sum(),
            conditional_rejections=(group.p_cc < .05).sum(),
            worse_than_raw=(group.QS > metrics.xs('Raw', level='method').QS).sum())
        for k, v in vals.items(): audit.check_close(summary.loc[method, k], v); checks += 1
    point = np.array([a.mean(0) for a in arrays]).mean(0)
    family = evaluation['family']; left = [columns.index(a) for a, b in family]; right = [columns.index(b) for a, b in family]
    assert len(family) == 22
    estimate = point[left]-point[right]
    intervals = pd.read_csv(OUT/'intervals.csv').set_index(['block_length', 'lhs', 'rhs'])
    for length in (20, 60):
        counts = np.load(BASE/f'bootstrap_L{length}.npz')['counts']
        rng = np.random.default_rng(20260910+length)
        for draw in range(999):
            starts = rng.integers(len(calendar), size=math.ceil(len(calendar)/length))
            sampled = np.concatenate([np.arange(t, t+length) % len(calendar) for t in starts])[:len(calendar)]
            np.testing.assert_array_equal(counts[draw], np.bincount(sampled, minlength=len(calendar)))
        means = sum((counts[:, p]@a)/counts[:, p].sum(1)[:, None] for p, a in zip(positions, arrays))/24
        saved = np.load(OUT/f'bootstrap_L{length}.npz')
        audit.check_close(saved['means'], means); audit.check_close(saved['estimate'], estimate)
        delta = means[:, left]-means[:, right]; sd = delta.std(0, ddof=1)
        critical = np.quantile(np.max(np.abs((delta-estimate)/sd), axis=1), .95)
        audit.check_close(saved['delta'], delta); audit.check_close(saved['sd'], sd)
        audit.check_close(saved['critical'], critical)
        for j, (a, b) in enumerate(family):
            row = intervals.loc[(length, a, b)]
            lo, hi = np.quantile(delta[:, j], [.025, .975])
            expected = [estimate[j], estimate[j]-critical*sd[j], estimate[j]+critical*sd[j], lo, hi]
            actual = row[['estimate', 'simultaneous_lower', 'simultaneous_upper', 'percentile_lower', 'percentile_upper']]
            audit.check_close(actual, np.array(expected)*10000); checks += 5
    dump(ROOT/'validation.json', dict(status='passed', validator_sha256=sha(__file__),
        independent_metric_code_sha256=sha(PROJECT/'research/r8_model_extension/validate_ten.py'),
        run_complete_sha256=sha(ROOT/'run/complete.json'), replay_complete_sha256=sha(ROOT/'replay/complete.json'),
        evaluation_complete_sha256=sha(OUT/'complete.json'), all_fit_outputs_exact=True,
        independent_features_and_tree_forecasts=forecasts, annual_fits=len(fits),
        scalar_metric_and_interval_checks=checks, independently_rebuilt_bootstrap_draws=1998,
        family_contrasts=22, panels=24, methods=3))
    print((ROOT/'validation.json').read_text(), flush=True)


if __name__ == '__main__': main()
