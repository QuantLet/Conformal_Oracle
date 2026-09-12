"""Exploratory matched-loss state study from immutable R8 financial paths.

No fits or canonical outputs. Run --run, then --validate. The protocol is
fixed in artifacts/r8_shape_cost/financial/PROTOCOL.json before execution.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / 'artifacts/r8_shape_cost/financial'
REPORT = PROJECT / 'docs/shape_cost_20260911/FINANCIAL_RESULTS.md'
sys.path.insert(0, str(PROJECT / 'research/r8_commodity_etp'))
import panel_scope as scope
METHODS = ['Shift-ERM', 'Vol-ERM']
CONTRASTS = ['high', 'other', 'high_minus_other', 'matched_all']
ALPHA = .01


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def loss(y, q):
    error = y - q
    return (ALPHA - (error < 0)) * error


def load_inputs():
    inputs, pairs, checks = {}, [], {'stored_metric_cells': 0, 'daily_identity_rows': 0}

    def bind(p):
        inputs[str(p.relative_to(PROJECT))] = sha(p)
        return p

    for p in [Path(scope.__file__), PROJECT/'source/scripts/extension_20260831/panel_statistics.py',
              PROJECT/'research/r8_commodity_etp/controlled_comparisons.py',
              PROJECT/'research/r8_commodity_etp/full_candidates.py']:
        bind(p)
    for model in sorted(scope.MODELS):
        directory, suffix = scope.MODELS[model]
        for asset in scope.ASSETS:
            base = scope.source(asset)
            forecast_path = base/'data'/directory/(f'{asset}_{suffix}.parquet' if suffix else f'{asset}.parquet')
            return_path = base/'data/returns'/f'{asset}.csv'
            folder = scope.controlled(asset)/f'{model}__{asset}'/'full'
            done = json.loads(bind(folder/'complete.json').read_text())
            for filename, digest in done['outputs'].items():
                assert sha(bind(folder/filename)) == digest, (model, asset, filename)
            assert sha(bind(forecast_path)) == done['binding']['forecast_sha256']
            assert sha(bind(return_path)) == done['binding']['return_sha256']
            frame = pd.read_parquet(folder/'daily.parquet')
            forecast = pd.read_parquet(forecast_path)
            returns = pd.read_csv(return_path, index_col='date', parse_dates=True).log_return
            assert forecast.index.is_unique and forecast.index.is_monotonic_increasing
            assert frame.index.is_unique and frame.index.is_monotonic_increasing
            aligned = returns.loc[forecast.index].to_numpy()
            nc = int(.7*len(aligned))
            assert frame.index.equals(forecast.index[nc:])
            np.testing.assert_array_equal(frame.r.to_numpy(), aligned[nc:])
            sigma = returns.shift(1).rolling(20, min_periods=20).std(ddof=1).reindex(forecast.index).to_numpy()
            sigma = np.maximum(sigma, 1e-8)
            assert np.isfinite(sigma).all()
            np.testing.assert_array_equal(frame.sigma.to_numpy(), sigma[nc:])
            high = sigma[nc:] > np.quantile(sigma[:nc], .9)
            np.testing.assert_array_equal(frame.high_volatility.to_numpy(), high)
            scale = float(np.std(aligned[:nc], ddof=1))
            scale_independent = float(np.sqrt(sum((float(x)-float(np.mean(aligned[:nc])))**2 for x in aligned[:nc])/(nc-1)))
            assert scale > 0
            np.testing.assert_allclose(scale, scale_independent, rtol=2e-14, atol=1e-16)
            y = frame.r.to_numpy()
            q = np.column_stack([frame[f'0.01/{m}'].to_numpy() for m in METHODS])
            assert np.isfinite(q).all() and np.isfinite(y).all()
            losses = loss(y[:, None], q)
            alternate = np.where(y[:, None] < q, .99*(q-y[:, None]), .01*(y[:, None]-q))
            np.testing.assert_allclose(losses, alternate, rtol=2e-15, atol=2e-18)
            v = -q
            overshoot = np.maximum(q-y[:, None], 0.)
            np.testing.assert_allclose(losses, ALPHA*(v+y[:, None])+overshoot, rtol=2e-14, atol=2e-17)
            delta = losses[:, 1]-losses[:, 0]
            threshold = ALPHA*(v[:, 1]-v[:, 0])
            over = overshoot[:, 1]-overshoot[:, 0]
            np.testing.assert_allclose(delta, threshold+over, rtol=1e-11, atol=4e-17)
            metrics = pd.read_csv(folder/'metrics.csv')
            for j, method in enumerate(METHODS):
                for state, mask in [('All', np.ones(len(frame), bool)), ('High', high), ('Other', ~high)]:
                    if not mask.any():
                        continue
                    row = metrics[(metrics.alpha == .01)&(metrics.method == method)&(metrics.state == state)].iloc[0]
                    assert row.n_test == mask.sum()
                    assert row.viol == (y[mask] < q[mask, j]).sum()
                    np.testing.assert_allclose(row.QS, losses[mask, j].mean(), rtol=2e-12, atol=2e-16)
                    checks['stored_metric_cells'] += 1
            checks['daily_identity_rows'] += len(y)*2
            pairs.append(dict(key=f'{model}__{asset}', model=model, asset=asset, dates=frame.index,
                              y=y, q=q, loss=losses, high=high, scale=scale, n_cal=nc,
                              cal_first=str(forecast.index[0].date()), cal_last=str(forecast.index[nc-1].date()),
                              delta=delta, threshold=threshold, overshoot=over,
                              folder=str(folder.relative_to(PROJECT))))
    assert len(pairs) == 168
    assert sum(p['high'].any() and (~p['high']).any() for p in pairs) == 161
    return pairs, inputs, checks


def describe(pairs):
    rows = []
    for p in pairs:
        for state, mask in [('high', p['high']), ('other', ~p['high']), ('all', np.ones(len(p['y']), bool))]:
            if not mask.any():
                continue
            row = dict(key=p['key'], model=p['model'], asset=p['asset'], state=state,
                       n=int(mask.sum()), n_cal=p['n_cal'], calibration_sd=p['scale'],
                       matched=bool(p['high'].any() and (~p['high']).any()))
            for j, method in enumerate(METHODS):
                tag = 'shift' if j == 0 else 'vol'
                row[f'{tag}_qs'] = float(p['loss'][mask, j].mean())
                row[f'{tag}_pi'] = float((p['y'][mask] < p['q'][mask, j]).mean())
                row[f'{tag}_signed_var'] = float((-p['q'][mask, j]).mean())
                row[f'{tag}_overshoot'] = float(np.maximum(p['q'][mask, j]-p['y'][mask], 0.).mean())
            for field in ['delta', 'threshold', 'overshoot']:
                row[field] = float(p[field][mask].mean())
                row[field+'_normalized'] = row[field]/p['scale']
            rows.append(row)
    per_pair = pd.DataFrame(rows)
    rows = []
    for population, selected in [('matched161', per_pair[per_pair.matched]), ('full168', per_pair[per_pair.state == 'all'])]:
        for state, frame in selected.groupby('state', sort=True):
            row = dict(population=population, state=state, pairs=len(frame), pair_dates=int(frame.n.sum()))
            for name in ['shift_qs','vol_qs','shift_pi','vol_pi','shift_signed_var','vol_signed_var',
                         'shift_overshoot','vol_overshoot','delta','threshold','overshoot',
                         'delta_normalized','threshold_normalized','overshoot_normalized']:
                row[name] = float(frame[name].mean())
            row['shift_qs_normalized'] = float((frame.shift_qs/frame.calibration_sd).mean())
            row['vol_qs_normalized'] = float((frame.vol_qs/frame.calibration_sd).mean())
            rows.append(row)
    summary = pd.DataFrame(rows)
    matched = summary[summary.population == 'matched161'].set_index('state')
    point = np.array([matched.loc['high','delta_normalized'], matched.loc['other','delta_normalized'],
                      matched.loc['high','delta_normalized']-matched.loc['other','delta_normalized'],
                      matched.loc['all','delta_normalized']])
    return per_pair, summary, point


def calendar_arrays(pairs):
    matched = [p for p in pairs if p['high'].any() and (~p['high']).any()]
    dates = pd.date_range(min(p['dates'][0] for p in matched), max(p['dates'][-1] for p in matched), freq='D')
    available = np.zeros((len(dates), len(matched), 3), dtype=np.float64)
    values = np.zeros_like(available)
    for i, p in enumerate(matched):
        positions = dates.get_indexer(p['dates'])
        assert (positions >= 0).all()
        for j, mask in enumerate([p['high'], ~p['high'], np.ones(len(p['high']), bool)]):
            available[positions, i, j] = mask
            values[positions, i, j] = mask*p['delta']/p['scale']
    return matched, dates, available, values


def seed_sequence(block, protocol):
    tag = int.from_bytes(hashlib.sha256(protocol['bootstrap']['namespace'].encode()).digest()[:4], 'little')
    return np.random.SeedSequence([protocol['bootstrap']['seed_base'], tag, block])


def make_counts(T, block, protocol):
    rng = np.random.default_rng(seed_sequence(block, protocol))
    rows = []
    for _ in range(protocol['bootstrap']['draws']):
        begins = rng.integers(0, T, size=int(np.ceil(T/block)))
        ix = ((begins[:, None]+np.arange(block)) % T).ravel()[:T]
        rows.append(np.bincount(ix, minlength=T).astype(np.uint16))
    return np.stack(rows)


def support_gate(denominator):
    # One empty cell anywhere aborts this entire four-contrast family.
    return not bool((denominator <= 0).any())


def bootstrap(pairs, point, protocol):
    matched, dates, available, values = calendar_arrays(pairs)
    status, intervals, empty_rows = [], [], []
    for block in protocol['bootstrap']['block_calendar_days']:
        counts = make_counts(len(dates), block, protocol)
        denominator = (counts @ available.reshape(len(dates), -1)).reshape(len(counts), len(matched), 3)
        bad = np.argwhere(denominator == 0)
        for b, i, state in bad:
            empty_rows.append(dict(block_calendar_days=block, draw=int(b), key=matched[i]['key'], state=['high','other','all'][state]))
        passed = support_gate(denominator)
        if passed:
            numerator = (counts @ values.reshape(len(dates), -1)).reshape(denominator.shape)
            means = (numerator/denominator).mean(axis=1)
            draws = np.column_stack([means[:, 0], means[:, 1], means[:, 0]-means[:, 1], means[:, 2]])
            sd = draws.std(axis=0, ddof=1)
            assert (sd > 0).all()
            critical = float(np.quantile(np.max(np.abs((draws-point)/sd), axis=1), .95))
            for j, name in enumerate(CONTRASTS):
                intervals.append(dict(block_calendar_days=block, contrast=name, point=point[j],
                                      lower=point[j]-critical*sd[j], upper=point[j]+critical*sd[j],
                                      family_size=4, draws=999, critical=critical, bootstrap_sd=sd[j]))
        else:
            draws = np.empty((0, 4))
        np.savez_compressed(OUT/f'bootstrap_{block}.npz', counts=counts, denominator=denominator,
                            draws=draws, point=point, keys=np.array([p['key'] for p in matched]),
                            calendar=dates.to_numpy())
        status.append(dict(block_calendar_days=block, inference='available' if passed else 'aborted_empty_state',
                           total_draws=len(counts), draws_with_empty_cells=int(np.unique(bad[:, 0]).size),
                           empty_cells=len(bad), pairs_ever_empty=len(np.unique(bad[:, 1])),
                           no_pairs_dropped=True, no_draws_redrawn=True, primary_family_size=4))
        print('financial bootstrap support', block, status[-1], flush=True)
    return status, pd.DataFrame(intervals, columns=['block_calendar_days','contrast','point','lower','upper','family_size','draws','critical','bootstrap_sd']), pd.DataFrame(empty_rows, columns=['block_calendar_days','draw','key','state'])


def rational_checks():
    alpha = Fraction(1, 100)
    grid = [Fraction(-3), Fraction(-1, 100), Fraction(0), Fraction(1, 100), Fraction(2)]
    count = 0
    for y in grid:
        for q0 in grid:
            for q1 in grid:
                qs = [(alpha-int(y < q))*(y-q) for q in [q0, q1]]
                accounting = alpha*(-q1+q0)+max(q1-y, 0)-max(q0-y, 0)
                assert qs[1]-qs[0] == accounting
                count += 1
    good = np.ones((3, 2, 3))
    assert support_gate(good)
    bad = good.copy(); bad[1, 0, 0] = 0
    assert not support_gate(bad)
    return dict(exact_rational_identity_cases=count, empty_support_negative_control=True,
                identity_covers_positive_negative_zero_thresholds_and_ties=True)


def write_report(summary, point, status, checks):
    matched = summary[summary.population == 'matched161'].set_index('state')
    rows = []
    for state in ['high','other','all']:
        x = matched.loc[state]
        rows.append(f"| {state} | {int(x.pairs)} | {int(x.pair_dates):,} | {x.delta_normalized:.8f} | {x.delta*1e4:.5f} | {x.shift_pi*100:.3f}% | {x.vol_pi*100:.3f}% |")
    accounting = []
    for state in ['high','other','all']:
        x = matched.loc[state]
        accounting.append(f"| {state} | {x.threshold_normalized:.8f} | {x.overshoot_normalized:.8f} | {x.delta_normalized:.8f} |")
    notes = '\n'.join(f"- {x['block_calendar_days']}-day blocks: {x['inference']}; {x['draws_with_empty_cells']}/999 draws contain empty pair-state cells, involving {x['pairs_ever_empty']} pairs and {x['empty_cells']} empty cells." for x in status)
    full = summary[summary.population == 'full168'].iloc[0]
    text = f'''# Existing-path financial study: constant versus volatility-scaled correction

Date: 11 September 2026. Exploratory analysis of previously inspected outcomes. Methods, state labels, normalizer, matched population, inference family, seed and empty-state rule were fixed in `artifacts/r8_shape_cost/financial/PROTOCOL.json` before this computation. No forecasts, corrections or models were fitted, and no financial data were downloaded. Existing artifacts and canonical manuscript sources were not modified.

## Result

The matched comparison is Vol-ERM minus Shift-ERM. Both fit one coefficient by original-return pinball loss. All 161 pairs with both original volatility states are retained. Negative differences favour Vol-ERM.

| State/horizon | Pairs | Pair-dates | Normalized QS difference | QS difference times 10,000 | Shift-ERM violations | Vol-ERM violations |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The high-minus-other normalized loss interaction is **{point[2]:.8f}**. This is a descriptive within-pair comparison on a matched population, unlike a subtraction of the original table's 161-pair high-state and 168-pair other-state means. The full 168-pair horizon remains a separate descriptive sensitivity: normalized difference {full.delta_normalized:.8f}; raw-unit difference times 10,000 {full.delta*1e4:.5f}.

## The prespecified inference rule

{notes}

An empty denominator causes the entire four-contrast primary family for that block length to be suppressed. No pair was dropped from that population; no draws were rejected and replaced; no state was pooled or reweighted. Consequently, an aborted family supplies **no confidence band and no significance claim**, including for the matched whole-horizon contrast in that family. The saved bootstrap files preserve all support counts, and `empty_state_events.csv` identifies every occurrence. This is a completed, transparently limited analysis, not a failed numerical run.

The state labels are past-only, but their realised test frequency can be very low. The fixed support audit identified seven Bitcoin pairs without any high-state date and excluded them by the prespecified both-state criterion; some retained high states have only four or five observations. Prior knowledge of that sparsity was recorded in the protocol before loss computation.

## Exact score accounting

For log-return loss L=-r and signed threshold v=-q, daily QS equals 0.01(v-L)+(L-v)+. Hence the between-method loss difference equals a signed-threshold component plus the difference in realised overshoot. Numbers below are normalized by each pair's fixed calibration-return standard deviation and then pair-averaged:

| State/horizon | 0.01 times signed-threshold difference | Overshoot difference | Sum: QS difference |
|---|---:|---:|---:|
{chr(10).join(accounting)}

These components explain where the reported threshold changes and where realised losses beyond it change. Overshoot is not Expected Shortfall; the target weight 0.01 is not an estimated funding cost. Log-return units do not equal exact cash P&L on a fictitious notional. No capital, welfare, liquidity-buffer or trading-profit advantage is inferred.

## Interpretation and limits

This study checks where the two fixed correction forms differ on existing market paths. It does not estimate oracle loss, the tail density or the long-run tail variance, and it does not test the quantitative decision boundary of the new known-scale experiment. The two-state averages do not establish date-conditional calibration or violation independence. Both datasets and overall method results were already inspected, so the new calculation is not an untouched external test. Constant-to-scaled correction and stressed-state robustness are already studied by Zhong (2026); the intended scientific distinction remains estimator-specific expected cost, not the presence of a state comparison.

## Verification and reproduction

`research/r8_shape_cost/financial.py --run` creates this analysis. `--validate` independently replays every stored pair-state score and normalizer, exact rational identity cases, all 1,998 common-calendar bootstrap count vectors and every pair-state denominator. Separate sum/weighted-count implementations verify the bootstrap support, while synthetic empty-state controls ensure the family-abort rule is active. Current completion is authoritative only in `artifacts/r8_shape_cost/financial/validation.json`.

Input, protocol and producer SHA-256 hashes are in `receipt.json`; all outputs and the report are bound in `validation.json`. Existing original completion receipts are verified before analysis. {checks['stored_metric_cells']} stored score cells and {checks['daily_identity_rows']:,} daily method-loss rows were checked. Calibration uses the first floor(0.7 N) returns on each model's original forecast calendar, with sample standard deviation ddof=1. High-state masks exactly replay the original past-20-observation volatility calculation and original calibration90th percentile. There are no manuscript or PDF edits in this research-stage task.
'''
    REPORT.write_text(text)


def run():
    protocol = json.loads((OUT/'PROTOCOL.json').read_text())
    assert protocol['methods'] == METHODS and protocol['primary_family'] == CONTRASTS
    assert not (OUT/'receipt.json').exists(), 'Use a fresh output directory; do not overwrite a completed research run.'
    pairs, inputs, checks = load_inputs()
    per_pair, summary, point = describe(pairs)
    per_pair.to_csv(OUT/'pair_state_metrics.csv', index=False, float_format='%.17g')
    summary.to_csv(OUT/'summary.csv', index=False, float_format='%.17g')
    support = [dict(key=p['key'], model=p['model'], asset=p['asset'], n_cal=p['n_cal'],
                    calibration_first=p['cal_first'], calibration_last=p['cal_last'], calibration_sd=p['scale'],
                    test_first=str(p['dates'][0].date()), test_last=str(p['dates'][-1].date()),
                    high_days=int(p['high'].sum()), other_days=int((~p['high']).sum()),
                    matched=bool(p['high'].any() and (~p['high']).any()), input_folder=p['folder']) for p in pairs]
    pd.DataFrame(support).to_csv(OUT/'support.csv', index=False, float_format='%.17g')
    status, intervals, empty = bootstrap(pairs, point, protocol)
    intervals.to_csv(OUT/'intervals.csv', index=False, float_format='%.17g')
    empty.to_csv(OUT/'empty_state_events.csv', index=False)
    dump(OUT/'inference_status.json', status)
    checks.update(rational_checks())
    dump(OUT/'checks.json', checks)
    write_report(summary, point, status, checks)
    dump(OUT/'receipt.json', dict(status='analysis_complete_validation_pending', inputs=inputs,
                                protocol_sha256=sha(OUT/'PROTOCOL.json'),
                                protocol_md_sha256=sha(OUT/'PROTOCOL.md'), producer_sha256=sha(__file__),
                                outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix not in ['.log'] and p.name != 'receipt.json'},
                                report_sha256=sha(REPORT), pairs=168, matched_pairs=161))
    print('Financial existing-path analysis complete; validation pending.', flush=True)


def validate():
    record = json.loads((OUT/'receipt.json').read_text())
    assert sha(__file__) == record['producer_sha256']
    assert sha(OUT/'PROTOCOL.json') == record['protocol_sha256']
    assert sha(OUT/'PROTOCOL.md') == record['protocol_md_sha256']
    assert sha(REPORT) == record['report_sha256']
    for name, digest in record['outputs'].items():
        assert sha(OUT/name) == digest, name
    for name, digest in record['inputs'].items():
        assert sha(PROJECT/name) == digest, name
    pairs, inputs, checks = load_inputs()
    assert inputs == record['inputs']
    per_pair, summary, point = describe(pairs)
    for name, frame in [('pair_state_metrics.csv', per_pair), ('summary.csv', summary)]:
        saved = pd.read_csv(OUT/name, float_precision='round_trip')
        pd.testing.assert_frame_equal(frame, saved, check_dtype=False, rtol=1e-14, atol=1e-16)
    rational = rational_checks()
    protocol = json.loads((OUT/'PROTOCOL.json').read_text())
    matched, dates, available, values = calendar_arrays(pairs)
    status = json.loads((OUT/'inference_status.json').read_text())
    empty_events = pd.read_csv(OUT/'empty_state_events.csv')
    replay_events = []
    denominator_cells = 0
    for spec in status:
        block = spec['block_calendar_days']
        saved = np.load(OUT/f'bootstrap_{block}.npz')
        counts = saved['counts']
        np.testing.assert_array_equal(counts, make_counts(len(dates), block, protocol))
        np.testing.assert_array_equal(counts.sum(axis=1), np.full(999, len(dates)))
        np.testing.assert_array_equal(saved['point'], point)
        np.testing.assert_array_equal(saved['calendar'], dates.to_numpy())
        independent = np.empty_like(saved['denominator'])
        for i, p in enumerate(matched):
            positions = dates.get_indexer(p['dates'])
            # Independent indexed sums, not the producer's dense calendar matmul.
            weights = counts[:, positions]
            independent[:, i, 0] = weights[:, p['high']].sum(axis=1)
            independent[:, i, 1] = weights[:, ~p['high']].sum(axis=1)
            independent[:, i, 2] = weights.sum(axis=1)
        np.testing.assert_array_equal(saved['denominator'], independent)
        denominator_cells += independent.size
        bad = np.argwhere(independent == 0)
        assert (spec['inference'] == 'available') == support_gate(independent)
        assert spec['draws_with_empty_cells'] == len(np.unique(bad[:,0]))
        assert spec['empty_cells'] == len(bad)
        if len(bad):
            assert saved['draws'].shape == (0,4)
        else:
            independently_weighted = np.zeros((999, 3))
            for i, p in enumerate(matched):
                weights = counts[:, dates.get_indexer(p['dates'])]
                for j, mask in enumerate([p['high'], ~p['high'], np.ones(len(p['high']), bool)]):
                    independently_weighted[:,j] += (weights[:,mask] @ (p['delta'][mask]/p['scale']))/independent[:,i,j]/len(matched)
            expected = np.column_stack([independently_weighted[:,0],independently_weighted[:,1],independently_weighted[:,0]-independently_weighted[:,1],independently_weighted[:,2]])
            np.testing.assert_allclose(saved['draws'], expected, rtol=1e-11, atol=2e-17)
        for b,i,s in bad:
            replay_events.append(dict(block_calendar_days=block,draw=int(b),key=matched[i]['key'],state=['high','other','all'][s]))
    pd.testing.assert_frame_equal(empty_events, pd.DataFrame(replay_events), check_dtype=False)
    for p, digest in record['inputs'].items():
        assert sha(PROJECT/p) == digest
    intervals = pd.read_csv(OUT/'intervals.csv')
    for spec in status:
        group = intervals[intervals.block_calendar_days == spec['block_calendar_days']]
        assert len(group) == (4 if spec['inference'] == 'available' else 0)
    outcome = dict(status='passed', classification='exploratory', inference_status=status,
                   input_files=len(inputs), input_hashes_unchanged=True, pairs=168, matched_pairs=161,
                   bootstrap_draws_replayed=1998, bootstrap_denominator_cells_replayed=denominator_cells,
                   exact_checks=rational, score_checks=checks, receipt_sha256=sha(OUT/'receipt.json'),
                   producer_sha256=sha(__file__), report_sha256=sha(REPORT),
                   outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix != '.log' and p.name != 'validation.json'})
    dump(OUT/'validation.json', outcome)
    print(json.dumps(outcome, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    assert args.run != args.validate, 'Choose exactly one operation.'
    run() if args.run else validate()
