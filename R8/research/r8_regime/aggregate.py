"""Complete-scenario, paired Monte Carlo summaries from bound simulation blocks."""
import json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import engine as e
_spec = importlib.util.spec_from_file_location('regime_runner', Path(__file__).with_name('run.py'))
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)

KEYS = ['innovation', 'scenario', 'alpha', 'period', 'method']
METRICS = ('risk', 'hit_probability', 'realised_loss', 'realised_hits', 'absolute_error', 'excess_loss')


def mean_se(values):
    x = np.asarray(values)
    return float(x.mean()), float(x.std(ddof=1)/np.sqrt(len(x)))


def main():
    binding = run.initialise()
    files, frames, decisions = {}, [], []
    for kind in ('normal', 't5'):
        for start in range(0, e.REPS, run.BLOCK):
            folder = run.OUT/'blocks'/f'{kind}_{start:03d}_{start+run.BLOCK:03d}'
            receipt = json.loads((folder/'complete.json').read_text())
            assert receipt['binding_sha256'] == e.sha(run.OUT/'binding.json')
            for name, h in receipt['outputs'].items():
                assert e.sha(folder/name) == h, (folder, name)
            files[str(folder.relative_to(run.OUT)/'complete.json')] = e.sha(folder/'complete.json')
            frames.append(pd.read_csv(folder/'replications.csv'))
            decisions.append(pd.read_csv(folder/'decisions.csv'))
    df = pd.concat(frames, ignore_index=True)
    d = pd.concat(decisions, ignore_index=True)
    assert len(df) == 2*len(e.SCENARIOS)*2*e.REPS*len(e.METHODS)*len(e.PERIODS)
    assert not df.duplicated(KEYS+['replication']).any()
    assert not d.duplicated(['innovation', 'scenario', 'alpha', 'replication']).any()
    assert np.isfinite(df[list(METRICS)]).all().all()
    records = []
    reference = df.pivot(index=KEYS[:4]+['replication'], columns='method', values='risk')
    for keys, g in df.groupby(KEYS, sort=True):
        assert len(g) == e.REPS and set(g.replication) == set(range(e.REPS))
        record = dict(zip(KEYS, keys))
        record['independent_histories'] = len(g)
        for field in METRICS:
            record[field], record[field+'_MCSE'] = mean_se(g[field])
        matched = reference.loc[keys[:4]].sort_index()
        for ref in ('Raw', 'Static-CP', 'Rolling500', 'PastSelectedRolling'):
            delta = matched[keys[-1]]-matched[ref]
            record['difference_vs_'+ref], record['MCSE_vs_'+ref] = mean_se(delta)
        records.append(record)
    out = run.OUT/'results'
    out.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(out/'summary.csv', index=False)
    df.to_parquet(out/'replications.parquet', index=False)
    d.to_csv(out/'decisions.csv', index=False)
    # Changes relative to matched unchanged-regime controls share seeds.
    comparisons = []
    refs = {'appears': 'correct', 'disappears': 'biased', 'reverses': 'biased',
            'jump_oracle': 'correct', 'jump_ewma': 'steady_ewma'}
    for kind in ('normal', 't5'):
        for alpha in e.ALPHAS:
            for scenario, reference_scenario in refs.items():
                for period in e.PERIODS:
                    g = df[(df.innovation == kind)&(df.alpha == alpha)&(df.period == period)]
                    left = g[g.scenario == scenario].set_index(['method', 'replication'])
                    right = g[g.scenario == reference_scenario].set_index(['method', 'replication'])
                    for method in e.METHODS:
                        row = {'innovation': kind, 'alpha': alpha, 'scenario': scenario,
                               'reference_scenario': reference_scenario, 'period': period, 'method': method}
                        for metric in ('risk', 'hit_probability', 'excess_loss'):
                            delta = left.loc[method, metric]-right.loc[method, metric]
                            row[metric+'_change'], row[metric+'_MCSE'] = mean_se(delta)
                        comparisons.append(row)
    pd.DataFrame(comparisons).to_csv(out/'scenario_contrasts.csv', index=False)
    curves = []
    for kind in ('normal', 't5'):
        for scenario in e.SCENARIOS:
            for alpha in e.ALPHAS:
                sums = {}
                for start in range(0, e.REPS, run.BLOCK):
                    path = run.OUT/'blocks'/f'{kind}_{start:03d}_{start+run.BLOCK:03d}'/f'{scenario}_{alpha:g}_moments.npz'
                    a = np.load(path)
                    for key in a.files:
                        sums[key] = sums.get(key, 0)+a[key]
                for metric in ('risk', 'hit_probability', 'excess_loss'):
                    mean = sums[metric+'_sum']/e.REPS
                    variance = np.maximum((sums[metric+'_squared_sum']-e.REPS*mean**2)/(e.REPS-1), 0)
                    days = np.arange(e.LENGTH-e.START)-(e.BREAK-e.START)+1
                    curves.append(pd.DataFrame({'innovation': kind, 'scenario': scenario, 'alpha': alpha,
                        'method': np.tile(e.METHODS, len(days)), 'metric': metric,
                        'day_from_break': np.repeat(days, len(e.METHODS)),
                        'mean': mean.ravel(), 'MCSE': np.sqrt(variance/e.REPS).ravel()}))
    pd.concat(curves, ignore_index=True).to_parquet(out/'curves.parquet', index=False)
    paths = sorted(p for p in out.iterdir() if p.is_file())
    receipt = {'producer_sha256': e.sha(__file__), 'binding_sha256': e.sha(run.OUT/'binding.json'),
               'source_receipts': files, 'period_method_replications': len(df),
               'independent_histories': 1000,
               'outputs': {str(p.relative_to(run.OUT)): e.sha(p) for p in paths}}
    (run.OUT/'aggregation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('Aggregated', len(df), 'period-method records from 1,000 independent histories.')


if __name__ == '__main__':
    main()
