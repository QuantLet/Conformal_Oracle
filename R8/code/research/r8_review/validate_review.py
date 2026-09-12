"""Validate the complete scientific extension and its links to the current panel."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from controlled_comparisons import PROJECT,ROOT,OUT,MODELS,sha,loss


def main():
    review=PROJECT/'artifacts/review_20260909';records={};count=0
    for model in MODELS:
        for returns in sorted((ROOT/'data/returns').glob('*.csv')):
            folder=OUT/f'{model}__{returns.stem}'
            for part in [folder,folder/'full']:
                meta=json.loads((part/'complete.json').read_text())
                assert meta['binding']['return_sha256']==sha(returns)
                for name,want in meta['outputs'].items():assert sha(part/name)==want
                records[str(part.relative_to(PROJECT))]=sha(part/'complete.json')
            native=pd.read_parquet(ROOT/'posthoc'/f'{model}__{returns.stem}.parquet')
            full=pd.read_parquet(folder/'full/daily.parquet')
            policy=pd.read_parquet(folder/'policy_daily.parquet');policy=policy[policy.origin=='Original']
            assert native.index.equals(full.index) and native.index.equals(policy.index)
            for name,column in [('Raw','Raw'),('Conformal','Shift-CP')]:
                np.testing.assert_allclose(native[name],full[f'0.01/{column}'],rtol=0,atol=1e-14)
            np.testing.assert_allclose(native['rolling'],policy.Rolling250,rtol=0,atol=1e-14)
            assert json.loads((folder/'complete.json').read_text())['past_only_selection_checked']
            count+=1
    assert count==216
    for p in sorted((review/'complexity_mc').glob('*/complete.json')):
        record=json.loads(p.read_text());assert record['binding']['producer_sha256']==sha(Path(__file__).with_name('complexity_simulation.py'))
        for name,want in record['outputs'].items():assert sha(p.parent/name)==want
        records[str(p.parent.relative_to(PROJECT))]=sha(p)
    assert len(list((review/'complexity_mc').glob('*/complete.json')))==32
    reports={}
    for scope,n in [('empirical',216),('simulation',32)]:
        path=review/'quality'/f'fresh_{scope}_replay.json';r=json.loads(path.read_text())
        assert r['complete'] and len(r['rows'])==n and r['checker_sha256']==sha(Path(__file__).with_name('verify_extensions.py'))
        reports[scope]=sha(path)
    aggregate=review/'results/complete.json';r=json.loads(aggregate.read_text())
    assert r['producer_sha256']==sha(Path(__file__).with_name('aggregate_extensions.py'))
    for name,want in r['outputs'].items():assert sha(aggregate.parent/name)==want
    for p in (ROOT/'results/review').glob('*.csv'):
        origin=review/'results'/p.name
        if p.name=='complexity_mc.csv':origin=review/'complexity_mc/summary.csv'
        elif p.name.startswith('calendar_'):origin=review/'calendar'/p.name
        assert sha(p)==sha(origin),p
    # Bootstrap point estimates independently recovered from reported per-pair losses.
    full=pd.read_csv(review/'results/full_pairs.csv');full=full[(full.alpha==.01)&(full.state=='All')]
    means=full.groupby('method').QS.mean()*1e4
    intervals=pd.read_csv(review/'results/paired_intervals.csv')
    for r in intervals.itertuples():
        if r.comparator in means and r.reference in means:
            np.testing.assert_allclose(r.difference,means[r.comparator]-means[r.reference],rtol=0,atol=1e-11)
    output=dict(passed=True,pairs=count,simulation_configurations=32,simulated_histories=1000,
                exact_fresh_replay_reports=reports,verified_result_bindings=records,
                bootstrap_aggregation_sha256=sha(aggregate),checker_sha256=sha(__file__))
    (review/'quality/review_validation.json').write_text(json.dumps(output,indent=2)+'\n')
    print('PASS: 216 matched pairs, 32 controlled simulation cells, current-panel links and fresh replay bindings',flush=True)


if __name__=='__main__':main()
