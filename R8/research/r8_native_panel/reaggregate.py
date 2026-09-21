"""Restrict existing results and replay their original aggregation, without fitting."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '2'
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / 'source/scripts/extension_20260831'
sys.path.insert(0, str(SCRIPTS))
from paper_scope import ARCHIVE, ROOT, DECISION, MODELS, N_PAIRS, EXCLUDED
from panel_statistics import scores
import analyse_panel as base

ART = ROOT.parent
INPUTS = {}
CHECKS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    path = Path(path)
    INPUTS[str(path.relative_to(PROJECT))] = sha(path)
    return path


def read(path):
    path = bind(path)
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)


def selected(frame):
    return frame[~frame.model.isin(EXCLUDED)].copy() if 'model' in frame else frame.copy()


def save(frame, name, directory=ROOT/'results'):
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory/name, index=False)


def copy(path, dest):
    bind(path); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)


def load_module(name, path):
    bind(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def verify_metrics(frame, metrics, names, expected=False):
    """Independent daily-to-pair arithmetic, including expected randomised loss."""
    y = frame.r.to_numpy()
    for name in names:
        row = metrics[metrics.method == name].iloc[0]
        if expected and name == 'DtACI-projected-expected':
            value = frame['DtACI-expected/loss'].mean()
        else:
            q = frame[name].to_numpy()
            value = np.mean((.01-(y < q))*(y-q))
            assert int((y < q).sum()) == row.viol, (name, 'violations')
        np.testing.assert_allclose(value, row.QS, rtol=1e-11, atol=1e-15)


def base_results():
    # Pair results are invariant to removing other forecasters.
    for path in sorted((ARCHIVE/'results').glob('*.csv')):
        frame = read(path)
        if 'model' in frame:
            save(selected(frame), path.name)
    for name in ['primary_ready.json', 'quality/asset_inventory.csv',
                 'results/monte_carlo/grid.csv', 'results/predictive_sampling/draws.csv',
                 'results/predictive_sampling/propagation.csv',
                 'results/review/complexity_mc.csv', 'results/review/calendar_scores.csv']:
        copy(ARCHIVE/name, ROOT/name)
    strata = pd.read_csv(ROOT/'results/strata.csv'); rows = []
    for excluded in [None] + list(base.CLASSES):
        d = strata if excluded is None else strata[strata.asset_class != excluded]
        for method, g in d.groupby('method'):
            rows.append(dict(excluded_class=excluded or 'None', method=method, n_pairs=len(g),
                             QS=g.QS.mean(), normalised_QS=g.normalised_QS.mean()))
    save(pd.DataFrame(rows), 'class_sensitivity.csv')
    quotation = pd.read_csv(ROOT/'results/quotation_period_sensitivity.csv')
    summary = quotation.groupby(['variant','method']).agg(n=('asset','size'), QS=('QS','mean'), pi=('pihat','mean')).reset_index()
    save(summary, 'quotation_period_summary.csv')
    ledger = pd.read_csv(ROOT/'results/indication.csv'); metrics = pd.read_csv(ROOT/'results/posthoc.csv')
    assert len(metrics) == N_PAIRS*10 and len(ledger) == N_PAIRS*4
    frames = []
    for asset in sorted(base.CLASS):
        supports = []
        for model in MODELS:
            path = ARCHIVE/'posthoc'/f'{model}__{asset}.parquet'; frame = read(path)
            supports.append(frame.index)
            pair = metrics[(metrics.model == model)&(metrics.asset == asset)]
            verify_metrics(frame, pair, base.METHODS)
            r = ledger[(ledger.model == model)&(ledger.asset == asset)&(ledger.alpha == .01)].iloc[0]
            gate = r.p_kup_cal < .05 or r.TL_cal != 'Green'
            frame['Gate-static'] = frame.Conformal if gate else frame.Raw
            frame['Gate-rolling'] = frame['rolling'] if gate else frame.Raw
            frames.append(frame)
            if asset == 'SP500': copy(path, ROOT/'posthoc'/path.name)
        dates = supports[0]
        for index in supports[1:]: dates = dates.intersection(index)
        for model in ['CAViaR-SAV','CAViaR-AS','GAS-t']:
            path = ARCHIVE/'data/dynamic'/f'{asset}_{model}.parquet'; dynamic = read(path)
            dates = dates.intersection(dynamic.index[int(.7*len(dynamic)):])
        common = pd.read_csv(ROOT/'results/common_support.csv')
        support = common[common.asset == asset]
        assert support.n_test.eq(len(dates)).all() and support['first'].eq(str(dates[0].date())).all()
        assert support['last'].eq(str(dates[-1].date())).all()
    assert len(frames) == N_PAIRS
    CHECKS['base_daily_metrics'] = N_PAIRS*10
    CHECKS['common_support_unchanged_assets'] = len(base.CLASS)
    base.ROOT = ROOT
    print('Replaying broad-method intervals on 168 pairs', flush=True)
    base.bootstrap(frames)
    del frames; gc.collect()


def review_results():
    sys.path.insert(0, str(PROJECT/'research/r8_review'))
    module = load_module('native_review_aggregate', PROJECT/'research/r8_review/aggregate_extensions.py')
    original = PROJECT/'artifacts/review_20260909/results'; dest = ROOT/'results/review'
    dest.mkdir(parents=True, exist_ok=True)
    for name, keys in [('sweep',['alpha','n_cal','method']), ('sweep_states',['n_cal','method','state']),
                       ('policies',['origin','method','state']), ('full',['alpha','method','state'])]:
        frame = selected(read(original/f'{name}_pairs.csv')); save(frame, f'{name}_pairs.csv', dest)
        if name == 'sweep': frame = frame[frame.method != 'Full-static']
        summary = frame.groupby(keys).agg(pairs=('asset','size'), QS=('QS','mean'), violation_rate=('pihat','mean'),
            mean_absolute_threshold=('width','mean'), test_observations=('n_test','sum'), violations=('viol','sum'),
            kupiec_rejections=('p_kup',lambda x:int((x<.05).sum())), green=('TL',lambda x:int((x=='Green').sum())))
        save(summary.reset_index(), f'{name}_summary.csv', dest)
    save(selected(read(original/'decisions.csv')), 'decisions.csv', dest)
    frames = []
    for model in MODELS:
        for asset in sorted(base.CLASS):
            folder = module.OUT/f'{model}__{asset}'
            full = read(folder/'full/daily.parquet'); policy = read(folder/'policy_daily.parquet')
            policy = policy[policy.origin == 'Original']
            assert full.index.equals(policy.index) and np.array_equal(full.r, policy.r)
            frame = pd.DataFrame({'r':full.r}, index=full.index)
            for name in module.FAMILY: frame[name] = full[f'0.01/{name}']
            for name in module.POLICIES: frame[name] = policy[name]
            metrics = read(folder/'full/metrics.csv')
            metrics = metrics[(metrics.alpha == .01)&(metrics.state == 'All')]
            verify_metrics(frame, metrics, module.FAMILY)
            frames.append(frame)
    CHECKS['controlled_daily_metrics'] = N_PAIRS*len(module.FAMILY)
    module.RESULTS = dest
    print('Replaying controlled/policy intervals on 168 pairs', flush=True)
    module.bootstrap(frames)
    del frames; gc.collect()


def decision_results():
    sys.path.insert(0, str(PROJECT/'research/r8_decision'))
    module = load_module('native_decision_aggregate', PROJECT/'research/r8_decision/aggregate.py')
    original = PROJECT/'artifacts/r8_decision/results'; DECISION.mkdir(parents=True, exist_ok=True)
    pairs = selected(read(original/'pairs.csv')); save(pairs, 'pairs.csv', DECISION)
    for name in ['decisions.csv','diagnostics.csv','seed_metrics.csv']:
        save(selected(read(original/name)), name, DECISION)
    raw = pairs[pairs.method == 'Raw'][['model','asset','QS']].rename(columns={'QS':'raw_QS'})
    merged = pairs.merge(raw,on=['model','asset'],validate='many_to_one')
    summary = merged.groupby('method').agg(pairs=('asset','size'), QS=('QS','mean'), normalised_QS=('normalised_QS','mean'),
        violation_rate=('pihat','mean'), width=('width','mean'), kupiec_rejections=('p_kup',lambda x:int((x<.05).sum())),
        kupiec_available=('p_kup','count'), test_observations=('n_test','sum'))
    summary['worse_than_raw'] = merged.assign(worse=merged.QS>merged.raw_QS).groupby('method').worse.sum()
    summary['QS_x10000'] = summary.QS*1e4
    save(summary.reset_index(), 'summary.csv', DECISION)
    save(pairs.groupby(['model','method']).agg(QS=('QS','mean'),violation_rate=('pihat','mean')).reset_index(), 'by_model.csv', DECISION)
    frames = []; scales = []; noncrypto = []
    for model in MODELS:
        for asset in sorted(base.CLASS):
            frame = read(PROJECT/'artifacts/r8_decision/pairs'/f'{model}__{asset}'/'daily.parquet')
            metric = pairs[(pairs.model == model)&(pairs.asset == asset)]
            verify_metrics(frame, metric, module.METHODS, expected=True)
            frames.append(frame); scales.append(metric.calibration_scale.iloc[0]); noncrypto.append(asset not in ['BTC','ETH'])
    CHECKS['strong_comparator_daily_metrics'] = N_PAIRS*len(module.METHODS)
    module.RESULTS = DECISION
    print('Replaying stronger-comparator intervals on 168 pairs', flush=True)
    intervals = module.bootstrap(frames, np.array(scales), np.array(noncrypto))
    save(intervals, 'intervals.csv', DECISION)
    del frames; gc.collect()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for path in [Path(__file__), Path(__file__).with_name('PROTOCOL.md'), SCRIPTS/'paper_scope.py',
                 SCRIPTS/'analyse_panel.py', SCRIPTS/'panel_statistics.py',
                 PROJECT/'research/r8_decision/methods.py']:
        bind(path)
    base_results(); review_results(); decision_results()
    before = json.loads((ART/'before.json').read_text())
    changed = []
    for name, digest in before['protected'].items():
        current = (PROJECT/name).read_bytes()
        if name == 'source/sections_r8/theory.tex':
            # The sole authorised edit in this file is the empirical example's
            # denominator; every theorem, assumption and equation stays exact.
            current = current.replace(b'Across \\nAugPairs{} pairs,', b'Across 216 pairs,')
        if hashlib.sha256(current).hexdigest() != digest: changed.append(name)
    assert not changed, changed
    generated = {'paper_outputs_manifest.json','r8_validation.json','forecast_structure.csv'}
    outputs = {str(p.relative_to(ART)):sha(p) for directory in [ROOT,DECISION]
               for p in directory.rglob('*') if p.is_file() and p.name not in generated}
    receipt = {'models':list(MODELS), 'excluded':list(EXCLUDED), 'pairs':N_PAIRS, 'inputs':INPUTS, 'outputs':outputs,
               'checks':CHECKS,
               'historical_files_byte_identical':sum(sha(PROJECT/p)==h for p,h in before['protected'].items()),
               'theory_unchanged_except_empirical_denominator':True,
               'new_inference':False, 'new_model_fitting':False, 'new_simulated_paths':False}
    (ART/'aggregation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['inputs','outputs']},indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Require exact replay of the saved aggregation receipt.')
    args = parser.parse_args()
    previous = json.loads((ART/'aggregation.json').read_text()) if args.check else None
    main()
    if previous is not None:
        current = json.loads((ART/'aggregation.json').read_text())
        assert current == previous, 'Aggregation replay changed inputs, outputs or validation scope'
        print('Exact aggregation replay passed', flush=True)
