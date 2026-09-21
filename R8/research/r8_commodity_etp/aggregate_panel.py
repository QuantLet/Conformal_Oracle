"""Reaggregate controlled and decision experiments on the fixed replacement panel."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '2'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from panel_scope import PROJECT, ART, ROOT, DECISION, MODELS, ASSETS, controlled, decisions

INPUTS = {}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    INPUTS[str(p.relative_to(PROJECT))] = sha(p)
    if p.suffix == '.json': return json.loads(p.read_text())
    return pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)


def module(name, path):
    INPUTS[str(path.relative_to(PROJECT))] = sha(path)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def verify(folder):
    record = read(folder/'complete.json')
    for name, digest in record['outputs'].items():
        assert sha(folder/name) == digest, (folder,name)
    return record


def verify_scores(frame, metrics, names, expected=False):
    y = frame.r.to_numpy()
    for name in names:
        r = metrics[metrics.method == name].iloc[0]
        if expected and name == 'DtACI-projected-expected':
            loss = frame['DtACI-expected/loss'].to_numpy()
        else:
            q = frame[name].to_numpy()
            loss = np.where(y < q, .99*(q-y), .01*(y-q))
            assert int((y < q).sum()) == int(r.viol), (name,'violations')
        np.testing.assert_allclose(loss.mean(), r.QS, atol=1e-15, rtol=1e-11)


def review():
    mod = module('etp_review_aggregation', PROJECT/'research/r8_review/aggregate_extensions.py')
    dest = ROOT/'results/review'; dest.mkdir(parents=True,exist_ok=True)
    mod.RESULTS = dest
    collections = {k:[] for k in ['sweep','sweep_states','policies','full']}
    frames, choices = [], []
    for model in MODELS:
        for asset in ASSETS:
            folder = controlled(asset)/f'{model}__{asset}'
            verify(folder); verify(folder/'full')
            for name,file in [('sweep','controlled.csv'), ('sweep_states','controlled_states.csv'),
                              ('policies','policies.csv'), ('full','full/metrics.csv')]:
                collections[name].append(read(folder/file))
            full = read(folder/'full/daily.parquet')
            policy = read(folder/'policy_daily.parquet'); policy = policy[policy.origin == 'Original']
            assert full.index.equals(policy.index) and np.array_equal(full.r,policy.r)
            frame = pd.DataFrame({'r':full.r},index=full.index)
            for name in mod.FAMILY: frame[name] = full[f'0.01/{name}']
            for name in mod.POLICIES: frame[name] = policy[name]
            metric = collections['full'][-1]
            verify_scores(frame, metric[(metric.alpha == .01)&(metric.state == 'All')], mod.FAMILY)
            frames.append(frame)
            fit = read(folder/'fits.json')
            for origin, choice in fit['policies'].items():
                choices.append(dict(model=model,asset=asset,origin=origin,window=choice['selected_window'],
                    gate=choice['gate'],validation_start=choice['validation_start'],calibration_stop=choice['train_stop_exclusive']))
    assert len(frames) == 168
    for name,items in collections.items():
        df = pd.concat(items,ignore_index=True); df.to_csv(dest/f'{name}_pairs.csv',index=False)
        keys = {'sweep':['alpha','n_cal','method'], 'sweep_states':['n_cal','method','state'],
                'policies':['origin','method','state'], 'full':['alpha','method','state']}[name]
        if name == 'sweep': df = df[df.method != 'Full-static']
        summary = df.groupby(keys).agg(pairs=('asset','size'),QS=('QS','mean'),violation_rate=('pihat','mean'),
            mean_absolute_threshold=('width','mean'),test_observations=('n_test','sum'),violations=('viol','sum'),
            kupiec_rejections=('p_kup',lambda s:int((s<.05).sum())),green=('TL',lambda s:int((s=='Green').sum())))
        summary.reset_index().to_csv(dest/f'{name}_summary.csv',index=False)
    pd.DataFrame(choices).to_csv(dest/'decisions.csv',index=False)
    mod.bootstrap(frames)
    return dest


def decision():
    mod = module('etp_decision_aggregation', PROJECT/'research/r8_decision/aggregate.py')
    DECISION.mkdir(parents=True,exist_ok=True); mod.RESULTS = DECISION
    frames, metrics, choices, diagnostics, seeds, scales, noncrypto = [],[],[],[],[],[],[]
    for model in MODELS:
        for asset in ASSETS:
            folder = decisions(asset)/f'{model}__{asset}'; verify(folder)
            frame = read(folder/'daily.parquet'); metric = read(folder/'metrics.csv'); fit = read(folder/'fits.json')
            assert set(metric.method) == set(mod.METHODS)
            verify_scores(frame,metric,mod.METHODS,expected=True)
            metric['normalised_QS'] = metric.QS/metric.calibration_scale
            frames.append(frame); metrics.append(metric); scales.append(metric.calibration_scale.iloc[0]); noncrypto.append(asset not in ['BTC','ETH'])
            one = read(folder/'dtaci_seed_metrics.csv'); one['model']=model; one['asset']=asset; seeds.append(one)
            gate = fit['gate']
            choices.append(dict(model=model,asset=asset,gate_selected=gate['selected'],past_minimum_selected=gate['past_minimum_selected'],
                state_p=fit['state']['selected_p'],state_penalty=fit['state']['selected_penalty'],
                pot_shift_threshold=fit['POT-Shift']['selected_threshold'],pot_vol_threshold=fit['POT-Vol']['selected_threshold']))
            certs = [t['fit']['certificate'] for t in fit['state']['trials']]+[fit['state']['fit']['certificate']]
            pots = [fit[n]['fit'] for n in ['POT-Shift','POT-Vol']]
            allpots = pots+[t['fit'] for n in ['POT-Shift','POT-Vol'] for t in fit[n]['trials']]
            diagnostics.append(dict(model=model,asset=asset,lp_max_gap=max(c['gap'] for c in certs),
                lp_max_dual_violation=max(c['dual_violation'] for c in certs),pot_final_fallbacks=sum(p['fallback'] for p in pots),
                pot_all_fallbacks=sum(p['fallback'] for p in allpots),pot_final_irregular_shapes=sum(p.get('irregular_shape',False) for p in pots),
                dtaci_projection_events=sum(fit['dtaci_projected']['projection_counts']),**fit['dtaci_unprojected_audit']))
    assert len(frames) == 168
    pairs = pd.concat(metrics,ignore_index=True); pairs.to_csv(DECISION/'pairs.csv',index=False)
    raw = pairs[pairs.method=='Raw'][['model','asset','QS']].rename(columns={'QS':'raw_QS'})
    merged = pairs.merge(raw,on=['model','asset'],validate='many_to_one')
    summary = merged.groupby('method').agg(pairs=('asset','size'),QS=('QS','mean'),normalised_QS=('normalised_QS','mean'),
        violation_rate=('pihat','mean'),width=('width','mean'),kupiec_rejections=('p_kup',lambda s:int((s<.05).sum())),
        kupiec_available=('p_kup','count'),test_observations=('n_test','sum'))
    summary['worse_than_raw'] = merged.assign(worse=merged.QS>merged.raw_QS).groupby('method').worse.sum()
    summary['QS_x10000'] = summary.QS*1e4
    summary.reset_index().to_csv(DECISION/'summary.csv',index=False)
    pd.DataFrame(choices).to_csv(DECISION/'decisions.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(DECISION/'diagnostics.csv',index=False)
    pd.concat(seeds,ignore_index=True).to_csv(DECISION/'seed_metrics.csv',index=False)
    pairs.groupby(['model','method']).agg(QS=('QS','mean'),violation_rate=('pihat','mean')).reset_index().to_csv(DECISION/'by_model.csv',index=False)
    mod.bootstrap(frames,np.array(scales),np.array(noncrypto)).to_csv(DECISION/'intervals.csv',index=False)
    return DECISION


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['review','decision'],required=True);p.add_argument('--check',action='store_true');args=p.parse_args()
    target=ART/f'aggregation_{args.kind}.json'
    previous=json.loads(target.read_text()) if args.check else None
    dest = review() if args.kind=='review' else decision()
    # Stable simulation/calendar files in review are bound by stage.json.
    outputs={str(p.relative_to(ART)):sha(p) for p in dest.iterdir() if p.is_file()}
    record=dict(producer_sha256=sha(__file__),scope_sha256=sha(Path(__file__).with_name('panel_scope.py')),
                inputs=INPUTS,outputs=outputs,pairs=168,unchanged_assets=21,reestimated_assets=3)
    if previous is not None: assert record==previous, 'Aggregation replay changed'
    target.write_text(json.dumps(record,indent=2)+'\n')
    print(args.kind,'replacement aggregation complete; exact replay:',args.check,flush=True)


if __name__ == '__main__': main()
