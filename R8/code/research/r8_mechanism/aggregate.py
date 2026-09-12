"""Paired Monte Carlo summaries; no outcome-dependent configuration removal."""
import json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import engine as e
_spec=importlib.util.spec_from_file_location('mechanism_runner',Path(__file__).with_name('run.py'))
run=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)

GROUP=['module','innovation','phi','n_cal','alpha','truth']


def fit_diagnostics(block):
    rows=[]
    def scan(node,path,rep):
        if isinstance(node,dict):
            if 'n_tail' in node and 'fallback' in node:
                rows.append({'block':block.name,'replication':rep,'kind':'POT','path':path,
                    **{k:node.get(k) for k in ['threshold_level','n_fit','n_tail','fallback','reason','shape','scale','irregular_shape','support_margin']}})
            if 'selected_penalty' in node:
                rows.append({'block':block.name,'replication':rep,'kind':'L1','path':path,
                    'selected_p':node['selected_p'],'selected_penalty':node['selected_penalty'],'inner_split':node['inner_split']})
            if 'gap' in node and 'dual_violation' in node:
                rows.append({'block':block.name,'replication':rep,'kind':'LP','path':path,
                    'gap':node['gap'],'dual_violation':node['dual_violation']})
            for key,value in node.items():scan(value,path+'/'+str(key),rep)
        elif isinstance(node,list):
            for j,value in enumerate(node):scan(value,path+'/'+str(j),rep)
    for line in (block/'fits.jsonl').read_text().splitlines():
        entry=json.loads(line);scan(entry['parameters'],'',entry['replication'])
    return rows


def main():
    results=e.OUT/'results';results.mkdir(exist_ok=True)
    expected=list(run.tasks());assert len(expected)==640
    frames=[];receipts={};diagnostics=[];moments={};path=np.load(e.OUT/'paths.npz');bound=run.binding()
    for module,kind,phi,n,start,stop in expected:
        name=f'{module}_{kind}_{phi:g}_{n}_{start:03d}_{stop:03d}'
        block=e.OUT/'blocks'/name;complete=json.loads((block/'complete.json').read_text())
        assert complete['binding']==bound
        assert all(e.sha(block/p)==h for p,h in complete['outputs'].items())
        receipts[name]=e.sha(block/'complete.json')
        frame=pd.read_csv(block/'replications.csv');frames.append(frame)
        diagnostics.extend(fit_diagnostics(block))
        stored=np.load(block/'moments.npz')
        for key in stored.files:
            mk=f'{module}/{kind}/{phi:g}/{n}/{key}'
            moments[mk]=moments.get(mk,0)+stored[key]
    data=pd.concat(frames,ignore_index=True)
    assert len(data)==552000 and not data.duplicated(GROUP+['replication','method']).any()
    assert len(data[GROUP].drop_duplicates())==144
    assert data.groupby(GROUP+['method']).size().eq(500).all()
    data.to_parquet(results/'replications.parquet',index=False)
    summary=[];paired=[];countrows=[];momentrows=[]
    for keys,df in data.groupby(GROUP,sort=True):
        meta=dict(zip(GROUP,keys));table=df.pivot(index='replication',columns='method',values='expected_QS')
        module,kind,phi,n,alpha,truth=keys
        sigma=e.V0 if module=='ar' else path[f'test_sigma_{kind}']
        oracle=e.old.conditional_quantile(kind,sigma,alpha)
        for method,g in df.groupby('method',sort=True):
            item={**meta,'method':method,'replications':len(g),
                'mean_expected_QS':float(g.expected_QS.mean()),'mean_excess_QS':float(g.excess_QS.mean()),
                'expected_QS_MCSE':float(g.expected_QS.std(ddof=1)/np.sqrt(len(g))),
                'median_expected_QS':float(g.expected_QS.median()),
                'p90_expected_QS':float(g.expected_QS.quantile(.9)),
                'p99_expected_QS':float(g.expected_QS.quantile(.99)),
                'top_five_share_of_total_regret':float(g.excess_QS.nlargest(5).sum()/g.excess_QS.sum()) if g.excess_QS.sum()>0 else 0.,
                'mean_expected_violation':float(g.expected_violation.mean()),
                'mean_prediction_MSE':float(g.prediction_MSE.mean()),
                'max_absolute_prediction':float(g.max_absolute_prediction.max())}
            for ref in ('Raw','Shift-CP','Vol-ERM'):
                if ref not in table:continue
                diff=table[method]-table[ref];mean=float(diff.mean());se=float(diff.std(ddof=1)/np.sqrt(len(diff)))
                item[f'difference_vs_{ref}']=mean;item[f'MCSE_vs_{ref}']=se
                paired.append({**meta,'method':method,'reference':ref,'difference':mean,'MCSE':se,
                               'lower95':mean-1.96*se,'upper95':mean+1.96*se,
                               'replicate_win_fraction':float((diff<0).mean())})
            summary.append(item)
            key=f'{module}/{kind}/{phi:g}/{n}/{alpha:g}/{truth}/{method}'
            mean,second,mse=moments[key]/e.REPS
            variance=np.maximum(second-mean*mean,0.);bias2=(mean-oracle)**2
            assert np.max(np.abs(mse-variance-bias2))<1e-10
            assert abs(np.mean(mse)-g.prediction_MSE.mean())<1e-10
            momentrows.append({**meta,'method':method,'integrated_variance':float(np.mean(variance)),
                               'integrated_squared_bias':float(np.mean(bias2)),'integrated_MSE':float(np.mean(mse))})
        if truth=='none':
            counts=df[df.method=='Raw'].sort_values('replication').oracle_tail_count.to_numpy()
            theorem=e.count_theory(alpha,phi,n) if module=='ar' else e.count_theory(alpha,0,n)
            centered2=(counts-n*alpha)**2
            countrows.append({**meta,**theorem,'mean_count':float(counts.mean()),
                'empirical_centered_second_moment':float(centered2.mean()),
                'second_moment_MCSE':float(centered2.std(ddof=1)/np.sqrt(e.REPS)),
                'empirical_count_variance':float(counts.var(ddof=1)),
                'zero_count_fraction':float((counts==0).mean())})
    pd.DataFrame(summary).to_csv(results/'summary.csv',index=False)
    pd.DataFrame(paired).to_csv(results/'paired.csv',index=False)
    pd.DataFrame(momentrows).to_csv(results/'decomposition.csv',index=False)
    pd.DataFrame(countrows).to_csv(results/'counts.csv',index=False)
    diag=pd.DataFrame(diagnostics);diag.to_parquet(results/'diagnostics.parquet',index=False)
    # Same expected risk should reproduce the seven original GARCH references.
    comparisons=[]
    for kind in ('normal','t5'):
        for truth in ('constant','state'):
            for n in e.SIZES:
                for alpha in e.ALPHAS:
                    old=e.old.OUT/f'{kind}_{truth}_{n}_{alpha:g}'/'replications.csv'
                    prior=pd.read_csv(old)
                    current=data[(data.module=='garch')&(data.innovation==kind)&(data.truth==truth)&(data.n_cal==n)&(data.alpha==alpha)]
                    joined=prior.merge(current,on=['replication','method'],suffixes=('_old','_new'),validate='one_to_one')
                    assert len(joined)==500*7
                    for method,g in joined.groupby('method'):
                        maximum=float(np.max(np.abs(g.expected_QS_old-g.expected_QS_new)))
                        comparisons.append({'innovation':kind,'truth':truth,'n_cal':n,'alpha':alpha,'method':method,
                                            'max_absolute_QS_difference':maximum,'source_sha256':e.sha(old)})
    pd.DataFrame(comparisons).to_csv(results/'prior_reference_comparison.csv',index=False)
    # Equivariance is exact mathematically, with numerical tolerance for LP representations.
    assert max(r['max_absolute_QS_difference'] for r in comparisons)<1e-8
    assert diag[diag.kind=='LP'].gap.max()<1e-6
    before=json.loads((e.OUT/'before.json').read_text())
    assert all(e.sha(e.PROJECT/p)==h for p,h in before['canonical'].items())
    files=sorted(p for p in results.iterdir() if p.is_file() and p.name!='complete.json')
    (results/'complete.json').write_text(json.dumps({'blocks':receipts,'rows':len(data),'configurations':144,
        'replications_per_configuration':500,'independent_histories':1500,
        'outputs':{p.name:e.sha(p) for p in files},'producer_sha256':e.sha(__file__),
        'canonical_unchanged':len(before['canonical'])},indent=2)+'\n')
    print(json.dumps({'rows':len(data),'configurations':144,'summary_rows':len(summary),
        'max_prior_QS_difference':max(r['max_absolute_QS_difference'] for r in comparisons),
        'max_LP_gap':float(diag[diag.kind=='LP'].gap.max())}),flush=True)


if __name__=='__main__':main()
