"""Paired calendar-block inference and full diagnostics for declared extension."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='2'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run import PROJECT,ROOT,OUT,MODELS,sha,load
import methods as m

METHODS=['Raw','Shift-CP','Shift-ERM','Vol-CP','Vol-ERM','State2-ERM','State4-ERM',
         'State-L1','State-L1-clipped','POT-Shift','POT-Vol','ACI-existing',
         'DtACI-projected-seed','DtACI-projected-expected','Rolling250','Rolling500',
         'Selected-rolling','Gate-selected-rolling','Loss-gate','Past-minimum']
FAMILY=['State-L1','POT-Shift','POT-Vol','DtACI-projected-expected','Loss-gate','Past-minimum']
RESULTS=OUT/'results'


def load_all():
    frames=[];rows=[];decisions=[];bindings={};seedrows=[];diagnostics=[]
    assets=sorted(p.stem for p in (ROOT/'data/returns').glob('*.csv'))
    for model in MODELS:
        for asset in assets:
            key=f'{model}__{asset}';folder=OUT/'pairs'/key
            done=json.loads((folder/'complete.json').read_text())
            assert all(sha(folder/k)==h for k,h in done['outputs'].items())
            assert done['binding']==load(model,asset)[-1]
            frame=pd.read_parquet(folder/'daily.parquet')
            metric=pd.read_csv(folder/'metrics.csv');fit=json.loads((folder/'fits.json').read_text())
            metric['normalised_QS']=metric.QS/metric.calibration_scale
            seed=pd.read_csv(folder/'dtaci_seed_metrics.csv');seed['model']=model;seed['asset']=asset
            seedrows.append(seed);rows.append(metric);frames.append(frame)
            gate=fit['gate']
            decisions.append({'model':model,'asset':asset,'gate_selected':gate['selected'],
                'past_minimum_selected':gate['past_minimum_selected'],
                'state_p':fit['state']['selected_p'],'state_penalty':fit['state']['selected_penalty'],
                'pot_shift_threshold':fit['POT-Shift']['selected_threshold'],
                'pot_vol_threshold':fit['POT-Vol']['selected_threshold']})
            statecert=[trial['fit']['certificate'] for trial in fit['state']['trials']]+[fit['state']['fit']['certificate']]
            pots=[fit[name]['fit'] for name in ['POT-Shift','POT-Vol']]
            allpots=pots+[trial['fit'] for name in ['POT-Shift','POT-Vol'] for trial in fit[name]['trials']]
            diagnostics.append({'model':model,'asset':asset,'lp_max_gap':max(c['gap'] for c in statecert),
                'lp_max_dual_violation':max(c['dual_violation'] for c in statecert),
                'pot_final_fallbacks':sum(p['fallback'] for p in pots),
                'pot_all_fallbacks':sum(p['fallback'] for p in allpots),
                'pot_final_irregular_shapes':sum(p.get('irregular_shape',False) for p in pots),
                'dtaci_projection_events':sum(fit['dtaci_projected']['projection_counts']),
                **fit['dtaci_unprojected_audit']})
            bindings[key]=sha(folder/'complete.json')
    assert len(frames)==216
    return frames,pd.concat(rows,ignore_index=True),pd.DataFrame(decisions),pd.DataFrame(diagnostics),pd.concat(seedrows,ignore_index=True),bindings


def bootstrap(frames,scales,without_crypto):
    dates=pd.date_range(min(f.index[0] for f in frames),max(f.index[-1] for f in frames),freq='D')
    T,P,M=len(dates),len(frames),len(METHODS)
    valid=np.zeros((T,P));values=np.zeros((T,P,M))
    for j,f in enumerate(frames):
        ix=dates.get_indexer(f.index);assert (ix>=0).all();valid[ix,j]=1
        for k,name in enumerate(METHODS):
            values[ix,j,k]=f['DtACI-expected/loss'] if name=='DtACI-projected-expected' else m.loss(f.r,f[name])
    point=(values.sum(axis=0)/valid.sum(axis=0)[:,None]).mean(axis=0)
    rows=[];sensitivity_rows=[];family_idx=[METHODS.index(n) for n in FAMILY];ref=METHODS.index('Shift-CP')
    per_pair=values.sum(axis=0)/valid.sum(axis=0)[:,None]
    sensitivity_points={'normalised':(per_pair/scales[:,None]).mean(axis=0),
                        'without_crypto':per_pair[without_crypto].mean(axis=0)}
    for block in (20,60):
        rng=np.random.default_rng(m.seed_for('panel-calendar',block));output=[]
        sensitivity_draws={key:[] for key in sensitivity_points}
        for start in range(0,999,25):
            B=min(25,999-start);counts=np.empty((B,T))
            for b in range(B):
                begins=rng.integers(0,T,size=int(np.ceil(T/block)))
                idx=((begins[:,None]+np.arange(block))%T).ravel()[:T]
                counts[b]=np.bincount(idx,minlength=T)
            denom=counts@valid;assert (denom>0).all()
            numerator=(counts@values.reshape(T,-1)).reshape(B,P,M)
            pair_means=numerator/denom[:,:,None]
            output.append(pair_means.mean(axis=1))
            sensitivity_draws['normalised'].append((pair_means/scales[None,:,None]).mean(axis=1))
            sensitivity_draws['without_crypto'].append(pair_means[:,without_crypto].mean(axis=1))
        draws=np.concatenate(output)
        dc=draws[:,family_idx]-draws[:,ref,None];center=point[family_idx]-point[ref]
        sd=dc.std(axis=0,ddof=1);assert (sd>0).all()
        crit=float(np.quantile(np.max(np.abs((dc-center)/sd),axis=1),.95))
        comparisons=[(name,'Shift-CP') for name in METHODS if name!='Shift-CP']
        comparisons += [(name,refname) for name in ['Loss-gate','Past-minimum','State-L1','POT-Vol','POT-Shift','DtACI-projected-expected'] for refname in ['Raw','Vol-ERM','Rolling500','Selected-rolling']]
        comparisons += [('State-L1-clipped','State-L1'),('DtACI-projected-seed','DtACI-projected-expected')]
        for name,reference in comparisons:
            i=METHODS.index(name);j=METHODS.index(reference);delta=draws[:,i]-draws[:,j]
            record={'method':name,'reference':reference,'block_calendar_days':block,'draws':999,
                    'difference':float((point[i]-point[j])*1e4),
                    'lower':float(np.quantile(delta,.025)*1e4),'upper':float(np.quantile(delta,.975)*1e4)}
            if reference=='Shift-CP' and name in FAMILY:
                half=crit*sd[FAMILY.index(name)]*1e4
                record.update(simultaneous_lower=record['difference']-half,simultaneous_upper=record['difference']+half,
                              simultaneous_family_size=len(FAMILY))
            rows.append(record)
        for label,chunks in sensitivity_draws.items():
            simulations=np.concatenate(chunks);pt=sensitivity_points[label]
            sf=simulations[:,family_idx]-simulations[:,ref,None];pc=pt[family_idx]-pt[ref]
            ss=sf.std(axis=0,ddof=1)
            critical=float(np.quantile(np.max(np.abs((sf-pc)/ss),axis=1),.95))
            factor=1e4 if label=='without_crypto' else 1.
            for pos,name in enumerate(FAMILY):
                d=sf[:,pos];point_delta=pc[pos]*factor;half=critical*ss[pos]*factor
                sensitivity_rows.append({'sensitivity':label,'method':name,'reference':'Shift-CP',
                    'block_calendar_days':block,'difference':point_delta,'lower':float(np.quantile(d,.025)*factor),
                    'upper':float(np.quantile(d,.975)*factor),'simultaneous_lower':point_delta-half,
                    'simultaneous_upper':point_delta+half})
        np.savez_compressed(RESULTS/f'bootstrap_{block}.npz',draws=draws,point=point,methods=np.array(METHODS))
        print('Calendar bootstrap complete',block,flush=True)
    pd.DataFrame(sensitivity_rows).to_csv(RESULTS/'sensitivity_intervals.csv',index=False)
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(exist_ok=True)
    frames,pairs,decisions,diagnostics,seeds,bindings=load_all()
    pairs.to_csv(RESULTS/'pairs.csv',index=False)
    raw=pairs[pairs.method=='Raw'][['model','asset','QS']].rename(columns={'QS':'raw_QS'})
    pairs=pairs.merge(raw,on=['model','asset'],validate='many_to_one')
    summary=pairs.groupby('method').agg(pairs=('asset','size'),QS=('QS','mean'),normalised_QS=('normalised_QS','mean'),
        violation_rate=('pihat','mean'),width=('width','mean'),kupiec_rejections=('p_kup',lambda x:int((x<.05).sum())),
        kupiec_available=('p_kup','count'),test_observations=('n_test','sum'))
    summary['worse_than_raw']=pairs.assign(worse=pairs.QS>pairs.raw_QS).groupby('method').worse.sum()
    summary['QS_x10000']=summary.QS*1e4
    summary.to_csv(RESULTS/'summary.csv')
    decisions.to_csv(RESULTS/'decisions.csv',index=False);diagnostics.to_csv(RESULTS/'diagnostics.csv',index=False)
    seeds.to_csv(RESULTS/'seed_metrics.csv',index=False)
    exclusions=[]
    groups={'commodities':['DJCI','GOLD','NATGAS','WTI'],'crypto':['BTC','ETH'],
            'bonds':['TLT','IBGL','CBU0'],'fx':['AUDUSD','EURUSD','GBPUSD','USDJPY']}
    for label,assets in groups.items():
        sub=pairs[~pairs.asset.isin(assets)]
        for name,g in sub.groupby('method'):
            exclusions.append({'excluded':label,'method':name,'pairs':len(g),'QS':g.QS.mean(),'normalised_QS':g.normalised_QS.mean()})
    for model in MODELS:
        sub=pairs[pairs.model!=model]
        for name,g in sub.groupby('method'):
            exclusions.append({'excluded':model,'method':name,'pairs':len(g),'QS':g.QS.mean(),'normalised_QS':g.normalised_QS.mean()})
    pd.DataFrame(exclusions).to_csv(RESULTS/'exclusions.csv',index=False)
    pairs.groupby(['model','method']).agg(QS=('QS','mean'),violation_rate=('pihat','mean')).to_csv(RESULTS/'by_model.csv')
    # Identical pair order to load_all; no selection of a preferred sensitivity.
    identities=list(bindings)
    scales=np.array([pairs[(pairs.model==key.split('__')[0])&(pairs.asset==key.split('__')[1])].calibration_scale.iloc[0] for key in identities])
    without_crypto=np.array([key.split('__')[1] not in ['BTC','ETH'] for key in identities])
    intervals=bootstrap(frames,scales,without_crypto);intervals.to_csv(RESULTS/'intervals.csv',index=False)
    before=json.loads((OUT/'before.json').read_text())
    changed=[f for f,h in before['canonical'].items() if sha(PROJECT/f)!=h];assert not changed,changed
    (RESULTS/'complete.json').write_text(json.dumps({'pairs':216,'protocol_sha256':before['protocol_sha256'],
        'bindings':bindings,'producer_sha256':sha(__file__),'canonical_files_unchanged':len(before['canonical']),
        'outputs':{p.name:sha(p) for p in RESULTS.iterdir() if p.is_file() and p.name!='complete.json'}},indent=2)+'\n')
    print(summary[['pairs','QS_x10000','violation_rate','worse_than_raw']].to_string(),flush=True)


if __name__=='__main__':main()
