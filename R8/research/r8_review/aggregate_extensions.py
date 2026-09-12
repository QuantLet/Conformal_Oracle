"""Complete-panel aggregation and calendar-block uncertainty for the repair."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from controlled_comparisons import PROJECT,ROOT,OUT,MODELS,loss,sha

RESULTS=PROJECT/'artifacts/review_20260909/results'
FAMILY=['Raw','Shift-CP','Shift-ERM','Vol-CP','Vol-ERM','State2-ERM','State4-ERM']
POLICIES=['Gate-static','Rolling125','Rolling250','Rolling500','Gate-rolling125',
          'Gate-rolling250','Gate-rolling500','Selected-rolling','Gate-selected-rolling']
NAMES=FAMILY+POLICIES


def verify(folder):
    record=json.loads((folder/'complete.json').read_text())
    for name,digest in record['outputs'].items():assert sha(folder/name)==digest,(folder,name)
    return record


def bootstrap(frames):
    first=min(frame.index[0] for frame in frames);last=max(frame.index[-1] for frame in frames)
    dates=pd.date_range(first,last,freq='D');P=len(frames);T=len(dates);M=len(NAMES)
    available=np.zeros((T,P));values=np.zeros((T,P,M))
    for j,frame in enumerate(frames):
        assert frame.index.is_unique and frame.index.is_monotonic_increasing
        index=dates.get_indexer(frame.index);assert (index>=0).all()
        available[index,j]=1
        for k,name in enumerate(NAMES):values[index,j,k]=loss(frame.r.to_numpy(),frame[name].to_numpy(),.01)
    point=(values.sum(axis=0)/available.sum(axis=0)[:,None]).mean(axis=0)
    records=[];rng=np.random.default_rng(20260909)
    comparisons=[(name,'Shift-CP') for name in NAMES if name!='Shift-CP']
    comparisons += [('Gate-selected-rolling','Selected-rolling'),('Gate-rolling250','Rolling250'),
                    ('Gate-rolling500','Rolling500'),('Selected-rolling','Rolling250'),
                    ('Vol-CP','Vol-ERM'),('State2-ERM','Shift-ERM'),('State4-ERM','State2-ERM')]
    for L in [20,60]:
        replicates=[]
        for begin in range(0,999,25):
            batch=min(25,999-begin);counts=np.empty((batch,T))
            for b in range(batch):
                starts=rng.integers(0,T,size=int(np.ceil(T/L)))
                ix=((starts[:,None]+np.arange(L))%T).ravel()[:T]
                counts[b]=np.bincount(ix,minlength=T)
            denominator=counts@available;assert (denominator>0).all()
            numerator=(counts@values.reshape(T,-1)).reshape(batch,P,M)
            replicates.append((numerator/denominator[:,:,None]).mean(axis=1))
        draws=np.concatenate(replicates)
        # One simultaneous family: the six fixed full-calibration competitors
        # against Shift-CP. Other policy intervals remain pointwise.
        family_indices=[NAMES.index(name) for name in FAMILY if name!='Shift-CP']
        reference=NAMES.index('Shift-CP')
        delta_family=draws[:,family_indices]-draws[:,reference,None]
        center_family=point[family_indices]-point[reference]
        sd=delta_family.std(axis=0,ddof=1)
        assert (sd>0).all()
        critical=float(np.quantile(np.max(np.abs((delta_family-center_family)/sd),axis=1),.95))
        for name,reference_name in comparisons:
            i=NAMES.index(name);j=NAMES.index(reference_name);delta=draws[:,i]-draws[:,j]
            row={'comparator':name,'reference':reference_name,'block_calendar_days':L,'B':999,
                 'difference':float((point[i]-point[j])*1e4),
                 'lower':float(np.quantile(delta,.025)*1e4),'upper':float(np.quantile(delta,.975)*1e4)}
            if reference_name=='Shift-CP' and name in FAMILY:
                pos=family_indices.index(i);half=critical*sd[pos]*1e4
                row.update(simultaneous_lower=row['difference']-half,simultaneous_upper=row['difference']+half,
                           simultaneous_family_size=6,simultaneous_critical=critical)
            records.append(row)
        np.savez_compressed(RESULTS/f'bootstrap_L{L}.npz',methods=np.array(NAMES),means=draws,point=point)
        print('Block intervals',L,'days complete',flush=True)
    pd.DataFrame(records).to_csv(RESULTS/'paired_intervals.csv',index=False)


def main():
    ready=json.loads((PROJECT/'artifacts/review_20260909/calendar/reduction.json').read_text())
    assert ready['complete'] and ready['n_assets']==24,'Calendar panel incomplete'
    assets=sorted(p.stem for p in (ROOT/'data/returns').glob('*.csv'))
    sweeps=[];states=[];policies=[];full=[];frames=[];bindings={};decisions=[]
    for model in MODELS:
        for asset in assets:
            folder=OUT/f'{model}__{asset}'
            one=verify(folder);two=verify(folder/'full')
            expected_code=Path(__file__).with_name('controlled_comparisons.py')
            assert one['binding']['producer_sha256']==sha(expected_code)
            assert two['binding']['candidate_sha256']==sha(expected_code)
            bindings[f'{model}__{asset}']={'sweep_complete_sha256':sha(folder/'complete.json'),
                                           'full_complete_sha256':sha(folder/'full/complete.json')}
            sweeps.append(pd.read_csv(folder/'controlled.csv'))
            states.append(pd.read_csv(folder/'controlled_states.csv'))
            policy=pd.read_csv(folder/'policies.csv');policies.append(policy)
            full.append(pd.read_csv(folder/'full/metrics.csv'))
            path=pd.read_parquet(folder/'full/daily.parquet')
            frame=pd.DataFrame({'r':path.r},index=path.index)
            for name in FAMILY:frame[name]=path[f'.01/{name}'] if f'.01/{name}' in path else path[f'0.01/{name}']
            policy_path=pd.read_parquet(folder/'policy_daily.parquet')
            policy_path=policy_path[policy_path.origin=='Original']
            assert frame.index.equals(policy_path.index) and np.array_equal(frame.r,policy_path.r)
            assert np.allclose(frame['Shift-CP'],policy_path.Static,rtol=0,atol=1e-14)
            for name in POLICIES:frame[name]=policy_path[name]
            frames.append(frame)
            fit=json.loads((folder/'fits.json').read_text())
            for origin,choice in fit['policies'].items():
                decisions.append({'model':model,'asset':asset,'origin':origin,'window':choice['selected_window'],
                                  'gate':choice['gate'],'validation_start':choice['validation_start'],
                                  'calibration_stop':choice['train_stop_exclusive']})
    assert len(frames)==216
    RESULTS.mkdir(parents=True,exist_ok=True)
    for name,data in [('sweep',sweeps),('sweep_states',states),('policies',policies),('full',full)]:
        frame=pd.concat(data,ignore_index=True);frame.to_csv(RESULTS/f'{name}_pairs.csv',index=False)
        keys={'sweep':['alpha','n_cal','method'],'sweep_states':['n_cal','method','state'],
              'policies':['origin','method','state'],'full':['alpha','method','state']}[name]
        if name=='sweep':frame=frame[frame.method!='Full-static']
        summary=frame.groupby(keys).agg(pairs=('asset','size'),QS=('QS','mean'),violation_rate=('pihat','mean'),
            mean_absolute_threshold=('width','mean'),test_observations=('n_test','sum'),violations=('viol','sum'),
            kupiec_rejections=('p_kup',lambda x:int((x<.05).sum())),green=('TL',lambda x:int((x=='Green').sum())))
        summary.to_csv(RESULTS/f'{name}_summary.csv')
    pd.DataFrame(decisions).to_csv(RESULTS/'decisions.csv',index=False)
    bootstrap(frames)
    outputs={p.name:sha(p) for p in RESULTS.iterdir() if p.is_file() and p.name!='complete.json'}
    (RESULTS/'complete.json').write_text(json.dumps({'producer_sha256':sha(__file__),'pairs':216,
        'inputs':bindings,'outputs':outputs,'calendar_reduction_sha256':sha(PROJECT/'artifacts/review_20260909/calendar/reduction.json')},indent=2)+'\n')
    print('Complete 216-pair extension aggregated',flush=True)


if __name__=='__main__':main()
