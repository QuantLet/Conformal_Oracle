"""Rebuild each external pair loss and all calendar draws without the aggregator."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import hashlib
import json
import numpy as np
import pandas as pd
from corrections import OUT,MODELS,ASSETS,METHODS,sha


def close(x,y):np.testing.assert_allclose(x,y,rtol=3e-10,atol=2e-13)


def main():
    results=OUT/'results';done=json.loads((results/'complete.json').read_text())
    assert done['producer_sha256']==sha(__import__('pathlib').Path(__file__).with_name('aggregate.py'))
    for p,h in done['outputs'].items():assert sha(results/p)==h,p
    primary=json.loads((results/'primary_decisions.json').read_text());families=[('correction',['State-L1','POT-Shift','POT-Vol','DtACI-projected-expected'],'Shift-CP'),('indication',['Gate-selected-rolling','Loss-gate','Past-minimum'],'Selected-rolling')]
    frames=[];losses=[];scales=[];metric_rows=[]
    for model in MODELS:
        for asset in ASSETS:
            key=f'{model}__{asset}';folder=OUT/'pairs'/key;assert sha(folder/'complete.json')==done['pair_receipts'][key]
            f=pd.read_parquet(folder/'daily.parquet');mt=pd.read_csv(folder/'metrics.csv').set_index('method');scales.append(mt.calibration_scale.iloc[0]);frames.append(f)
            one=[]
            for name in METHODS:
                e=f.r.to_numpy()-f[name].to_numpy() if name!='DtACI-projected-expected' else None
                v=f['DtACI-expected/loss'].to_numpy() if e is None else np.maximum(.01*e,-.99*e)
                close(v.mean(),mt.loc[name,'QS']);one.append(v)
                metric_rows.append(dict(model=model,asset=asset,method=name,QS=v.mean(),normalised_QS=v.mean()/scales[-1],
                    pi=mt.loc[name,'pihat'],width=mt.loc[name,'width'],kup=mt.loc[name,'p_kup']))
            losses.append(np.column_stack(one))
    assert all(f.index.equals(frames[0].index) for f in frames)
    values=np.stack(losses,axis=1);scales=np.array(scales);index=frames[0].index
    table=pd.DataFrame(metric_rows);summary=pd.read_csv(results/'summary.csv').set_index('method');raw=table[table.method=='Raw'].QS.to_numpy()
    for name,g in table.groupby('method',sort=False):
        s=summary.loc[name];assert s.pairs==48
        for key,col in [('QS','QS'),('normalised_QS','normalised_QS'),('violation_rate','pi'),('width','width')]:close(s[key],g[col].mean())
        assert s.kupiec_rejections==(g.kup<.05).sum() and s.kupiec_available==g.kup.count()
        assert s.worse_than_raw==(g.QS.to_numpy()>raw).sum()
    period=pd.read_csv(results/'period_pairs.csv').set_index(['model','asset','period','method'])
    for label,lo,hi in [('2015-2019',2015,2019),('2020-2021',2020,2021),('2022-July2026',2022,2026)]:
        mask=(index.year>=lo)&(index.year<=hi)
        for i,(model,asset) in enumerate((m,a) for m in MODELS for a in ASSETS):
            for j,name in enumerate(METHODS):
                r=period.loc[(model,asset,label,name)];assert r.n==mask.sum();close(r.QS,values[mask,i,j].mean());close(r.normalised_QS,r.QS/scales[i])
    calendar=pd.date_range(index[0],index[-1]);positions=calendar.get_indexer(index);T=len(calendar);actual=values.mean(axis=0)
    points=np.stack([actual.mean(0),(actual/scales[:,None]).mean(0)]);intervals=pd.read_csv(results/'intervals.csv')
    checks=0
    for block in [20,60]:
        seed=int.from_bytes(hashlib.sha256(f'20260909/panel-calendar/{block}'.encode()).digest()[:4],'little');rng=np.random.default_rng(seed)
        recreated=[]
        for start in range(0,999,25):
            counts=[]
            for _ in range(min(25,999-start)):
                origins=rng.integers(0,T,size=int(np.ceil(T/block)))
                sampled=np.concatenate([(origin+np.arange(block))%T for origin in origins])[:T]
                counts.append(np.bincount(sampled,minlength=T)[positions])
            counts=np.array(counts);means=(counts@values.reshape(len(index),-1)).reshape(len(counts),48,len(METHODS))/counts.sum(1)[:,None,None]
            recreated.append(np.stack([means.mean(1),(means/scales[None,:,None]).mean(1)],axis=1))
        replay=np.concatenate(recreated)
        with np.load(results/f'bootstrap_{block}.npz') as old:close(old['draws'],replay);close(old['point'],points);assert old['methods'].tolist()==METHODS
        checks+=replay.size
        for unit,label in enumerate(['return','normalised']):
            factor=1e4 if unit==0 else 1.
            for family,names,reference in families:
                ix=[METHODS.index(n) for n in names];ref=METHODS.index(reference)
                delta=replay[:,unit,ix]-replay[:,unit,ref,None];center=points[unit,ix]-points[unit,ref];sd=delta.std(0,ddof=1)
                c=np.quantile(np.abs((delta-center)/sd).max(1),.95)
                for j,name in enumerate(names):
                    rows=intervals[(intervals.units==label)&(intervals.family==family)&(intervals.method==name)&(intervals.block_calendar_days==block)];assert len(rows)==1
                    row=rows.iloc[0];assert row.family_size==len(names) and row.draws==999 and row.reference==reference
                    expected=dict(difference=center[j]*factor,lower=np.quantile(delta[:,j],.025)*factor,upper=np.quantile(delta[:,j],.975)*factor,
                                  simultaneous_lower=(center[j]-c*sd[j])*factor,simultaneous_upper=(center[j]+c*sd[j])*factor)
                    for key,v in expected.items():close(row[key],v)
    for r in primary:
        rows=intervals[(intervals.units=='normalised')&(intervals.family==r['family'])&(intervals.method==r['method'])]
        assert len(rows)==2 and r['passes_primary_transfer_criterion']==bool((rows.simultaneous_upper<0).all())
        assert r['higher_loss_both_bands']==bool((rows.simultaneous_lower>0).all())
    record=dict(status='passed',producer_sha256=sha(__file__),result_complete_sha256=sha(results/'complete.json'),
                pairs=48,methods=len(METHODS),daily_losses_rebuilt=int(values.size),independent_calendar_draws=1998,
                bootstrap_means_verified=checks,period_pair_method_rows=len(period),simultaneous_interval_rows=len(intervals))
    (OUT/'aggregation_validation.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)


if __name__=='__main__':main()
