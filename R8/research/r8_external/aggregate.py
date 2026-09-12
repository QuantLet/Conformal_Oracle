"""External primary endpoint and the two protocol-fixed simultaneous families."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import json
import numpy as np
import pandas as pd
from corrections import OUT,MODELS,ASSETS,METHODS,load,m,sha
RESULTS=OUT/'results'
FAMILIES={'correction':(['State-L1','POT-Shift','POT-Vol','DtACI-projected-expected'],'Shift-CP'),
          'indication':(['Gate-selected-rolling','Loss-gate','Past-minimum'],'Selected-rolling')}


def arrays():
    points=[];frames=[];scales=[];receipts={};metrics=[];decisions=[];periods=[]
    for model in MODELS:
        for asset in ASSETS:
            key=f'{model}__{asset}';folder=OUT/'pairs'/key
            done=json.loads((folder/'complete.json').read_text());assert done['binding']==load(model,asset)[-1]
            for p,h in done['outputs'].items():assert sha(folder/p)==h
            f=pd.read_parquet(folder/'daily.parquet');mt=pd.read_csv(folder/'metrics.csv');fit=json.loads((folder/'fits.json').read_text())
            scale=mt.calibration_scale.iloc[0];assert (mt.calibration_scale==scale).all()
            mt['normalised_QS']=mt.QS/scale;metrics.append(mt)
            daily=np.column_stack([f['DtACI-expected/loss'] if name=='DtACI-projected-expected' else m.loss(f.r,f[name]) for name in METHODS])
            points.append(daily);frames.append(f);scales.append(scale);receipts[key]=sha(folder/'complete.json')
            decisions.append(dict(model=model,asset=asset,loss_gate=fit['gate']['selected'],past_minimum=fit['gate']['past_minimum_selected'],
                                  selected_window=fit['window']['selected_window'],coverage_gate=fit['window']['gate'],
                                  state_p=fit['state']['selected_p'],state_penalty=fit['state']['selected_penalty'],
                                  pot_shift_threshold=fit['POT-Shift']['selected_threshold'],pot_vol_threshold=fit['POT-Vol']['selected_threshold'],
                                  pot_final_fallbacks=sum(fit[n]['fit']['fallback'] for n in ['POT-Shift','POT-Vol'])))
            for label,low,high in [('2015-2019',2015,2019),('2020-2021',2020,2021),('2022-July2026',2022,2026)]:
                ix=(f.index.year>=low)&(f.index.year<=high)
                for j,name in enumerate(METHODS):periods.append(dict(model=model,asset=asset,period=label,method=name,n=int(ix.sum()),QS=float(daily[ix,j].mean()),normalised_QS=float(daily[ix,j].mean()/scale)))
    assert len(frames)==48 and all(f.index.equals(frames[0].index) for f in frames)
    return np.stack(points,axis=1),np.array(scales),frames[0].index,pd.concat(metrics,ignore_index=True),pd.DataFrame(decisions),pd.DataFrame(periods),receipts


def main():
    for name in ['base_validation.json','correction_validation.json']:assert json.loads((OUT/name).read_text())['status']=='passed'
    RESULTS.mkdir(exist_ok=True)
    losses,scales,index,metrics,decisions,periods,receipts=arrays()
    raw=metrics[metrics.method=='Raw'][['model','asset','QS']].rename(columns={'QS':'raw_QS'})
    metrics=metrics.merge(raw,on=['model','asset'],validate='many_to_one');metrics['worse']=metrics.QS>metrics.raw_QS
    summary=metrics.groupby('method',sort=False).agg(pairs=('asset','size'),QS=('QS','mean'),normalised_QS=('normalised_QS','mean'),
        violation_rate=('pihat','mean'),width=('width','mean'),kupiec_rejections=('p_kup',lambda x:int((x<.05).sum())),
        kupiec_available=('p_kup','count'),worse_than_raw=('worse','sum'))
    summary['QS_x10000']=summary.QS*1e4;summary.to_csv(RESULTS/'summary.csv')
    metrics.to_csv(RESULTS/'pairs.csv',index=False);decisions.to_csv(RESULTS/'decisions.csv',index=False)
    periods.to_csv(RESULTS/'period_pairs.csv',index=False)
    periods.groupby(['period','method']).agg(pairs=('asset','size'),QS=('QS','mean'),normalised_QS=('normalised_QS','mean')).to_csv(RESULTS/'period_summary.csv')
    metrics.groupby(['model','method']).agg(pairs=('asset','size'),QS=('QS','mean'),normalised_QS=('normalised_QS','mean'),violation_rate=('pihat','mean')).to_csv(RESULTS/'by_model.csv')
    calendar=pd.date_range(index[0],index[-1],freq='D');positions=calendar.get_indexer(index);T=len(calendar)
    valid=np.zeros(T);valid[positions]=1
    # Identical support for all pairs makes daily averaging commute with
    # pair-equal calendar bootstrap means. Retain both units of the loss.
    daily=np.zeros((T,2,len(METHODS)));daily[positions,0]=losses.mean(axis=1)
    daily[positions,1]=(losses/scales[None,:,None]).mean(axis=1)
    point=np.stack([losses.mean(axis=(0,1)),(losses/scales[None,:,None]).mean(axis=(0,1))])
    rows=[]
    for block in [20,60]:
        rng=np.random.default_rng(m.seed_for('panel-calendar',block));draws=[]
        for _ in range(999):
            begins=rng.integers(0,T,size=int(np.ceil(T/block)))
            ix=((begins[:,None]+np.arange(block))%T).ravel()[:T]
            counts=np.bincount(ix,minlength=T);den=counts@valid;assert den>0
            draws.append((counts@daily.reshape(T,-1)).reshape(2,len(METHODS))/den)
        draws=np.stack(draws);np.savez_compressed(RESULTS/f'bootstrap_{block}.npz',draws=draws,point=point,methods=np.array(METHODS))
        for unit,tag in enumerate(['return','normalised']):
            factor=1e4 if tag=='return' else 1.
            for family,(names,reference) in FAMILIES.items():
                ids=[METHODS.index(n) for n in names];ref=METHODS.index(reference)
                dc=draws[:,unit,ids]-draws[:,unit,ref,None];center=point[unit,ids]-point[unit,ref]
                sd=dc.std(axis=0,ddof=1);assert (sd>0).all()
                critical=float(np.quantile(np.abs((dc-center)/sd).max(axis=1),.95))
                for j,name in enumerate(names):
                    diff=center[j]*factor;half=critical*sd[j]*factor
                    rows.append(dict(units=tag,family=family,method=name,reference=reference,block_calendar_days=block,draws=999,
                        family_size=len(names),difference=diff,lower=np.quantile(dc[:,j],.025)*factor,upper=np.quantile(dc[:,j],.975)*factor,
                        simultaneous_lower=diff-half,simultaneous_upper=diff+half))
        print('External calendar bootstrap',block,'complete',flush=True)
    intervals=pd.DataFrame(rows);intervals.to_csv(RESULTS/'intervals.csv',index=False)
    decisions=[]
    for (family,name),g in intervals[intervals.units=='normalised'].groupby(['family','method']):
        assert set(g.block_calendar_days)=={20,60}
        decisions.append(dict(family=family,method=name,reference=g.reference.iloc[0],
            passes_primary_transfer_criterion=bool((g.simultaneous_upper<0).all()),
            higher_loss_both_bands=bool((g.simultaneous_lower>0).all())))
    (RESULTS/'primary_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    (RESULTS/'complete.json').write_text(json.dumps(dict(status='complete',producer_sha256=sha(__file__),pairs=48,methods=METHODS,
        pair_receipts=receipts,base_validation_sha256=sha(OUT/'base_validation.json'),correction_validation_sha256=sha(OUT/'correction_validation.json'),
        outputs={p.name:sha(p) for p in RESULTS.iterdir() if p.is_file() and p.name!='complete.json'}),indent=2)+'\n')
    print('All external results saved; independent aggregate validation remains.',flush=True)


if __name__=='__main__':main()
