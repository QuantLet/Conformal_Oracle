"""Independent losses, ranks, selections and mixtures, plus all 48 fresh replays."""
import hashlib
import json
import math
import sys
import numpy as np
import pandas as pd
from scipy.stats import genpareto
from corrections import PROJECT,OUT,MODELS,ASSETS,METHODS,load,compute,sha
sys.path.insert(0,str(PROJECT/'research/r8_model_extension'))
from validate_ten import metric,check_metric,check_close


def loss(y,q):return np.maximum(.01*(y-q),-.99*(y-q))


def main():
    scalars=values=draw_count=lp_count=pot_count=0;receipts={}
    for model in MODELS:
        for asset in ASSETS:
            key=f'{model}__{asset}';base=OUT/'pairs'/key;replay=OUT/'correction_replay'/key
            y,q,sigma,index,nc,binding=load(model,asset)
            for folder in [base,replay]:
                done=json.loads((folder/'complete.json').read_text());assert done['binding']==binding
                for p,h in done['outputs'].items():assert sha(folder/p)==h
            for name in ['metrics.csv','fits.json','dtaci_seed_metrics.csv']:assert (base/name).read_bytes()==(replay/name).read_bytes()
            f=pd.read_parquet(base/'daily.parquet');pd.testing.assert_frame_equal(f,pd.read_parquet(replay/'daily.parquet'),check_exact=True)
            fit=json.loads((base/'fits.json').read_text());table=pd.read_csv(base/'metrics.csv').set_index('method');assert set(table.index)==set(METHODS)
            assert f.index.equals(index[nc:]);np.testing.assert_array_equal(f.r,y[nc:])
            for name in METHODS:
                if name=='DtACI-projected-expected':continue
                for col,value in metric(y[nc:],f[name].to_numpy()).items():
                    if col=='TL':assert table.loc[name,col]==value
                    else:check_metric(col,table.loc[name,col],value);scalars+=1
            with np.load(base/'dtaci_experts.npz') as a,np.load(replay/'dtaci_experts.npz') as b:
                assert a.files==b.files
                for name in a.files:np.testing.assert_array_equal(a[name],b[name]);values+=a[name].size
                p=a['projected_p'][nc:];experts=a['projected_q'][nc:]
                assert (p>0).all();check_close(p.sum(1),1.)
                for name,array in [('loss',(p*loss(y[nc:,None],experts)).sum(1)),('hits',(p*(y[nc:,None]<experts)).sum(1)),('width',(p*np.abs(experts)).sum(1)),('mean_threshold',(p*experts).sum(1))]:check_close(f['DtACI-expected/'+name],array)
                check_close(table.loc['DtACI-projected-expected','QS'],f['DtACI-expected/loss'].mean())
                assert table.loc['DtACI-projected-expected',['p_kup','p_ind','p_cc']].isna().all()
                for col,keymix in [('pihat','hits'),('width','width')]:check_close(table.loc['DtACI-projected-expected',col],f['DtACI-expected/'+keymix].mean())
                seedtable=pd.read_csv(base/'dtaci_seed_metrics.csv').set_index('replicate')
                for rep in range(8):
                    seed=int.from_bytes(hashlib.sha256(f'20260909/{key}/{rep}'.encode()).digest()[:4],'little')
                    u=np.random.default_rng(seed).random(len(y));choices=(u[:,None]>np.cumsum(np.nan_to_num(a['projected_p']),axis=1)).sum(1).clip(0,6)
                    pred=a['projected_q'][np.arange(nc,len(y)),choices[nc:]]
                    for col,value in metric(y[nc:],pred).items():
                        if col=='TL':assert seedtable.loc[rep,col]==value
                        else:check_metric(col,seedtable.loc[rep,col],value);scalars+=1
            s=q-y;rolls={}
            for w in [125,250,500]:
                z=np.full(len(y),np.nan);rank=math.ceil((w+1)*.99)-1
                for t in range(w,len(y)):z[t]=np.sort(s[t-w:t])[rank]
                rolls[w]=z
            np.testing.assert_array_equal(f['Rolling500'],q[nc:]-rolls[500][nc:])
            v=int(.7*nc);assert v==fit['inner_split']==fit['window']['validation_start']
            trials={w:loss(y[v:nc],q[v:nc]-z[v:nc]).mean() for w,z in rolls.items()}
            selected=min(trials,key=lambda w:(trials[w],w));assert selected==fit['window']['selected_window']
            for w,val in trials.items():check_close(fit['window']['validation_loss'][str(w)],val)
            cal=metric(y[:nc],q[:nc]);gate=bool(cal['p_kup']<.05 or cal['TL']!='Green');assert gate==fit['window']['gate']
            np.testing.assert_array_equal(f['Selected-rolling'],q[nc:]-rolls[selected][nc:])
            np.testing.assert_array_equal(f['Gate-selected-rolling'],f['Selected-rolling'] if gate else q[nc:])
            for stop,label in [(v,'inner'),(nc,'full')]:
                shift=np.sort(s[:stop])[math.ceil((stop+1)*.99)-1]
                order=np.argsort(s[:stop]/sigma[:stop],kind='stable');c=np.cumsum(sigma[:stop][order])
                vol=(s[:stop]/sigma[:stop])[order[np.searchsorted(c,.99*c[-1],side='left')]]
                saved=fit['gate'] if label=='inner' else fit['full_static'];check_close(saved['shift'],shift);check_close(saved['vol_coefficient'],vol)
            np.testing.assert_array_equal(f['Shift-CP'],q[nc:]-fit['full_static']['shift'])
            np.testing.assert_array_equal(f['Vol-ERM'],q[nc:]-fit['full_static']['vol_coefficient']*sigma[nc:])
            g=fit['gate'];paths=[q,q-g['shift'],q-g['vol_coefficient']*sigma,q-rolls[500]]
            matrix=np.column_stack([loss(y[v:nc],a[v:nc])-loss(y[v:nc],q[v:nc]) for a in paths[1:]])
            means=matrix.mean(0);bounds=[]
            for block in [20,60]:
                seed=int.from_bytes(hashlib.sha256(f'20260909/{key}/{block}'.encode()).digest()[:4],'little')
                rng=np.random.default_rng(seed);n=len(matrix);sim=[]
                for _ in range(499):
                    starts=rng.integers(0,n,size=int(np.ceil(n/block)))
                    ix=np.concatenate([(s+np.arange(block))%n for s in starts])[:n]
                    sim.append(matrix[ix].mean(0));draw_count+=1
                sim=np.stack(sim);sd=sim.std(0,ddof=1);positive=sd>1e-15;z=np.zeros_like(sim);z[:,positive]=(sim[:,positive]-means[positive])/sd[positive]
                critical=max(0.,float(np.quantile(z.max(1),.95)));bounds.append(means+critical*sd)
            upper=np.maximum(*bounds);check_close(g['upper_bounds'],upper)
            names=['Raw','Inner-Shift','Inner-Vol','Rolling500'];selected=names[int(np.argmin(upper))+1] if upper.min()<0 else 'Raw'
            assert selected==g['selected'];np.testing.assert_array_equal(f['Loss-gate'],paths[names.index(selected)][nc:])
            past=min(range(4),key=lambda j:(loss(y[v:nc],paths[j][v:nc]).mean(),j));assert names[past]==g['past_minimum_selected']
            np.testing.assert_array_equal(f['Past-minimum'],paths[past][nc:])
            for trial in fit['state']['trials']+[{'fit':fit['state']['fit']}]:
                a=trial['fit'];n=a['certificate']['n_fit'];x=(np.log(sigma[:n])-a['center'])/a['spread'];X=np.column_stack([x**j for j in range(a['p'])]);b=np.array(a['coef'])
                objective=loss((y[:n]-q[:n])/a['scale'],X@b).sum()+n*a['certificate']['penalty']*np.abs(b[1:]).sum()
                np.testing.assert_allclose(objective,a['certificate']['primal'],rtol=1e-10,atol=1e-9);lp_count+=1
            for name,normal in [('POT-Shift',False),('POT-Vol',True)]:
                z=s/sigma if normal else s
                for a in [fit[name]['fit']]+[t['fit'] for t in fit[name]['trials']]:
                    ex=z[:a['n_fit']];assert (ex>a['threshold']).sum()==a['n_tail']
                    quantile=np.sort(ex)[math.ceil((len(ex)+1)*.99)-1] if a['fallback'] else a['threshold']+genpareto.ppf(1-.01/a['tail_fraction'],a['shape'],scale=a['scale'])
                    np.testing.assert_allclose(quantile,a['quantile'],rtol=1e-8,atol=1e-10);pot_count+=1
            receipts[key]=sha(base/'complete.json')
        print(model,'independent correction checks passed',flush=True)
    perturbations=[]
    for model in MODELS:
        key=f'{model}__NoDur';y,q,sigma,index,nc,_=load(model,'NoDur');changed=y.copy();changed[nc:]+=np.linspace(.1,1,len(y)-nc)
        pred,fit,proj,orig,_,_=compute(changed,q,sigma,nc,key)
        folder=OUT/'pairs'/key;old=json.loads((folder/'fits.json').read_text());daily=pd.read_parquet(folder/'daily.parquet')
        for name in ['state','POT-Shift','POT-Vol','gate','window','full_static']:assert json.loads(json.dumps(fit[name]))==old[name]
        for name in ['State-L1','State-L1-clipped','POT-Shift','POT-Vol','Raw','Shift-CP','Vol-ERM']:np.testing.assert_array_equal(pred[name],daily[name])
        for name in ['Rolling500','Selected-rolling','Gate-selected-rolling','Loss-gate','Past-minimum']:assert pred[name][0]==daily[name].iloc[0]
        with np.load(folder/'dtaci_experts.npz') as z:
            np.testing.assert_array_equal(proj['predictions'][:nc+1],z['projected_q'][:nc+1]);np.testing.assert_array_equal(proj['probabilities'][:nc+1],z['projected_p'][:nc+1])
        perturbations.append(key)
    record=dict(status='passed',producer_sha256=sha(__file__),pairs=48,full_fresh_replays=48,independent_scalar_checks=scalars,
        exact_array_values=values,independent_gate_bootstrap_draws=draw_count,LP_objectives=lp_count,POT_inversions=pot_count,
        future_outcome_perturbations=perturbations,pair_receipts=receipts)
    (OUT/'correction_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print({k:v for k,v in record.items() if k!='pair_receipts'},flush=True)


if __name__=='__main__':main()
