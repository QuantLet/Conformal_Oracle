"""Full replay, independent losses/mixtures, and information-boundary checks."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import genpareto
from run import PROJECT,OUT,MODELS,ASSETS,METHODS,load,compute,sha
from validate_ten import metric,check_metric,check_close


def pinball(y,q):
    e=np.asarray(y)-np.asarray(q)
    return np.maximum(.01*e,-.99*e)


def main():
    checks=values=lp_checks=pot_checks=0;receipts={}
    for model in MODELS:
        for asset in ASSETS:
            key=f'{model}__{asset}';one=OUT/'pairs'/key;two=OUT/'replay'/key
            y,q,sigma,index,binding=load(model,asset);nc=int(.7*len(y))
            for folder in [one,two]:
                c=json.loads((folder/'complete.json').read_text());assert c['binding']==binding
                for name,h in c['outputs'].items():assert sha(folder/name)==h,(key,name)
            for name in ['fits.json','metrics.csv','dtaci_seed_metrics.csv']:
                assert (one/name).read_bytes()==(two/name).read_bytes(),(key,name)
            f=pd.read_parquet(one/'daily.parquet')
            pd.testing.assert_frame_equal(f,pd.read_parquet(two/'daily.parquet'),check_exact=True)
            assert f.index.equals(index[nc:]);np.testing.assert_array_equal(f.r,y[nc:])
            with np.load(one/'dtaci_experts.npz') as a,np.load(two/'dtaci_experts.npz') as b:
                assert a.files==b.files
                for name in a.files:np.testing.assert_array_equal(a[name],b[name]);values+=a[name].size
                experts=a['projected_q'];prob=a['projected_p'];states=a['projected_levels']
                np.testing.assert_allclose(prob[500:].sum(1),1.,atol=1e-14,rtol=1e-14)
                assert (prob[500:]>0).all() and np.isfinite(experts[500:]).all()
                expected_states=np.clip(states[500:-1]+np.array([.001,.002,.004,.008,.016,.032,.064])*(.01-(y[500:-1,None]<experts[500:-1])),1/501,500/501)
                np.testing.assert_allclose(states[501:],expected_states,rtol=0,atol=1e-14)
                mixtures={'loss':(prob[nc:]*pinball(y[nc:,None],experts[nc:])).sum(1),
                          'hits':(prob[nc:]*(y[nc:,None]<experts[nc:])).sum(1),
                          'width':(prob[nc:]*np.abs(experts[nc:])).sum(1),
                          'mean_threshold':(prob[nc:]*experts[nc:]).sum(1)}
                for name,array in mixtures.items():check_close(f[f'DtACI-expected/{name}'],array)
                seedtable=pd.read_csv(one/'dtaci_seed_metrics.csv').set_index('replicate')
                for rep in range(8):
                    seed=int.from_bytes(hashlib.sha256(f'20260909/{key}/{rep}'.encode()).digest()[:4],'little')
                    u=np.random.default_rng(seed).random(len(y));choices=(u[:,None]>np.cumsum(np.nan_to_num(prob),axis=1)).sum(1).clip(0,6)
                    target=experts[np.arange(nc,len(y)),choices[nc:]]
                    for name,value in metric(y[nc:],target).items():
                        if name=='TL':assert seedtable.loc[rep,name]==value
                        else:check_metric(name,seedtable.loc[rep,name],value);checks+=1
            table=pd.read_csv(one/'metrics.csv').set_index('method');assert set(table.index)==set(METHODS)
            for name in METHODS:
                if name=='DtACI-projected-expected':
                    for col,value in [('QS',mixtures['loss'].mean()),('pihat',mixtures['hits'].mean()),('width',mixtures['width'].mean()),('viol',mixtures['hits'].sum())]:check_close(table.loc[name,col],value);checks+=1
                    assert table.loc[name,['p_kup','p_ind','p_cc']].isna().all()
                else:
                    np.testing.assert_array_equal(y[nc:]<f[name],-y[nc:]>-f[name])
                    for col,value in metric(y[nc:],f[name].to_numpy()).items():
                        if col=='TL':assert table.loc[name,col]==value
                        else:check_metric(col,table.loc[name,col],value);checks+=1
            fit=json.loads((one/'fits.json').read_text());v=fit['inner_split']
            for trial in fit['state']['trials']+[{'fit':fit['state']['fit']}]:
                p=trial['fit'];n=p['certificate']['n_fit'];x=(np.log(sigma[:n])-p['center'])/p['spread']
                X=np.column_stack([x**j for j in range(p['p'])]);b=np.array(p['coef'])
                cost=pinball((y[:n]-q[:n])/p['scale'],X@b).sum()+n*p['certificate']['penalty']*np.abs(b[1:]).sum()
                np.testing.assert_allclose(cost,p['certificate']['primal'],rtol=1e-10,atol=1e-9)
                assert p['certificate']['gap']<1e-6*max(1,abs(cost)) and p['certificate']['dual_violation']<1e-7
                lp_checks+=1
            for name,normalised in [('POT-Shift',False),('POT-Vol',True)]:
                s=(q-y)/sigma if normalised else q-y
                for p in [fit[name]['fit']]+[x['fit'] for x in fit[name]['trials']]:
                    z=s[:p['n_fit']];ex=z[z>p['threshold']]-p['threshold']
                    assert len(ex)==p['n_tail'];check_close(len(ex)/len(z),p['tail_fraction'])
                    if not p['fallback']:
                        implied=p['threshold']+genpareto.ppf(1-.01/p['tail_fraction'],p['shape'],scale=p['scale'])
                        np.testing.assert_allclose(implied,p['quantile'],rtol=1e-8,atol=1e-10)
                        assert (1+p['shape']*ex/p['scale']>0).all()
                    else:check_close(p['quantile'],sorted(z)[math.ceil((len(z)+1)*.99)-1])
                    pot_checks+=1
            receipts[key]=sha(one/'complete.json')
    perturbations=[]
    for model,asset in [('Chronos-2','USO'),('PatchTST-FM','BTC'),('TS-ICL','SP500')]:
        key=f'{model}__{asset}';y,q,sigma,index,_=load(model,asset);nc=int(.7*len(y))
        changed=y.copy();changed[nc:]+=np.linspace(.1,1,len(y)-nc)
        pred,fit,proj,orig,mix,seeds=compute(changed,q,sigma,nc,key)
        folder=OUT/'pairs'/key;expected=json.loads((folder/'fits.json').read_text());daily=pd.read_parquet(folder/'daily.parquet')
        for name in ['state','POT-Shift','POT-Vol','gate','controlled_full','window','aci']:assert json.loads(json.dumps(fit[name]))==expected[name],(key,name)
        for name in ['State-L1','State-L1-clipped','POT-Shift','POT-Vol','Shift-CP','Shift-ERM','Vol-CP','Vol-ERM','State2-ERM','State4-ERM']:
            np.testing.assert_array_equal(pred[name],daily[name])
        with np.load(folder/'dtaci_experts.npz') as z:
            for name,array in [('projected_q',proj['predictions']),('projected_p',proj['probabilities']),('unprojected_q',orig['predictions'])]:np.testing.assert_array_equal(array[:nc+1],z[name][:nc+1])
        for name in ['Loss-gate','ACI-existing','Selected-rolling','Gate-selected-rolling']:assert pred[name][0]==daily[name].iloc[0]
        perturbations.append(key)
    before=json.loads((OUT/'before.json').read_text())
    for p,h in before['canonical'].items():assert sha(PROJECT/p)==h,p
    record=dict(status='passed',producer_sha256=sha(__file__),pairs=240,full_fresh_replays=240,
        numerical_array_values_replayed=values,independent_scalar_checks=checks,lp_objectives=lp_checks,
        pot_inversions_or_fallbacks=pot_checks,future_outcome_perturbations=perturbations,pair_receipts=receipts)
    (OUT/'validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print({k:v for k,v in record.items() if k!='pair_receipts'},flush=True)


if __name__=='__main__':main()
