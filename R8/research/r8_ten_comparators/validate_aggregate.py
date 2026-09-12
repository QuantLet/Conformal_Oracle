"""Independent calendar resampling with one shared support per asset."""
import hashlib
import json
import math
import numpy as np
import pandas as pd
from run import OUT,RESULTS,MODELS,ASSETS,METHODS,sha
from validate_ten import check_close

FAMILY=['State-L1','POT-Shift','POT-Vol','DtACI-projected-expected','Loss-gate','Past-minimum']


def main():
    complete=json.loads((RESULTS/'complete.json').read_text());assert complete['pairs']==240
    for name,h in complete['outputs'].items():assert sha(RESULTS/name)==h,name
    pairs=pd.read_csv(RESULTS/'pairs.csv');assert len(pairs)==240*20
    assert not pairs.duplicated(['model','asset','method']).any()
    summary=pd.read_csv(RESULTS/'summary.csv').set_index('method')
    scalar_checks=0
    for method,g in pairs.groupby('method'):
        row=summary.loc[method]
        for key,value in dict(pairs=len(g),QS=g.QS.mean(),normalised_QS=g.normalised_QS.mean(),
            violation_rate=g.pihat.mean(),width=g.width.mean(),kupiec_rejections=(g.p_kup<.05).sum(),
            kupiec_available=g.p_kup.notna().sum(),test_observations=g.n_test.sum(),QS_x10000=g.QS.mean()*1e4).items():
            check_close(row[key],value);scalar_checks+=1
    frames={};scales={}
    for asset in ASSETS:
        model_losses=[];index=None;scale=None
        for model in MODELS:
            key=f'{model}__{asset}';folder=OUT/'pairs'/key
            assert sha(folder/'complete.json')==complete['bindings'][key]
            frame=pd.read_parquet(folder/'daily.parquet')
            if index is not None:assert index.equals(frame.index)
            index=frame.index
            rows=pairs[(pairs.model==model)&(pairs.asset==asset)].set_index('method')
            c=rows.calibration_scale.iloc[0]
            if scale is not None:assert c==scale
            scale=c;arrays=[]
            for method in METHODS:
                if method=='DtACI-projected-expected':loss=frame['DtACI-expected/loss'].to_numpy()
                else:
                    error=frame.r.to_numpy()-frame[method].to_numpy();loss=np.maximum(.01*error,-.99*error)
                check_close(loss.mean(),rows.loc[method,'QS']);scalar_checks+=1;arrays.append(loss)
            model_losses.append(np.column_stack(arrays))
        # Every model has identical dates within an asset. Averaging models
        # before resampling is therefore exactly the same linear statistic
        # as averaging all 240 pair means after resampling.
        frames[asset]=pd.DataFrame(np.mean(model_losses,axis=0),index=index,columns=METHODS)
        scales[asset]=scale
    calendar=pd.date_range(min(f.index[0] for f in frames.values()),max(f.index[-1] for f in frames.values()))
    per_asset=np.array([frames[a].mean(0).to_numpy() for a in ASSETS])
    scale=np.array([scales[a] for a in ASSETS]);noncrypto=np.array([a not in ['BTC','ETH'] for a in ASSETS])
    points={'raw':per_asset.mean(0),'normalised':(per_asset/scale[:,None]).mean(0),
            'without_crypto':per_asset[noncrypto].mean(0)}
    intervals=pd.read_csv(RESULTS/'intervals.csv').set_index(['block_calendar_days','method','reference'])
    sensitivity=pd.read_csv(RESULTS/'sensitivity_intervals.csv').set_index(['block_calendar_days','sensitivity','method'])
    lhs=[METHODS.index(n) for n in FAMILY];ref=METHODS.index('Shift-CP')
    for block in [20,60]:
        seed=int.from_bytes(hashlib.sha256(f'20260909/panel-calendar/{block}'.encode()).digest()[:4],'little')
        rng=np.random.default_rng(seed);counts=[]
        for draw in range(999):
            starts=rng.integers(0,len(calendar),size=math.ceil(len(calendar)/block))
            idx=np.concatenate([np.arange(s,s+block)%len(calendar) for s in starts])[:len(calendar)]
            counts.append(np.bincount(idx,minlength=len(calendar)))
        counts=np.array(counts);means=[]
        for asset in ASSETS:
            frame=frames[asset];pos=calendar.get_indexer(frame.index);weights=counts[:,pos]
            means.append((weights@frame.to_numpy())/weights.sum(1)[:,None])
        means=np.stack(means,axis=1)
        draws={'raw':means.mean(1),'normalised':(means/scale[None,:,None]).mean(1),
               'without_crypto':means[:,noncrypto].mean(1)}
        saved=np.load(RESULTS/f'bootstrap_{block}.npz')
        assert saved['methods'].tolist()==METHODS
        check_close(saved['draws'],draws['raw']);check_close(saved['point'],points['raw'])
        difference=draws['raw'][:,lhs]-draws['raw'][:,ref,None]
        center=points['raw'][lhs]-points['raw'][ref];sd=difference.std(0,ddof=1)
        critical=np.quantile(np.max(np.abs((difference-center)/sd),axis=1),.95)
        for (method,reference),row in intervals.xs(block).iterrows():
            i=METHODS.index(method);j=METHODS.index(reference);delta=draws['raw'][:,i]-draws['raw'][:,j]
            lo,hi=np.quantile(delta,[.025,.975]);est=points['raw'][i]-points['raw'][j]
            for key,value in dict(difference=est*1e4,lower=lo*1e4,upper=hi*1e4).items():check_close(row[key],value);scalar_checks+=1
            if reference=='Shift-CP' and method in FAMILY:
                half=critical*sd[FAMILY.index(method)]
                check_close(row.simultaneous_lower,(est-half)*1e4);check_close(row.simultaneous_upper,(est+half)*1e4)
        for label in ['normalised','without_crypto']:
            sample=draws[label];point=points[label];delta=sample[:,lhs]-sample[:,ref,None];estimate=point[lhs]-point[ref]
            sd=delta.std(0,ddof=1);critical=np.quantile(np.max(np.abs((delta-estimate)/sd),axis=1),.95)
            factor=1. if label=='normalised' else 1e4
            for i,method in enumerate(FAMILY):
                row=sensitivity.loc[(block,label,method)];lo,hi=np.quantile(delta[:,i],[.025,.975]);half=critical*sd[i]
                for key,value in dict(difference=estimate[i]*factor,lower=lo*factor,upper=hi*factor,
                    simultaneous_lower=(estimate[i]-half)*factor,simultaneous_upper=(estimate[i]+half)*factor).items():
                    check_close(row[key],value);scalar_checks+=1
    result=dict(status='passed',producer_sha256=sha(__file__),complete_sha256=sha(RESULTS/'complete.json'),
        pairs=240,methods=20,independent_scalar_checks=scalar_checks,independent_calendar_draws=1998,
        primary_bootstrap_means_verified=1998*20,normalised_and_noncrypto_bands_verified=True)
    (OUT/'aggregation_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)


if __name__=='__main__':main()
