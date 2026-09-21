"""Known-target GARCH experiment separating approximation bias and variance."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import time
import zlib
import numpy as np
import pandas as pd
from scipy import stats
from controlled_comparisons import candidates,sha

OUT=Path(__file__).resolve().parents[2]/'artifacts/review_20260909/complexity_mc'
OMEGA,A,B,BURN,REPS=1e-5,.10,.85,2000,500
V0=np.sqrt(OMEGA/(1-A-B))


def garch(kind,count,seed):
    rng=np.random.default_rng(seed)
    e=(rng.standard_normal(count+BURN) if kind=='normal' else rng.standard_t(5,count+BURN)*np.sqrt(3/5))
    s2=V0**2;last=0.;y=np.empty(count);sigma=np.empty(count)
    for i,innovation in enumerate(e):
        s2=OMEGA+A*last**2+B*s2
        last=np.sqrt(s2)*innovation
        if i>=BURN:
            y[i-BURN]=last;sigma[i-BURN]=np.sqrt(s2)
    return y,sigma


def conditional_quantile(kind,sigma,alpha):
    z=stats.norm.ppf(alpha) if kind=='normal' else np.sqrt(3/5)*stats.t.ppf(alpha,5)
    return sigma*z


def expected_loss(kind,q,sigma,alpha):
    if kind=='normal':
        z=q/sigma
        return sigma*(stats.norm.pdf(z)+z*(stats.norm.cdf(z)-alpha))
    scale=sigma*np.sqrt(3/5);z=q/scale
    return scale*((5+z*z)/4*stats.t.pdf(z,5)+z*(stats.t.cdf(z,5)-alpha))


def distortion(sigma,truth):
    return np.full_like(sigma,.25*V0) if truth=='constant' else .25*V0+.75*V0*np.log(sigma/V0)


def run(kind,truth,n,alpha):
    started=time.monotonic()
    folder=OUT/f'{kind}_{truth}_{n}_{alpha:g}'
    folder.mkdir(parents=True,exist_ok=True)
    binding={'producer_sha256':sha(__file__),'candidate_sha256':sha(Path(__file__).with_name('controlled_comparisons.py')),
             'innovation':kind,'truth':truth,'n_cal':n,'alpha':alpha,'replications':REPS,'test_states':1024}
    done=folder/'complete.json'
    if done.exists():
        record=json.loads(done.read_text())
        assert record['binding']==binding
        assert all(sha(folder/k)==v for k,v in record['outputs'].items())
        return folder.name,'verified'
    test_seed=zlib.crc32(f'20260909|test|{kind}'.encode())&0xffffffff
    _,test_sigma=garch(kind,1024,test_seed)
    oracle=conditional_quantile(kind,test_sigma,alpha)
    raw=oracle+distortion(test_sigma,truth)
    optimal_loss=expected_loss(kind,oracle,test_sigma,alpha)
    rows=[];param_rows=[];sums={};squares={};squared_errors={}
    for rep in range(REPS):
        seed=zlib.crc32(f'20260909|cal|{kind}|{rep}'.encode())&0xffffffff
        y,sigma=garch(kind,1000,seed)
        y=y[-n:];sigma=sigma[-n:]
        q=conditional_quantile(kind,sigma,alpha)+distortion(sigma,truth)
        pred,params=candidates(y,q,sigma,raw,test_sigma,alpha)
        param_rows.append({'replication':rep,'seed':seed,'parameters':params})
        for method,p in pred.items():
            risk=expected_loss(kind,p,test_sigma,alpha)
            assert np.min(risk-optimal_loss)>-1e-12
            rows.append({'replication':rep,'method':method,'expected_QS':float(risk.mean()),
                         'excess_QS':float((risk-optimal_loss).mean()),'prediction_MSE':float(np.mean((p-oracle)**2))})
            sums[method]=sums.get(method,0)+p
            squares[method]=squares.get(method,0)+p*p
            squared_errors[method]=squared_errors.get(method,0)+(p-oracle)**2
    data=pd.DataFrame(rows);data.to_csv(folder/'replications.csv',index=False)
    (folder/'parameters.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in param_rows))
    decomposition=[];state_rows=[]
    reference=data[data.method=='Shift-ERM'].set_index('replication').expected_QS
    for method,g in data.groupby('method'):
        center=sums[method]/REPS
        variance=np.maximum(squares[method]/REPS-center**2,0)
        bias2=(center-oracle)**2
        mse=squared_errors[method]/REPS
        assert np.max(np.abs(mse-bias2-variance))<1e-12
        delta=g.set_index('replication').expected_QS-reference
        decomposition.append({'innovation':kind,'truth':truth,'n_cal':n,'alpha':alpha,'method':method,
            'mean_expected_QS':g.expected_QS.mean(),'mean_excess_QS':g.excess_QS.mean(),
            'excess_QS_MCSE':g.excess_QS.std(ddof=1)/np.sqrt(REPS),
            'difference_vs_shift_ERM':delta.mean(),'difference_MCSE':delta.std(ddof=1)/np.sqrt(REPS),
            'integrated_squared_bias':bias2.mean(),'integrated_variance':variance.mean(),'integrated_MSE':mse.mean()})
        for i in range(len(test_sigma)):
            state_rows.append({'method':method,'state':i,'sigma':test_sigma[i],'oracle_quantile':oracle[i],
                               'mean_prediction':center[i],'variance':variance[i],'squared_bias':bias2[i]})
    pd.DataFrame(decomposition).to_csv(folder/'summary.csv',index=False)
    pd.DataFrame(state_rows).to_parquet(folder/'prediction_moments.parquet',index=False)
    files=['replications.csv','parameters.jsonl','summary.csv','prediction_moments.parquet']
    done.write_text(json.dumps({'binding':binding,'test_seed':test_seed,'outputs':{p:sha(folder/p) for p in files},
                               'elapsed_seconds':time.monotonic()-started},indent=2)+'\n')
    return folder.name,round(time.monotonic()-started,1)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        tasks=[pool.submit(run,kind,truth,n,alpha) for kind in ['normal','t5']
               for truth in ['constant','state'] for n in [125,250,500,1000] for alpha in [.01,.05]]
        for task in as_completed(tasks):print(*task.result(),flush=True)
    frames=[pd.read_csv(p) for p in sorted(OUT.glob('*/summary.csv'))]
    data=pd.concat(frames,ignore_index=True)
    assert len(data)==32*7
    data.to_csv(OUT/'summary.csv',index=False)


if __name__=='__main__':main()
