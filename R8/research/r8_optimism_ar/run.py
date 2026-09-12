"""Minimal archived Normal-AR control. No random draws, fitting or panel reader."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'research/r8_optimism_ar'
OUT=ROOT/'results/optimism_ar';SAVED=ROOT/'artifacts/r8_mechanism';LOCK=OUT/'lock.json'
SIGMA=np.sqrt(.0002);ALPHA=.01;Z=float(norm.ppf(ALPHA));Q=SIGMA*Z
F=float(norm.pdf(Z)/SIGMA);V=.0099

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def bind(path):
    p=Path(path);s=p.stat()
    return dict(path=str(p.resolve()),sha256=sha(p),size=s.st_size,mtime_ns=s.st_mtime_ns)

def dump(path,x):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def validate(path=LOCK):
    lock=json.loads(Path(path).read_text())
    for rec in lock['inputs']:
        actual=bind(rec['path'])
        assert all(actual[k]==rec[k] for k in ('sha256','size','mtime_ns')),rec['path']
    return lock

def initialise():
    assert not LOCK.exists(),'Do not overwrite a lock'
    commit=subprocess.check_output(['git','-C',str(BASE/'protocol_repository'),'rev-parse','HEAD'],text=True).strip()
    assert subprocess.check_output(['git','-C',str(BASE/'protocol_repository'),'show',commit+':PROTOCOL.md'])==(BASE/'PROTOCOL.md').read_bytes()
    assert sha(SAVED/'paths.npz')==json.loads((SAVED/'paths.json').read_text())['sha256']
    prior=json.loads((SAVED/'results/complete.json').read_text())
    for name in ('counts.csv','replications.parquet'):
        assert sha(SAVED/'results'/name)==prior['outputs'][name]
    paths=[BASE/'PROTOCOL.md',Path(__file__),SAVED/'paths.npz',SAVED/'paths.json',
        SAVED/'results/counts.csv',SAVED/'results/replications.parquet',SAVED/'results/complete.json',
        ROOT/'results/theory_loop/synthetic/history_bootstrap_indices.npy',ROOT/'results/theory_loop_v3/manifest.json']
    dump(LOCK,dict(protocol_commit=commit,inputs=[bind(p) for p in paths]))

def rho(u): return np.where(np.asarray(u)>=0,.01*np.asarray(u),-.99*np.asarray(u))

def risk(c):
    q=Q-np.asarray(c);z=q/SIGMA
    return SIGMA*norm.pdf(z)+q*(norm.cdf(z)-ALPHA)

def rank(n):return (99*(n+1)+99)//100

def omega(phi,n):
    if phi==0:return V,n*V,0,0.,0.
    bound=lambda h:phi**(h+1)/(np.pi*np.sqrt(1-phi*phi)*(1-phi))
    L=1
    while bound(L)>1e-13:L+=1
    cov=[];errs=[]
    for j in range(1,max(L,n-1)+1):
        v,e=quad(lambda u:np.exp(-Z*Z/(1+u))/(2*np.pi*np.sqrt(1-u*u)),0,phi**j,epsabs=1e-14,epsrel=1e-12)
        cov.append(v);errs.append(e)
    cov=np.asarray(cov)
    return V+2*cov[:L].sum(),n*V+2*np.dot(n-np.arange(1,n),cov[:n-1]),L,bound(L),2*sum(errs[:L])

def test(checks,name,bad,good,predicate):
    def accepted(x):
        try:return bool(predicate(x))
        except (AssertionError,ValueError,KeyError):return False
    fail=not accepted(bad);passed=accepted(good) if fail else False
    checks.append(dict(name=name,defect_rejected=fail,valid_accepted=passed))
    assert fail and passed,name

def close(a,b):return np.allclose(a,b,atol=1e-12,rtol=1e-10)

def run(output):
    validate();output.mkdir(parents=True,exist_ok=True)
    assert not (output/'histories.csv').exists()
    dump(output/'fixtures.json',dict(rank_sample=list(range(250)),wrong_future='prefix[0:214]',
        wrong_penalty='iid variance at phi .8',wrong_family='drop or duplicate final cell'))
    checks=[];grid=np.arange(250.)
    test(checks,'actual_wrong_rank',grid[int(np.ceil(.99*250))-1],grid[rank(250)-1],lambda x:x==248.)
    u=np.array([-1.,1.]);test(checks,'actual_wrong_loss_sign',-rho(u),rho(u),lambda x:close(x,[.99,.01]))
    path=np.arange(1000.)
    test(checks,'actual_future_leakage',path[:214],path[500:714],lambda x:np.array_equal(x,np.arange(500.,714.)))
    bad=json.loads(LOCK.read_text());bad['inputs'][0]['sha256']='0'*64
    badpath=output/'stale_lock.json';dump(badpath,bad);rejected=False
    try:validate(badpath)
    except AssertionError:rejected=True
    assert rejected;validate();checks.append(dict(name='stale_input_hash',defect_rejected=True,valid_accepted=True))
    for c in (-.03,-.005,0.,.005,.03):
        ref=quad(lambda z:float(rho(SIGMA*z-Q+c))*norm.pdf(z),-np.inf,np.inf,
            points=None,epsabs=1e-13,epsrel=1e-12)[0]
        test(checks,'analytic_loss_'+str(c),risk(c)+.01,risk(c),lambda x:close(x,ref))
    paths=np.load(SAVED/'paths.npz');old=pd.read_parquet(SAVED/'results/replications.parquet')
    counts=pd.read_csv(SAVED/'results/counts.csv');rows=[];families=[];vectors=[];dep=[]
    for phi in (0.,.8):
        all_y=paths[f'ar_normal_{phi:g}'];assert all_y.shape==(500,1000)
        for n in (500,1000):
            y=all_y[:,:n];s=Q-y;k=rank(n);c=np.sort(s,axis=1)[:,k-1]
            J=np.mean(rho(c[:,None]-s)-rho(-s),axis=1)
            om,countvar,L,tail,qerr=omega(phi,n);A=om/(2*n*F);Aiid=V/(2*n*F)
            saved=counts[(counts.module=='ar')&(counts.innovation=='normal')&(counts.phi==phi)&(counts.n_cal==n)&(counts.alpha==.01)&(counts.truth=='none')].iloc[0]
            test(checks,f'population_LRV_{phi}_{n}',om+.01,om,lambda x:close(x,saved.omega))
            test(checks,f'finite_count_variance_{phi}_{n}',countvar+1,countvar,lambda x:close(x,saved.finite_count_variance))
            if phi:
                test(checks,'actual_omitted_covariances_'+str(n),V,om,lambda x:close(x,saved.omega))
            dep.append(dict(phi=phi,n=n,omega=om,finite_count_variance=countvar,finite_omega=countvar/n,
                long_run_inflation=om/V,f_true=F,lags=L,tail_bound=tail,quadrature_error=qerr,A0=A,A_iid=Aiid))
            vint=risk(c)-risk(0.)
            if n==1000:
                st=old[(old.module=='ar')&(old.innovation=='normal')&(old.phi==phi)&(old.n_cal==n)&(old.alpha==.01)&(old.truth=='none')&(old.method=='Shift-CP')].sort_values('replication')
                test(checks,'old_full_window_risk_'+str(phi),risk(c)+.01,risk(c),lambda x:close(x,st.expected_QS.to_numpy()))
            modes=[('independent',vint,0)]
            if n==500:
                fs=Q-all_y[:,500:714]
                modes.append(('contiguous',np.mean(rho(c[:,None]-fs)-rho(-fs),axis=1),214))
            for mode,val,H in modes:
                O=val-J;families.append((phi,n,mode));vectors.append(O/(2*A))
                for i in range(500):rows.append(dict(phi=phi,n=n,evaluation=mode,H=H,rep=i,k=k,
                    C=c[i],J=J[i],V=val[i],optimism=O[i],A0=A,A_iid=Aiid,
                    ratio=O[i]/(2*A),ratio_iid=O[i]/(2*Aiid)))
    expected={(phi,n,mode) for phi in (0.,.8) for n in (500,1000) for mode in ('independent','contiguous') if mode=='independent' or n==500}
    complete=lambda x:len(x)==6 and set(x)==expected
    test(checks,'missing_family',families[:-1],families,complete)
    test(checks,'duplicate_family',families[:-1]+families[:1],families,complete)
    order=sorted(range(6),key=lambda i:families[i]);families=[families[i] for i in order]
    X=np.column_stack([vectors[i] for i in order]);idx=np.load(ROOT/'results/theory_loop/synthetic/history_bootstrap_indices.npy')
    test(checks,'bootstrap_shape',idx[:-1],idx,lambda x:x.shape==(999,500) and x.min()>=0 and x.max()<500)
    means=X.mean(0);se=X.std(0,ddof=1)/np.sqrt(500)
    test(checks,'standard_errors',np.zeros(6),se,lambda x:np.isfinite(x).all() and (x>0).all())
    maxt=np.max(np.abs(X[idx].mean(1)-means)/se,axis=1);crit=float(np.quantile(maxt,.95,method='higher'))
    bands=[]
    for j,key in enumerate(families):
        phi,n,mode=key;d=next(r for r in dep if r['phi']==phi and r['n']==n)
        low=means[j]-crit*se[j];high=means[j]+crit*se[j];iid=V/d['omega']
        bands.append(dict(phi=phi,n=n,evaluation=mode,mean_ratio=means[j],se=se[j],lower=low,upper=high,
            iid_reference=iid,includes_LRV_reference=bool(low<=1<=high),includes_iid_reference=bool(low<=iid<=high),
            critical_value=crit,family_size=6,bootstrap=999))
    pd.DataFrame(rows).to_csv(output/'histories.csv',index=False)
    pd.DataFrame(dep).to_csv(output/'dependence.csv',index=False)
    pd.DataFrame(bands).to_csv(output/'bands.csv',index=False)
    pd.DataFrame(dict(bootstrap=np.arange(999),max_t=maxt)).to_csv(output/'bootstrap.csv',index=False)
    pd.DataFrame([dict(phi=phi,n=1000,evaluation='contiguous',status='NOT_AVAILABLE') for phi in (0.,.8)]).to_csv(output/'unavailable.csv',index=False)
    dump(output/'checks.json',dict(status='PASS',checks=checks,financial_panel='NOT_RUN',new_histories=0))
    validate();print('Completed 6-cell archived AR control, 3000 derived rows.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bind',action='store_true');p.add_argument('--output',type=Path,default=OUT/'primary');a=p.parse_args()
    if a.bind:initialise();print('Protocol and original artifact bindings verified.')
    else:run(a.output)
