"""Frozen-input diagnostic of expectation-level training optimism. No generator."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'research/r8_theory_loop_v3'
OUT=ROOT/'results/theory_loop_v3'
SYN=ROOT/'results/theory_loop/synthetic'
LOCK=OUT/'lock.json'
SIZES=(250,500,700,1000,2000)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def record(path):
    p=Path(path);s=p.stat()
    return dict(path=str(p.resolve()),sha256=sha(p),mtime_ns=s.st_mtime_ns,size=s.st_size)

def dump(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def bind_inputs():
    assert not LOCK.exists(),'Do not overwrite an existing execution lock'
    commit=subprocess.check_output(['git','-C',str(BASE/'protocol_repository'),'rev-parse','HEAD'],text=True).strip()
    committed=subprocess.check_output(['git','-C',str(BASE/'protocol_repository'),'show',commit+':PROTOCOL.md'])
    assert committed==(BASE/'PROTOCOL.md').read_bytes()
    paths=[BASE/'PROTOCOL.md',Path(__file__),ROOT/'source/sections_r8/risk.tex',ROOT/'source/sections_r8/risk_proofs.tex',
        ROOT/'results/theory_loop/provenance_manifest.json',ROOT/'results/theory_loop_v2/manifest.json',
        ROOT/'results/theory_loop_v2/risk/uniform_exact.csv']
    paths +=[SYN/name for name in ('estimators.csv','loss_histories.csv','expansion.csv','truth.csv',
        'validation_summary.csv','admission.json','history_bootstrap_indices.npy')]
    paths +=[SYN/f'{name}_{law}.npz' for name in ('calibration','truth','mixture_integration') for law in ('normal','t5')]
    dump(LOCK,dict(protocol_commit=commit,files=[record(p) for p in paths]))

def validate_lock(path=LOCK):
    lock=json.loads(Path(path).read_text())
    for rec in lock['files']:
        new=record(rec['path'])
        assert all(new[k]==rec[k] for k in ('sha256','mtime_ns','size')),rec['path']
    return lock

def rho(u):
    u=np.asarray(u)
    return np.where(u>=0,.01*u,-.99*u)

def rank(n): return (99*(n+1)+99)//100

def integral(x,c):
    return np.mean(np.maximum(c-x,0)-np.maximum(-x,0),axis=-1)-.99*np.asarray(c).squeeze()

def suffix(x,n):
    H=3*n//7
    if x.shape[-1]<n+H:raise ValueError('Saved contiguous test is unavailable')
    return x[...,n:n+H]

def expected_keys():
    return {(law,n,mode) for law in ('normal','t5') for n in SIZES
            for mode in ('independent','contiguous') if mode=='independent' or n<=1000}

def complete(keys): return len(keys)==18 and set(keys)==expected_keys()

def critical_value(x,idx):
    means=x.mean(axis=0);se=x.std(axis=0,ddof=1)/np.sqrt(len(x))
    if not np.isfinite(se).all() or (se<=0).any():raise ValueError('Invalid standard error')
    bm=x[idx].mean(axis=1)
    maxima=(np.abs(bm-means)/se).max(axis=1)
    return float(np.quantile(maxima,.95,method='higher')),means,se,maxima

def check(rows,name,bad,good,predicate):
    def accepts(value):
        try:return bool(predicate(value))
        except (ValueError,AssertionError,KeyError,IndexError):return False
    rejected=not accepts(bad)
    passed=accepts(good) if rejected else False
    rows.append(dict(name=name,negative_control_rejected=rejected,valid_case_accepted=passed))
    if not rejected or not passed:raise AssertionError(rows[-1])

def close(x,y): return np.allclose(x,y,atol=1e-12,rtol=1e-10)

def preflight(output):
    fixtures=dict(loss_errors=[-2.,-.5,0.,.5,2.],n_rank=250,
        stale_sha256='0'*64,drop_factor_two=True,future_slice_mutant='prefix instead of suffix')
    dump(output/'fixtures.json',fixtures)
    rows=[];x=np.array(fixtures['loss_errors']);wanted=np.array([1.98,.495,0,.005,.02])
    check(rows,'actual_wrong_tail_loss',np.where(x>=0,.99*x,-.01*x),rho(x),lambda z:close(z,wanted))
    grid=np.linspace(-1,1,250);k=rank(250)
    check(rows,'actual_empirical_instead_of_conformal_rank',grid[int(np.ceil(.99*250))-1],grid[k-1],lambda z:z==grid[248])
    # Reflection must map the existing threshold, not retake the upper tail.
    check(rows,'actual_reflected_wrong_tail',np.sort(-grid)[k-1],-grid[k-1],lambda z:z==-grid[248])
    c=grid[k-1];direct=np.mean(rho(c-grid)-rho(-grid));I=integral(grid,c)
    check(rows,'actual_reversed_empirical_loss_sign',-I,I,lambda z:close(z,direct))
    sample=np.arange(2000.)
    check(rows,'actual_leaking_future_prefix',sample[:3*700//7],suffix(sample,700),lambda z:np.array_equal(z,np.arange(700.,1000.)))
    keys=sorted(expected_keys())
    check(rows,'missing_family_cell',keys[:-1],keys,complete)
    check(rows,'duplicate_family_cell',keys[:-1]+keys[:1],keys,complete)
    A=.0099/(2*1000)
    check(rows,'actual_missing_optimism_factor',A,2*A,lambda z:z==.0099/1000)
    xx=np.array([[1.,3.],[2.,7.],[3.,5.],[6.,9.]])
    idx=np.array([[0,0,1,2],[1,2,3,3],[0,1,2,3]])
    q,mean,se,maxima=critical_value(xx,idx)
    expected=np.max(np.abs(np.array([xx[r].mean(0)-mean for r in idx]))/se,axis=1)
    mutant=np.max(np.abs(xx[idx].mean(1)-1)/se,axis=1)
    check(rows,'actual_bootstrap_null_instead_of_mean_center',mutant,maxima,lambda z:close(z,expected))
    shifted=critical_value(xx+20,idx)[0]
    check(rows,'bootstrap_translation_invariance',shifted+1,shifted,lambda z:close(z,q))
    lock=json.loads(LOCK.read_text());lock['files'][0]['sha256']='0'*64
    fixture=output/'stale_lock.json';dump(fixture,lock)
    rejected=False
    try:validate_lock(fixture)
    except AssertionError:rejected=True
    if not rejected:raise AssertionError('Stale lock accepted')
    validate_lock();rows.append(dict(name='stale_input_binding',negative_control_rejected=True,valid_case_accepted=True))
    dump(output/'preflight.json',dict(status='PASS',checks=rows));return rows

def run(output):
    validate_lock();output.mkdir(parents=True,exist_ok=True)
    assert not (output/'histories.csv').exists(),'Refuse to overwrite a completed calculation'
    checks=preflight(output)
    estimates=pd.read_csv(SYN/'estimators.csv')
    stored=pd.read_csv(SYN/'loss_histories.csv')
    truth=pd.read_csv(SYN/'truth.csv').set_index('law')
    indices=np.load(SYN/'history_bootstrap_indices.npy')
    check(checks,'bootstrap_fixed_shape_range',indices[:-1],indices,
        lambda z:z.shape==(999,500) and z.min()>=0 and z.max()<500 and np.issubdtype(z.dtype,np.integer))
    expected_hash=next(r['sha256'] for r in json.loads(LOCK.read_text())['files'] if r['path'].endswith('history_bootstrap_indices.npy'))
    check(checks,'bootstrap_bound_bytes','0'*64,sha(SYN/'history_bootstrap_indices.npy'),lambda z:z==expected_hash)
    rows=[];rankrows=[];family=[];columns=[];missing=[]
    for law in ('normal','t5'):
        z=np.load(SYN/f'calibration_{law}.npz');scores=z['scores']
        check(checks,'finite_saved_scores_'+law,scores[:499],scores,lambda z:z.shape==(500,2000) and np.isfinite(z).all())
        fref=float(truth.loc[law,'f_true']);A=lambda n:.0099/(2*n*fref)
        mixture=np.load(SYN/f'mixture_integration_{law}.npz')
        poly=np.polynomial.Chebyshev(mixture['coef127'],domain=mixture['domain'])
        for n in SIZES:
            x=scores[:,:n];k=rank(n);c=np.sort(x,axis=1)[:,k-1]
            full_train=rho(c[:,None]-x).mean(axis=1);oracle_train=rho(-x).mean(axis=1)
            J=full_train-oracle_train
            sample=estimates[(estimates.law==law)&(estimates.n==n)&estimates.bias.eq(0)].sort_values('rep')
            check(checks,f'500_unique_estimates_{law}_{n}',sample.rep.to_numpy()[:-1],sample.rep.to_numpy(),lambda z:np.array_equal(z,np.arange(500)))
            check(checks,f'rank_replay_{law}_{n}',sample.C.to_numpy()+.01,c,lambda z:close(z,sample.C))
            base=stored[(stored.law==law)&(stored.n==n)&stored.bias.eq(0)].sort_values('rep')
            check(checks,f'500_unique_stored_losses_{law}_{n}',base.rep.to_numpy()[:-1],base.rep.to_numpy(),lambda z:np.array_equal(z,np.arange(500)))
            independent_V=base.delta.to_numpy();oracle_risk=base.raw_loss.to_numpy()
            check(checks,f'mixture_replay_{law}_{n}',poly(-c)+.01,poly(-c),
                lambda z:np.allclose(z,base.static_loss,atol=4e-11,rtol=4e-11))
            ahat=sample.omega.to_numpy()/(2*n*sample.f_sj.to_numpy())
            check(checks,f'original_cost_replay_{law}_{n}',ahat*2,ahat,lambda z:close(z,sample.A_hat))
            aho=.0099/(2*n*sample.f_sj.to_numpy());ahf=sample.omega.to_numpy()/(2*n*fref)
            qhit=(x<=0).sum(axis=1);chit=(x<=c[:,None]).sum(axis=1)
            for rep in range(500):rankrows.append(dict(law=law,n=n,rep=rep,k=k,k_minus_np=k-.99*n,
                C=c[rep],true_quantile_hits=qhit[rep],estimated_quantile_hits=chit[rep]))
            modes=[('independent',independent_V,0,None)]
            if n<=1000:
                future=suffix(scores,n);H=future.shape[1]
                V=(rho(c[:,None]-future)-rho(-future)).mean(axis=1)
                modes.append(('contiguous',V,H,future))
            else:missing.append(dict(law=law,n=n,evaluation='contiguous',status='NOT_AVAILABLE',reason='No saved future after 2000 observations'))
            for mode,V,H,future in modes:
                O=V-J;family.append((law,n,mode));columns.append(O/(2*A(n)))
                for bias in (0.,.25*np.sqrt(1e-5/(1-.10-.85))):
                    ss=estimates[(estimates.law==law)&(estimates.n==n)&np.isclose(estimates.bias,bias,atol=1e-15,rtol=0)].sort_values('rep')
                    shifted=x+bias;cs=c+bias
                    I=(rho(cs[:,None]-shifted)-rho(-shifted)).mean(axis=1)
                    Kcal=(rho(-x)-rho(-shifted)).mean(axis=1)
                    directI=integral(shifted,cs[:,None])
                    check(checks,f'empirical_integral_{law}_{n}_{mode}_{bias}',-directI,directI,lambda z:close(z,I))
                    check(checks,f'original_I_replay_{law}_{n}_{mode}_{bias}',I+.01,I,lambda z:close(z,ss.signed_ecdf_loss))
                    if mode=='independent':
                        st=stored[(stored.law==law)&(stored.n==n)&np.isclose(stored.bias,bias,atol=1e-15,rtol=0)].sort_values('rep')
                        delta=st.delta.to_numpy();Ktest=oracle_risk-st.raw_loss.to_numpy()
                    else:
                        delta=(rho(cs[:,None]-(future+bias))-rho(-(future+bias))).mean(axis=1)
                        Ktest=(rho(-future)-rho(-(future+bias))).mean(axis=1)
                    uncentred=delta-I
                    check(checks,f'centering_identity_{law}_{n}_{mode}_{bias}',O+Ktest+Kcal,O+Ktest-Kcal,
                        lambda z:close(z,uncentred)) if bias else check(checks,f'centering_identity_{law}_{n}_{mode}_{bias}',O+1,O,lambda z:close(z,uncentred))
                    for rep in range(500):rows.append(dict(law=law,n=n,H=H,evaluation=mode,bias=bias,rep=rep,
                        C=cs[rep],J=J[rep],V=V[rep],I=I[rep],delta=delta[rep],Kcal=Kcal[rep],Ktest=Ktest[rep],
                        optimism_centred=O[rep],optimism_uncentred=uncentred[rep],A0_ref=A(n),Ahat=ahat[rep],
                        A_exact_omega=aho[rep],A_reference_density=ahf[rep],
                        prediction_one_A0=I[rep]+A(n),prediction_two_A0=I[rep]+2*A(n),prediction_two_Ahat=I[rep]+2*ahat[rep]))
    check(checks,'complete_primary_family',family[:-1],family,complete)
    order=sorted(range(len(family)),key=lambda i:family[i]);family=[family[i] for i in order]
    matrix=np.column_stack([columns[i] for i in order])
    crit,means,se,maxima=critical_value(matrix,indices)
    bands=[]
    for j,(law,n,mode) in enumerate(family):
        lo=means[j]-crit*se[j];hi=means[j]+crit*se[j]
        bands.append(dict(law=law,n=n,evaluation=mode,H=3*n//7 if mode=='contiguous' else 0,
            mean_optimism_over_2A0=means[j],standard_error=se[j],lower=lo,upper=hi,
            includes_first_order_reference=bool(lo<=1<=hi),relative_discrepancy=means[j]-1,
            critical_value=crit,family_size=18,bootstrap_replicates=999))
    allrows=pd.DataFrame(rows);allrows.to_csv(output/'histories.csv',index=False)
    pd.DataFrame(rankrows).to_csv(output/'rank_diagnostics.csv',index=False)
    pd.DataFrame(bands).to_csv(output/'simultaneous_bands.csv',index=False)
    pd.DataFrame(missing).to_csv(output/'unavailable.csv',index=False)
    pd.DataFrame(dict(bootstrap=np.arange(999),max_t=maxima)).to_csv(output/'bootstrap_maxima.csv',index=False)
    summaries=[]
    for key,g in allrows.groupby(['law','n','evaluation','bias'],sort=True):
        a=g.A0_ref.iloc[0];err=g.Ahat/a-1;eo=g.A_reference_density/a-1;ef=g.A_exact_omega/a-1
        check(checks,'nuisance_factorization_'+str(key),eo+ef,eo+ef+eo*ef,lambda z:close(z,err))
        q=g.C-g.bias
        summaries.append(dict(law=key[0],n=key[1],evaluation=key[2],bias=key[3],histories=len(g),
            A0_ref=a,mean_J_over_A0=g.J.mean()/a,mean_V_over_A0=g.V.mean()/a,
            mean_I=g.I.mean(),mean_delta=g.delta.mean(),
            mean_optimism_centred=g.optimism_centred.mean(),mean_optimism_uncentred=g.optimism_uncentred.mean(),
            one_A0_error=(g.prediction_one_A0-g.delta).mean(),two_A0_error=(g.prediction_two_A0-g.delta).mean(),
            two_Ahat_error=(g.prediction_two_Ahat-g.delta).mean(),
            centred_two_A0_error=(g.J+2*a-g.V).mean(),centred_two_Ahat_error=(g.J+2*g.Ahat-g.V).mean(),
            mean_Ahat_over_A0=g.Ahat.mean()/a,mean_exact_omega_cost_over_A0=g.A_exact_omega.mean()/a,
            mean_reference_density_cost_over_A0=g.A_reference_density.mean()/a,
            median_Ahat_relative_error=float(np.median(np.abs(err))),
            mean_omega_relative_error=eo.mean(),mean_inverse_density_relative_error=ef.mean(),
            mean_nuisance_interaction=(eo*ef).mean(),mean_C=q.mean(),squared_bias_C=q.mean()**2,
            variance_C=q.var(ddof=1),mse_C=(q*q).mean()))
    pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False)
    dump(output/'checks.json',dict(status='PASS',checks=checks,primary_family=18,
        statistical_admission='FAIL_UNCHANGED',financial_panel='NOT_RUN',new_random_histories=0))
    validate_lock();print('Completed',len(rows),'stored-history diagnostic rows;',len(bands),'simultaneous ratios.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bind',action='store_true');parser.add_argument('--output',type=Path,default=OUT/'diagnostic')
    args=parser.parse_args()
    if args.bind:bind_inputs();print('Committed protocol and inputs bound.')
    else:run(args.output)
