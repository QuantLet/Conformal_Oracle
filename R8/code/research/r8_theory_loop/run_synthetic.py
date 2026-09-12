"""Execute the committed synthetic gate. Contains no financial-panel reader."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import zlib
import numpy as np
import pandas as pd
from scipy import stats
import scipy
import estimators as e
from checks import admission,record_check,sha

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/theory_loop'
SYN=OUT/'synthetic'
SIZES=(250,500,700,1000,2000)


def seed(s):
    return zlib.crc32(s.encode())&0xffffffff


def dump(p,data):
    Path(p).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def wilson(hits,n):
    z=stats.norm.ppf(.975);p=hits/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return float(center-radius),float(center+radius)


def empirical_lrv(series,L=1000):
    x=series-series.mean(axis=1,keepdims=True);n=x.shape[1]
    nfft=1<<(2*n-1).bit_length()
    fft=np.fft.rfft(x,nfft,axis=1)
    ac=np.fft.irfft(fft*fft.conj(),nfft,axis=1)[:,:L+1]/n
    return ac[:,0]+2*(ac[:,1:]*(1-np.arange(1,L+1)/(L+1))).sum(axis=1)


def run():
    lock=json.loads((OUT/'lock.json').read_text())
    if sha(ROOT/'analysis_plan_theory_loop.md')!=lock['plan_sha256']:
        raise RuntimeError('Changed protocol')
    preflight=json.loads((OUT/'preflight.json').read_text())
    if preflight['status']!='PASS':
        raise RuntimeError('Preflight failed')
    if (SYN/'admission.json').exists():
        raise RuntimeError('Immutable run exists; use replay, not overwrite')
    sources=[*sorted((ROOT/'research/r8_theory_loop').glob('*.py')),
             ROOT/'research/r8_theory_loop/sj_batch.R',ROOT/'research/r8_theory_loop/failing_cases.json',
             ROOT/'research/r8_review/complexity_simulation.py',ROOT/'analysis_plan_theory_loop.md']
    dump(OUT/'execution_binding.json',dict(sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
          python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,
          platform=platform.platform(),protocol_commit=lock['protocol_commit'],financial_panel_reads=0))
    r_info=subprocess.check_output(['Rscript','--vanilla','-e',
                  'print(sessionInfo()); print(stats::bw.SJ)'],text=True)
    (SYN/'R_runtime_and_selector.txt').write_text(r_info)
    bootstrap=np.random.default_rng(2026091141).integers(0,500,size=(999,500))
    np.save(SYN/'history_bootstrap_indices.npy',bootstrap)
    all_rows=[]; truth_rows=[]; summaries=[];expansion=[];loss_histories=[];runchecks=[]
    finite_expansion=[]; finite_losses=[]
    for law in ('normal','t5'):
        print('Generate and save:',law,flush=True)
        seeds=[seed(f'20260909|cal|{law}|{rep}') for rep in range(500)]
        y,sigma=e.garch_batch(law,2000,seeds)
        d=stats.norm() if law=='normal' else stats.t(5,scale=np.sqrt(3/5))
        z=float(d.ppf(.01));scores=z*sigma-y
        np.savez_compressed(SYN/f'calibration_{law}.npz',returns=y,sigma=sigma,scores=scores,seeds=seeds)
        binary=SYN/f'scores_{law}.bin';scores.astype('<f8').tofile(binary)
        # Reconstruct one historical seed with the literal scalar pipeline.
        sys.path.insert(0,str(ROOT/'research/r8_review'))
        import complexity_simulation as old
        ry,rs=old.garch(law,1000,seeds[0])
        good=np.column_stack((y[0,:1000],sigma[0,:1000]));bad=good.copy();bad[0,1]+=.01
        record_check(runchecks,'existing_generator_replay_'+law,bad,good,
                     lambda arr:np.array_equal(arr,np.column_stack((ry,rs))))
        truth_seeds=[seed(f'20260911|theory_loop|truth|{law}|{b}') for b in range(20)]
        ty,ts=e.garch_batch(law,50000,truth_seeds)
        truth_scores=z*ts-ty
        np.savez_compressed(SYN/f'truth_{law}.npz',scores=truth_scores,sigma=ts,seeds=truth_seeds)
        f_blocks=d.pdf(z)*np.mean(1/ts,axis=1)
        ftrue=float(f_blocks.mean());fse=float(f_blocks.std(ddof=1)/np.sqrt(20))
        # A precision failure is recorded, not repaired with more reference draws.
        precision=fse/ftrue<=.02
        lrvs=empirical_lrv((truth_scores<=0).astype(float))
        truth_rows.append(dict(law=law,omega_true=.0099,omega_empirical=float(lrvs.mean()),
                          omega_empirical_mcse=float(lrvs.std(ddof=1)/np.sqrt(20)),
                          f_true=ftrue,f_true_mcse=fse,f_relative_mcse=fse/ftrue,
                          truth_precision_pass=bool(precision),reference_histories=20,
                          observations_per_history=50000))
        _,test_sigma=e.garch_batch(law,1024,[seed(f'20260909|test|{law}')])
        test_sigma=test_sigma[0]
        np.save(SYN/f'test_sigma_{law}.npy',test_sigma)
        oracle=z*test_sigma
        finite_f=float(d.pdf(z)*np.mean(1/test_sigma))
        oracle_risk=float(np.mean(e.expected_loss(law,oracle,test_sigma)))
        print('Sheather–Jones:',law,'2500 calibration windows',flush=True)
        subprocess.run(['Rscript','--vanilla',str(ROOT/'research/r8_theory_loop/sj_batch.R'),
                        str(binary),str(SYN/f'sj_{law}.csv'),'2000','500'],check=True)
        sj=pd.read_csv(SYN/f'sj_{law}.csv')
        # Deviation02: common long-reference marginal law for risk and density.
        displacements=np.r_[-sj.C.to_numpy(),0.,.25*np.sqrt(1e-5/(1-.10-.85))]
        domain=[float(displacements.min()),float(displacements.max())]
        flat_sigma=ts.ravel(); flat_oracle=z*flat_sigma
        def mixture_risk(a):
            return float(np.mean(e.expected_loss(law,flat_oracle+float(a),flat_sigma)))
        nodes={}
        def values(x):
            vals=np.array([mixture_risk(a) for a in x])
            nodes[len(x)]=(x.copy(),vals.copy())
            return vals
        print('Integrate common reference law:',law,flush=True)
        poly63=np.polynomial.Chebyshev.interpolate(values,63,domain=domain)
        poly127=np.polynomial.Chebyshev.interpolate(values,127,domain=domain)
        agrees=lambda x:np.allclose(x,poly127(displacements),rtol=4e-11,atol=4e-11)
        record_check(runchecks,'mixture_integration_degree_check_'+law,
                     poly63(displacements)+.01,poly63(displacements),agrees)
        probe=np.linspace(*domain,9)
        direct=np.array([mixture_risk(a) for a in probe])
        record_check(runchecks,'mixture_integration_direct_'+law,
                     poly127(probe)+.01,poly127(probe),
                     lambda x:np.allclose(x,direct,rtol=4e-11,atol=4e-11))
        np.savez(SYN/f'mixture_integration_{law}.npz',domain=domain,
                 nodes63=nodes[64][0],values63=nodes[64][1],coef63=poly63.coef,
                 nodes127=nodes[128][0],values127=nodes[128][1],coef127=poly127.coef,
                 probe=probe,probe_exact=direct,max_degree_discrepancy=
                 float(np.max(np.abs(poly63(displacements)-poly127(displacements)))))
        long_oracle_risk=mixture_risk(0.)
        for n in SIZES:
            sample=scores[:,:n]
            shifts=np.partition(sample,e.rank(n)-1,axis=1)[:,e.rank(n)-1]
            selected=sj[sj.n==n].sort_values('rep')
            if len(selected)!=500 or not (selected.status=='OK').all():
                dump(SYN/'implementation_failure.json',dict(law=law,n=n,
                     failures=selected[selected.status!='OK'].fillna('').to_dict('records')))
                raise RuntimeError('Primary SJ execution failed; no panel access')
            f=selected.f.to_numpy();bw=selected.bw.to_numpy()
            record_check(runchecks,f'SJ_rank_replay_{law}_{n}',selected.C.to_numpy()+.001,
                         selected.C.to_numpy(),lambda x:np.allclose(x,shifts,rtol=1e-12,atol=1e-14))
            direct=e.gaussian_density(sample[0],shifts[0],bw[0])
            record_check(runchecks,f'R_density_replay_{law}_{n}',f[0]*2,f[0],
                         lambda x:np.isclose(x,direct,rtol=1e-12,atol=1e-14))
            hits=(sample<=shifts[:,None]).astype(float);L=int(np.floor(4*(n/100)**(2/9)))
            omegas=[];rowset=[]
            for rep in range(500):
                omega,unfloor=e.hac(hits[rep],L)
                try:
                    auto,autoraw,autobw,rho=e.andrews(hits[rep]);astatus='OK'
                except ValueError as exc:
                    auto=autoraw=autobw=rho=np.nan;astatus=str(exc)
                omegas.append(omega);sp=e.spacing(sample[rep])
                for bias in (0.,.25*np.sqrt(1e-5/(1-.10-.85))):
                    c=shifts[rep]+bias;integral=e.ecdf_integral(sample[rep]+bias,c)
                    a=omega/(2*n*f[rep]);signal=e.h2(c,n,omega,f[rep])
                    rowset.append(dict(law=law,n=n,rep=rep,bias=bias,C=c,omega=omega,
                         omega_unfloored=unfloor,omega_floor=unfloor<=.00099,NW_lags=L,
                         omega_andrews=auto,andrews_raw=autoraw,andrews_bw=autobw,
                         andrews_rho=rho,andrews_status=astatus,f_sj=f[rep],sj_bw=bw[rep],
                         f_sbg=sp['f'],sbg_status=sp['status'],sbg_m=sp['m'],sbg_lo=sp['lower'],
                         sbg_hi=sp['upper'],h2_naive=n*c*c,h2=signal,h2_floor=signal==0,
                         B_hat=-integral,signed_ecdf_loss=integral,A_hat=a,
                         prediction=integral+a,literal_prompt_prediction=-integral+a,
                         lambda_hat=e.shrinkage(c,n,omega,f[rep]),variance_hat=omega/(n*f[rep]**2)))
            all_rows.extend(rowset);omegas=np.array(omegas)
            vh=omegas/(n*f*f);empvar=float(shifts.var(ddof=1));ratio=float(vh.mean()/empvar)
            oe=abs(omegas/.0099-1);fe=abs(f/ftrue-1)
            covered=abs(shifts)<=stats.norm.ppf(.975)*np.sqrt(vh)
            blo,bhi=wilson(int(covered.sum()),500)
            boot_oe=np.median(oe[bootstrap],axis=1);boot_fe=np.median(fe[bootstrap],axis=1)
            boot_ratio=vh[bootstrap].mean(axis=1)/shifts[bootstrap].var(axis=1,ddof=1)
            quant=lambda a:list(map(float,np.quantile(a,[.025,.975])))
            ol,oh=quant(boot_oe);fl,fh=quant(boot_fe);vl,vu=quant(boot_ratio)
            summary=dict(law=law,n=n,finite_histories=500,
                median_omega_relative_error=float(np.median(oe)),omega_error_lo=ol,omega_error_hi=oh,
                median_density_relative_error=float(np.median(fe)),density_error_lo=fl,density_error_hi=fh,
                empirical_variance=empvar,mean_variance_hat=float(vh.mean()),variance_ratio=ratio,
                variance_ratio_lo=vl,variance_ratio_hi=vu,
                coverage=float(covered.mean()),coverage_lo=blo,coverage_hi=bhi,
                omega_pass=bool(np.median(oe)<.15),density_pass=bool(np.median(fe)<.15),
                variance_ratio_pass=bool(.85<=ratio<=.98),coverage_alternative_pass=bool(.85<=covered.mean()<=.98),
                omega_floor_fraction=float(np.mean(omegas<=.00099)),
                sbg_available=0)
            summary['joint_pass']=summary['omega_pass'] and summary['density_pass'] and summary['variance_ratio_pass']
            summaries.append(summary)
            # Exact conditional loss over the fixed original 1024 test states.
            corrected=np.mean(e.expected_loss(law,oracle[None,:]-shifts[:,None],test_sigma[None,:]),axis=1)
            for bias in (0.,.25*np.sqrt(1e-5/(1-.10-.85))):
                raw=float(np.mean(e.expected_loss(law,oracle+bias,test_sigma)))
                removed=raw-oracle_risk;delta=corrected-raw
                predicted=-removed+.0099/(2*n*finite_f)
                pred_long=-removed+.0099/(2*n*ftrue)
                observed=float(delta.mean());mcse=float(delta.std(ddof=1)/np.sqrt(500))
                finite_expansion.append(dict(law=law,n=n,bias=bias,B_true_reference_states=removed,
                     f_reference_states=finite_f,f_long_reference=ftrue,observed_delta=observed,
                     observed_delta_x1e4=observed*1e4,prediction=predicted,prediction_x1e4=predicted*1e4,
                     prediction_long_density=pred_long,
                     corrected_two_density_prediction=-removed+.0099*finite_f/(2*n*ftrue**2),
                     mcse=mcse,discrepancy=observed-predicted,
                     relative_discrepancy=(observed-predicted)/abs(predicted) if abs(predicted)>=1e-12 else None,
                     n_scaled_discrepancy=n*(observed-predicted)))
                finite_losses.extend(dict(law=law,n=n,bias=bias,rep=rep,static_loss=float(corrected[rep]),
                                          raw_loss=raw,delta=float(delta[rep])) for rep in range(500))
                long_raw=mixture_risk(bias)
                long_B=long_raw-long_oracle_risk
                long_corrected=poly127(-shifts)
                long_delta=long_corrected-long_raw
                long_pred=-long_B+.0099/(2*n*ftrue)
                mean=float(long_delta.mean());se=float(long_delta.std(ddof=1)/np.sqrt(500))
                expansion.append(dict(law=law,n=n,bias=bias,B_true_reference_states=long_B,
                     f_reference_states=ftrue,f_long_reference=ftrue,observed_delta=mean,
                     observed_delta_x1e4=mean*1e4,prediction=long_pred,prediction_x1e4=long_pred*1e4,
                     prediction_long_density=long_pred,mcse=se,discrepancy=mean-long_pred,
                     relative_discrepancy=(mean-long_pred)/abs(long_pred) if abs(long_pred)>=1e-12 else None,
                     n_scaled_discrepancy=n*(mean-long_pred),reference_states=1000000,
                     evaluation='common_long_reference_marginal'))
                loss_histories.extend(dict(law=law,n=n,bias=bias,rep=rep,
                     static_loss=float(long_corrected[rep]),raw_loss=long_raw,
                     delta=float(long_delta[rep])) for rep in range(500))
            print(law,n,'density median error',round(summary['median_density_relative_error'],4),
                  'variance ratio',round(ratio,4),'admission cell',summary['joint_pass'],flush=True)
    pd.DataFrame(all_rows).to_csv(SYN/'estimators.csv',index=False)
    pd.DataFrame(truth_rows).to_csv(SYN/'truth.csv',index=False)
    pd.DataFrame(summaries).to_csv(SYN/'validation_summary.csv',index=False)
    pd.DataFrame(expansion).to_csv(SYN/'expansion.csv',index=False)
    pd.DataFrame(loss_histories).to_csv(SYN/'loss_histories.csv',index=False)
    pd.DataFrame(finite_expansion).to_csv(SYN/'finite_reference_diagnostic.csv',index=False)
    pd.DataFrame(finite_losses).to_csv(SYN/'finite_loss_histories.csv',index=False)
    passed=admission(summaries) and all(r['truth_precision_pass'] for r in truth_rows)
    report=dict(status='PASS' if passed else 'FAIL',financial_panel_access_allowed=passed,
          financial_panel_reads=0,requested_deliverables_status='NOT_RUN',
          reason='Synthetic admission failed' if not passed else 'Admission passed; financial execution pending',
          required_cells=[r for r in summaries if r['n'] in (700,1000)],
          truth_precision=[dict(law=r['law'],passed=r['truth_precision_pass']) for r in truth_rows],
          secondary_SBG='OUT_OF_RANGE at all requested sizes; sensitivity unavailable')
    dump(SYN/'admission.json',report)
    dump(SYN/'runtime_checks.json',dict(status='PASS',checks=runchecks,
          negative_controls=len(runchecks)))
    print(json.dumps(dict(synthetic_admission=report['status'],financial_panel_reads=0)),flush=True)


if __name__=='__main__':
    run()
