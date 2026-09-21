"""Fresh-process array and summary replay, with rejection-first controls."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from scipy import stats
from checks import record_check,sha,require_panel_admission

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/theory_loop';SYN=OUT/'synthetic'


def main():
    records=[]
    binding=json.loads((OUT/'execution_binding.json').read_text())['sources']
    bad=dict(binding);bad[next(iter(bad))]='0'*64
    record_check(records,'execution_sources_unchanged',bad,binding,
                 lambda d:all(sha(ROOT/p)==h for p,h in d.items()))
    data=pd.read_csv(SYN/'estimators.csv');summ=pd.read_csv(SYN/'validation_summary.csv')
    truth=pd.read_csv(SYN/'truth.csv');exp=pd.read_csv(SYN/'expansion.csv')
    losses=pd.read_csv(SYN/'loss_histories.csv')
    record_check(records,'complete_design',data.iloc[1:],data,
                 lambda d:len(d)==10000 and d.groupby(['law','n','bias']).size().eq(500).all())
    boot=np.load(SYN/'history_bootstrap_indices.npy')
    refboot=np.random.default_rng(2026091141).integers(0,500,size=(999,500))
    record_check(records,'bootstrap_seed_replay',boot+1,boot,lambda x:np.array_equal(x,refboot))
    close=lambda a,b:np.allclose(a,b,rtol=1e-11,atol=1e-13)
    sys.path.insert(0,str(ROOT/'research/r8_shape_cost'))
    from validate_simulation import reference_loss,quadrature_loss
    for law in ('normal','t5'):
        path=np.load(SYN/f'calibration_{law}.npz');scores=path['scores']
        long=np.load(SYN/f'truth_{law}.npz');ts=long['sigma']
        distribution=stats.norm() if law=='normal' else stats.t(5,scale=np.sqrt(3/5))
        z=distribution.ppf(.01)
        ftrue=float(np.mean(distribution.pdf(z)/ts))
        actual=float(truth[truth.law==law].iloc[0].f_true)
        record_check(records,'density_truth_'+law,actual*2,actual,lambda x:close(x,ftrue))
        record_check(records,'calibration_score_identity_'+law,scores+.01,scores,
                     lambda x:np.array_equal(x,z*path['sigma']-path['returns']))
        # Full R rerun, with an intentionally changed byte rejected first.
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/'sj.csv'
            subprocess.run(['Rscript','--vanilla',str(ROOT/'research/r8_theory_loop/sj_batch.R'),
                            str(SYN/f'scores_{law}.bin'),str(output),'2000','500'],check=True)
            original=(SYN/f'sj_{law}.csv').read_bytes()
            replay=output.read_bytes()
            record_check(records,'SJ_fresh_process_'+law,replay+b'x',replay,lambda b:b==original)
        test_sigma=np.load(SYN/f'test_sigma_{law}.npy');q=z*test_sigma
        grid=np.load(SYN/f'mixture_integration_{law}.npz')
        polynomial=np.polynomial.Chebyshev(grid['coef127'],domain=grid['domain'])
        # Independent direct integration of the stored long reference, not the producer.
        flat_sigma=ts.ravel();flat_oracle=z*flat_sigma
        def mixture(a):
            return float(reference_loss(law,flat_oracle+a,flat_sigma).mean())
        direct=np.array([mixture(a) for a in grid['probe']])
        record_check(records,'long_reference_integration_'+law,polynomial(grid['probe'])+.01,
                     polynomial(grid['probe']),lambda x:np.allclose(x,direct,rtol=4e-11,atol=4e-11))
        for n in (250,500,700,1000,2000):
            s=scores[:,:n];k=(99*(n+1)+99)//100
            c=np.sort(s,axis=1)[:,k-1]
            indicators=(s<=c[:,None]).astype(float)
            centered=indicators-indicators.mean(axis=1,keepdims=True)
            L=int(4*(n/100)**(2/9))
            omega=np.mean(centered*centered,axis=1)
            for lag in range(1,L+1):
                omega+=2*(1-lag/(L+1))*np.sum(centered[:,lag:]*centered[:,:-lag],axis=1)/n
            omega=np.maximum(omega,.00099)
            frame=data[(data.law==law)&(data.n==n)&(data.bias==0)].sort_values('rep')
            f=frame.f_sj.to_numpy();bw=frame.sj_bw.to_numpy()
            direct=np.mean(np.exp(-.5*((s-c[:,None])/bw[:,None])**2),axis=1)/(bw*np.sqrt(2*np.pi))
            columns=np.column_stack((frame.C,frame.omega,f))
            reference=np.column_stack((c,omega,direct))
            bad=columns.copy();bad[0,2]*=2
            record_check(records,f'estimator_replay_{law}_{n}',bad,columns,lambda x:close(x,reference))
            vh=omega/(n*f*f);oe=abs(omega/.0099-1);fe=abs(f/ftrue-1)
            ref=np.array([np.median(oe),np.median(fe),np.var(c,ddof=1),np.mean(vh),
                          np.mean(vh)/np.var(c,ddof=1),np.mean(abs(c)<=stats.norm.ppf(.975)*np.sqrt(vh))])
            row=summ[(summ.law==law)&(summ.n==n)].iloc[0]
            actual=row[['median_omega_relative_error','median_density_relative_error','empirical_variance',
                        'mean_variance_hat','variance_ratio','coverage']].to_numpy(dtype=float)
            bad=actual.copy();bad[1]+=.01
            record_check(records,f'summary_replay_{law}_{n}',bad,actual,lambda x:close(x,ref))
            interval_ref=np.concatenate([np.quantile(np.median(oe[boot],axis=1),[.025,.975]),
                         np.quantile(np.median(fe[boot],axis=1),[.025,.975]),
                         np.quantile(vh[boot].mean(axis=1)/c[boot].var(axis=1,ddof=1),[.025,.975])])
            actual=row[['omega_error_lo','omega_error_hi','density_error_lo','density_error_hi',
                        'variance_ratio_lo','variance_ratio_hi']].to_numpy(dtype=float)
            record_check(records,f'bootstrap_summary_{law}_{n}',actual+.01,actual,
                         lambda x:close(x,interval_ref))
            for bias in data[(data.law==law)&(data.n==n)].bias.unique():
                g=data[(data.law==law)&(data.n==n)&(data.bias==bias)].sort_values('rep')
                corrected_c=c+bias
                h=np.maximum(0,n*corrected_c**2-omega/f**2)
                lam=f*f*h/(omega+f*f*h)
                def pinball(x):
                    return .01*np.maximum(x,0)+.99*np.maximum(-x,0)
                delta=(pinball(corrected_c[:,None]-(s+bias))-pinball(-(s+bias))).mean(axis=1)
                actual=np.column_stack((g.h2,g.lambda_hat,g.signed_ecdf_loss,g.prediction))
                ref=np.column_stack((h,lam,delta,delta+omega/(2*n*f)))
                bad=actual.copy();bad[0,3]+=.01
                record_check(records,f'derived_replay_{law}_{n}_{bias:g}',bad,actual,lambda x:close(x,ref))
                raw=mixture(bias)
                corrected=polynomial(-c)
                g=losses[(losses.law==law)&(losses.n==n)&np.isclose(losses.bias,bias)].sort_values('rep')
                record_check(records,f'loss_replay_{law}_{n}_{bias:g}',g.delta.to_numpy()+.01,
                             g.delta.to_numpy(),lambda x:close(x,corrected-raw))
                row=exp[(exp.law==law)&(exp.n==n)&np.isclose(exp.bias,bias)].iloc[0]
                B=raw-mixture(0.)
                prediction=-B+.0099/(2*n*ftrue)
                obs=float(np.mean(corrected-raw))
                ref=np.array([B,obs,prediction,obs-prediction,(obs-prediction)/abs(prediction)])
                actual=row[['B_true_reference_states','observed_delta','prediction','discrepancy',
                            'relative_discrepancy']].to_numpy(dtype=float)
                record_check(records,f'expansion_summary_{law}_{n}_{bias:g}',actual+.01,actual,
                             lambda x:close(x,ref))
            # Independent numerical integration at one actual saved threshold.
            threshold=float(q[0]-c[0]);scale=float(test_sigma[0])
            val=float(reference_loss(law,threshold,scale))
            quad=quadrature_loss(law,threshold,scale,.01)
            record_check(records,f'quadrature_{law}_{n}',val*2,val,
                         lambda x:np.isclose(x,quad,rtol=4e-11,atol=4e-11))
    record_check(records,'SBG_all_failures_retained',['OK']*len(data),data.sbg_status.tolist(),
                 lambda x:len(x)==10000 and all(s=='OUT_OF_RANGE' for s in x))
    actual=json.loads((SYN/'admission.json').read_text())
    required=summ[summ.n.isin([700,1000])]
    correct=bool((required.median_omega_relative_error<.15).all()
         and (required.median_density_relative_error<.15).all()
         and required.variance_ratio.between(.85,.98).all())
    record_check(records,'admission_verdict_replay',not correct,
                 actual['status']=='PASS',lambda x:x==correct)
    # Observe the execution barrier refusing the real failed admission.
    try:
        require_panel_admission()
        denied=False
    except RuntimeError:
        denied=True
    record_check(records,'financial_execution_barrier',False,denied,lambda x:x is True)
    preserved=json.loads((OUT/'preserved_R8.json').read_text());bad=dict(preserved)
    bad[next(iter(bad))]='0'*64
    record_check(records,'R8_still_unchanged',bad,preserved,
                 lambda d:all(sha(ROOT/p)==h for p,h in d.items()))
    report=dict(computational_checks='PASS',statistical_admission=actual['status'],
                checks=records,negative_controls=len(records),preserved_R8_files=len(preserved),
                financial_panel_reads=0)
    (SYN/'independent_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))


if __name__=='__main__':
    main()
