"""Separate numerical selector variation, KDE error and quantile-location error."""
import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from common import ROOT,OUT,validate_lock,save_json

SETTINGS=('default_1000','tight_1000','tight_4096','tight_16384','solvercheck_16384')
TRANSFORMS=('identity','plus_bias','minus_bias','times_100','reflection')
KEYS=['law','n','rep','setting','transformation']

def density(x,c,bw):
    return float(np.mean(np.exp(-.5*((np.asarray(x)-c)/bw)**2))/(bw*np.sqrt(2*np.pi)))

def complete_frame(df):
    expected=set(itertools.product(('normal','t5'),(700,1000),(0,17,499),SETTINGS,TRANSFORMS))
    return len(df)==300 and not df.duplicated(KEYS).any() and set(map(tuple,df[KEYS].to_numpy()))==expected

def record_check(records,name,bad,good,predicate,required=True):
    try: bad_pass=bool(predicate(bad))
    except (ValueError,AssertionError,KeyError): bad_pass=False
    if bad_pass: raise AssertionError('Negative control escaped: '+name)
    try: passed=bool(predicate(good))
    except (ValueError,AssertionError,KeyError): passed=False
    records.append(dict(name=name,negative_control_rejected=True,status='PASS' if passed else 'FAIL'))
    if required and not passed: raise AssertionError('Required check failed: '+name)

def reference_density(law,s,sigma):
    if law=='normal':
        z=stats.norm.ppf(.01)-s/sigma
        return float(np.mean(stats.norm.pdf(z)/sigma))
    scale=np.sqrt(3/5);z=stats.t.ppf(.01,5)*scale-s/sigma
    return float(np.mean(stats.t.pdf(z/scale,5)/(scale*sigma)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--sj',type=Path,default=OUT/'sj')
    p.add_argument('--output',type=Path,default=OUT/'diagnostic');a=p.parse_args()
    validate_lock();a.output.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.sj/'summary.csv');records=[]
    record_check(records,'complete_unique_300_cells_missing',df.iloc[:-1],df,complete_frame)
    duplicate=pd.concat([df.iloc[:-1],df.iloc[:1]],ignore_index=True)
    record_check(records,'complete_unique_300_cells_duplicate',duplicate,df,complete_frame)
    record_check(records,'public_and_independent_reconstruction',False,
        (df.status.eq('OK') & df.reconstruction_status.eq('OK') & df.pair_counts_identical.eq(True)).all(),bool)
    record_check(records,'finite_positive_bandwidths_densities',np.array([0.]),df[['bw','f_C','f_cstar']].to_numpy(),
        lambda v:np.isfinite(v).all() and (v>0).all())
    record_check(records,'independent_root_reconstruction',1.,
        df.public_vs_independent_root_over_hmax.abs().max(),lambda x:x<1e-8)
    old=pd.read_csv(ROOT/'results/theory_loop/synthetic/estimators.csv')
    old=old[old.bias.eq(0)]
    truth=pd.read_csv(ROOT/'results/theory_loop/synthetic/truth.csv').set_index('law')
    scores={};refs={};stats_rows=[]
    for law in ('normal','t5'):
        with np.load(ROOT/f'results/theory_loop/synthetic/calibration_{law}.npz') as z:
            scores[law]=z['scores'].copy()
        with np.load(ROOT/f'results/theory_loop/synthetic/truth_{law}.npz') as z:
            sigma=z['sigma'].reshape(-1)
        f0=reference_density(law,0,sigma)
        record_check(records,law+'_reference_matches_original',f0*2,f0,
            lambda x:abs(x-truth.loc[law,'f_true'])<1e-12)
        for n in (700,1000):
            vals=old[(old.law==law)&(old.n==n)]
            assert len(vals)==500
            stats_rows.append(dict(law=law,n=n,histories=500,mean_C=vals.C.mean(),
                squared_bias=vals.C.mean()**2,variance_C=vals.C.var(ddof=1),mse_C=(vals.C**2).mean()))
            for rep in (0,17,499):
                row=vals[vals.rep==rep].iloc[0]
                refs[law,n,rep]=dict(f0=f0,fC=reference_density(law,row.C,sigma),old=row)
    variance=pd.DataFrame(stats_rows);variance.to_csv(a.output/'variance_reference.csv',index=False)
    vmap=variance.set_index(['law','n']).variance_C.to_dict();decomposition=[];equiv=[]
    for row in df.itertuples(index=False):
        ref=refs[row.law,row.n,row.rep];x=scores[row.law][row.rep,:row.n]
        scale=abs(row.transform_scale);bw=row.bw/scale;fC=row.f_C*scale;f0=row.f_cstar*scale
        direct=density(x,row.C_original,bw)
        record_check(records,'density_replay_'+str(len(decomposition)),direct*2,direct,
            lambda v:np.isclose(v,fC,rtol=1e-10,atol=1e-12))
        om=ref['old'].omega;vemp=vmap[row.law,row.n]
        decomposition.append(dict(law=row.law,n=row.n,rep=row.rep,setting=row.setting,transformation=row.transformation,
            C=row.C_original,bw_original_units=bw,fhat_C=fC,fhat_cstar=f0,f_ref_C=ref['fC'],f_ref_cstar=ref['f0'],
            total_ratio=fC/ref['f0'],kernel_ratio=fC/ref['fC'],location_ratio=ref['fC']/ref['f0'],
            relative_error_fhat_C=abs(fC/ref['f0']-1),relative_error_fhat_cstar=abs(f0/ref['f0']-1),
            omega_hat=om,omega_exact=.0099,empirical_variance_C=vemp,
            variance_both_estimated=om/(row.n*fC*fC),variance_exact_omega=.0099/(row.n*fC*fC),
            variance_reference_density=om/(row.n*ref['f0']**2),variance_both_reference=.0099/(row.n*ref['f0']**2)))
        if row.transformation!='identity':
            identity=df[(df.law==row.law)&(df.n==row.n)&(df.rep==row.rep)&(df.setting==row.setting)&df.transformation.eq('identity')].iloc[0]
            mappedx=row.transform_scale*x+row.transform_offset
            fixed=density(mappedx,row.C_mapped,abs(row.transform_scale)*identity.bw)*scale
            correct=density(x,row.C_original,identity.bw)
            # Actual dimensional-error mutant: bandwidth is not rescaled when x is multiplied.
            wrong=density(100*x,100*row.C_original,identity.bw)*100
            record_check(records,'fixed_bandwidth_equivariance_'+str(len(equiv)),wrong,fixed,
                lambda v:np.isclose(v,correct,rtol=1e-10,atol=1e-12))
            equiv.append(dict(law=row.law,n=row.n,rep=row.rep,setting=row.setting,transformation=row.transformation,
                selector_bw_relative_change=bw/identity.bw-1,selector_fC_relative_change=fC/identity.f_C-1,
                selector_fcstar_relative_change=f0/identity.f_cstar-1,
                fixed_kernel_relative_error=fixed/correct-1))
    dec=pd.DataFrame(decomposition);dec.to_csv(a.output/'density_decomposition.csv',index=False)
    equiv=pd.DataFrame(equiv);equiv.to_csv(a.output/'equivariance.csv',index=False)
    ratio=dec.kernel_ratio*dec.location_ratio
    record_check(records,'multiplicative_decomposition',ratio+1,ratio,
        lambda v:np.allclose(v,dec.total_ratio,rtol=1e-12,atol=0))
    comparisons=[]
    for name,left,right,tol in [('bins','tight_4096','tight_16384',.001),
                               ('root','tight_16384','solvercheck_16384',1e-6),
                               ('default_tolerance','default_1000','tight_1000',None),
                               ('total_numerical_change','default_1000','tight_16384',None)]:
        l=dec[dec.setting==left].set_index(['law','n','rep','transformation'])
        r=dec[dec.setting==right].set_index(['law','n','rep','transformation'])
        for key in l.index:
            vals={c:abs(l.loc[key,c]/r.loc[key,c]-1) for c in ('bw_original_units','fhat_C','fhat_cstar')}
            maximum=max(vals.values())
            if tol is not None:
                record_check(records,name+'_convergence_'+str(key),tol*2,maximum,lambda v:v<=tol,required=False)
            comparisons.append(dict(law=key[0],n=key[1],rep=key[2],transformation=key[3],comparison=name,
                left=left,right=right,tolerance=tol,max_relative_difference=maximum,
                status='DESCRIPTIVE' if tol is None else ('PASS' if maximum<=tol else 'FAIL'),**vals))
    cmp=pd.DataFrame(comparisons);cmp.to_csv(a.output/'numerical_comparisons.csv',index=False)
    identity=dec[dec.transformation=='identity'].copy()
    selected=identity.groupby(['law','n','setting']).agg(
        windows=('rep','size'),median_relative_error_at_C=('relative_error_fhat_C','median'),
        median_relative_error_at_cstar=('relative_error_fhat_cstar','median'),
        median_kernel_ratio=('kernel_ratio','median'),median_location_ratio=('location_ratio','median'))
    selected.to_csv(a.output/'selected_window_summary.csv')
    original_comparison=[]
    for row in identity[identity.setting=='default_1000'].itertuples():
        ref=refs[row.law,row.n,row.rep]['old']
        original_comparison.append(max(abs(row.fhat_C/ref.f_sj-1),abs(row.bw_original_units/ref.sj_bw-1)))
    record_check(records,'default_reproduces_original_selected_windows',1.,max(original_comparison),lambda v:v<1e-12)
    save_json(a.output/'checks.json',dict(status='COMPLETED_WITH_NUMERICAL_FAILURE' if cmp.status.eq('FAIL').any() else 'PASS',checks=records,
        required_checks_passed=True,numerical_convergence_failures=int(cmp.status.eq('FAIL').sum()),
        panel_status='NOT_RUN',old_admission='FAIL',interpretation='Twelve selected windows are diagnostic only.'))
    validate_lock();print('Diagnostic complete:',len(dec),'rows;',int(cmp.status.eq('FAIL').sum()),'numerical convergence failures.')

if __name__=='__main__': main()
