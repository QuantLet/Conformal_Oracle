"""Replay, equivariance and independent iid integration checks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import quad
import engine as e


def main():
    results=e.OUT/'results';complete=json.loads((results/'complete.json').read_text())
    assert all(e.sha(results/f)==h for f,h in complete['outputs'].items())
    data=pd.read_parquet(results/'replications.parquet');summary=pd.read_csv(results/'summary.csv')
    inputs=np.load(e.OUT/'paths.npz');pathmeta=json.loads((e.OUT/'paths.json').read_text())
    assert not any(pathmeta['cdf_endpoint_clipping'].values())
    assert e.sha(e.OUT/'paths.npz')==pathmeta['sha256']
    assert len(set(pathmeta['seeds'].values()))==len(pathmeta['seeds'])
    for kind in ('normal','t5'):
        _,sigma=e.old.garch(kind,1024,pathmeta['seeds'][f'test/{kind}'])
        assert np.array_equal(sigma,inputs[f'test_sigma_{kind}'])
    # Fully regenerate all independent paths from their recorded seeds.
    for rep in range(e.REPS):
        eps=np.random.default_rng(pathmeta['seeds'][f'ar/{rep}']).standard_normal(1000)
        assert np.array_equal(eps,inputs['ar_innovations'][rep])
        for phi in (0.,.5,.8):
            z=e.ar_path(eps,phi);assert np.array_equal(z,inputs[f'ar_z_{phi:g}'][rep])
            for kind in ('normal','t5'):
                y,_=e.marginal(z,kind);assert np.array_equal(y,inputs[f'ar_{kind}_{phi:g}'][rep])
        for kind in ('normal','t5'):
            y,s=e.old.garch(kind,1000,pathmeta['seeds'][f'garch/{kind}/{rep}'])
            assert np.array_equal(y,inputs[f'garch_y_{kind}'][rep])
            assert np.array_equal(s,inputs[f'garch_sigma_{kind}'][rep])
    # The two marginal transformations have exactly the same tail counts.
    keys=['phi','n_cal','alpha','truth','replication','method']
    ar=data[data.module=='ar']
    a=ar[ar.innovation=='normal'].set_index(keys).oracle_tail_count
    b=ar[ar.innovation=='t5'].set_index(keys).oracle_tail_count
    assert a.equals(b)
    # Full-sample translation/fallback checks on stored results.
    max_equivariance=0.
    for module in ('ar','garch'):
        df=data[data.module==module]
        ix=['innovation','phi','n_cal','alpha','replication','method']
        a=df[df.truth=='none'].set_index(ix).expected_QS
        b=df[df.truth=='constant'].set_index(ix).expected_QS
        names=['Shift-CP','Shift-ERM','POT80-Shift','POT90-Shift']
        if module=='garch':names+=['State2-ERM','State4-ERM','State-L1','State-L1-clipped']
        delta=(a-b).reset_index();delta=delta[delta.method.isin(names)]
        maximum=float(delta.expected_QS.abs().max());assert maximum<1e-12
        max_equivariance=max(max_equivariance,maximum)
    ix=['module','innovation','phi','n_cal','alpha','truth','replication']
    table=data[data.n_cal==125].pivot(index=ix,columns='method',values='expected_QS')
    assert np.max(np.abs(table['POT90-Shift']-table['Shift-CP']))<1e-12
    g=table.dropna(subset=['POT90-Vol'])
    assert np.max(np.abs(g['POT90-Vol']-g['Vol-CP']))<1e-12
    # Independent exact iid order-statistic risk integration, without sample paths.
    iid=[]
    for kind in ('normal','t5'):
        for n in e.SIZES:
            for alpha in e.ALPHAS:
                for method in ('Shift-CP','Shift-ERM'):
                    k=int(np.ceil((n+1)*(1-alpha))) if method=='Shift-CP' else int(np.ceil(n*(1-alpha)))
                    a=n+1-k;b=k
                    def integrand(u):
                        q=e.old.conditional_quantile(kind,e.V0,u)
                        return e.old.expected_loss(kind,q,e.V0,alpha)*stats.beta.pdf(u,a,b)
                    exact,error=quad(integrand,0,1,epsabs=1e-11,epsrel=1e-9,limit=300,
                                     points=[a/(a+b),min(.1,5*a/(a+b))])
                    row=summary[(summary.module=='ar')&(summary.innovation==kind)&(summary.phi==0)&
                        (summary.n_cal==n)&(summary.alpha==alpha)&(summary.truth=='none')&(summary.method==method)].iloc[0]
                    iid.append({'innovation':kind,'n_cal':n,'alpha':alpha,'method':method,
                        'integrated_expected_QS':exact,'quadrature_error':error,
                        'simulation_expected_QS':row.mean_expected_QS,'simulation_MCSE':row.expected_QS_MCSE,
                        'difference_in_MCSE':(row.mean_expected_QS-exact)/row.expected_QS_MCSE})
    pd.DataFrame(iid).to_csv(results/'iid_integration_check.csv',index=False)
    replays=[]
    for key in ('ar_normal_0.8_125_000_025','ar_t5_0.8_1000_475_500',
                'garch_normal_0_1000_000_025','garch_t5_0_1000_475_500'):
        old=e.OUT/'blocks'/key;new=e.OUT/'replay'/key
        assert (new/'complete.json').exists()
        assert (old/'fits.jsonl').read_bytes()==(new/'fits.jsonl').read_bytes()
        pd.testing.assert_frame_equal(pd.read_csv(old/'replications.csv'),pd.read_csv(new/'replications.csv'),check_exact=True)
        x=np.load(old/'moments.npz');y=np.load(new/'moments.npz')
        assert x.files==y.files
        assert all(np.array_equal(x[k],y[k]) for k in x.files)
        replays.append(key)
    before=json.loads((e.OUT/'before.json').read_text())
    assert all(e.sha(e.PROJECT/f)==h for f,h in before['canonical'].items())
    record={'independent_calibration_paths_regenerated':1500,'test_state_paths_regenerated':2,'cdf_endpoint_clipping':0,
            'identical_normal_t5_tail_counts':True,'max_location_equivariance_QS_error':max_equivariance,
            'POT90_n125_equals_CP':True,'iid_integrals':len(iid),
            'maximum_absolute_iid_MC_standardised_error':max(abs(r['difference_in_MCSE']) for r in iid),
            'max_quadrature_error':max(r['quadrature_error'] for r in iid),
            'exact_fresh_process_replay_blocks':replays,'canonical_unchanged':len(before['canonical']),
            'producer_sha256':e.sha(__file__),'aggregate_receipt_sha256':e.sha(results/'complete.json')}
    (e.OUT/'validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
