"""Full fresh fitting replay plus independent forecast/recursion reconstruction."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model
from prepare import PROJECT,OUT,ASSETS,sha,prepare


def main():
    warnings.filterwarnings('ignore');prepare(check=True)
    refs={};rows=0;years=0;fixed=0;flags=0;fallbacks=0
    levels=np.array([.01,.025,.05,.1]);z0=stats.norm.ppf(levels)
    for asset in ASSETS:
        rp=OUT/'data/returns'/f'{asset}.csv';ret=pd.read_csv(rp,index_col='date',parse_dates=True).log_return;y=ret.to_numpy()
        for name in ['hs','gjr_t']:
            fp=OUT/'data/benchmarks'/f'{asset}_{name}.parquet';pp=OUT/'parameters'/name/f'{asset}.parquet'
            meta=OUT/'provenance'/name/f'{asset}.json';done=json.loads(meta.read_text())
            assert done['binding']['input_sha256']==sha(rp) and done['forecast_sha256']==sha(fp) and done['parameters_sha256']==sha(pp)
            assert done['binding']['producer_sha256']==sha(PROJECT/'source/scripts/extension_20260831/classical.py')
            f=pd.read_parquet(fp);p=pd.read_parquet(pp);assert f.index.equals(ret.index[250:]) and p.index.equals(f.index)
            for path in [fp,pp]:pd.testing.assert_frame_equal(pd.read_parquet(path),pd.read_parquet(OUT/'classical_replay'/path.relative_to(OUT)),check_exact=True)
            lastnu=np.nan
            for j,(date,row) in enumerate(p.iterrows()):
                t=j+250;w=y[t-250:t]
                assert row.context_start==ret.index[t-250] and row.context_end==ret.index[t-1]
                assert row.context_sha256==hashlib.sha256(w.astype('<f8').tobytes()).hexdigest()
                if name=='hs':
                    ordered=np.sort(w);h=(len(w)-1)*levels;lo=h.astype(int);weight=h-lo
                    pred=ordered[lo]+weight*(ordered[np.ceil(h).astype(int)]-ordered[lo])
                else:
                    if row.get('t_rejected',True)==False:prefix='t_';dist='t'
                    elif row.get('normal_rejected',True)==False:prefix='normal_';dist='normal'
                    else:prefix=None
                    if prefix:
                        names=['mu','omega','alpha[1]','gamma[1]','beta[1]']+(['nu'] if dist=='t' else [])
                        parameters=np.array([row[prefix+n] for n in names])
                        am=arch_model(ret.iloc[t-250:t]*100,vol='GARCH',p=1,o=1,q=1,dist=dist)
                        fc=am.fix(parameters).forecast(horizon=1,reindex=False)
                        mu=float(fc.mean.iloc[-1,0])/100;sd=float(np.sqrt(fc.variance.iloc[-1,0]))/100
                        nu=parameters[-1] if dist=='t' else np.nan;fixed+=1
                    else:mu=0.;sd=float(np.std(w*100,ddof=1))/100;nu=np.nan;fallbacks+=1
                    if np.isfinite(nu) and nu>2.10:lastnu=nu
                    if np.isfinite(lastnu):np.testing.assert_allclose(row.nu_used,lastnu,rtol=0,atol=0)
                    else:assert np.isnan(row.nu_used)
                    z=stats.t.ppf(levels,lastnu)/np.sqrt(lastnu/(lastnu-2)) if np.isfinite(lastnu) else z0
                    pred=mu+sd*z
                np.testing.assert_allclose(f.loc[date,[f'VaR_{a:g}' for a in levels]],pred,rtol=1e-11,atol=1e-13)
                rows+=1
            refs[str(meta.relative_to(OUT))]=sha(meta)
        for name in ['CAViaR-AS','GAS-t']:
            base=OUT/'dynamic'/f'{name}__{asset}';replay=OUT/'dynamic_replay'/f'{name}__{asset}'
            done=json.loads((base/'complete.json').read_text());assert done['binding']['input_sha256']==sha(rp)
            assert done['binding']['producer_sha256']==sha(PROJECT/'research/r8_external/dynamic.py')
            assert done['binding']['original_sha256']==sha(PROJECT/'source/analysis/phase3_dynamic/run_dynamic_var.py')
            assert done['forecast_sha256']==sha(base/'forecasts.parquet')
            pd.testing.assert_frame_equal(pd.read_parquet(base/'forecasts.parquet'),pd.read_parquet(replay/'forecasts.parquet'),check_exact=True)
            for year in range(2000,2027):
                info=json.loads((base/f'{year}.json').read_text());assert info==json.loads((replay/f'{year}.json').read_text())
                assert done['years'][str(year)]==sha(base/f'{year}.json')
                f=pd.read_parquet(base/f'{year}.parquet');assert sha(base/f'{year}.parquet')==info['forecast_sha256']
                pd.testing.assert_frame_equal(f,pd.read_parquet(replay/f'{year}.parquet'),check_exact=True)
                positions=np.flatnonzero(ret.index.year==year);start=int(positions[0]);stop=int(positions[-1])+1
                assert info['context_end']==str(ret.index[start-1].date()) and info['context_start']==str(ret.index[start-1250].date())
                assert info['context_sha256']==hashlib.sha256(y[start-1250:start].astype('<f8').tobytes()).hexdigest()
                z=y[start-1250:stop];th=np.array(info['theta']);ref=np.empty(len(z))
                if name=='CAViaR-AS':
                    ref[0]=info['initial_quantile']
                    for i in range(1,len(z)):ref[i]=th[0]+th[1]*ref[i-1]+th[2]*max(z[i-1],0)+th[3]*max(-z[i-1],0)
                    np.testing.assert_array_equal(f['VaR_0.01'],ref[1250:])
                else:
                    ref[0]=np.log(max(np.std(z[:250]),1e-8))
                    for i in range(1,len(z)):
                        e=z[i-1]/np.exp(ref[i-1]);s=(th[3]+1)/(th[3]+e**2)*e**2-1
                        ref[i]=th[0]+th[2]*ref[i-1]+th[1]*s
                    sd=np.exp(ref[1250:]);np.testing.assert_allclose(f.student_t_scale,sd,rtol=5e-13,atol=1e-15)
                    np.testing.assert_allclose(f['VaR_0.01'],sd*stats.t.ppf(.01,th[3]),rtol=5e-13,atol=1e-15)
                flags+=int(not info['selected_success']);years+=1
            refs[str((base/'complete.json').relative_to(OUT))]=sha(base/'complete.json')
        print(asset,'base replay and independent reconstruction passed',flush=True)
    record=dict(status='passed',producer_sha256=sha(__file__),admission_sha256=sha(OUT/'admission.json'),
                classical_forecast_rows_reconstructed=rows,fixed_parameter_gjr_reconstructions=fixed,
                gjr_window_sd_fallbacks=fallbacks,dynamic_yearly_fits_replayed=years,dynamic_selected_nonconvergence_flags=flags,
                base_series=48,full_fresh_replays=48,receipts=refs)
    (OUT/'base_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print({k:v for k,v in record.items() if k!='receipts'},flush=True)


if __name__=='__main__':main()
