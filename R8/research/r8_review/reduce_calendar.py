"""Reduce corrected native outputs and isolate calendar effects at fixed laws."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
from scipy.stats import t

PROJECT=Path(__file__).resolve().parents[2]
OLD=PROJECT/'artifacts/extension_20260831'
ROOT=PROJECT/'artifacts/review_20260909/calendar'
sys.path.insert(0,str(PROJECT/'source/scripts/extension_20260831'))
from panel_statistics import ALPHAS,qshift,scores


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ready-only',action='store_true');a=ap.parse_args()
    summaries=[];metrics=[];bindings={}
    for returns in sorted((OLD/'data/returns').glob('*.csv')):
        asset=returns.stem;folder=ROOT/'native'/asset
        if not (folder/'complete.json').exists():
            if a.ready_only:continue
            raise RuntimeError(f'Incomplete calendar inference: {asset}')
        complete=json.loads((folder/'complete.json').read_text())
        assert sha(folder/'binding.json')==complete['binding_sha256']
        old=OLD/'data/lagllama'/f'{asset}.parquet'
        backup=ROOT/'legacy_forecasts'/old.name;backup.parent.mkdir(exist_ok=True)
        if not backup.exists():shutil.copy2(old,backup)
        original=pd.read_parquet(backup)
        ret=pd.read_csv(returns,index_col='date',parse_dates=True).log_return
        rows=[];params=[];chunks={}
        for p in sorted(folder.glob('*.npz')):
            meta=json.loads(p.with_suffix('.json').read_text())
            assert sha(p)==meta['sha256'] and meta['legacy_samples_exact']
            chunks[p.name]=meta['sha256']
            with np.load(p) as z:
                index=pd.DatetimeIndex(z['date'].astype('datetime64[ns]'),name='date')
                assert index.equals(ret.index[z['positions']])
                samples=z['actual_samples']
                quantiles=np.percentile(samples,np.array(ALPHAS)*100,axis=1).T
                values=pd.DataFrame({'mean':samples.mean(axis=1),'std':samples.std(axis=1)},index=index)
                for j,alpha in enumerate(ALPHAS):values[f'VaR_{alpha:g}']=quantiles[:,j]
                rows.append(values)
                block=pd.DataFrame(index=index)
                for mode in ['legacy','actual']:
                    parameters=z[f'{mode}_parameters'].astype(float)
                    for j,name in enumerate(['df','mu','scale']):block[f'{mode}_{name}']=parameters[:,j]
                    for alpha in ALPHAS:
                        block[f'{mode}_{alpha:g}']=t.ppf(alpha,parameters[:,0],loc=parameters[:,1],scale=parameters[:,2])
                params.append(block)
        corrected=pd.concat(rows);law=pd.concat(params)
        assert corrected.index.equals(original.index) and len(corrected)==complete['rows']
        assert corrected.index.equals(ret.index[512:])
        forecast=ROOT/'data/lagllama'/f'{asset}.parquet';forecast.parent.mkdir(parents=True,exist_ok=True)
        corrected.to_parquet(forecast)
        parpath=ROOT/'parameters'/f'{asset}.parquet';parpath.parent.mkdir(exist_ok=True)
        law.to_parquet(parpath)
        y=ret.loc[corrected.index].to_numpy();nc=int(.7*len(y))
        for alpha in ALPHAS:
            oldq=original[f'VaR_{alpha:g}'].to_numpy();newq=corrected[f'VaR_{alpha:g}'].to_numpy()
            oldlaw=law[f'legacy_{alpha:g}'].to_numpy();newlaw=law[f'actual_{alpha:g}'].to_numpy()
            denom=np.maximum(np.abs(oldlaw),1e-12)
            summaries.append({'asset':asset,'alpha':alpha,'n':len(y),'n_cal':nc,
                'changed_distribution_quantiles':int((np.abs(newlaw-oldlaw)>1e-12).sum()),
                'median_absolute_law_change_relative_to_old_threshold':float(np.median(np.abs(newlaw-oldlaw)/denom)),
                'p95_absolute_law_change_relative_to_old_threshold':float(np.quantile(np.abs(newlaw-oldlaw)/denom,.95)),
                'median_signed_law_change':float(np.median(newlaw-oldlaw)),
                'legacy_static_shift':qshift(oldq[:nc]-y[:nc],alpha),
                'actual_static_shift':qshift(newq[:nc]-y[:nc],alpha)})
            for mode,q in [('Legacy-sampled',oldq),('Actual-sampled',newq),('Legacy-law',oldlaw),('Actual-law',newlaw)]:
                shift=qshift(q[:nc]-y[:nc],alpha)
                for method,p in [('Raw',q[nc:]),('Static',q[nc:]-shift)]:
                    metrics.append({'asset':asset,'alpha':alpha,'interface':mode,'method':method,**scores(y[nc:],p,alpha)})
        bindings[asset]={'native_binding_sha256':sha(folder/'binding.json'),'chunks':chunks,
                         'return_sha256':sha(returns),'legacy_forecast_sha256':sha(backup),
                         'actual_forecast_sha256':sha(forecast),'law_parameters_sha256':sha(parpath)}
    pd.DataFrame(summaries).to_csv(ROOT/'calendar_effects.csv',index=False)
    pd.DataFrame(metrics).to_csv(ROOT/'calendar_scores.csv',index=False)
    (ROOT/'reduction.json').write_text(json.dumps({'producer_sha256':sha(__file__),'assets':bindings,
        'complete':len(bindings)==24,'same_test_support':True,'n_assets':len(bindings)},indent=2)+'\n')
    print('Calendar assets reduced:',len(bindings),'/ 24',flush=True)


if __name__=='__main__':main()
