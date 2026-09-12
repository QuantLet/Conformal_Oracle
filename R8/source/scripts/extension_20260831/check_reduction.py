#!/usr/bin/env python3
"""Reconstruct every TSFM quantile from native samples or saved closure parameters."""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import t
from panel_statistics import ROOT,ALPHAS
from reduce_native import sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ready-only',action='store_true');a=ap.parse_args();rows=[]
    for model in ['moirai','moirai2','timesfm25','lagllama']:
        for ret in sorted((ROOT/'data/returns').glob('*.csv')):
            file=ROOT/'data'/model/(ret.stem+'.parquet');pf=ROOT/'parameters'/model/file.name
            if not file.exists() and a.ready_only:continue
            pred=pd.read_parquet(file);par=pd.read_parquet(pf)
            meta=json.loads((ROOT/'provenance'/model/(ret.stem+'.json')).read_text())
            assert meta['forecast_sha256']==sha(file) and meta['parameters_sha256']==sha(pf)
            native=(ROOT.parents[1]/meta['native_directory']) if 'native_directory' in meta else ROOT/'native'/model/ret.stem
            if model in ['moirai','lagllama']:
                chunks=[]
                for path in sorted(native.glob('*.npz')):
                    assert sha(path)==meta['native_chunks'][path.name]
                    with np.load(path) as z:
                        chunks.append(np.percentile(z[meta.get('native_array','native')],np.array(ALPHAS)*100,axis=1).T)
                expected=np.concatenate(chunks)
            else:
                expected=t.ppf(np.array(ALPHAS)[None,:],par.nu_used.to_numpy()[:,None],
                               loc=par.mu.to_numpy()[:,None],scale=par.sigma_used.to_numpy()[:,None])
                if model=='timesfm25':expected[:,-1]=par['q_0.1']
            actual=pred[[f'VaR_{alpha:g}' for alpha in ALPHAS]].to_numpy()
            delta=float(np.max(np.abs(actual-expected)))
            rows.append(dict(model=model,asset=ret.stem,rows=len(pred),cells=actual.size,max_abs=delta))
            assert np.array_equal(expected,actual),(model,ret.stem,delta)
            print(model,ret.stem,'PASS',len(pred)*4,flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'quality/quantile_replay.csv',index=False)
    print('PASS',sum(r['cells'] for r in rows),'quantile cells',flush=True)


if __name__=='__main__':main()
