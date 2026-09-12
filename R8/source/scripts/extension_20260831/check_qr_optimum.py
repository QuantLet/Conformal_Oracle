#!/usr/bin/env python3
"""Audit affine calibration against the global pinball-loss linear programme."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse
from panel_statistics import ROOT,MODELS,load_pair


def fit_qr(y,q,alpha=.01):
    # Scaling y and q improves conditioning; the intercept is mapped back.
    scale=max(float(np.std(y)),1e-8);Y=y/scale;Q=q/scale;n=len(y)
    X=sparse.csc_matrix(np.c_[np.ones(n),Q]);I=sparse.eye(n,format='csc')
    A=sparse.hstack([X,I,-I],format='csc')
    r=linprog(np.r_[0.,0.,np.full(n,alpha),np.full(n,1-alpha)],A_eq=A,b_eq=Y,
              bounds=[(None,None)]*2+[(0,None)]*(2*n),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not r.success:raise ValueError(r.message)
    return np.array([r.x[0]*scale,r.x[1]]),r


def main():
    rows=[]
    for model in MODELS:
        for path in sorted((ROOT/'data/returns').glob('*.csv')):
            asset=path.stem;y,f=load_pair(model,asset);n=int(.7*len(y));q=f['VaR_0.01'].to_numpy()
            old=json.loads((ROOT/'posthoc'/f'{model}__{asset}.json').read_text())['fits']['qr']
            beta,res=fit_qr(y[:n],q[:n]);e=y[:n]-beta[0]-beta[1]*q[:n]
            value=float(np.sum((.01-(e<0))*e))
            rows.append(dict(model=model,asset=asset,old_loss=old['loss'],optimal_loss=value,improvement=old['loss']-value,
                             intercept=beta[0],slope=beta[1],max_test_quantile_change=float(np.max(np.abs((beta[0]-old['params'][0])+(beta[1]-old['params'][1])*q[n:])))))
    pd.DataFrame(rows).to_csv(ROOT/'quality/qr_global_optimum.csv',index=False)
    print(pd.DataFrame(rows)[['improvement','max_test_quantile_change']].describe().to_string(),flush=True)


if __name__=='__main__':main()
