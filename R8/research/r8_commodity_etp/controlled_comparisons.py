"""Matched-loss correction families and past-only deployment comparisons."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'source/scripts/extension_20260831'))
from panel_statistics import ROOT, MODELS, ALPHAS, load_pair, qshift, scores
from posthoc import rolling

ROOT = PROJECT/'artifacts/r8_commodity_etp'
MODELS = {k:v for k,v in MODELS.items() if k not in ['TimesFM-2.5','Moirai-2.0']}
OUT = ROOT/'controlled'
SIZES = [125,250,500,1000]
WINDOWS = [125,250,500]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loss(y, q, alpha):
    error = y-q
    return (alpha-(error<0))*error


def weighted_quantile(values, weights, p):
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    return float(values[order[min(np.searchsorted(cumulative, p*cumulative[-1], side='left'), len(order)-1)]])


def qr_fit(X, target, alpha):
    n,p = X.shape
    objective = np.r_[np.zeros(p), np.full(n,alpha), np.full(n,1-alpha)]
    A = sparse.hstack([sparse.csc_matrix(X), sparse.eye(n,format='csc'), -sparse.eye(n,format='csc')], format='csc')
    fit = linprog(objective, A_eq=A, b_eq=target,
                  bounds=[(None,None)]*p+[(0,None)]*(2*n), method='highs',
                  options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not fit.success:
        raise RuntimeError(f'Quantile LP failed: {fit.message}')
    beta = fit.x[:p]
    primal = float(np.sum(loss(target, X@beta, alpha)))
    assert abs(primal-fit.fun) < 1e-6*max(1,abs(fit.fun))
    return beta, {'objective':primal,'iterations':int(fit.nit),'success':True}


def candidates(y, q, sigma, test_q, test_sigma, alpha):
    s = q-y
    ref = float(np.median(sigma))
    assert ref > 0 and (sigma>0).all() and (test_sigma>0).all()
    plain = float(np.quantile(s,1-alpha,method='inverted_cdf'))
    cp = qshift(s,alpha)
    normalised = qshift(s/sigma,alpha)
    weighted = weighted_quantile(s/sigma,sigma,1-alpha)
    predictions = {'Raw':test_q, 'Shift-CP':test_q-cp, 'Shift-ERM':test_q-plain,
                   'Vol-CP':test_q-normalised*test_sigma, 'Vol-ERM':test_q-weighted*test_sigma}
    train = {'Raw':q, 'Shift-CP':q-cp, 'Shift-ERM':q-plain,
             'Vol-CP':q-normalised*sigma, 'Vol-ERM':q-weighted*sigma}
    params = {'shift_cp':cp,'shift_erm':plain,'vol_cp':normalised,'vol_erm':weighted,'reference_scale':ref}
    center = float(np.mean(np.log(sigma)))
    spread = max(float(np.std(np.log(sigma))),1e-6)
    x = (np.log(sigma)-center)/spread
    xt = (np.log(test_sigma)-center)/spread
    params.update(log_vol_center=center,log_vol_spread=spread)
    for p in [2,4]:
        X = np.column_stack([x**j for j in range(p)])
        Xt = np.column_stack([xt**j for j in range(p)])
        coef,meta = qr_fit(X,(y-q)/ref,alpha)
        name = f'State{p}-ERM'
        train[name] = q+ref*(X@coef)
        predictions[name] = test_q+ref*(Xt@coef)
        params[name] = {'coef':coef.tolist(),**meta}
    train_loss = {key:float(np.mean(loss(y,value,alpha))) for key,value in train.items()}
    assert train_loss['State2-ERM'] <= train_loss['Shift-ERM']+1e-9
    assert train_loss['State4-ERM'] <= train_loss['State2-ERM']+1e-9
    for key,value in predictions.items():
        assert np.isfinite(value).all(), key
    params['training_loss'] = train_loss
    return predictions,params


def past_volatility(ret, index):
    # Date t receives observations through t-1 only; each native calendar
    # remains unchanged and the full available pre-forecast history is used.
    sigma = ret.shift(1).rolling(20,min_periods=20).std(ddof=1).reindex(index)
    assert sigma.notna().all()
    return np.maximum(sigma.to_numpy(),1e-8)


def gate_and_window(y,q,nc,rolls,alpha=.01):
    cal = scores(y[:nc],q[:nc],alpha)
    gate = bool(cal['p_kup']<.05 or cal['TL']!='Green')
    start = max(max(WINDOWS),int(.7*nc))
    assert start < nc
    trials = {w:float(np.mean(loss(y[start:nc],q[start:nc]-rolls[w][start:nc],alpha))) for w in WINDOWS}
    selected = min(WINDOWS,key=lambda w:(trials[w],w))
    return gate,selected,{'validation_start':start,'validation_stop_exclusive':nc,
                          'validation_loss':trials,'gate_p_kup':cal['p_kup'],'gate_zone':cal['TL']}


def work(model,asset):
    started = time.monotonic()
    directory,suffix = MODELS[model]
    forecast_path = ROOT/'data'/directory/(f'{asset}_{suffix}.parquet' if suffix else f'{asset}.parquet')
    if model == 'Lag-Llama':
        forecast_path = ROOT/'data/lagllama'/f'{asset}.parquet'
    return_path = ROOT/'data/returns'/f'{asset}.csv'
    binding = {'producer_sha256':sha(__file__),'forecast_sha256':sha(forecast_path),
               'return_sha256':sha(return_path),'statistics_sha256':sha(PROJECT/'source/scripts/extension_20260831/panel_statistics.py'),
               'rolling_sha256':sha(PROJECT/'source/scripts/extension_20260831/posthoc.py')}
    folder = OUT/f'{model}__{asset}'
    folder.mkdir(parents=True,exist_ok=True)
    done = folder/'complete.json'
    if done.exists():
        old = json.loads(done.read_text())
        assert old['binding'] == binding
        assert all(sha(folder/key)==value for key,value in old['outputs'].items())
        return model,asset,'verified'
    ret = pd.read_csv(return_path,index_col='date',parse_dates=True).log_return
    forecast = pd.read_parquet(forecast_path)
    y = ret.loc[forecast.index].to_numpy()
    sigma = past_volatility(ret,forecast.index)
    n = len(y); nc = int(.7*n)
    assert nc>=max(SIZES)
    daily = pd.DataFrame({'r':y[nc:]},index=forecast.index[nc:])
    rows=[];params={};state_rows=[]
    high = sigma[nc:]>np.quantile(sigma[:nc],.9)
    for alpha in ALPHAS:
        q = forecast[f'VaR_{alpha:g}'].to_numpy()
        full = q[nc:]-qshift(q[:nc]-y[:nc],alpha)
        rows.append(dict(model=model,asset=asset,alpha=alpha,n_cal=nc,method='Full-static',**scores(y[nc:],full,alpha)))
        for size in SIZES:
            a = slice(nc-size,nc)
            preds,fit = candidates(y[a],q[a],sigma[a],q[nc:],sigma[nc:],alpha)
            params[f'{alpha:g}/{size}'] = fit
            for method,pred in preds.items():
                rows.append(dict(model=model,asset=asset,alpha=alpha,n_cal=size,method=method,**scores(y[nc:],pred,alpha)))
                if alpha==.01:
                    daily[f'{size}/{method}']=pred
                    for state,mask in [('High',high),('Other',~high)]:
                        if mask.sum():
                            state_rows.append(dict(model=model,asset=asset,n_cal=size,method=method,state=state,**scores(y[nc:][mask],pred[mask],alpha)))
    policy_rows=[];decisions={};policy_paths=[]
    q = forecast['VaR_0.01'].to_numpy()
    rolls = {w:rolling(q-y,w=w) for w in WINDOWS}
    for tag,start,stop in [('Original',nc,n)]+[(f'Origin{o}',n*o//100,n*(o+10)//100) for o in [50,60,70,80]]:
        gate,w,decision = gate_and_window(y,q,start,rolls)
        shift = qshift(q[:start]-y[:start])
        ix = slice(start,stop)
        pred = {'Raw':q[ix],'Static':q[ix]-shift,
                'Gate-static':q[ix]-shift if gate else q[ix]}
        for window in WINDOWS:
            corrected = q[ix]-rolls[window][ix]
            pred[f'Rolling{window}']=corrected
            pred[f'Gate-rolling{window}']=corrected if gate else q[ix]
        pred['Selected-rolling']=pred[f'Rolling{w}']
        pred['Gate-selected-rolling']=pred[f'Gate-rolling{w}']
        decision.update(selected_window=w,gate=gate,train_stop_exclusive=start,test_stop_exclusive=stop,static_shift=shift)
        decisions[tag]=decision
        # Recompute selection after changing every test outcome. Decisions
        # must agree; this is a dependency check, not a performance test.
        perturbed=y.copy();perturbed[start:]+=1
        perturbed_rolls={window:rolling(q-perturbed,w=window) for window in WINDOWS}
        gate2,w2,decision2=gate_and_window(perturbed,q,start,perturbed_rolls)
        assert gate2==gate and w2==w
        assert decision2['validation_loss']==decision['validation_loss']
        state_high=sigma[ix]>np.quantile(sigma[:start],.9)
        paths=pd.DataFrame({'r':y[ix],'sigma':sigma[ix],'high_volatility':state_high},index=forecast.index[ix])
        paths['origin']=tag
        for method,p in pred.items():
            paths[method]=p
            for state,mask in [('All',np.ones(stop-start,dtype=bool)),('High',state_high),('Other',~state_high)]:
                if mask.sum():
                    policy_rows.append(dict(model=model,asset=asset,origin=tag,method=method,state=state,
                                            selected_window=w,gate=gate,**scores(y[ix][mask],p[mask])))
        policy_paths.append(paths)
    pd.DataFrame(rows).to_csv(folder/'controlled.csv',index=False)
    pd.DataFrame(state_rows).to_csv(folder/'controlled_states.csv',index=False)
    pd.DataFrame(policy_rows).to_csv(folder/'policies.csv',index=False)
    daily.to_parquet(folder/'controlled_daily.parquet')
    pd.concat(policy_paths).to_parquet(folder/'policy_daily.parquet')
    (folder/'fits.json').write_text(json.dumps({'controlled':params,'policies':decisions},indent=2)+'\n')
    files=['controlled.csv','controlled_states.csv','policies.csv','controlled_daily.parquet','policy_daily.parquet','fits.json']
    done.write_text(json.dumps({'binding':binding,'outputs':{name:sha(folder/name) for name in files},
                               'past_only_selection_checked':True,'elapsed_seconds':time.monotonic()-started},indent=2)+'\n')
    return model,asset,round(time.monotonic()-started,1)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--models',nargs='+')
    ap.add_argument('--workers',type=int,default=3)
    args=ap.parse_args()
    assets=sorted(p.stem for p in (ROOT/'data/returns').glob('*.csv'))
    models=args.models or list(MODELS)
    OUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks=[pool.submit(work,m,a) for m in models for a in assets]
        for task in as_completed(tasks):
            print(*task.result(),flush=True)


if __name__=='__main__':main()
