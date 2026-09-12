"""Integrate future Normal AR losses on saved histories; no random generation."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'research/r8_mechanism'))
import engine as old
OUT = PROJECT / 'artifacts/r8_horizon_bridge/horizon'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def loss(q, mu, sd, alpha):
    d = mu - q
    z = d / sd
    return sd * np.exp(-0.5*z*z)/np.sqrt(2*np.pi) + d*(ndtr(z) + alpha - 1)


def calculate(folder):
    folder.mkdir(parents=True, exist_ok=False)
    source = old.OUT / 'paths.npz'
    assert sha(source) == json.loads((old.OUT/'paths.json').read_text())['sha256']
    inputs = {str(p.relative_to(PROJECT)):sha(p) for p in [source, old.OUT/'paths.json',
        Path(__file__), Path(__file__).with_name('HORIZON_PROTOCOL.md'), Path(old.__file__),
        Path(old.old.__file__), Path(old.m.__file__)]}
    paths = np.load(source)
    rows, reps, old_error, quad_error = [], [], 0., 0.
    for phi in (0., .5, .8):
        z = paths[f'ar_z_{phi:g}']
        assert z.shape == (500, 1000)
        for n in (125, 250, 500, 1000):
            tables = [old.OUT/'blocks'/f'ar_normal_{phi:g}_{n}_{k:03d}_{k+25:03d}'/'replications.csv'
                      for k in range(0,500,25)]
            inputs.update({str(p.relative_to(PROJECT)):sha(p) for p in tables})
            archived = pd.concat([pd.read_csv(p) for p in tables], ignore_index=True)
            for alpha in (.01, .05):
                raw0 = old.V0 * norm.ppf(alpha)
                rank = int(np.ceil((n+1)*(1-alpha)))
                q = -np.sort(-old.V0*z[:,-n:], axis=1)[:,rank-1]
                indep_corrected = loss(q, 0., old.V0, alpha)
                oracle_loss = float(loss(raw0, 0., old.V0, alpha))
                omega = old.count_theory(alpha, phi, n)['omega']
                first_order = omega/(2*n*(norm.pdf(norm.ppf(alpha))/old.V0))
                horizons = [1,20,3*n//7,n,4*n]
                powers = phi**np.arange(1,max(horizons)+1)
                mu = old.V0*z[:,-1,None]*powers
                sd = old.V0*np.sqrt(1-powers*powers)
                future_corrected = loss(q[:,None], mu, sd, alpha)
                for truth in ('none','constant'):
                    raw = raw0 + (0 if truth=='none' else .25*old.V0)
                    c = raw-q
                    assert np.allclose(c, np.sort(raw-old.V0*z[:,-n:],axis=1)[:,rank-1],rtol=0,atol=1e-16)
                    independent = indep_corrected-loss(raw,0.,old.V0,alpha)
                    known = archived[(archived.alpha==alpha)&(archived.truth==truth)&(archived.method=='Shift-CP')].sort_values('replication')
                    err = float(np.max(np.abs(known.expected_QS.to_numpy()-indep_corrected)))
                    old_error = max(old_error,err)
                    assert len(known)==500 and err<2e-16
                    future = future_corrected-loss(raw,mu,sd,alpha)
                    total = np.cumsum(future,axis=1)
                    for H in horizons:
                        average = total[:,H-1]/H
                        boundary = average-independent
                        if phi==0: assert np.max(np.abs(boundary))<2e-16
                        assert np.max(np.abs(average-future[:,:H].mean(axis=1)))<2e-16
                        row = dict(phi=phi,n_cal=n,alpha=alpha,truth=truth,H=H,replications=500,
                            expected_change=float(average.mean()), independent_change=float(independent.mean()),
                            boundary=float(boundary.mean()), boundary_se=float(boundary.std(ddof=1)/np.sqrt(500)),
                            independent_cost=float((indep_corrected-oracle_loss).mean()), leading_cost=first_order)
                        rows.append(row)
                        reps.extend(dict(phi=phi,n_cal=n,alpha=alpha,truth=truth,H=H,replication=i,
                                         contiguous_change=average[i],independent_change=independent[i],boundary=boundary[i])
                                    for i in range(500))
                    # Direct numerical integration at lag one on the first saved history.
                    for forecast in (raw,q[0]):
                        mean,scale = mu[0,0],sd[0]
                        cut = (forecast-mean)/scale
                        fun = lambda u: ((mean+scale*u-forecast)*(alpha-((mean+scale*u-forecast)<0)))*norm.pdf(u)
                        value = quad(fun,-np.inf,cut,epsabs=1e-13)[0]+quad(fun,cut,np.inf,epsabs=1e-13)[0]
                        quad_error=max(quad_error,abs(value-float(loss(forecast,mean,scale,alpha))))
    assert len(rows)==240 and quad_error<1e-11
    pd.DataFrame(rows).to_csv(folder/'summary.csv',index=False)
    pd.DataFrame(reps).to_csv(folder/'replications.csv',index=False)
    record = dict(status='passed',summary_cells=len(rows),histories_per_cell=500,
                  archived_loss_max_error=old_error,quadrature_max_error=quad_error,
                  no_new_random_draws=True,no_model_fits=True,inputs=inputs,
                  outputs={n:sha(folder/n) for n in ('summary.csv','replications.csv')})
    (folder/'validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    target=OUT.with_name('horizon_replay') if args.replay else OUT
    calculate(target)
    if args.replay:
        a=json.loads((OUT/'validation.json').read_text());b=json.loads((target/'validation.json').read_text())
        assert a==b
        (target/'exact_replay.json').write_text(json.dumps({'status':'passed','outputs':b['outputs']},indent=2)+'\n')
