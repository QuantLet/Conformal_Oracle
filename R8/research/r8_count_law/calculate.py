"""Deterministic order-statistic loss under pairwise-independent renewal blocks."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import integrate,optimize,special,stats

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/r8_count_law'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def triple(v):
    """PGF coefficients for the number of ABOVE-threshold scores in a triple."""
    t=(v*v+np.maximum(2*v-1,0)**2)/2
    return np.stack([t,3*v*v-3*t,3*v-6*v*v+3*t,1-3*v+3*v*v-t],axis=-1)


def multiply(a,b,limit):
    ans=np.zeros((a.shape[0],limit+1))
    for j in range(min(b.shape[1],limit+1)):
        ans[:,j:]+=a[:,:limit+1-j]*b[:,j,None]
    return ans


def count_pgf(v,n,theta,limit):
    """Stationary window PGF, retaining coefficients through `limit`."""
    v=np.asarray(v).reshape(-1)
    p1=np.stack([v,1-v],axis=-1);p3=triple(v)
    q0=np.zeros((len(v),limit+1));q0[:,0]=1
    q1=multiply(q0,p1,limit)
    if n==1:return q1
    q2=multiply(q1,p1,limit)
    history=[q0,q1,q2]
    for t in range(3,n+1):
        q=(1-theta)*multiply(history[-1],p1,limit)+theta*multiply(history[-3],p3,limit)
        history=[history[-2],history[-1],q]
    # Boundary, second position of a triple, third position of a triple.
    result=(history[-1]+theta*multiply(history[-2],p1,limit)
            +theta*multiply(multiply(history[-3],p1,limit),p1,limit))/(1+2*theta)
    assert result.min()>-1e-11 and result.sum(axis=1).max()<1+1e-10
    return result


def cost(n,alpha,k,theta,nodes):
    points,weights=special.roots_legendre(nodes)
    z=stats.norm.ppf(1-alpha);p=1-alpha
    answer=0.
    for lo,hi in [(-12.,0.),(0.,z),(z,12.)]:
        x=lo+(points+1)*(hi-lo)/2
        v=stats.norm.cdf(x)
        h=count_pgf(v,n,theta,n-k).sum(axis=1)
        integrand=(p-v)*h if hi<=z else (v-p)*(1-h)
        answer+=float(weights@integrand*(hi-lo)/2)
    return answer


def normal_risk(q,alpha):
    return stats.norm.pdf(q)+q*(stats.norm.cdf(q)-alpha)


def checks():
    v=np.array([.001,.01,.1,.3,.5,.7,.9,.99,.999])
    errors=dict(iid_beta=0.,three_window=0.,normalisation=0.,mean=0.,variance=0.,copula_integral=0.)
    for n in (3,10,25):
        for theta in (0.,.5,.9):
            pgf=count_pgf(v,n,theta,n);ks=np.arange(n+1)
            errors['normalisation']=max(errors['normalisation'],float(np.max(np.abs(pgf.sum(axis=1)-1))))
            mean=pgf@ks;variance=pgf@(ks*ks)-mean*mean
            errors['mean']=max(errors['mean'],float(np.max(np.abs(mean-n*(1-v)))))
            errors['variance']=max(errors['variance'],float(np.max(np.abs(variance-n*v*(1-v)))))
            if theta==0:
                for k in range(1,n+1):
                    errors['iid_beta']=max(errors['iid_beta'],float(np.max(np.abs(pgf[:,:n-k+1].sum(axis=1)-special.betainc(k,n+1-k,v)))))
            if n==3:
                iid=stats.binom.pmf(ks[None,:],3,1-v[:,None]);w=theta/(1+2*theta)
                errors['three_window']=max(errors['three_window'],float(np.max(np.abs(pgf-((1-w)*iid+w*triple(v))))))
    for u in v:
        # For first coordinate x, length of admissible second coordinates.
        def section(x):return max(0.,min(u,u-x))+max(0.,u-max(0.,1-x))
        direct=integrate.quad(section,0,u,points=[max(0.,min(u,1-u))],epsabs=1e-13)[0]
        errors['copula_integral']=max(errors['copula_integral'],abs(direct-triple(np.array([u]))[0,0]))
    assert max(errors.values())<2e-11,errors
    return errors


def main(replay=False):
    folder=OUT.with_name('r8_count_law_replay') if replay else OUT
    folder.mkdir(parents=True,exist_ok=False)
    errors=checks();rows=[]
    for alpha in (.01,.05):
        q=stats.norm.ppf(alpha);minimum=normal_risk(q,alpha)
        for n in (125,250,500,1000):
            for theta in (0.,.5,.9):
                for conformal in (False,True):
                    k=int(np.ceil((n+int(conformal))*(1-alpha)))
                    a=cost(n,alpha,k,theta,512);b=cost(n,alpha,k,theta,256)
                    gain=lambda c:normal_risk(q+c,alpha)-minimum-a
                    c=optimize.brentq(gain,0.,20.,xtol=1e-13)
                    rows.append(dict(alpha=alpha,n=n,theta=theta,method='Shift-CP' if conformal else 'Shift-ERM',rank=k,
                        estimation_cost=a,quadrature_difference=abs(a-b),omega=alpha*(1-alpha),
                        upper_shift_boundary=c,upper_violation_boundary=stats.norm.cdf(q+c)))
    data=pd.DataFrame(rows)
    assert len(data)==48 and data.quadrature_difference.max()<1e-9
    old=ROOT/'artifacts/r8_frontier/normal_exact_break_even.csv'
    merged=data[data.theta==0].merge(pd.read_csv(old),on=['alpha','n','method'],suffixes=('_new','_old'))
    difference=float(np.max(np.abs(merged.estimation_cost_new-merged.estimation_cost_old)))
    assert len(merged)==16 and difference<5e-9
    data.to_csv(folder/'risk.csv',index=False)
    record=dict(status='passed',configurations=len(data),checks=errors,old_iid_max_difference=difference,
        quadrature_max_difference=float(data.quadrature_difference.max()),
        normal_tail_remainder=2*1000*(stats.norm.pdf(12)-12*stats.norm.sf(12)),
        no_random_draws=True,no_model_fits=True,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('PROTOCOL.md'),old]},
        outputs={'risk.csv':sha(folder/'risk.csv')})
    (folder/'validation.json').write_text(json.dumps(record,indent=2)+'\n')
    if replay:assert record==json.loads((OUT/'validation.json').read_text())
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');args=p.parse_args();main(args.replay)
