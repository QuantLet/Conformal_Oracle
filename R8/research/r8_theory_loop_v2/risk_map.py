"""Primary squared-error quadrature of the locked Gaussian local experiment."""
import argparse
import csv
from fractions import Fraction as F
import math
from pathlib import Path
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr
from common import OUT,validate_lock

def action(x):
    return 0. if abs(x)<=1 else x-1/x

def phi(x): return math.exp(-x*x/2)/math.sqrt(2*math.pi)

TAIL_BOUND=2*((12+2)*phi(12)+2*ndtr(-12))

def risk(a):
    cuts=sorted(set([-12.,12.]+[x for x in (-a-1,-a+1) if -12<x<12]))
    parts=[quad(lambda z:(action(a+z)-a)**2*phi(z),l,r,epsabs=5e-12,
                epsrel=1e-11,limit=250) for l,r in zip(cuts[:-1],cuts[1:])]
    return sum(v for v,e in parts),sum(e for v,e in parts)

def active_probability(a): return float(ndtr(-1-a)+ndtr(a-1))

def benchmarks(a):
    return dict(raw=a*a,full=1.,half=(a*a+1)/4,fixed_oracle=a*a/(1+a*a))

def uniform_row(n):
    p=F(99,100);v=p*(1-p);k=(99*(n+1)+99)//100;delta=k-n*p
    train=(delta**2-delta-n*v)/(2*n*(n+1))
    test=(F(k*(k+1),(n+1)*(n+2))-2*p*F(k,n+1)+p*p)/2
    A=v/(2*n)
    return dict(n=n,k=k,delta=delta,training_delta=train,test_delta=test,
        leading_A=A,one_A_prediction=train+A,two_A_prediction=train+2*A,
        mean_C=F(k,n+1)-p,variance_C=F(k*(n+1-k),(n+1)**2*(n+2)))

def write_csv(path,rows):
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT/'risk')
    args=parser.parse_args();validate_lock();args.output.mkdir(parents=True,exist_ok=True)
    grid=sorted(set([i/20 for i in range(-160,161)]+[s*a for a in (16,32,64,128) for s in (-1,1)]))
    rows=[]
    for a in grid:
        val,err=risk(a);b=benchmarks(a)
        rows.append(dict(a=a,risk=val,quad_error=err,tail_bound=TAIL_BOUND,
            active_probability=active_probability(a),**b,**{f'minus_{k}':val-v for k,v in b.items()}))
    write_csv(args.output/'risk_map.csv',rows)
    roots=[]
    positive=[r for r in rows if 0<=r['a']<=8]
    for name in benchmarks(0):
        for l,r in zip(positive[:-1],positive[1:]):
            if l['minus_'+name]*r['minus_'+name]<0:
                fun=lambda a:risk(a)[0]-benchmarks(a)[name]
                root=brentq(fun,l['a'],r['a'],xtol=2e-13,rtol=1e-13)
                roots.append(dict(comparator=name,left=l['a'],right=r['a'],root=root,residual=fun(root)))
    write_csv(args.output/'crossings.csv',roots)
    exact=[uniform_row(n) for n in (250,500,700,1000,2000,10000)]
    write_csv(args.output/'uniform.csv',[{k:float(v) if isinstance(v,F) else v for k,v in r.items()} for r in exact])
    write_csv(args.output/'uniform_exact.csv',[{k:str(v) for k,v in r.items()} for r in exact])
    validate_lock();print('Computed',len(rows),'risk points;',len(roots),'positive-axis bracketed crossings.')

if __name__=='__main__': main()
