"""Exact rational certificate for a 1% conformal-rank counterexample."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/r8_count_law'


def add(a,b):
    result=[F(0)]*max(len(a),len(b))
    for i,x in enumerate(a):result[i]+=x
    for i,x in enumerate(b):result[i]+=x
    return result


def multiply(a,b):
    result=[F(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):result[i+j]+=x*y
    return result


def scale(a,x):return [v*x for v in a]


def maximum_cdf(n,theta,upper):
    single=[F(0),F(1)]
    triple=[F(1,2),F(-2),F(5,2)] if upper else [F(0),F(0),F(1,2)]
    history=[[F(1)],single,[F(0),F(0),F(1)]]
    for t in range(3,n+1):
        q=add(scale(multiply(single,history[-1]),1-theta),scale(multiply(triple,history[-3]),theta))
        history=history[-2:]+[q]
    return scale(add(add(history[-1],scale(multiply(single,history[-2]),theta)),
                     scale(multiply(multiply(single,single),history[-3]),theta)),1/(1+2*theta))


def integral(coefficients,lo,hi):
    return sum(x*(hi**(j+1)-lo**(j+1))/F(j+1) for j,x in enumerate(coefficients))


def cost(n,p,theta):
    value=F(0)
    for upper,(lo,hi) in enumerate([(F(0),F(1,2)),(F(1,2),F(1))]):
        value+=integral(multiply(maximum_cdf(n,theta,upper),[-p,F(1)]),lo,hi)
    return (1-p)**2/2-value


def run():
    n=125;p=F(99,100);delta=F(813,100000)
    iid=cost(n,p,F(0));dependent=cost(n,p,F(1,2));raw=delta*delta/2
    assert iid==F(589,17780000)
    assert F(3292333,10**11)<dependent<F(3292334,10**11)
    assert dependent<raw<iid
    assert cost(3,F(3,4),F(0))==F(3,160)
    assert cost(3,F(3,4),F(1,2))==F(37,1920)
    return dict(status='passed',arithmetic='exact rational',n=n,p=str(p),delta=str(delta),
                iid_cost=str(iid),renewal_cost=str(dependent),raw_regret=str(raw),
                iid_loss_change=str(iid-raw),renewal_loss_change=str(dependent-raw),
                iid_loss_change_decimal=float(iid-raw),renewal_loss_change_decimal=float(dependent-raw),
                producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                interpretation='Constructed population witness, not a fitted financial rule; all pairwise covariances match.')


if __name__=='__main__':
    target=OUT/'exact_witness.json';result=run()
    if target.exists():assert json.loads(target.read_text())==result
    else:target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
