"""Locked calibration-only definitions. No market-data loader in this module."""
import numpy as np
from scipy import stats

ALPHA=.01
V=.0099


def rank(n):
    k=(99*(n+1)+99)//100
    if not 1<=k<=n:
        raise ValueError('Inadmissible conformal rank')
    return k


def rank_shift(scores):
    scores=np.asarray(scores,float)
    if scores.ndim!=1 or not np.isfinite(scores).all():
        raise ValueError('Invalid scores')
    return float(np.partition(scores,rank(len(scores))-1)[rank(len(scores))-1])


def ecdf_integral(scores,c):
    # Antiderivative of the ECDF, evaluated at c and 0.
    s=np.asarray(scores,float)
    return float(np.mean(np.maximum(c-s,0)-np.maximum(-s,0))-.99*c)


def hac(hits,L,apply_floor=True):
    x=np.asarray(hits,float);x=x-x.mean();n=len(x)
    raw=float(x@x/n+2*sum((1-j/(L+1))*(x[j:]@x[:-j])/n for j in range(1,L+1)))
    return (max(raw,V/10) if apply_floor else raw), raw


def andrews(hits,apply_floor=True):
    x=np.asarray(hits,float);x=x-x.mean();n=len(x)
    den=float(x[:-1]@x[:-1])
    if den<=0:
        raise ValueError('Undefined Andrews AR(1)')
    rho=float(x[1:]@x[:-1]/den)
    if abs(rho)>=1:
        raise ValueError('Nonstationary Andrews AR(1) estimate')
    bw=1.1447*(4*rho*rho/(1-rho*rho)**2*n)**(1/3)
    raw=float(x@x/n)
    if bw>0:
        raw+=2*sum((1-j/bw)*float(x[j:]@x[:-j])/n for j in range(1,min(n,int(np.ceil(bw)))))
    return (max(raw,V/10) if apply_floor else raw),raw,bw,rho


def gaussian_density(scores,point,bw):
    if not np.isfinite(bw) or bw<=0:
        raise ValueError('Invalid kernel bandwidth')
    z=(np.asarray(scores)-point)/bw
    return float(np.mean(np.exp(-z*z/2))/(bw*np.sqrt(2*np.pi)))


def spacing(scores,p=.99,m=None):
    x=np.sort(scores);n=len(x);m=int(n**.8) if m is None else int(m)
    j=int(np.floor(n*p));lo=j-m+1;hi=j+m
    out=dict(m=m,lower=lo,upper=hi)
    if lo<1 or hi>n:
        return dict(out,status='OUT_OF_RANGE',f=None)
    gap=float(x[hi-1]-x[lo-1])
    if gap<=0:
        return dict(out,status='NONPOSITIVE_SPACING',f=None)
    return dict(out,status='OK',f=(2*m/n)/gap)


def h2(c,n,omega,f):
    return float(max(0,n*c*c-omega/(f*f)))


def shrinkage(c,n,omega,f):
    signal=f*f*h2(c,n,omega,f)
    return float(signal/(omega+signal))


def boundary(sigma,g,h2_z):
    sigma=np.asarray(sigma,float)
    A=sigma.mean();B=np.mean(sigma*sigma)/A;C=1/np.mean(1/sigma)
    if A-C<=1e-12*max(A,C):
        raise ValueError('DEGENERATE_SCALE')
    return float(g*g*h2_z-V*(B-C)/(A-C))


def expected_loss(law,q,sigma):
    q=np.asarray(q);sigma=np.asarray(sigma)
    if law=='normal':
        z=q/sigma
        return sigma*(stats.norm.pdf(z)+z*(stats.norm.cdf(z)-.01))
    scale=sigma*np.sqrt(3/5);z=q/scale
    return scale*((5+z*z)/4*stats.t.pdf(z,5)+z*(stats.t.cdf(z,5)-.01))


def garch_batch(law,count,seeds):
    """Vectorised histories, exactly the existing scalar generator recurrence."""
    eps=[]
    for seed in seeds:
        rng=np.random.default_rng(seed)
        eps.append(rng.standard_normal(count+2000) if law=='normal'
                   else rng.standard_t(5,count+2000)*np.sqrt(3/5))
    eps=np.asarray(eps);s2=np.full(len(seeds),1e-5/(1-.10-.85));last=np.zeros(len(seeds))
    y=np.empty((len(seeds),count));sigma=np.empty_like(y)
    for t in range(count+2000):
        s2=1e-5+.10*last**2+.85*s2
        last=np.sqrt(s2)*eps[:,t]
        if t>=2000:
            y[:,t-2000]=last;sigma[:,t-2000]=np.sqrt(s2)
    return y,sigma
