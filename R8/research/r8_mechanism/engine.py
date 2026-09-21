"""Predeclared mechanism experiment; independent-future expected losses."""
import os
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_k]='1'
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy import stats
from scipy.integrate import quad

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'research/r8_decision'))
import methods as m
sys.path.insert(0,str(PROJECT/'research/r8_review'))
import complexity_simulation as old
from controlled_comparisons import candidates

OUT=PROJECT/'artifacts/r8_mechanism'
REPS=500
SIZES=(125,250,500,1000)
ALPHAS=(.01,.05)
V0=old.V0


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ar_path(innovations,phi):
    z=np.empty(len(innovations));z[0]=innovations[0]
    scale=np.sqrt(1-phi**2)
    for j in range(1,len(z)):z[j]=phi*z[j-1]+scale*innovations[j]
    return z


def marginal(z,kind):
    if kind=='normal':return V0*z,0
    u=stats.norm.cdf(z);clipped=(u==0)|(u==1)
    u=np.clip(u,np.nextafter(0.,1.),np.nextafter(1.,0.))
    return V0*np.sqrt(3/5)*stats.t.ppf(u,5),int(clipped.sum())


def hit_probability(kind,q,sigma):
    return stats.norm.cdf(q/sigma) if kind=='normal' else stats.t.cdf(q/(sigma*np.sqrt(3/5)),5)


def covariance(alpha,r):
    z=stats.norm.ppf(alpha)
    return quad(lambda u:np.exp(-z*z/(1+u))/(2*np.pi*np.sqrt(1-u*u)),0,r,
                epsabs=1e-14,epsrel=1e-12)[0]


def count_theory(alpha,phi,n):
    base=alpha*(1-alpha)
    if phi==0:return {'omega':base,'remainder_bound':0.,'lags':0,'finite_count_variance':n*base}
    H=1
    def bound(h):return phi**(h+1)/(np.pi*np.sqrt(1-phi**2)*(1-phi))
    while bound(H)>1e-13:H+=1
    cov=np.array([covariance(alpha,phi**h) for h in range(1,max(H,n-1)+1)])
    finite=n*base+2*np.dot(n-np.arange(1,n),cov[:n-1])
    return {'omega':base+2*cov[:H].sum(),'remainder_bound':bound(H),'lags':H,
            'finite_count_variance':float(finite)}


def state_fit(y,q,sigma,p,penalty,alpha):
    X,params=m.design(sigma,p=p);scale=float(np.median(sigma))
    coef,cert=m.l1_fit(X,(y-q)/scale,penalty,alpha)
    return {**params,'scale':scale,'coef':coef.tolist(),'certificate':cert,'alpha':alpha}


def state_select(y,q,sigma,alpha):
    v=int(.7*len(y));trials=[]
    for p in (2,4):
        for penalty in m.LAMBDAS:
            fit=state_fit(y[:v],q[:v],sigma[:v],p,penalty,alpha)
            pred=m.predict_state(q[v:],sigma[v:],fit)
            trials.append({'p':p,'penalty':penalty,'validation_loss':float(m.loss(y[v:],pred,alpha).mean()),'fit':fit})
    best=min(trials,key=lambda a:(a['validation_loss'],a['p'],-a['penalty']))
    fitted=state_fit(y,q,sigma,best['p'],best['penalty'],alpha)
    return fitted,{'inner_split':v,'selected_p':best['p'],'selected_penalty':best['penalty'],'trials':trials}


def distort(sigma,truth):
    return np.zeros_like(sigma) if truth=='none' else old.distortion(sigma,truth)


def basic(y,q,sigma,qt,st,alpha):
    s=q-y
    pars={'shift_cp':m.cp(s,alpha),'shift_erm':float(np.quantile(s,1-alpha,method='inverted_cdf')),
          'vol_cp':m.cp(s/sigma,alpha),'vol_erm':m.weighted_quantile(s/sigma,sigma,1-alpha)}
    pred={'Raw':qt,'Shift-CP':qt-pars['shift_cp'],'Shift-ERM':qt-pars['shift_erm'],
          'Vol-CP':qt-st*pars['vol_cp'],'Vol-ERM':qt-st*pars['vol_erm']}
    return pred,pars


def pot_at(fit,s,alpha):
    if fit['fallback']:return m.cp(s,alpha)
    return m.gpd_quantile(fit['threshold'],fit['shape'],fit['scale'],fit['tail_fraction'],alpha)


def metric(kind,pred,sigma,alpha,oracle):
    risk=old.expected_loss(kind,pred,sigma,alpha)
    opt=old.expected_loss(kind,oracle,sigma,alpha)
    assert np.isfinite(pred).all() and np.isfinite(risk).all()
    assert np.min(risk-opt)>-1e-12
    return {'expected_QS':float(np.mean(risk)),'excess_QS':float(np.mean(risk-opt)),
            'expected_violation':float(np.mean(hit_probability(kind,pred,sigma))),
            'prediction_MSE':float(np.mean((pred-oracle)**2)),
            'max_absolute_prediction':float(np.max(np.abs(pred)))}


def one_ar(y,kind,alpha,offset):
    # Only simple reference used in tests; production shares translation-invariant fits.
    q=float(old.conditional_quantile(kind,V0,alpha))+offset
    return q-m.cp(q-y,alpha)


def ar_calculation(y,kind):
    params={};predictions={}
    for tau in (.8,.9):params[f'POT{int(tau*100)}']=m.fit_pot(-y,tau,.01)
    for alpha in ALPHAS:
        oracle=float(old.conditional_quantile(kind,V0,alpha))
        corrected={'Shift-CP':-m.cp(-y,alpha),
                   'Shift-ERM':-float(np.quantile(-y,1-alpha,method='inverted_cdf'))}
        for tau in (.8,.9):
            name=f'POT{int(tau*100)}'
            corrected[name+'-Shift']=-pot_at(params[name],-y,alpha)
        for truth in ('none','constant'):
            raw=oracle+(0 if truth=='none' else .25*V0)
            predictions[(alpha,truth)]={'Raw':np.asarray(raw),**{k:np.asarray(v) for k,v in corrected.items()}}
    return predictions,params


def garch_calculation(y,sigma,st,kind):
    predictions={};parameters={}
    # Normalised scores differ across alpha by a constant; keep that translation explicit.
    volfits={}
    for truth in ('none','constant','state'):
        x=(-y+distort(sigma,truth))/sigma
        volfits[truth]={f'POT{int(tau*100)}':m.fit_pot(x,tau,.01) for tau in (.8,.9)}
    parameters['normalised_tail_fits']=volfits
    for alpha in ALPHAS:
        oracle=old.conditional_quantile(kind,st,alpha)
        qc=old.conditional_quantile(kind,sigma,alpha)
        original,unpen=candidates(y,qc,sigma,oracle,st,alpha)
        parameters[f'{alpha:g}/unpenalised_reference']=unpen
        fitted,selection=state_select(y,qc,sigma,alpha)
        l1pred=m.predict_state(oracle,st,fitted)
        l1clip=m.predict_state(oracle,st,fitted,clip=True)
        parameters[f'{alpha:g}/none/L1']={'fit':fitted,**selection}
        shiftfits={}
        for truth in ('none','constant','state'):
            q=qc+distort(sigma,truth);qt=oracle+distort(st,truth)
            pred,pars=basic(y,q,sigma,qt,st,alpha)
            # The prescribed state distortion is in both unpenalised column spaces.
            pred.update({name:original[name] for name in ('State2-ERM','State4-ERM')})
            pars['unpenalised_reuse']={'source':f'{alpha:g}/unpenalised_reference',
                                       'subtract_distortion_from_fitted_correction':truth}
            if truth=='state':
                fit,sel=state_select(y,q,sigma,alpha)
                pred['State-L1']=m.predict_state(qt,st,fit)
                pred['State-L1-clipped']=m.predict_state(qt,st,fit,clip=True)
                parameters[f'{alpha:g}/state/L1']={'fit':fit,**sel}
            else:
                pred['State-L1']=l1pred;pred['State-L1-clipped']=l1clip
                if truth=='constant':pars['L1_reuse']={'source':f'{alpha:g}/none/L1','intercept_offset':-.25*V0}
            for tau in (.8,.9):
                name=f'POT{int(tau*100)}'
                if truth=='constant':
                    sf=shiftfits['none'][name]
                    pred[name+'-Shift']=predictions[(alpha,'none')][name+'-Shift']
                    pars[name+'-Shift']={'reuse':f'{alpha:g}/none/{name}-Shift','score_translation':.25*V0}
                else:
                    sf=m.fit_pot(q-y,tau,alpha)
                    shiftfits.setdefault(truth,{})[name]=sf
                    pred[name+'-Shift']=qt-sf['quantile'];pars[name+'-Shift']=sf
                vf=volfits[truth][name];x=(-y+distort(sigma,truth))/sigma
                z=float(old.conditional_quantile(kind,1.,alpha))
                c=z+pot_at(vf,x,alpha)
                pred[name+'-Vol']=qt-st*c
                pars[name+'-Vol']={'source':f'normalised_tail_fits/{truth}/{name}',
                                  'score_translation':z,'quantile':c}
            predictions[(alpha,truth)]=pred
            parameters[f'{alpha:g}/{truth}/parameters']=pars
    return predictions,parameters
