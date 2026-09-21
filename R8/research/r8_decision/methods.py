"""Research comparators: exact L1 quantile LP, score POT, and DtACI variants."""
import os
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import hashlib
import numpy as np
from scipy import sparse
from scipy.optimize import linprog, minimize
from scipy.stats import genpareto

ALPHA=.01
LAMBDAS=(0.,1e-4,1e-3,1e-2,.1,1.)
GAMMAS=np.array([.001,.002,.004,.008,.016,.032,.064])


def seed_for(key, replicate=0):
    return int.from_bytes(hashlib.sha256(f'20260909/{key}/{replicate}'.encode()).digest()[:4],'little')


def loss(y,q,alpha=ALPHA):
    e=np.asarray(y)-np.asarray(q)
    return np.where(e>=0,alpha*e,(alpha-1)*e)


def cp(s,alpha=ALPHA):
    k=int(np.ceil((len(s)+1)*(1-alpha)))
    if not 1<=k<=len(s):raise ValueError('No finite conformal rank')
    return float(np.partition(s,k-1)[k-1])


def weighted_quantile(x,w,p):
    order=np.argsort(x,kind='stable'); c=np.cumsum(w[order])
    return float(x[order[min(np.searchsorted(c,p*c[-1],side='left'),len(x)-1)]])


def design(sigma,params=None,p=4,clip=False):
    z=np.log(sigma)
    if params is None:
        params={'center':float(z.mean()),'spread':max(float(z.std()),1e-6),
                'minimum':float(z.min()),'maximum':float(z.max()),'p':p}
    if clip:z=np.clip(z,params['minimum'],params['maximum'])
    x=(z-params['center'])/params['spread']
    return np.column_stack([x**j for j in range(params['p'])]),params


def l1_fit(X,y,penalty,alpha=ALPHA):
    """Mean pinball + L1 slopes; unpenalised intercept; primal/dual audit."""
    n,p=X.shape
    costs=np.r_[0.,np.full(p-1,n*penalty)]
    objective=np.r_[costs,costs,np.full(n,alpha),np.full(n,1-alpha)]
    A=sparse.hstack([sparse.csc_matrix(X),-sparse.csc_matrix(X),
                    sparse.eye(n,format='csc'),-sparse.eye(n,format='csc')],format='csc')
    fit=linprog(objective,A_eq=A,b_eq=y,bounds=(0,None),method='highs',
                options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not fit.success:raise RuntimeError(f'L1 LP: {fit.message}')
    coef=fit.x[:p]-fit.x[p:2*p]
    primal=float(loss(y,X@coef,alpha).sum()+n*penalty*np.abs(coef[1:]).sum())
    dual=float(np.dot(y,fit.eqlin.marginals))
    feasibility=float(np.max(A.T@fit.eqlin.marginals-objective))
    gap=abs(primal-dual)
    assert gap<1e-6*max(1,abs(primal)),(gap,primal,dual)
    assert feasibility<1e-7,feasibility
    return coef,{'primal':primal,'dual':dual,'gap':gap,'dual_violation':feasibility,
                 'iterations':int(fit.nit),'penalty':penalty,'n_fit':n}


def fit_state(y,q,sigma,p,penalty):
    X,params=design(sigma,p=p);scale=float(np.median(sigma))
    coef,cert=l1_fit(X,(y-q)/scale,penalty)
    return {**params,'scale':scale,'coef':coef.tolist(),'certificate':cert}


def predict_state(q,sigma,params,clip=False):
    X,_=design(sigma,params,clip=clip)
    return q+params['scale']*(X@np.asarray(params['coef']))


def select_state(y,q,sigma,v,nc):
    trials=[]
    for p in (2,4):
        for penalty in LAMBDAS:
            fit=fit_state(y[:v],q[:v],sigma[:v],p,penalty)
            pred=predict_state(q[v:nc],sigma[v:nc],fit)
            trials.append({'p':p,'penalty':penalty,'validation_loss':float(loss(y[v:nc],pred).mean()),'fit':fit})
    selected=min(trials,key=lambda r:(r['validation_loss'],r['p'],-r['penalty']))
    fitted=fit_state(y[:nc],q[:nc],sigma[:nc],selected['p'],selected['penalty'])
    return fitted,{'trials':trials,'selected_p':selected['p'],'selected_penalty':selected['penalty']}


def gpd_quantile(threshold,shape,scale,tail_fraction,alpha=ALPHA):
    if not (scale>0 and 0<alpha<tail_fraction):raise ValueError('Invalid GPD quantile inputs')
    log_ratio=np.log(tail_fraction/alpha)
    if abs(shape)<1e-8:return float(threshold+scale*log_ratio)
    return float(threshold+scale*np.expm1(shape*log_ratio)/shape)


def fit_pot(s,tau,alpha=ALPHA):
    threshold=float(np.quantile(s,tau));excess=np.asarray(s)[s>threshold]-threshold
    meta={'threshold_level':tau,'threshold':threshold,'n_fit':len(s),'n_tail':len(excess),
          'tail_fraction':len(excess)/len(s),'fallback':False,'attempts':[]}
    def fallback(reason):
        meta.update(fallback=True,reason=reason,quantile=cp(s,alpha));return meta
    if len(excess)<20:return fallback('fewer than 20 strict exceedances')
    normaliser=float(excess.mean())
    if not np.isfinite(normaliser) or normaliser<=0:return fallback('invalid excess scale')
    x=excess/normaliser;estimates=[]
    for start in (-.1,.1,.5):
        records=[]
        def optimizer(fun,x0,args=(),disp=0):
            result=minimize(fun,x0,args=args,method='Nelder-Mead',
                            options={'maxiter':4000,'xatol':1e-9,'fatol':1e-9})
            records.append({'start':np.asarray(x0).tolist(),'params':result.x.tolist(),
                            'objective':float(result.fun),'success':bool(result.success),
                            'iterations':int(result.nit)})
            return result.x
        try:
            shape,loc,scale=genpareto.fit(x,start,floc=0,scale=1.,optimizer=optimizer)
            support=bool(scale>0 and np.all(1+shape*x/scale>0))
            nll=float(-genpareto.logpdf(x,shape,loc=0,scale=scale).sum())
            ok=bool(records and records[-1]['success'] and support and np.isfinite(nll))
            meta['attempts'].append({'initial_shape':start,'optimizer':records,'valid':ok})
            if ok:estimates.append((nll,float(shape),float(scale)))
        except (ValueError,FloatingPointError,OverflowError) as e:
            meta['attempts'].append({'initial_shape':start,'optimizer':records,'valid':False,'error':str(e)})
    if not estimates:return fallback('no converged valid GPD fit')
    nll,shape,scale=min(estimates,key=lambda z:z[0]);scale*=normaliser
    quantile=gpd_quantile(threshold,shape,scale,len(excess)/len(s),alpha)
    if not np.isfinite(quantile):return fallback('nonfinite GPD quantile')
    meta.update(shape=shape,scale=scale,normalised_nll=nll,normaliser=normaliser,
                support_margin=float(np.min(1+shape*excess/scale)),
                irregular_shape=bool(shape<=-.5),quantile=quantile)
    return meta


def select_pot(y,q,sigma,v,nc,normalised):
    z=(q-y)/sigma if normalised else q-y
    trials=[]
    for tau in (.9,.95):
        fit=fit_pot(z[:v],tau)
        correction=fit['quantile']*(sigma[v:nc] if normalised else 1.)
        trials.append({'threshold_level':tau,'validation_loss':float(loss(y[v:nc],q[v:nc]-correction).mean()),'fit':fit})
    choice=min(trials,key=lambda r:(r['validation_loss'],r['threshold_level']))
    fit=fit_pot(z[:nc],choice['threshold_level'])
    return fit,{'normalised':normalised,'selected_threshold':choice['threshold_level'],'trials':trials}


def quantile_inverse(sorted_scores,score):
    """Inverse of the linear empirical quantile; actual hits handle ties."""
    j=int(np.searchsorted(sorted_scores,score,side='left'));n=len(sorted_scores)
    if j==0:return 1.
    if j==n:return 0.
    if sorted_scores[j]==score:return float(1-j/(n-1))
    fraction=(score-sorted_scores[j-1])/(sorted_scores[j]-sorted_scores[j-1])
    return float(1-(j-1+fraction)/(n-1))


def dtaci(s,q,projected=True,w=500,alpha=ALPHA):
    n=len(s);k=len(GAMMAS);levels=np.full(k,alpha);weights=np.full(k,1/k)
    pred=np.full((n,k),np.nan);prob=np.full((n,k),np.nan)
    states=np.full((n,k),np.nan);beta=np.full(n,np.nan)
    projections=np.zeros(k,dtype=int);ties=0
    eta=float(np.sqrt(3*(np.log(2*k*500)+1)/(500*(alpha*(1-alpha))**2)))
    for t in range(w,n):
        window=np.sort(s[t-w:t]);states[t]=levels;prob[t]=weights
        shifts=np.interp(1-np.clip(levels,0,1),np.linspace(0,1,w),window)
        if not projected:
            shifts[levels<=0]=np.inf;shifts[levels>=1]=-np.inf
        pred[t]=q[t]-shifts
        # Return violation r<q_corrected is exactly score>shift.
        hits=s[t]>shifts
        beta[t]=quantile_inverse(window,s[t]);ties+=int(np.any(np.diff(window)==0))
        expert_loss=loss(beta[t],levels,alpha)
        logw=np.log(weights)-eta*expert_loss;logw-=logw.max()
        updated=np.exp(logw);updated/=updated.sum()
        weights=(1-.001)*updated+.001/k
        levels=levels+GAMMAS*(alpha-hits.astype(float))
        if projected:
            clipped=np.clip(levels,1/(w+1),w/(w+1))
            projections+=(clipped!=levels);levels=clipped
    return {'predictions':pred,'probabilities':prob,'states':states,'beta':beta,
            'meta':{'projected':projected,'window':w,'eta':eta,'share':.001,
                    'gammas':GAMMAS.tolist(),'projection_counts':projections.tolist(),
                    'windows_with_ties':ties,'level_bounds':[1/(w+1),w/(w+1)] if projected else None}}


def mixture_path(result,key,replicate=0):
    p=result['probabilities'];q=result['predictions'];valid=np.isfinite(p).all(axis=1)
    rng=np.random.default_rng(seed_for(key,replicate));u=rng.random(len(p))
    choices=np.sum(u[:,None]>np.cumsum(np.nan_to_num(p),axis=1),axis=1)
    choices=np.minimum(choices,p.shape[1]-1)
    out=np.full(len(p),np.nan);out[valid]=q[np.arange(len(p))[valid],choices[valid]]
    return out,choices


def circular_means(values,block,B,rng):
    n,d=values.shape;blocks=int(np.ceil(n/block));remainder=n-(blocks-1)*block
    starts=rng.integers(0,n,size=(B,blocks))
    extended=np.concatenate([values,values[:block]],axis=0)
    prefix=np.vstack([np.zeros(d),np.cumsum(extended,axis=0)])
    sums=prefix[starts+block]-prefix[starts]
    sums[:,-1]=prefix[starts[:,-1]+remainder]-prefix[starts[:,-1]]
    return sums.sum(axis=1)/n


def loss_gate(validation_losses,key):
    names=list(validation_losses);assert names[0]=='Raw'
    differences=np.column_stack([validation_losses[n]-validation_losses['Raw'] for n in names[1:]])
    means=differences.mean(axis=0);bands=[]
    for block in (20,60):
        draws=circular_means(differences,block,499,np.random.default_rng(seed_for(key,block)))
        sd=draws.std(axis=0,ddof=1);positive=sd>1e-15
        standard=np.zeros_like(draws);standard[:,positive]=(draws[:,positive]-means[positive])/sd[positive]
        critical=max(0.,float(np.quantile(standard.max(axis=1),.95)))
        upper=means+critical*sd
        bands.append({'block_observations':block,'critical':critical,'upper':upper.tolist(),'sd':sd.tolist()})
    upper=np.maximum(bands[0]['upper'],bands[1]['upper']);best=int(np.argmin(upper))
    selected=names[best+1] if upper[best]<0 else 'Raw'
    plain=min(names,key=lambda name:(float(np.mean(validation_losses[name])),names.index(name)))
    return {'selected':selected,'past_minimum_selected':plain,'candidates':names,
            'mean_differences':means.tolist(),'upper_bounds':upper.tolist(),'bands':bands,
            'bootstrap_draws':499,'guarantee':'empirical heuristic only'}
