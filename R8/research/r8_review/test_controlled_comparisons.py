import numpy as np
from scipy.integrate import quad
from scipy import stats
from controlled_comparisons import candidates,qr_fit,weighted_quantile,loss
from complexity_simulation import expected_loss


def test_one_parameter_optima_against_linear_program():
    rng=np.random.default_rng(18)
    y=rng.standard_t(5,300)*.01
    q=np.full(300,-.02)
    sigma=np.exp(rng.normal(-4,.4,300))
    alpha=.01
    predictions,params=candidates(y,q,sigma,q,sigma,alpha)
    beta,_=qr_fit(np.ones((len(y),1)),y-q,alpha)
    assert abs(loss(y,q+beta[0],alpha).sum()-loss(y,predictions['Shift-ERM'],alpha).sum())<1e-9
    beta,_=qr_fit(sigma[:,None],y-q,alpha)
    assert abs(loss(y,q+sigma*beta[0],alpha).sum()-loss(y,predictions['Vol-ERM'],alpha).sum())<1e-9


def test_expected_pinball_against_independent_quadrature():
    for kind in ['normal','t5']:
        distribution=stats.norm(scale=.02) if kind=='normal' else stats.t(5,scale=.02*np.sqrt(3/5))
        for alpha in [.01,.05]:
            for q in [-.08,-.02,.01]:
                integrand=lambda y: float((alpha-(y<q))*(y-q)*distribution.pdf(y))
                numeric=quad(integrand,-np.inf,q,epsabs=1e-11)[0]+quad(integrand,q,np.inf,epsabs=1e-11)[0]
                analytic=float(expected_loss(kind,np.array([q]),np.array([.02]),alpha)[0])
                assert abs(numeric-analytic)<1e-9


def test_weighted_quantile_subgradient_with_ties():
    values=np.array([-2.,-2.,0.,1.,5.])
    weights=np.array([1.,4.,2.,3.,1.])
    for p in [.01,.5,.99]:
        value=weighted_quantile(values,weights,p)
        assert weights[values<value].sum()<=p*weights.sum()+1e-12
        assert weights[values<=value].sum()>=p*weights.sum()-1e-12
