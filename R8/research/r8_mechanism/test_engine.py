import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad
import engine as e


@pytest.mark.parametrize('kind',['normal','t5'])
@pytest.mark.parametrize('alpha',[.01,.05])
def test_expected_loss_integral(kind,alpha):
    scale=.02
    density=(lambda x:stats.norm.pdf(x,scale=scale)) if kind=='normal' else (lambda x:stats.t.pdf(x/(scale*np.sqrt(3/5)),5)/(scale*np.sqrt(3/5)))
    q=float(e.old.conditional_quantile(kind,scale,alpha))+.006
    left=quad(lambda x:(1-alpha)*(q-x)*density(x),-np.inf,q,epsabs=1e-12)[0]
    right=quad(lambda x:alpha*(x-q)*density(x),q,np.inf,epsabs=1e-12)[0]
    assert abs(e.old.expected_loss(kind,q,scale,alpha)-left-right)<1e-11


def test_stationary_ar_and_monotone_hits():
    x=np.array([.4,-.5,1.,-.2]);z=e.ar_path(x,.8)
    assert z[0]==x[0]
    assert np.allclose(z[1:],.8*z[:-1]+.6*x[1:])
    assert np.array_equal(e.ar_path(x,0),x)
    for kind in ('normal','t5'):
        y,clipped=e.marginal(z,kind);assert clipped==0
        for alpha in e.ALPHAS:
            q=e.old.conditional_quantile(kind,e.V0,alpha)
            assert np.array_equal(y<q,z<stats.norm.ppf(alpha))


def test_covariance_and_finite_count_identity():
    r=.7
    assert abs(e.covariance(.5,r)-np.arcsin(r)/(2*np.pi))<1e-13
    assert e.count_theory(.01,0,250)['finite_count_variance']==250*.01*.99
    result=e.count_theory(.05,.8,2)
    assert abs(result['finite_count_variance']-(2*.05*.95+2*e.covariance(.05,.8)))<1e-13
    assert result['remainder_bound']<1e-13


@pytest.mark.parametrize('alpha',[.01,.05])
def test_location_equivariance_and_pot(alpha):
    y=np.random.default_rng(129).normal(size=250)
    for d in (-.4,0.,.3):
        assert abs(d-e.m.cp(d-y,alpha)+e.m.cp(-y,alpha))<1e-14
    s=-y
    for tau in (.8,.9):
        f=e.m.fit_pot(s,tau,alpha);g=e.m.fit_pot(s+.25,tau,alpha)
        assert abs(e.pot_at(f,s,alpha)+.25-g['quantile'])<1e-6
    f=e.m.fit_pot(s[:125],.9,alpha)
    assert f['fallback'] and f['n_tail']==13
    assert e.pot_at(f,s[:125],alpha)==e.m.cp(s[:125],alpha)


@pytest.mark.parametrize('alpha',[.01,.05])
def test_state_alpha_and_translation(alpha):
    rng=np.random.default_rng(521);n=125;s=np.exp(.3*rng.normal(size=n))
    y=s*rng.normal(size=n);q=-2*s
    f=e.state_fit(y,q,s,2,.01,alpha)
    g=e.state_fit(y,q+.35,s,2,.01,alpha)
    a=e.m.predict_state(q,s,f);b=e.m.predict_state(q+.35,s,g)
    assert np.max(np.abs(a-b))<1e-8
    assert f['alpha']==alpha and f['certificate']['gap']<1e-8
    # A constant predictor must use the requested, not default, quantile level.
    X=np.ones((n,1));beta,_=e.m.l1_fit(X,y,0,alpha)
    assert abs(beta[0]-np.quantile(y,alpha,method='inverted_cdf'))<1e-8


def test_garch_oracle_and_state_span():
    y,s=e.old.garch('normal',125,321)
    q=e.old.conditional_quantile('normal',s,.01)
    raw=q+e.distort(s,'state')
    X,params=e.m.design(s,p=2)
    beta=np.linalg.lstsq(X,e.distort(s,'state'),rcond=None)[0]
    assert np.max(np.abs(X@beta-(raw-q)))<1e-14
    oracle=e.old.expected_loss('normal',q,s,.01)
    assert np.all(e.old.expected_loss('normal',q+.001,s,.01)>oracle)
    assert np.allclose(e.hit_probability('normal',q,s),.01)


def test_selection_does_not_use_test_states():
    y,s=e.old.garch('t5',125,823)
    test=np.array([.01,.02,.03])
    _,first=e.garch_calculation(y,s,test,'t5')
    _,second=e.garch_calculation(y,s,test*3,'t5')
    assert first==second


@pytest.mark.parametrize('large',[1.,1e8])
def test_unbounded_transfer_inequality(large):
    # Explicit finite joint law with very different correction magnitudes.
    pa=np.array([.999,.001]);conditional=np.array([[.9,.1],[.1,.9]])
    ps=pa@conditional;P=pa[:,None]*conditional;Q=pa[:,None]*ps
    c=np.array([.03,large])[:,None];s=np.array([-2.,3.])[None,:]
    D=e.m.loss(0.,s-c)-e.m.loss(0.,s)
    tv=.5*np.abs(P-Q).sum()
    actual=abs(np.sum((P-Q)*D))
    bound=2*np.sqrt(np.dot(pa,c[:,0]**2)*tv)
    assert actual<=bound
    variance=np.dot(pa,(c[:,0]-np.dot(pa,c[:,0]))**2)
    assert actual<=2*np.sqrt(variance*tv)
    fixed=e.m.loss(0.,s-.3)-e.m.loss(0.,s)
    assert abs(np.sum((P-Q)*fixed))<1e-12
