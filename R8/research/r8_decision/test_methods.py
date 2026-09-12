"""Independent identities and information-boundary tests for research methods."""
import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import genpareto
import methods as m


def test_l1_intercept_and_penalty_limit():
    x=np.linspace(-2,2,200);X=np.c_[np.ones(200),x,x*x,x*x*x]
    y=np.sin(x)+.2*np.cos(19*x)
    beta,cert=m.l1_fit(X,y,1e3)
    reference=np.quantile(y,.01,method='inverted_cdf')
    assert np.max(np.abs(beta[1:]))<1e-8
    assert abs(m.loss(y,X@beta).mean()-m.loss(y,reference).mean())<1e-10
    assert cert['gap']<1e-7


def test_unpenalised_matches_independent_lp():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'r8_review'))
    from controlled_comparisons import qr_fit
    x=np.linspace(-2,2,180);X=np.c_[np.ones(len(x)),x];y=.3+x*.2+np.sin(x*17)
    fitted,_=m.l1_fit(X,y,0.)
    reference,_=qr_fit(X,y,.01)
    assert abs(m.loss(y,X@fitted).sum()-m.loss(y,X@reference).sum())<1e-8


@pytest.mark.parametrize('shape',[-.4,0.,1e-10,.2,.8])
def test_gpd_inversion(shape):
    q=m.gpd_quantile(.7,shape,.3,.1)
    expected=.7+genpareto.ppf(.9,shape,scale=.3)
    assert q==pytest.approx(expected,abs=1e-9)
    assert .1*genpareto.sf(q-.7,shape,scale=.3)==pytest.approx(.01,abs=1e-10)


def test_pot_recovers_deterministic_exponential_quantile_grid():
    # Deterministic quantiles, not a stochastic financial simulation.
    x=-np.log(1-(np.arange(2000)+.5)/2000)
    fit=m.fit_pot(x,.9)
    assert not fit['fallback']
    assert abs(fit['quantile']-np.log(100))<.15
    assert fit['support_margin']>0


def test_dtaci_independent_one_step():
    s=np.linspace(-1,1,520);q=np.zeros(len(s))
    result=m.dtaci(s,q,w=500)
    initial=np.full(7,.01);weights=np.full(7,1/7)
    qs=np.quantile(s[:500],1-initial)
    np.testing.assert_allclose(result['predictions'][500],-qs,atol=1e-14)
    errors=s[500]>qs
    expected=np.clip(initial+m.GAMMAS*(.01-errors),1/501,500/501)
    np.testing.assert_allclose(result['states'][501],expected,atol=1e-14)
    beta=0.;ell=.01*(beta-initial)-np.minimum(0,beta-initial)
    w=weights*np.exp(-result['meta']['eta']*ell);w/=w.sum();w=.999*w+.001/7
    np.testing.assert_allclose(result['probabilities'][501],w,atol=1e-14)


def test_dtaci_strict_tie_and_extended_boundaries():
    s=np.zeros(650);q=np.zeros(len(s))
    projected=m.dtaci(s,q)
    # Exact equality never causes a strict violation.
    assert np.all(s[500:,None]<=-projected['predictions'][500:])
    np.testing.assert_allclose(projected['states'][501],.01+m.GAMMAS*.01)
    # An abrupt record makes some unprojected experts negative next day.
    s[500]=10.;s[501:]=0.
    original=m.dtaci(s,q,projected=False)
    assert np.isneginf(original['predictions'][501]).any()


def test_inverse_with_repeated_calibration_scores():
    values=np.array([-1.,-1.,-1.,0.,0.,1.,2.,2.])
    for score in [-2.,-1.,-.9,-.2,0.,.2,.9,1.,1.3,2.,3.]:
        beta=m.quantile_inverse(values,score)
        for level in np.linspace(.002,.998,123):
            shift=np.quantile(values,1-level)
            assert bool(score>shift)==bool(level>beta),(score,level,beta,shift)
    assert m.quantile_inverse(np.zeros(50),0.)==1.


def test_dtaci_prefix_invariance_and_mixture_loss():
    s=np.sin(np.arange(720)*.173);q=np.cos(np.arange(720)*.07)*.1
    one=m.dtaci(s,q);altered=s.copy();altered[660:]+=100
    two=m.dtaci(altered,q)
    np.testing.assert_array_equal(one['predictions'][:661],two['predictions'][:661])
    np.testing.assert_array_equal(one['probabilities'][:661],two['probabilities'][:661])
    y=q-s;t=620
    exact=sum(one['probabilities'][t,j]*float(m.loss(y[t],one['predictions'][t,j])) for j in range(7))
    vector=np.sum(one['probabilities'][t]*m.loss(y[t],one['predictions'][t]))
    assert exact==pytest.approx(vector)
    # Jensen: do not substitute the loss of the average threshold.
    assert m.loss(y[t],one['probabilities'][t]@one['predictions'][t])<=exact+1e-12


def test_gate_no_gain_and_known_deterministic_gain():
    z=np.linspace(.1,.2,600)
    no=m.loss_gate({'Raw':z,'A':z.copy(),'B':z+.01,'C':z+.02},'test')
    assert no['selected']=='Raw'
    yes=m.loss_gate({'Raw':z,'A':z-.02,'B':z+.01,'C':z.copy()},'test')
    assert yes['selected']=='A'
    assert yes['upper_bounds'][0]<0


def test_circular_block_sums_match_explicit_resampling():
    x=np.c_[np.arange(123),np.sin(np.arange(123))]
    fast=m.circular_means(x,20,7,np.random.default_rng(42))
    rng=np.random.default_rng(42);starts=rng.integers(0,123,size=(7,7))
    manual=[]
    for row in starts:
        ix=((row[:,None]+np.arange(20))%123).ravel()[:123]
        manual.append(x[ix].mean(axis=0))
    np.testing.assert_allclose(fast,manual,atol=1e-12)
