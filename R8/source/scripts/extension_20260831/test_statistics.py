import numpy as np
from scipy.optimize import linprog
from scipy.stats import chi2
from panel_statistics import fit_scale,isotonic_quantile,predict_isotonic,qshift,scores


def test_kupiec_boundary_and_christoffersen_no_adjacent_hits():
    y=np.ones(1000);q=np.zeros(1000)
    z=scores(y,q)
    assert np.isclose(z['p_kup'],chi2.sf(-2000*np.log(.99),1))
    z=scores(-y,q)
    assert z['p_kup']==0.
    y=np.ones(1000);y[::100]=-1
    z=scores(y,q)
    assert np.isfinite(z['p_ind']) and np.isfinite(z['p_cc'])
    assert 0<=z['p_ind']<=1


def test_scale_coverage_matches_brute_force_with_ties_and_mixed_signs():
    rng=np.random.default_rng(4)
    for _ in range(20):
        y=np.round(rng.normal(size=30),1);q=np.round(rng.normal(size=30),1)
        alpha=.1;c,p=fit_scale(y,q,alpha)
        events=np.unique((y[q!=0]/q[q!=0]));events=events[events>0]
        candidates=np.r_[1,events,events[0]/2,events[:-1]+np.diff(events)/2,events[-1]*2]
        best=min(abs(np.mean(y<t*q)-alpha) for t in candidates)
        assert np.isclose(abs(p-alpha),best)
        assert p==np.mean(y<c*q)


def test_quantile_pav_agrees_with_independent_linear_program():
    rng=np.random.default_rng(71)
    for alpha in [.01,.1,.5]:
        for _ in range(8):
            x=np.sort(rng.integers(0,8,size=20));y=rng.normal(size=20)
            levels,values=isotonic_quantile(x,y,alpha)
            q=predict_isotonic(levels,values,x)
            # Independent LP: q_i + u_i - v_i = y_i, q nondecreasing;
            # equality within every repeated covariate value.
            n=len(y);objective=np.r_[np.zeros(n),np.full(n,alpha),np.full(n,1-alpha)]
            eq=np.c_[np.eye(n),np.eye(n),-np.eye(n)]
            ub=np.zeros((n-1,3*n))
            for i in range(n-1):ub[i,i]=1;ub[i,i+1]=-1
            extra=[]
            for i in range(n-1):
                if x[i]==x[i+1]:extra.append(ub[i])
            eq=np.r_[eq,np.array(extra).reshape(-1,3*n)]
            result=linprog(objective,A_ub=ub,b_ub=np.zeros(n-1),A_eq=eq,
                           b_eq=np.r_[y,np.zeros(len(extra))],bounds=[(None,None)]*n+[(0,None)]*(2*n),method='highs')
            assert result.success
            loss=np.sum((alpha-(y<q))*(y-q))
            assert np.isclose(loss,result.fun,atol=1e-8)
            assert np.all(np.diff(q)>=0)


def test_finite_sample_order_statistic():
    assert qshift(np.arange(250),.01)==248
    assert qshift(np.arange(125),.01)==124
