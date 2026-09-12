"""Independent enumeration, integration, monotonicity and replay checks."""
from fractions import Fraction as F
import itertools
import json
import math
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import pandas as pd
from scipy.integrate import quad
from engine import finite, error, uniform_risk, overlap_error_by_k, minimum_length
from run import ROOT, OUT, sha


def enumerate_histories(n, alpha, epsilon, retention):
    """Enumerate observable refresh patterns and ordered distinct-draw signs.

    Unlike the producer, this does not use a binomial PMF, CDF, likelihood
    cutoff or a random-length sufficient-statistic aggregation.
    """
    t0, t1 = alpha*(1-epsilon)/F(3,4), alpha*(1+epsilon)/F(3,4)
    c0, c1 = 1-alpha/t0, 1-alpha/t1
    total0 = total1 = e = regret = F(0)
    affinity = 0.; states = 0; expected_k = F(0)
    for refresh in itertools.product([False, True], repeat=n-1):
        k = 1+sum(refresh)
        pattern = retention**(n-k)*(1-retention)**(k-1)
        if not pattern:
            continue
        expected_k += pattern*k
        for signs in itertools.product([False, True], repeat=k):
            j = sum(signs)
            p0 = pattern*t0**j*(1-t0)**(k-j)
            p1 = pattern*t1**j*(1-t1)**(k-j)
            total0 += p0; total1 += p1
            e += min(p0, p1)/2
            # Direct posterior correction and two squared-regret components.
            optimum = (t0*p0*c0+t1*p1*c1)/(t0*p0+t1*p1)
            regret += (t0*p0*(optimum-c0)**2+t1*p1*(optimum-c1)**2)/4
            affinity += math.sqrt(float(p0*p1))
            states += 1
    assert total0 == total1 == 1
    assert expected_k == 1+(n-1)*(1-retention)
    return e, regret, affinity, states


def enumeration_checks():
    cases = states = 0; max_error = 0.; certificates = []
    for alpha in [F(1,100), F(1,20)]:
        for epsilon in [F(1,10), F(1,5)]:
            for retention in [F(0), F(1,2), F(3,4)]:
                for n in range(1,8):
                    exact, risk, affinity, count = enumerate_histories(n, alpha, epsilon, retention)
                    result = finite(n, float(alpha), float(epsilon), float(retention))
                    differences = [abs(float(exact)-result['best_average_error']),
                                   abs(float(risk)-result['scalar_bayes_regret']),
                                   abs(affinity-result['affinity'])]
                    max_error = max(max_error, *differences)
                    assert max(differences) < 3e-13
                    cases += 1; states += count
                    if n == 7 and alpha == F(1,100) and epsilon == F(1,5):
                        certificates.append(dict(n=n, retention=str(retention),
                                                 error=str(exact), scalar_bayes_risk=str(risk)))
    (OUT/'rational_certificates.json').write_text(json.dumps(certificates, indent=2)+'\n')
    return dict(cases=cases, enumerated_states=states, maximum_error=max_error)


def integration_checks():
    errors = []; optimality = []
    for alpha in [.01, .05]:
        for epsilon in [.1, .2]:
            for theta in [alpha*(1-epsilon)/.75, alpha*(1+epsilon)/.75]:
                def numerical(c):
                    def loss(s):
                        u=c-s
                        return u*(alpha-(u<0))
                    total = 0.
                    for lo, hi, density in [(-1.,0.,1-theta),(0.,1.,theta)]:
                        points = [c] if lo < c < hi else None
                        total += density*quad(loss,lo,hi,points=points,epsabs=1e-13,epsrel=1e-13)[0]
                    return total
                for c in [-2.,-.5,0.,.1,.5,1.,2.]:
                    errors.append(abs(numerical(c)-uniform_risk(c,alpha,theta)))
                cstar=1-alpha/theta
                for c in [0.,.1,.5,1.]:
                    optimality.append(abs((numerical(c)-numerical(cstar))-.5*theta*(c-cstar)**2))
                expected=alpha*.5-theta*(.5-.5*.5**2)
                errors.append(abs(numerical(.5)-numerical(0)-expected))
    assert max(errors+optimality) < 1e-13
    return dict(risk_checks=len(errors), oracle_checks=len(optimality), maximum_error=max(errors+optimality))


def contiguous_checks():
    manifest=json.loads((OUT/'contiguous.json').read_text())
    assert manifest['producer_sha256']==sha(ROOT/'research/r8_information_limit/contiguous.py')
    for name,wanted in manifest['inputs'].items():assert sha(ROOT/name)==wanted
    assert manifest['output_sha256']==sha(OUT/'contiguous.csv')
    primary=pd.read_csv(OUT/'run/finite.csv')
    extra=pd.read_csv(OUT/'contiguous.csv')
    assert len(primary)==len(extra)==64
    for p,e in zip(primary.itertuples(),extra.itertuples()):
        assert (p.n,p.alpha,p.epsilon,p.retention)==(e.n,e.alpha,e.epsilon,e.retention)
        average=sum(p.retention**j for j in range(1,e.H+1))/e.H
        theta=(p.theta0+p.theta1)/2
        assert abs(e.binary_contiguous_lower-(p.binary_regret-theta*.375*average))<1e-15
        assert abs(e.scalar_contiguous_lower-(p.scalar_bayes_regret-theta*.5*average))<1e-15
    # Gauss nodes on intervals split at every action's loss kink. The
    # resulting finite transition matrix independently verifies the exact
    # Markov transfer and envelope for arbitrary history-dependent actions.
    # This is an algebra check, not a claim that a finite discrete grid is
    # identical to the nonatomic statistical experiment used in the proof.
    z,w=np.polynomial.legendre.leggauss(2)
    values=np.concatenate([(z+1)/2-1,(z+1)/4,(z+1)/4+.5])
    cases=0;maximum=0.
    for alpha in [.01,.05]:
        for theta in [alpha*.8/.75,alpha*1.2/.75]:
            marginal=np.concatenate([(1-theta)*w/2,theta*w/4,theta*w/4])
            for r in [0.,.5,.75]:
                transition=r*np.eye(6)+(1-r)*np.tile(marginal,(6,1))
                for n in [1,2,3]:
                    histories=np.asarray(list(itertools.product(range(6),repeat=n)),dtype=int)
                    probability=marginal[histories[:,0]].copy()
                    for j in range(1,n):probability*=transition[histories[:,j-1],histories[:,j]]
                    assert abs(probability.sum()-1)<1e-12
                    history_values=values[histories]
                    policies=[np.where(history_values.mean(axis=1)>0,.5,0),
                              np.where(history_values[:,-1]>0,0.,1.),
                              np.where(history_values[:,0]>history_values[:,-1],1.,.5)]
                    for action in policies:
                        L=float(action.max())
                        u=action[:,None]-values[None,:]
                        before=-values
                        difference=u*(alpha-(u<0))-before*(alpha-(before<0))
                        pop=difference@marginal
                        last=difference[np.arange(len(histories)),histories[:,-1]]
                        for H in [1,2,3,5]:
                            actual=0.
                            for j in range(1,H+1):
                                future=np.linalg.matrix_power(transition,j)[histories[:,-1]]
                                actual+=probability@np.sum(future*difference,axis=1)/H
                            factor=sum(r**j for j in range(1,H+1))/H
                            predicted=probability@pop+factor*(probability@(last-pop))
                            maximum=max(maximum,abs(actual-predicted))
                            assert abs(actual-probability@pop)<=theta*(L-L*L/2)*factor+1e-14
                            cases+=1
    assert maximum<1e-13
    return dict(configurations=64,markov_transfer_cases=cases,maximum_transfer_error=maximum,
                positive_binary_bounds=int((extra.binary_contiguous_lower>0).sum()),
                positive_scalar_bounds=int((extra.scalar_contiguous_lower>0).sum()),
                negative_bounds_retained=True,scalar_range_restriction='[0,1]')


def validation():
    primary=json.loads((OUT/'run/validation.json').read_text())
    assert primary == json.loads((OUT/'replay/validation.json').read_text())
    for name,wanted in primary['inputs'].items():
        assert sha(ROOT/name)==wanted,name
    for folder in ['run','replay']:
        for name,wanted in primary['outputs'].items():
            assert sha(OUT/folder/name)==wanted,(folder,name)
    before=json.loads((OUT/'before.json').read_text())
    assert all(sha(ROOT/name)==wanted for name,wanted in before['canonical'].items())
    frame=pd.read_csv(OUT/'run/finite.csv')
    # Ordered sample sizes and persistence levels in the complete grid.
    for _,part in frame.groupby(['alpha','epsilon','retention']):
        ordered=part.sort_values('n')
        assert np.all(np.diff(ordered.best_average_error)<=1e-12)
        assert np.all(np.diff(ordered.scalar_bayes_regret)<=1e-15)
    for _,part in frame.groupby(['alpha','epsilon','n']):
        ordered=part.sort_values('retention')
        assert np.all(np.diff(ordered.best_average_error)>=-1e-12)
        assert np.all(np.diff(ordered.scalar_bayes_regret)>=-1e-15)
    # Check full n=1,...,1000 trajectories, not only four displayed lengths.
    for alpha in [.01,.05]:
        for epsilon in [.1,.2]:
            theta0,theta1=alpha*(1-epsilon)/.75,alpha*(1+epsilon)/.75
            curve=overlap_error_by_k(np.arange(1,1001),theta0,theta1)
            assert np.all(np.diff(curve)<1e-13)
    lengths=pd.read_csv(OUT/'run/lengths.csv')
    for row in lengths.itertuples():
        n=int(row.minimum_n)
        assert error(n,row.alpha,row.epsilon,row.retention)<=row.target_error
        assert error(n-1,row.alpha,row.epsilon,row.retention)>row.target_error
    convergence=pd.read_csv(OUT/'run/convergence.csv')
    smallest=convergence[convergence.alpha==.0001]
    largest=convergence[convergence.alpha==.01]
    assert smallest.absolute_difference.max()<largest.absolute_difference.max()
    limits=pd.read_csv(OUT/'run/poisson.csv')
    assert limits.omitted_probability_bound.max()<1e-14
    assert (limits.poisson_error >= limits.zero_count_error_lower).all()
    with tempfile.TemporaryDirectory(prefix='irfa-information-whitespace-') as folder:
        a=Path(folder)/'empty';a.write_text('')
        for path in sorted((ROOT/'research/r8_information_limit').glob('*')):
            if not path.is_file():continue
            p=subprocess.run(['git','diff','--no-index','--check',str(a),str(path)],capture_output=True,text=True)
            assert not p.stdout and not p.stderr and p.returncode in (0,1),path
        b=Path(folder)/'bad';b.write_text('bad whitespace \n')
        negative=subprocess.run(['git','diff','--no-index','--check',str(a),str(b)],capture_output=True,text=True)
        assert 'trailing whitespace' in negative.stdout and negative.returncode not in (0,1)
    return dict(fresh_process_replay_exact=True, protected_files=len(before['canonical']),
                canonical_unchanged=True, minimum_length_checks=len(lengths),
                complete_iid_monotonicity_points=4000,
                rare_event_max_error_alpha_0001=float(smallest.absolute_difference.max()),
                rare_event_max_error_alpha_01=float(largest.absolute_difference.max()),
                poisson_tail_bound=float(limits.omitted_probability_bound.max()),
                whitespace_negative_control=True, git_metadata_present=(ROOT/'.git').exists())


if __name__=='__main__':
    result=dict(status='passed',enumeration=enumeration_checks(),integration=integration_checks(),
                contiguous=contiguous_checks(),
                validation=validation(),producer_sha256=sha(__file__),
                proof_sha256=sha(ROOT/'research/r8_information_limit/PROOF.md'))
    (OUT/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
