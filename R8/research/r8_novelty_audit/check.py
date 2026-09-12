"""Alternative deterministic checks; no imports from manuscript producers."""
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/r8_novelty_audit'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add(*polynomials):
    answer = [F(0)] * max(map(len, polynomials))
    for poly in polynomials:
        for i, value in enumerate(poly):
            answer[i] += value
    return answer


def multiply(a, b):
    answer = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            answer[i+j] += x*y
    return answer


def shifted(poly, power, coefficient):
    return [F(0)]*power + [coefficient*x for x in poly]


def composition_cdf(n, theta, upper):
    # Sum all completed block compositions, then allow a final partial
    # triple. This differs from the producer's order-three recurrence.
    triple = [F(1, 2), F(-2), F(5, 2)] if upper else [F(0), F(0), F(1, 2)]
    powers = [[F(1)]]
    for _ in range(n//3):
        powers.append(multiply(powers[-1], triple))

    @lru_cache(None)
    def completed(length):
        if length < 0:
            return [F(0)]
        parts = []
        for triples in range(length//3+1):
            singles = length-3*triples
            weight = F(math.comb(singles+triples, triples)) * (1-theta)**singles * theta**triples
            parts.append(shifted(powers[triples], singles, weight))
        return add(*parts)

    def start(length):
        return add(completed(length), shifted(completed(length-1), 1, theta),
                   shifted(completed(length-2), 2, theta))

    return [x/(1+2*theta) for x in add(start(n), shifted(start(n-1), 1, theta),
                                      shifted(start(n-2), 2, theta))]


def count_witness():
    n, p, d = 125, F(99, 100), F(813, 100000)
    costs = []
    for theta in [F(0), F(1, 2)]:
        integral = F(0)
        for upper, (lo, hi) in enumerate([(F(0), F(1, 2)), (F(1, 2), F(1))]):
            poly = multiply(composition_cdf(n, theta, bool(upper)), [-p, F(1)])
            integral += sum(x*(hi**(j+1)-lo**(j+1))/(j+1) for j, x in enumerate(poly))
        costs.append((1-p)**2/2-integral)
    expected = json.loads((ROOT/'artifacts/r8_count_law/exact_witness.json').read_text())
    assert costs == [F(expected['iid_cost']), F(expected['renewal_cost'])]
    raw = d*d/2
    assert costs[1] < raw < costs[0]
    # Substituting the iid count law for both processes erases the sign
    # reversal, despite identical pairwise independence and margins.
    assert not (costs[0] < raw < costs[0])
    return dict(exact_composition_certificate=True, iid_cost=str(costs[0]),
                dependent_cost=str(costs[1]), raw_regret=str(raw),
                iid_change=float(costs[0]-raw), dependent_change=float(costs[1]-raw),
                delta_interval=[math.sqrt(2*float(costs[1])), math.sqrt(2*float(costs[0]))],
                iid_count_substitution_rejected=True)


def pinball(u, alpha):
    return np.maximum(alpha*u, -(1-alpha)*u)


def markov_transfer():
    transition = np.array([[.6, .3, .1], [.2, .2, .6], [.5, .1, .4]])
    system = transition.T-np.eye(3)
    system[-1] = 1
    marginal = np.linalg.solve(system, np.array([0., 0., 1.]))
    assert np.max(abs(marginal@transition-marginal)) < 1e-15
    assert np.max(abs(marginal[:, None]*transition-(marginal[:, None]*transition).T)) > .01
    states = np.array([-2., .3, 2.])
    powers = [np.linalg.matrix_power(transition, j) for j in range(1, 18)]
    beta = np.array([marginal@np.abs(matrix-marginal).sum(axis=1)/2 for matrix in powers])
    rows = []
    for n in range(1, 6):
        paths = np.array(list(itertools.product(range(3), repeat=n)))
        probabilities = marginal[paths[:, 0]].copy()
        for t in range(1, n):
            probabilities *= transition[paths[:, t-1], paths[:, t]]
        assert abs(probabilities.sum()-1) < 1e-14
        values = states[paths]
        for alpha in [.01, .2, .5]:
            rank = min(n, math.ceil((n+1)*(1-alpha)))
            rules = [np.sort(values, axis=1)[:, rank-1], values.mean(axis=1),
                     values[:, -1], np.full(len(paths), .2)]
            for rule_id, action in enumerate(rules):
                variance = probabilities@(action-probabilities@action)**2
                loss = pinball(action[:, None]-states, alpha)-pinball(-states, alpha)
                independent = probabilities@(loss@marginal)
                actual = np.array([probabilities@np.sum(matrix[paths[:, -1]]*loss, axis=1)
                                   for matrix in powers])
                for H in [1, 3, 17]:
                    deviation = float(actual[:H].mean()-independent)
                    bound = float(math.sqrt(variance)*np.sqrt(beta[:H]).mean())
                    assert abs(deviation) <= bound+2e-14
                    rows.append(dict(n=n, alpha=alpha, rule=rule_id, H=H,
                                     deviation=deviation, bound=bound))
    frame = pd.DataFrame(rows)
    assert frame.deviation.abs().max() > .01, 'Omitting the boundary escaped detection'
    frame.to_csv(OUT/'nonreversible_transfer.csv', index=False)
    return dict(cases=len(rows), nonreversible=True,
                stationary_distribution=marginal.tolist(),
                omitted_boundary_rejected=True, maximum_absolute_boundary=float(frame.deviation.abs().max()))


def uniform_moments():
    rows = []
    for alpha in [F(1, 100), F(1, 20), F(1, 2)]:
        p = 1-alpha
        for n in [125, 250, 1000, 1000000, 100000000]:
            rank = min(n, math.ceil((n+1)*p))
            bias = F(rank, n+1)-p
            variance = F(rank*(n+1-rank), (n+1)**2*(n+2))
            second = variance+bias*bias
            expected = n*second/2
            leading = p*alpha/2
            for h, lam in itertools.product([0., .2], [0., .25, .5, 1.]):
                local = .5*(lam*lam*n*float(second)+2*lam*(lam-1)*h*math.sqrt(n)*float(bias)
                            +(lam*lam-2*lam)*h*h)
                limit = .5*(lam*lam*float(p*alpha)+(lam*lam-2*lam)*h*h)
                rows.append(dict(alpha=float(alpha), n=n, h=h, fraction=lam,
                                 scaled_exact=local, limit=limit, error=abs(local-limit)))
            if n == 100000000:
                assert abs(float(expected-leading)) < 1e-8
    frame = pd.DataFrame(rows)
    assert frame[frame.n == 100000000].error.max() < 2e-5
    selected = frame[(frame.alpha == .01)&(frame.h == 0)&(frame.fraction == 1)&(frame.n == 100000000)].iloc[0]
    assert abs(selected.scaled_exact-2*selected.limit) > 1e-3, 'Missing one-half factor escaped detection'
    frame.to_csv(OUT/'uniform_moments.csv', index=False)
    return dict(cases=len(rows), exact_order_statistic_moments=True,
                largest_n_error=float(frame[frame.n == 100000000].error.max()),
                missing_half_factor_rejected=True)


def zero_count_bound():
    a0, a1 = .8/.75, 1.2/.75
    delta = 1/a0-1/a1
    rows = []
    for tau, retention in itertools.product([1., 2.5, 5.], [0., .5, .75]):
        mu0, mu1 = tau*(1-retention)*a0, tau*(1-retention)*a1
        z0, z1 = math.exp(-mu0), math.exp(-mu1)
        constant = delta*delta*a0*a1*z0*z1/(4*(a0*z0+a1*z1))
        assert constant > 0
        for n in [125, 250, 1000, 10000]:
            alpha = tau/n
            k = np.arange(1, n+1)
            weight = binom.pmf(k-1, n-1, 1-retention)
            b0, b1 = np.exp(k*np.log1p(-a0*alpha)), np.exp(k*np.log1p(-a1*alpha))
            scaled = delta*delta*a0*a1*np.dot(weight, b0*b1/(a0*b0+a1*b1))/4
            H = (3*n)//7
            boundary = (a0+a1)/4*sum(retention**j for j in range(1, H+1))/H
            rows.append(dict(tau=tau, retention=retention, n=n, alpha=alpha,
                             scaled_zero_count=scaled, limit=constant,
                             scaled_contiguous_lower=scaled-boundary,
                             absolute_lower=alpha*scaled))
    frame = pd.DataFrame(rows)
    for _, group in frame.groupby(['tau', 'retention']):
        group = group.sort_values('n')
        assert abs(group.iloc[-1].scaled_zero_count-group.iloc[-1].limit) < abs(group.iloc[0].scaled_zero_count-group.iloc[0].limit)
        # The assertion concerns risk divided by alpha, not a constant
        # lower bound on unnormalised loss as alpha tends to zero.
        assert group.iloc[-1].absolute_lower < group.iloc[0].absolute_lower/20
    frame.to_csv(OUT/'zero_count.csv', index=False)
    return dict(cases=len(rows), positive_limit_cases=9,
                poisson_approximation_not_used=True, unnormalised_positive_limit_rejected=True)


if __name__ == '__main__':
    result = dict(status='passed', producer_sha256=sha(__file__),
                  inputs={p:sha(ROOT/p) for p in ['research/r8_novelty_audit/PROTOCOL.md',
                                                'artifacts/r8_count_law/exact_witness.json']},
                  witness=count_witness(), transfer=markov_transfer(),
                  fixed_level=uniform_moments(), rare_event=zero_count_bound(),
                  independent_implementation=True, independent_external_reviewer=False,
                  new_paths=0, new_forecasts=0)
    result['outputs']={p.name:sha(p) for p in sorted(OUT.glob('*.csv'))}
    target = OUT/'mathematical_checks.json'
    if target.exists():
        assert json.loads(target.read_text()) == result, 'Fresh-process audit differs'
    else:
        target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
