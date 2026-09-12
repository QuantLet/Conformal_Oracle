"""Deterministic checks of the proposed bridge; no simulation or model fitting."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import integrate, optimize, special, stats

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT/'artifacts/r8_horizon_bridge'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pinball(x, alpha):
    return np.maximum(alpha*x, -(1-alpha)*x)


def finite_markov():
    rows = []
    states = np.array([-1., 1.]); pi = np.array([.5, .5])
    for n, alpha, rho in itertools.product([1, 3, 8], [.01, .05, .5], [0., .4, .9]):
        transition = rho*np.eye(2)+(1-rho)*np.tile(pi, (2, 1))
        paths = np.array(list(itertools.product(range(2), repeat=n)))
        probabilities = np.full(len(paths), .5)
        for j in range(1, n): probabilities *= transition[paths[:, j-1], paths[:, j]]
        assert abs(probabilities.sum()-1) < 1e-14
        rank = min(n, math.ceil((n+1)*(1-alpha)))
        correction = np.sort(states[paths], axis=1)[:, rank-1]
        centre = probabilities@correction
        variance = probabilities@(correction-centre)**2
        loss = pinball(correction[:, None]-states, alpha)-pinball(-states, alpha)
        independent = probabilities@(loss@pi)
        for horizon in [1, 2, 10, 100]:
            differences = []
            for lag in range(1, horizon+1):
                future = np.linalg.matrix_power(transition, lag)[paths[:, -1]]
                actual = probabilities@np.sum(loss*future, axis=1)
                differences.append(actual-independent)
            beta = .5*rho**np.arange(1, horizon+1)
            for weighting in ['uniform', 'increasing']:
                weights = np.ones(horizon) if weighting=='uniform' else np.arange(1, horizon+1)
                weights = weights/weights.sum()
                error = float(weights@differences)
                bound = float(np.sqrt(variance)*(weights@np.sqrt(beta)))
                assert abs(error) <= bound+2e-14, (n, alpha, rho, horizon, weighting)
                rows.append(dict(n=n, alpha=alpha, rho=rho, horizon=horizon,
                    weighting=weighting, exact_transfer_error=error, upper_bound=bound))
    pd.DataFrame(rows).to_csv(OUT/'finite_markov.csv', index=False)
    return len(rows)


def continuous_refresh():
    rows = []; errors = []
    for alpha in [.01, .05, .5]:
        p = 1-alpha
        # Exact integration over the two triangles separated by C=S.
        def expected_at(c):
            return integrate.quad(lambda s: float(pinball(c-s, alpha)), -p, c,
                                  epsabs=1e-13)[0] + integrate.quad(
                lambda s: float(pinball(c-s, alpha)), c, 1-p, epsabs=1e-13)[0]
        raw = expected_at(0.)
        independent = integrate.quad(lambda c: expected_at(c)-raw, -p, 1-p, epsabs=1e-12)[0]
        diagonal = -raw
        errors += [abs(raw-alpha*p/2), abs(independent-(1/6-alpha*p/2)), abs(independent-diagonal-1/6)]
        for rho, horizon in itertools.product([.2, .5, .9], [1, 10, 10000]):
            actual_mean = np.mean([(rho**j)*diagonal+(1-rho**j)*independent for j in range(1, horizon+1)])
            exact_boundary = -rho*(1-rho**horizon)/(6*horizon*(1-rho))
            errors.append(abs(actual_mean-independent-exact_boundary))
            bound = np.sqrt(1/12)*sum(rho**(j/2) for j in range(1, horizon+1))/horizon
            assert abs(exact_boundary) <= bound+1e-14
            rolling_boundary = -rho/6
            if horizon==10000:
                assert abs(rolling_boundary) > bound, 'Invalid rolling extension was not rejected'
            rows.append(dict(alpha=alpha, rho=rho, horizon=horizon,
                static_boundary=exact_boundary, transfer_bound=bound,
                rolling_boundary=rolling_boundary))
    assert max(errors)<1e-11
    pd.DataFrame(rows).to_csv(OUT/'continuous_refresh.csv', index=False)
    return dict(cases=len(rows), max_quadrature_error=max(errors),
                invalid_rolling_extension_rejected=True)


def local_information():
    rows = []; max_error = 0.
    for alpha, epsilon in itertools.product([.01, .05], [.1, .25]):
        p=1-alpha; z=stats.norm.ppf(p); f=stats.norm.pdf(z); omega=p*alpha
        tau=np.sqrt(omega)/f; hm=(1-epsilon)*tau; hp=(1+epsilon)*tau
        middle=(hm+hp)/2
        integrated=integrate.quad(lambda t: stats.norm.pdf(t-hp), -np.inf, middle,
            epsabs=1e-12)[0]+integrate.quad(lambda t: stats.norm.pdf(t-hm), middle, np.inf, epsabs=1e-12)[0]
        bound=2*stats.norm.cdf(-abs(hp-hm)/2)
        max_error=max(max_error, abs(integrated-bound))
        rows.append(dict(alpha=alpha, relative_distance=epsilon, h_minus=hm, h_plus=hp,
            minimum_sum_of_errors=bound, minimum_equal_prior_error=bound/2,
            loss_limit_minus=(omega-f*f*hm*hm)/(2*f),
            loss_limit_plus=(omega-f*f*hp*hp)/(2*f)))
        assert rows[-1]['loss_limit_minus']>0 and rows[-1]['loss_limit_plus']<0
        for h in [hm, hp]:
            objective=lambda lam:(lam*lam*omega-(2*lam-lam*lam)*f*f*h*h)/(2*f)
            oracle=f*f*h*h/(omega+f*f*h*h)
            solved=optimize.minimize_scalar(objective, bounds=(0, 1), method='bounded').x
            assert abs(oracle-solved)<1e-7
    assert max_error<1e-12
    pd.DataFrame(rows).to_csv(OUT/'gaussian_information.csv', index=False)
    return dict(cases=len(rows), max_quadrature_error=max_error, oracle_minima_checked=8)


def main():
    OUT.mkdir(exist_ok=False)
    protected=[PROJECT/'source/main_R8.tex', PROJECT/'source/supplement_R8.tex',
               *sorted((PROJECT/'source/sections_r8').glob('*.tex')),
               PROJECT/'artifacts/r8_referee_revision/final_validation.json']
    before={str(p.relative_to(PROJECT)):sha(p) for p in protected}
    result=dict(status='passed', finite_markov_checks=finite_markov(),
                continuous_refresh=continuous_refresh(), gaussian_information=local_information(),
                producer_sha256=sha(__file__), note_sha256=sha(Path(__file__).with_name('NOTE.md')),
                existing_documents_unchanged=True, no_new_random_draws=True, no_model_fitting=True)
    assert all(sha(PROJECT/p)==h for p,h in before.items())
    result['protected_inputs']=before
    result['outputs']={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ['protected_inputs','outputs']},flush=True)


if __name__=='__main__':main()
