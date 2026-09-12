"""Deterministic finite experiments; no random observations or model fits."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom, poisson


def parameters(alpha, epsilon, action=.5):
    theta0 = alpha * (1-epsilon) / (1-action/2)
    theta1 = alpha * (1+epsilon) / (1-action/2)
    if not 0 < alpha < theta0 < theta1 < 1:
        raise ValueError('Both optimal corrections must lie strictly inside (0,1)')
    return theta0, theta1, 1-alpha/theta0, 1-alpha/theta1


def overlap_error_by_k(k, theta0, theta1):
    k = np.asarray(k, dtype=int)
    intercept = np.log1p(-theta0) - np.log1p(-theta1)
    slope = np.log(theta1/theta0) + intercept
    first_positive = np.floor(k * intercept / slope).astype(int) + 1
    return .5 * (binom.sf(first_positive-1, k, theta0)
                 + binom.cdf(first_positive-1, k, theta1))


def run_weights(n, retention):
    if not isinstance(n, (int, np.integer)) or n < 1 or not 0 <= retention < 1:
        raise ValueError('Invalid length or retention')
    return binom.pmf(np.arange(n), n-1, 1-retention)


def error(n, alpha, epsilon, retention, action=.5):
    theta0, theta1, _, _ = parameters(alpha, epsilon, action)
    if retention == 0:
        return float(overlap_error_by_k(n, theta0, theta1))
    k = np.arange(1, n+1)
    return float(run_weights(n, retention) @ overlap_error_by_k(k, theta0, theta1))


def costs_by_k(max_n, alpha, epsilon, action=.5):
    """Direct PMF sums independent of the likelihood-ratio CDF formula."""
    theta0, theta1, c0, c1 = parameters(alpha, epsilon, action)
    rows = []
    for k in range(1, max_n+1):
        j = np.arange(k+1)
        l0 = binom.logpmf(j, k, theta0)
        l1 = binom.logpmf(j, k, theta1)
        denominator = np.logaddexp(np.log(theta0)+l0, np.log(theta1)+l1)
        harmonic = np.exp(logsumexp(l0+l1-denominator))
        bayes = (c1-c0)**2 * theta0*theta1/4 * harmonic
        overlap = .5*np.exp(logsumexp(np.minimum(l0, l1)))
        affinity = np.exp(logsumexp((l0+l1)/2))
        # Direct posterior risk provides another algebraic representation.
        a = np.exp(np.log(theta0)+l0-denominator)
        posterior = a*c0 + (1-a)*c1
        direct = .25*np.sum(theta0*np.exp(l0)*(posterior-c0)**2
                            + theta1*np.exp(l1)*(posterior-c1)**2)
        rows.append((overlap, bayes, affinity, direct))
    return np.asarray(rows)


def finite(n, alpha, epsilon, retention, costs=None):
    theta0, theta1, c0, c1 = parameters(alpha, epsilon)
    costs = costs_by_k(n, alpha, epsilon) if costs is None else costs
    weights = run_weights(n, retention)
    e, bayes, affinity_direct, direct = weights @ costs[:n]
    q = np.sqrt(theta0*theta1) + np.sqrt((1-theta0)*(1-theta1))
    affinity = q*(retention+(1-retention)*q)**(n-1)
    # Stable form of (1-sqrt(1-affinity^2))/2.
    affinity_lower = affinity**2/(2*(1+np.sqrt(max(0., 1-affinity**2))))
    v = theta0*theta1*(c1-c0)**2/(2*(np.sqrt(theta0)+np.sqrt(theta1))**2)
    return dict(n=n, alpha=alpha, epsilon=epsilon, retention=retention,
                theta0=theta0, theta1=theta1, optimal0=c0, optimal1=c1,
                expected_distinct_runs=1+(n-1)*(1-retention),
                best_average_error=float(e), binary_regret=float(alpha*.5*epsilon*e),
                scalar_bayes_regret=float(bayes), scalar_bayes_regret_over_alpha=float(bayes/alpha),
                testing_scalar_lower=float(v*e), affinity=float(affinity),
                affinity_error_lower=float(affinity_lower),
                cdf_error=float(error(n, alpha, epsilon, retention)),
                direct_scalar_risk=float(direct), direct_affinity=float(affinity_direct))


def minimum_length(alpha, epsilon, retention, target):
    if not 0 < target < .5:
        raise ValueError('Target error must lie between zero and one half')
    lower, upper = 0, 1
    while error(upper, alpha, epsilon, retention) > target:
        lower, upper = upper, 2*upper
        if upper > 2**22:
            raise RuntimeError('Explicit search safety limit reached; no answer reported')
    while upper-lower > 1:
        middle = (lower+upper)//2
        if error(middle, alpha, epsilon, retention) <= target:
            upper = middle
        else:
            lower = middle
    achieved = error(upper, alpha, epsilon, retention)
    previous = error(upper-1, alpha, epsilon, retention) if upper > 1 else .5
    assert achieved <= target and previous > target
    return dict(alpha=alpha, epsilon=epsilon, retention=retention, target_error=target,
                minimum_n=upper, achieved_error=achieved, preceding_error=previous)


def poisson_limit(tau, epsilon, retention):
    a0, a1 = (1-epsilon)/.75, (1+epsilon)/.75
    c0, c1 = 1-1/a0, 1-1/a1
    mu0, mu1 = tau*(1-retention)*a0, tau*(1-retention)*a1
    stop = int(poisson.isf(1e-15, max(mu0, mu1))) + 2
    j = np.arange(stop+1)
    l0, l1 = poisson.logpmf(j, mu0), poisson.logpmf(j, mu1)
    e = .5*np.exp(logsumexp(np.minimum(l0, l1)))
    denominator = np.logaddexp(np.log(a0)+l0, np.log(a1)+l1)
    risk = (c1-c0)**2*a0*a1/4*np.exp(logsumexp(l0+l1-denominator))
    return dict(tau=tau, epsilon=epsilon, retention=retention,
                poisson_error=float(e), scalar_limit_over_alpha=float(risk),
                omitted_probability_bound=float(poisson.sf(stop, mu0)+poisson.sf(stop, mu1)),
                zero_count_error_lower=float(.5*np.exp(-max(mu0, mu1))))


def uniform_risk(c, alpha, theta):
    def interval(a, b):
        below = np.clip(c-a, 0, b-a)
        above = np.clip(b-c, 0, b-a)
        positive = below*(c-a-below/2)/(b-a)
        negative = above*(b-c-above/2)/(b-a)
        return alpha*positive+(1-alpha)*negative
    return (1-theta)*interval(-1., 0.) + theta*interval(0., 1.)
