"""Fixed-level, known-scale comparison; exact contiguous expected test loss."""
import hashlib
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/r8_shape_cost/simulation'
ALPHA = .01
REPS = 5000
SIZES = (250, 1000, 4000)
KINDS = ('normal', 't5')
FACTORS = (.5, 2.)
METHODS = ('Raw', 'Shift-ERM', 'Vol-ERM', 'Vol-UERM', 'Vol-CP', 'Shift-CP')
BASE_SEED = 20260911
BOOT_SEED = 2026091131
BOOT_DRAWS = 9999
STAY = .95
A, B, C = 1.5, 5/3, 4/3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def law(kind):
    return stats.norm if kind == 'normal' else stats.t(df=5, scale=np.sqrt(3/5))


def expected_loss(kind, q):
    """E rho_alpha(epsilon-q), for variance-one mean-zero innovations."""
    q = np.asarray(q)
    if kind == 'normal':
        return stats.norm.pdf(q) + q * (stats.norm.cdf(q) - ALPHA)
    scale = np.sqrt(3/5)
    z = q / scale
    return scale * ((5 + z*z) / 4 * stats.t.pdf(z, 5)
                    + z * (stats.t.cdf(z, 5) - ALPHA))


def occupancy(last_state, horizon, stay=STAY):
    """Mean probability of state 2 at future dates 1,...,H."""
    eigenvalue = 2 * stay - 1
    if eigenvalue == 1:
        return np.asarray(last_state, dtype=float)
    factor = eigenvalue * (1 - eigenvalue**horizon) / (horizon * (1-eigenvalue))
    return .5 + (np.asarray(last_state) - .5) * factor


def erm_quantile(values, weights=None):
    """Leftmost p=.99 empirical minimiser, including exact integer ties."""
    values = np.asarray(values)
    if weights is None:
        rank = (99 * len(values) + 99) // 100
        return float(np.partition(values, rank-1)[rank-1])
    weights = np.asarray(weights)
    assert np.all(weights == weights.astype(np.int64)) and np.all(weights > 0)
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order].astype(np.int64))
    index = np.searchsorted(100 * cumulative, 99 * cumulative[-1], side='left')
    return float(values[order[index]])


def cp_quantile(values):
    values = np.asarray(values)
    rank = (99 * (len(values)+1) + 99) // 100
    assert rank <= len(values)
    return float(np.partition(values, rank-1)[rank-1])


def history(rep):
    scale_seed = [BASE_SEED, 3101, int(rep)]
    innovation_seed = [BASE_SEED, 3102, int(rep)]
    rng = np.random.default_rng(np.random.SeedSequence(scale_seed))
    states = np.empty(max(SIZES), dtype=np.uint8)
    states[0] = rng.integers(0, 2)
    flips = (rng.random(max(SIZES)-1) >= STAY).astype(np.uint8)
    states[1:] = np.bitwise_xor.accumulate(flips) ^ states[0]
    uniform = np.random.default_rng(np.random.SeedSequence(innovation_seed)).random(max(SIZES))
    # RNG can represent zero; a one-ulp interior endpoint is specified a priori.
    uniform = np.clip(uniform, np.nextafter(0., 1.), np.nextafter(1., 0.))
    return states, uniform


def cells():
    result = []
    for kind in KINDS:
        dist = law(kind)
        z_alpha = float(dist.ppf(ALPHA))
        g = float(dist.pdf(z_alpha))
        h_star = np.sqrt(2 * ALPHA * (1-ALPHA)) / g
        for n in SIZES:
            for factor in FACTORS:
                h = factor * h_star
                d = h / np.sqrt(n)
                c_star = brentq(lambda c: .5 * (dist.cdf(z_alpha+d-c)
                    + dist.cdf(z_alpha+d-c/2)) - ALPHA, 0., 2*d,
                    xtol=1e-14, rtol=1e-14)
                result.append(dict(cell=f'{kind}_n{n}_h{factor:g}', kind=kind, n=n,
                    horizon=3*n//7, h_factor=factor, h=h, h_star=h_star, d=d,
                    z_alpha=z_alpha, g=g, constant_population_target=c_star,
                    primary=kind=='normal' and n==1000,
                    predicted_delta=(ALPHA*(1-ALPHA)*(B-C)/g-g*h*h*(A-C))/(2*n)))
    return result


def fit_all(z, sigma, d):
    standard = z + d
    scores = sigma * standard
    return {'Raw': 0., 'Shift-ERM': erm_quantile(scores),
            'Vol-ERM': erm_quantile(standard, sigma),
            'Vol-UERM': erm_quantile(standard), 'Vol-CP': cp_quantile(standard),
            'Shift-CP': cp_quantile(scores)}


def evaluate(info, method, coefficient, last_state):
    """Vectorised over history; coefficients held fixed for the full horizon."""
    coef = np.asarray(coefficient)
    z, d, kind = info['z_alpha'], info['d'], info['kind']
    scaled = method.startswith('Vol-')
    q1 = z + d - coef
    q2 = z + d - (coef if scaled else coef/2)
    high = occupancy(last_state, info['horizon'])
    low_loss = expected_loss(kind, q1)
    high_loss = 2 * expected_loss(kind, q2)
    low_hit, high_hit = law(kind).cdf(q1), law(kind).cdf(q2)
    optimal_loss = expected_loss(kind, z) * (1 + high)
    loss = (1-high)*low_loss + high*high_loss
    marginal_loss = .5*(low_loss+high_loss)
    return dict(loss=loss, marginal_loss=marginal_loss,
                excess_vs_conditional_oracle=loss-optimal_loss,
                violation=(1-high)*low_hit+high*high_hit,
                marginal_violation=.5*(low_hit+high_hit),
                low_state_loss=low_loss, high_state_loss=high_loss,
                low_state_violation=low_hit, high_state_violation=high_hit,
                forecast_state1=q1, forecast_state2=2*q2,
                mean_high_probability=high)
