"""Past-only regime controls; no empirical observations or manuscript mutation."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import hashlib
from pathlib import Path
import sys
import numpy as np
from scipy import stats, special

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'research/r8_decision'))
import methods as original

V0 = np.sqrt(1e-5/(1-.10-.85))
REPS, BURN, LENGTH, START, BREAK, VALIDATE = 500, 1000, 2500, 1250, 1500, 1000
WINDOWS = (125, 250, 500, 1000)
ALPHAS = (.01, .05)
SCENARIOS = ('correct', 'biased', 'appears', 'disappears', 'reverses',
             'jump_oracle', 'steady_ewma', 'jump_ewma')
METHODS = ('Raw', 'Static-CP', 'Rolling125', 'Rolling250', 'Rolling500',
           'Rolling1000', 'VolRolling250', 'VolRolling500',
           'DtACI500', 'PastSelectedRolling', 'InitialKupiecGate')
PERIODS = {'pre': (0, 250), 'acute': (250, 270), 'early': (270, 375),
           'middle': (375, 750), 'late': (750, 1250), 'post': (250, 1250)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def seed(kind, rep):
    return int.from_bytes(hashlib.sha256(f'20260910/regime/{kind}/{rep}'.encode()).digest()[:8], 'little')


def innovations(kind, reps):
    out = []
    for rep in reps:
        rng = np.random.default_rng(seed(kind, rep))
        x = rng.standard_normal(BURN+LENGTH) if kind == 'normal' else rng.standard_t(5, BURN+LENGTH)*np.sqrt(3/5)
        out.append(x)
    return np.asarray(out)


def unit_quantile(kind, alpha):
    return stats.norm.ppf(alpha) if kind == 'normal' else np.sqrt(3/5)*stats.t.ppf(alpha, 5)


def expected(kind, q, sigma, alpha):
    if kind == 'normal':
        z = q/sigma
        cdf = special.ndtr(z)
        risk = sigma*(np.exp(-z*z/2)/np.sqrt(2*np.pi)+z*(cdf-alpha))
    else:
        scale = sigma*np.sqrt(3/5)
        z = q/scale
        cdf = special.stdtr(5, z)
        pdf = np.exp(special.gammaln(3)-special.gammaln(2.5))/np.sqrt(5*np.pi)*(1+z*z/5)**-3
        risk = scale*((5+z*z)/4*pdf+z*(cdf-alpha))
    return risk, cdf


def environment(x, scenario, kind, alpha):
    scale = np.full(x.shape[1], V0)
    if scenario.startswith('jump_'):
        scale[BURN+BREAK:] *= 2
    y = x*scale
    if scenario.endswith('ewma'):
        s = np.empty_like(y)
        variance = np.full(len(x), V0**2)
        for t in range(y.shape[1]):
            s[:, t] = np.sqrt(variance)
            variance = .94*variance+.06*y[:, t]**2
    else:
        s = np.broadcast_to(scale, y.shape).copy()
    q = s*unit_quantile(kind, alpha)
    if scenario in ('biased', 'disappears', 'reverses'):
        q[:, :BURN+BREAK] += .5*V0
    if scenario in ('biased', 'appears'):
        q[:, BURN+BREAK:] += .5*V0
    if scenario == 'reverses':
        q[:, BURN+BREAK:] -= .5*V0
    return y[:, BURN:], q[:, BURN:], s[:, BURN:], np.broadcast_to(scale[BURN:], (len(x), LENGTH)).copy()


def cp_rows(s, alpha):
    k = int(np.ceil((s.shape[1]+1)*(1-alpha)))
    if not 1 <= k <= s.shape[1]:
        raise ValueError('No finite CP rank')
    return np.partition(s, k-1, axis=1)[:, k-1]


def inverse_rows(window, score):
    n = window.shape[1]
    j = np.sum(window < score[:, None], axis=1)
    jj = np.clip(j, 1, n-1)
    hi = window[np.arange(len(window)), jj]
    lo = window[np.arange(len(window)), jj-1]
    fraction = np.divide(score-lo, hi-lo, out=np.zeros_like(score), where=hi != lo)
    result = 1-(jj-1+fraction)/(n-1)
    result = np.where(hi == score, 1-jj/(n-1), result)
    return np.where(j == 0, 1., np.where(j == n, 0., result))


def dt_step(window, score, q, levels, weights, alpha):
    n, w = window.shape
    pos = (1-np.clip(levels, 0, 1))*(w-1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo+1, w-1)
    rows = np.arange(n)[:, None]
    shifts = window[rows, lo]+(pos-lo)*(window[rows, hi]-window[rows, lo])
    predictions = q[:, None]-shifts
    beta = inverse_rows(window, score)
    eta = np.sqrt(3*(np.log(2*len(original.GAMMAS)*500)+1)/(500*(alpha*(1-alpha))**2))
    expert_loss = original.loss(beta[:, None], levels, alpha)
    logw = np.log(weights)-eta*expert_loss
    new = np.exp(logw-logw.max(axis=1)[:, None])
    new /= new.sum(axis=1)[:, None]
    new = (1-.001)*new+.001/len(original.GAMMAS)
    unprojected = levels+original.GAMMAS[None, :]*(alpha-(score[:, None] > shifts))
    updated = np.clip(unprojected, 1/(w+1), w/(w+1))
    return predictions, updated, new, (unprojected != updated)


def kupiec_reject(hits, alpha):
    count = hits.sum(axis=1)
    n = hits.shape[1]
    mle = count/n
    alt = special.xlogy(count, mle)+special.xlogy(n-count, 1-mle)
    null = count*np.log(alpha)+(n-count)*np.log1p(-alpha)
    return stats.chi2.sf(2*(alt-null), 1) < .05


def policies(y, q, scale, alpha):
    n, T = y.shape
    assert T > START and np.all(scale > 0)
    scores = q-y
    normalised = scores/scale
    H = T-START
    pred = np.zeros((n, H, len(METHODS)))
    pred[:, :, 0] = q[:, START:]
    shift = cp_rows(scores[:, START-1000:START], alpha)
    pred[:, :, 1] = q[:, START:]-shift[:, None]
    experts = np.empty((n, H, len(original.GAMMAS)))
    probabilities = np.empty_like(experts)
    states = np.empty_like(experts)
    projections = np.zeros((n, len(original.GAMMAS)), dtype=int)
    levels = np.full((n, len(original.GAMMAS)), alpha)
    weights = np.full_like(levels, 1/len(original.GAMMAS))
    validation = np.zeros((n, len(WINDOWS)))
    for t in range(VALIDATE, T):
        rolling = np.empty((n, len(WINDOWS)))
        window500 = np.sort(scores[:, t-500:t], axis=1)
        for j, w in enumerate(WINDOWS):
            c = (window500[:, int(np.ceil((w+1)*(1-alpha)))-1] if w == 500
                 else cp_rows(scores[:, t-w:t], alpha))
            rolling[:, j] = q[:, t]-c
        if t < START:
            validation += original.loss(y[:, t, None], rolling, alpha)
        else:
            pred[:, t-START, 2:6] = rolling
            for j, w in enumerate((250, 500)):
                pred[:, t-START, 6+j] = q[:, t]-scale[:, t]*cp_rows(normalised[:, t-w:t], alpha)
        ep, newlevels, newweights, projected = dt_step(window500, scores[:, t], q[:, t], levels, weights, alpha)
        if t >= START:
            experts[:, t-START] = ep
            probabilities[:, t-START] = weights
            states[:, t-START] = levels
            pred[:, t-START, 8] = np.sum(weights*ep, axis=1)
        projections += projected
        levels, weights = newlevels, newweights
    # Reversing indices before argmin makes exact ties favour the larger window.
    choices = len(WINDOWS)-1-np.argmin(validation[:, ::-1], axis=1)
    gate = kupiec_reject(y[:, VALIDATE:START] < q[:, VALIDATE:START], alpha)
    pred[:, :, 9] = pred[np.arange(n)[:, None], np.arange(H)[None, :], 2+choices[:, None]]
    pred[:, :, 10] = np.where(gate[:, None], pred[:, :, 9], pred[:, :, 0])
    return {'prediction': pred, 'experts': experts, 'probabilities': probabilities,
            'states': states, 'projections': projections, 'selected': choices,
            'gate': gate, 'validation_loss': validation/(START-VALIDATE), 'static_shift': shift}


def evaluate(result, y, sigma, kind, alpha):
    pred = result['prediction']
    s = sigma[:, START:, None]
    oracle = s*unit_quantile(kind, alpha)
    risk, hit = expected(kind, pred, s, alpha)
    real = original.loss(y[:, START:, None], pred, alpha)
    actual_hit = (y[:, START:, None] < pred).astype(float)
    abs_error = np.abs(pred-oracle)
    er, eh = expected(kind, result['experts'], s, alpha)
    weights = result['probabilities']
    risk[:, :, 8] = (er*weights).sum(axis=2)
    hit[:, :, 8] = (eh*weights).sum(axis=2)
    real[:, :, 8] = (original.loss(y[:, START:, None], result['experts'], alpha)*weights).sum(axis=2)
    actual_hit[:, :, 8] = ((y[:, START:, None] < result['experts'])*weights).sum(axis=2)
    abs_error[:, :, 8] = (np.abs(result['experts']-oracle)*weights).sum(axis=2)
    optimum, _ = expected(kind, oracle, s, alpha)
    assert np.min(risk-optimum) > -1e-12
    return {'risk': risk, 'hit_probability': hit, 'realised_loss': real,
            'realised_hits': actual_hit, 'absolute_error': abs_error,
            'excess_loss': risk-optimum}
