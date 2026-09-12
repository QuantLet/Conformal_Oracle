"""Partial-shift rules and exact losses; no random generation or fitted base model."""
import numpy as np
from fractions import Fraction
from scipy import optimize, special, stats

V0 = np.sqrt(1e-5 / (1 - .10 - .85))
FRACTIONS = np.array([0., .25, .5, .75, 1.])
METHODS = ['Raw', 'Full-CP', 'Half-Full', 'Inner-CP', 'Half-Inner',
           'Selected-Inner', 'Oracle-Grid', 'Oracle-Continuous']


def pinball(residual, alpha):
    return residual * (alpha - (residual < 0))


def select(y, raw, alpha):
    """Only past returns and their raw forecasts enter this feasible rule."""
    y, raw = np.broadcast_arrays(np.atleast_2d(y), np.atleast_2d(raw))
    if not (np.isfinite(y).all() and np.isfinite(raw).all()):
        raise ValueError('Nonfinite calibration input')
    n = y.shape[1]
    m = max(100, int(.7 * n))
    if m >= n:
        raise ValueError('No chronological validation block')
    k_inner = int(np.ceil((m + 1) * (1 - alpha)))
    k_full = int(np.ceil((n + 1) * (1 - alpha)))
    if not (1 <= k_inner <= m and 1 <= k_full <= n):
        raise ValueError('Conformal rank is not finite')
    scores = raw - y
    inner = np.partition(scores[:, :m], k_inner - 1, axis=1)[:, k_inner - 1]
    full = np.partition(scores, k_full - 1, axis=1)[:, k_full - 1]
    residual = y[:, m:, None] - raw[:, m:, None] + inner[:, None, None] * FRACTIONS
    validation = pinball(residual, alpha).mean(axis=1)
    index = np.argmin(validation, axis=1)
    # A check-loss objective is flat between adjacent order statistics when
    # N*(1-alpha) is an integer. Floating argmin must not choose within that
    # interval by rounding noise instead of the prespecified smaller fraction.
    target = (1 - Fraction(str(alpha))) * (n - m)
    ordered = np.sort(scores[:, m:], axis=1)
    lower_rank = -(-target.numerator // target.denominator)
    upper_rank = target.numerator // target.denominator + 1
    lower = ordered[:, lower_rank - 1]
    upper = ordered[:, upper_rank - 1] if target.denominator == 1 else lower
    corrections = inner[:, None] * FRACTIONS
    flat = (corrections >= lower[:, None]) & (corrections <= upper[:, None])
    has_flat = np.any(flat, axis=1)
    index[has_flat] = np.argmax(flat[has_flat], axis=1)
    gap = np.partition(validation, 1, axis=1)[:, 1] - validation.min(axis=1)
    tolerance = 64 * np.finfo(float).eps * (np.max(np.abs(scores[:, m:]), axis=1) + np.abs(inner))
    exact_rows = np.flatnonzero((gap <= tolerance) & ~has_flat)
    a = Fraction(str(alpha))
    for row in exact_rows:
        exact_scores = [Fraction(float(s)) for s in scores[row, m:]]
        values = []
        for c in corrections[row]:
            residuals = [Fraction(float(c)) - s for s in exact_scores]
            values.append(sum((a - (r < 0)) * r for r in residuals))
        index[row] = min(range(len(values)), key=lambda j: (values[j], j))
    return dict(inner=inner, full=full, validation=validation, index=index,
                selected_fraction=FRACTIONS[index], fit_size=m,
                rank_inner=k_inner, rank_full=k_full, flat_minimum=has_flat,
                exact_fallback=np.isin(np.arange(len(inner)), exact_rows))


def quantile(kind, sigma, alpha):
    z = stats.norm.ppf(alpha) if kind == 'normal' else np.sqrt(3 / 5) * stats.t.ppf(alpha, 5)
    return sigma * z


def distortion(sigma, truth):
    sigma = np.asarray(sigma)
    if truth == 'none':
        return np.zeros_like(sigma)
    if truth == 'constant':
        return np.full_like(sigma, .25 * V0)
    if truth != 'state':
        raise ValueError(truth)
    return .25 * V0 + .75 * V0 * np.log(sigma / V0)


def expected_loss(kind, q, sigma, alpha):
    if kind == 'normal':
        z = q / sigma
        return sigma * (np.exp(-z * z / 2) / np.sqrt(2 * np.pi) + z * (special.ndtr(z) - alpha))
    scale = sigma * np.sqrt(3 / 5)
    z = q / scale
    return scale * ((5 + z * z) / 4 * stats.t.pdf(z, 5) + z * (stats.t.cdf(z, 5) - alpha))


def probability(kind, q, sigma):
    if kind == 'normal':
        return special.ndtr(q / sigma)
    return stats.t.cdf(q / (sigma * np.sqrt(3 / 5)), 5)


def constant_optimum(kind, raw, sigma, alpha):
    displacement = raw - quantile(kind, sigma, alpha)
    lower, upper = float(np.min(displacement)), float(np.max(displacement))
    if upper - lower < 1e-16:
        return (lower + upper) / 2
    gradient = lambda c: alpha - probability(kind, raw - c, sigma).mean()
    return optimize.brentq(gradient, lower, upper, xtol=1e-14)


def candidate_corrections(choice, optimum):
    inner = choice['inner']
    fraction = np.zeros_like(inner)
    np.divide(optimum, inner, out=fraction, where=inner != 0)
    fraction = np.clip(fraction, 0, 1)
    candidates = np.column_stack([inner[:, None] * FRACTIONS,
                                  choice['full'], .5 * choice['full'], fraction * inner])
    return candidates, fraction


def independent_metrics(kind, raw, sigma, candidates, alpha):
    raw, sigma = np.atleast_1d(raw), np.atleast_1d(sigma)
    q = raw[None, None, :] - candidates[:, :, None]
    return (expected_loss(kind, q, sigma, alpha).mean(axis=2),
            probability(kind, q, sigma).mean(axis=2))


def contiguous_metrics(raw, candidates, alpha, last_z, phi, horizon):
    powers = phi ** np.arange(1, horizon + 1)
    mean = V0 * last_z[:, None] * powers
    sd = V0 * np.sqrt(1 - powers * powers)
    centred_q = raw - candidates[:, :, None] - mean[:, None, :]
    return (expected_loss('normal', centred_q, sd, alpha).mean(axis=2),
            probability('normal', centred_q, sd).mean(axis=2))


def method_columns(risks, violations, choice):
    row = np.arange(len(risks))
    oracle = np.argmin(risks[:, :5], axis=1)
    indices = [np.zeros(len(row), dtype=int), np.full(len(row), 5),
               np.full(len(row), 6), np.full(len(row), 4), np.full(len(row), 2),
               choice['index'], oracle, np.full(len(row), 7)]
    return (np.column_stack([risks[row, i] for i in indices]),
            np.column_stack([violations[row, i] for i in indices]), oracle)


def accounting(method_risks, scalar_risk):
    raw, full, _, inner, _, selected, oracle, continuous = method_risks.T
    pieces = dict(removable_loss=raw - scalar_risk,
                  inner_estimation_cost=inner - scalar_risk,
                  oracle_shrinkage_gain=inner - oracle,
                  selection_regret=selected - oracle,
                  grid_discretisation_gap=oracle - continuous,
                  validation_reservation_effect=inner - full,
                  full_estimation_cost=full - scalar_risk,
                  selected_change=selected - raw)
    reconstructed = (-pieces['removable_loss'] + pieces['inner_estimation_cost']
                     - pieces['oracle_shrinkage_gain'] + pieces['selection_regret'])
    assert np.max(np.abs(reconstructed - pieces['selected_change'])) < 1e-14
    for name in ['removable_loss', 'inner_estimation_cost', 'oracle_shrinkage_gain',
                 'selection_regret', 'grid_discretisation_gap', 'full_estimation_cost']:
        assert np.min(pieces[name]) > -1e-13, name
    return pieces
