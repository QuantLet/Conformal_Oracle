"""Independent archived-path AR verification; no producer import or random draws."""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/optimism_ar'
PRIMARY = OUT/'primary'
SAVED = ROOT/'artifacts/r8_mechanism'
REPORT = OUT/'independent_verification.json'
SIGMA = math.sqrt(.0002)
Z = float(norm.ppf(.01))
Q = SIGMA*Z
F = float(norm.pdf(Z)/SIGMA)
CHECKS = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def close(a, b):
    return np.allclose(a, b, atol=1e-12, rtol=1e-10)


def check(name, bad, good, predicate, defect):
    def accepted(value):
        try:
            return bool(predicate(value))
        except (ValueError, AssertionError, KeyError, IndexError, TypeError):
            return False
    rejected = not accepted(bad)
    passed = accepted(good) if rejected else False
    CHECKS.append(dict(name=name, defect=defect, defective_evaluated_first=True,
                       defect_rejected=rejected, valid_accepted=passed))
    if not rejected or not passed:
        raise AssertionError(CHECKS[-1])


def valid_bindings(records):
    return all(sha(r['path']) == r['sha256'] and Path(r['path']).stat().st_size == r['size']
               and Path(r['path']).stat().st_mtime_ns == r['mtime_ns'] for r in records)


def hinge_change(c, scores):
    return .01*c + np.maximum(scores-c, 0)-np.maximum(scores, 0)


def total_risk_quadrature(c):
    cutoff = (Q-c)/SIGMA
    lower = quad(lambda z: .99*(Q-c-SIGMA*z)*norm.pdf(z), -np.inf, cutoff,
                 epsabs=1e-14, epsrel=1e-12)[0]
    upper = quad(lambda z: .01*(SIGMA*z-Q+c)*norm.pdf(z), cutoff, np.inf,
                 epsabs=1e-14, epsrel=1e-12)[0]
    return lower+upper


def regret_quadrature(c):
    # Integrate the population score-risk derivative instead of using the producer formula.
    return quad(lambda u: .01-norm.cdf((Q-u)/SIGMA), 0., float(c),
                epsabs=1e-14, epsrel=1e-12)[0]


def covariance_angle(phi, lag):
    upper = math.asin(phi**lag)
    value, error = quad(lambda theta: math.exp(-Z*Z/(1+math.sin(theta)))/(2*math.pi),
                        0., upper, epsabs=1e-14, epsrel=1e-12)
    return value, error


def dependence(phi, n):
    if phi == 0:
        return dict(omega=.0099, finite_count_variance=n*.0099, lags=0,
                    tail_bound=0., covariance=[], quadrature_error=0.)
    denominator = math.pi*math.sqrt(1-phi*phi)*(1-phi)
    L = math.ceil(math.log(1e-13*denominator)/math.log(phi)-1)
    tail = phi**(L+1)/denominator
    if not (tail <= 1e-13 and phi**L/denominator > 1e-13):
        raise AssertionError('Tail cutoff is not the first admissible integer')
    terms = [covariance_angle(phi, lag) for lag in range(1, max(L, n-1)+1)]
    covariance = np.array([v for v, _ in terms])
    return dict(omega=.0099+2*math.fsum(covariance[:L]),
                finite_count_variance=n*.0099+2*math.fsum(
                    (n-lag)*covariance[lag-1] for lag in range(1, n)),
                lags=L, tail_bound=tail, covariance=covariance,
                quadrature_error=2*math.fsum(error for _, error in terms[:L]))


def run():
    started = time.time_ns()
    lock = json.loads((OUT/'lock.json').read_text())
    broken = [dict(r) for r in lock['inputs']]
    broken[0]['sha256'] = '0'*64
    check('stale_input_hash_before', broken, lock['inputs'], valid_bindings,
          'Change first digest in complete execution lock')
    pathmeta = json.loads((SAVED/'paths.json').read_text())
    check('original_path_checksum', '0'*64, sha(SAVED/'paths.npz'),
          lambda h: h == pathmeta['sha256'], 'Replace archived path checksum')
    completion = json.loads((SAVED/'results/complete.json').read_text())
    for filename in ('counts.csv', 'replications.parquet'):
        check('original_completion_'+filename, '0'*64, sha(SAVED/'results'/filename),
              lambda h: h == completion['outputs'][filename], 'Replace original completion digest')
    h = pd.read_csv(PRIMARY/'histories.csv')
    dep = pd.read_csv(PRIMARY/'dependence.csv')
    bands = pd.read_csv(PRIMARY/'bands.csv')
    bootstrap = pd.read_csv(PRIMARY/'bootstrap.csv')
    unavailable = pd.read_csv(PRIMARY/'unavailable.csv')
    counts = pd.read_csv(SAVED/'results/counts.csv')
    old = pd.read_parquet(SAVED/'results/replications.parquet')
    paths = np.load(SAVED/'paths.npz')
    family = {(phi, n, mode) for phi in (0., .8) for n in (500, 1000)
              for mode in ('independent', 'contiguous') if mode == 'independent' or n == 500}
    family_keys = [(r.phi, r.n, r.evaluation) for r in bands.itertuples()]
    complete_family = lambda k: len(k) == 6 and len(set(k)) == 6 and set(k) == family
    check('missing_primary_cell', family_keys[:-1], family_keys, complete_family, 'Drop final primary cell')
    check('duplicated_primary_cell', family_keys[:-1]+family_keys[:1], family_keys, complete_family,
          'Duplicate first cell instead of final distinct cell')
    history_keys = [(r.phi, r.n, r.evaluation, r.rep) for r in h.itertuples()]
    expected_history = {(phi, n, mode, rep) for phi, n, mode in family for rep in range(500)}
    complete_history = lambda k: len(k) == 3000 and len(set(k)) == 3000 and set(k) == expected_history
    check('complete_3000_histories', history_keys[:-1], history_keys, complete_history, 'Drop one history')
    check('unique_3000_histories', history_keys[:-1]+history_keys[:1], history_keys,
          complete_history, 'Duplicate a history key')
    missing_keys = [(r.phi, r.n, r.evaluation, r.status) for r in unavailable.itertuples()]
    check('missing_future_reported', missing_keys[:1], missing_keys,
          lambda k: len(k) == 2 and set(k) == {(p, 1000, 'contiguous', 'NOT_AVAILABLE') for p in (0., .8)},
          'Omit one unavailable future cell')
    risk0 = total_risk_quadrature(0.)
    for c in (-.03, -.005, 0., .005, .03):
        direct = total_risk_quadrature(c)
        integrated = risk0+regret_quadrature(c)
        check('independent_normal_risk_'+str(c), -integrated, integrated,
              lambda x: close(x, direct), 'Reverse Normal expected-loss sign')
    normalised = {}
    dependence_values = {}
    maxima_error = dict(training=0., contiguous=0., independent_risk=0., covariance=0., finite_count_variance=0.)
    for phi in (0., .8):
        y = paths[f'ar_normal_{phi:g}']
        latent = paths[f'ar_z_{phi:g}']
        innovations = paths['ar_innovations']
        check('original_normal_margin_'+str(phi), -y, y, lambda a: close(a, SIGMA*latent),
              'Reverse the stored Normal margin')
        recurrence = phi*latent[:, :-1]+math.sqrt(1-phi*phi)*innovations[:, 1:]
        check('stored_AR_recurrence_'+str(phi), -latent[:, 1:], latent[:, 1:],
              lambda a: close(a, recurrence), 'Reverse the actual archived AR states')
        check('stationary_initial_state_'+str(phi), -latent[:, 0], latent[:, 0],
              lambda a: close(a, innovations[:, 0]), 'Reverse initial archived Gaussian state')
        if y.shape != (500, 1000):
            raise AssertionError('Wrong archived path shape')
        if phi:
            for lag in (1, 2, 5, 10):
                r = phi**lag
                joint = quad(lambda z: norm.pdf(z)*norm.cdf((Z-r*z)/math.sqrt(1-r*r)),
                             -np.inf, Z, epsabs=1e-14, epsrel=1e-12)[0]
                ref = joint-.0001
                transformed, _ = covariance_angle(phi, lag)
                check('independent_bivariate_hit_covariance_'+str(lag), 0., transformed,
                      lambda x: close(x, ref), 'Omit nonzero target-hit covariance')
                maxima_error['covariance'] = max(maxima_error['covariance'], abs(transformed-ref))
        for n in (500, 1000):
            score = Q-y[:, :n]
            product = 99*(n+1)
            k = product//100+int(product % 100 != 0)
            sorted_score = np.sort(score, axis=1)
            c = sorted_score[:, k-1]
            J = hinge_change(c[:, None], score).mean(1)
            independent_V = np.array([regret_quadrature(x) for x in c])
            population = dependence(phi, n)
            om = population['omega']
            dependence_values[(phi, n)] = om
            saved_dep = dep[(dep.phi == phi) & (dep.n == n)]
            if len(saved_dep) != 1:
                raise AssertionError('Missing or duplicate dependence row')
            saved_dep = saved_dep.iloc[0]
            old_dep = counts[(counts.module == 'ar') & (counts.innovation == 'normal')
                & counts.phi.eq(phi) & counts.n_cal.eq(n) & counts.alpha.eq(.01) & counts.truth.eq('none')]
            if len(old_dep) != 1:
                raise AssertionError('Missing or duplicate original population count row')
            old_dep = old_dep.iloc[0]
            keys = ['omega', 'finite_count_variance', 'lags', 'tail_bound']
            values = np.array([population[key] for key in keys])
            expected_old = np.array([old_dep.omega, old_dep.finite_count_variance,
                                     old_dep.lags, old_dep.remainder_bound])
            mutant = values.copy();mutant[0] = .0099 if phi else .0198
            check(f'original_population_counts_{phi}_{n}', mutant, values,
                  lambda a: close(a, expected_old), 'Use iid Omega under dependence, or double iid variance')
            check(f'new_population_counts_{phi}_{n}', values+.01, values,
                  lambda a: close(a, saved_dep[keys].to_numpy(float)), 'Corrupt dependence summary')
            maxima_error['finite_count_variance'] = max(maxima_error['finite_count_variance'],
                abs(population['finite_count_variance']-old_dep.finite_count_variance))
            A0 = om/(2*n*F)
            Aiid = .0099/(2*n*F)
            check(f'penalty_and_density_{phi}_{n}', [2*A0, Aiid, F], [A0, Aiid, F],
                  lambda a: close(a, saved_dep[['A0', 'A_iid', 'f_true']].to_numpy(float)),
                  'Double the population cost while retaining density')
            if n == 1000:
                st = old[(old.module == 'ar') & (old.innovation == 'normal') & old.phi.eq(phi)
                    & old.n_cal.eq(1000) & old.alpha.eq(.01) & old.truth.eq('none')
                    & old.method.eq('Shift-CP')].sort_values('replication')
                if len(st) != 500 or not np.array_equal(st.replication, np.arange(500)):
                    raise AssertionError('Historical replay set incomplete')
                reference = np.column_stack([risk0+independent_V, independent_V,
                    norm.cdf((Q-c)/SIGMA), c*c, abs(Q-c)])
                saved_metrics = st[['expected_QS', 'excess_QS', 'expected_violation',
                                   'prediction_MSE', 'max_absolute_prediction']].to_numpy()
                bad = saved_metrics.copy();bad[:, 0] *= -1
                check('historical_full_window_metrics_'+str(phi), bad, saved_metrics,
                      lambda a: close(a, reference), 'Reverse archived expected loss')
            modes = [('independent', independent_V, 0)]
            if n == 500:
                future = Q-y[:, 500:714]
                contiguous_V = hinge_change(c[:, None], future).mean(1)
                wrong_future = Q-y[:, :214]
                bad_V = hinge_change(c[:, None], wrong_future).mean(1)
                saved_V = h[h.phi.eq(phi) & h.n.eq(500) & h.evaluation.eq('contiguous')].sort_values('rep').V
                check('actual_future_prefix_leakage_'+str(phi), bad_V, contiguous_V,
                      lambda a: close(a, saved_V), 'Use calibration prefix as test observations')
                modes.append(('contiguous', contiguous_V, 214))
            for mode, V, H in modes:
                rows = h[h.phi.eq(phi) & h.n.eq(n) & h.evaluation.eq(mode)].sort_values('rep')
                check(f'actual_rank_{phi}_{n}_{mode}', sorted_score[:, math.ceil(.99*n)-1], c,
                      lambda a: close(a, rows.C), 'Use empirical instead of conformal rank')
                O = V-J
                expected = np.column_stack([np.full(500, H), np.full(500, k), c, J, V, O,
                    np.full(500, A0), np.full(500, Aiid), O/(2*A0), O/(2*Aiid)])
                names = ['H', 'k', 'C', 'J', 'V', 'optimism', 'A0', 'A_iid', 'ratio', 'ratio_iid']
                observed = rows[names].to_numpy()
                bad = observed.copy();bad[:, 3] *= -1
                check(f'all_raw_array_fields_{phi}_{n}_{mode}', bad, observed,
                      lambda a: close(a, expected), 'Reverse actual training loss')
                normalised[(phi, n, mode)] = O/(2*A0)
                maxima_error['training'] = max(maxima_error['training'], float(np.max(abs(J-rows.J))))
                key = 'independent_risk' if mode == 'independent' else 'contiguous'
                maxima_error[key] = max(maxima_error[key], float(np.max(abs(V-rows.V))))
    order = sorted(normalised)
    X = np.column_stack([normalised[key] for key in order])
    indices_path = ROOT/'results/theory_loop/synthetic/history_bootstrap_indices.npy'
    indices = np.load(indices_path)
    expected_hash = next(r['sha256'] for r in lock['inputs'] if r['path'] == str(indices_path))
    check('bootstrap_bound_file', '0'*64, sha(indices_path), lambda a: a == expected_hash,
          'Stale bootstrap file digest')
    altered = indices.copy();altered[0, 0] = (altered[0, 0]+1) % 500
    check('actual_altered_legal_index', altered, indices, lambda a: np.array_equal(a, indices),
          'Change one valid resampling index')
    means = np.sum(X, axis=0)/500
    fixed_se = np.sqrt(np.sum((X-means)**2, axis=0)/(499*500))
    check('zero_standard_error', np.zeros(6), fixed_se,
          lambda a: np.isfinite(a).all() and (a > 0).all(), 'Set every cell standard error to zero')
    frequency = np.array([np.bincount(row, minlength=500) for row in indices])
    bootstrap_mean = frequency@X/500
    max_t = np.max(abs(bootstrap_mean-means)/fixed_se, axis=1)
    wrong_max = np.max(abs(bootstrap_mean-1)/fixed_se, axis=1)
    check('actual_bootstrap_wrong_centre', wrong_max, max_t,
          lambda a: close(a, bootstrap.max_t), 'Centre bootstrap at first-order one instead of sample mean')
    critical = float(np.sort(max_t)[math.ceil(.95*998)])
    references = np.array([.0099/dependence_values[(phi, n)] for phi, n, _ in order])
    lower, upper = means-critical*fixed_se, means+critical*fixed_se
    actual_bands = bands.set_index(['phi', 'n', 'evaluation']).loc[order]
    expected_bands = np.column_stack([means, fixed_se, lower, upper, references,
                                     np.full(6, critical), np.full(6, 6), np.full(6, 999)])
    names = ['mean_ratio', 'se', 'lower', 'upper', 'iid_reference', 'critical_value', 'family_size', 'bootstrap']
    observed = actual_bands[names].to_numpy()
    bad = observed.copy();bad[:, 4] = 1.
    check('same_band_iid_reference', bad, observed, lambda a: close(a, expected_bands),
          'Use one as the iid benchmark even when target hits are dependent')
    included_lrv = (lower <= 1) & (upper >= 1)
    included_iid = (lower <= references) & (upper >= references)
    expected_flags = np.column_stack([included_lrv, included_iid])
    actual_flags = actual_bands[['includes_LRV_reference', 'includes_iid_reference']].to_numpy(bool)
    check('all_reference_compatibility_flags', ~expected_flags, expected_flags,
          lambda a: np.array_equal(a, actual_flags), 'Reverse both benchmark inclusion flags')
    check('stale_input_hash_after', broken, lock['inputs'], valid_bindings,
          'Recheck corrupted complete binding after computation')
    return dict(status='PASS', protocol_commit=lock['protocol_commit'],
        histories=3000, independent_cells=4, contiguous_cells=2, primary_family=6,
        critical_value=critical, includes_LRV_reference=int(included_lrv.sum()),
        includes_iid_reference=int(included_iid.sum()),
        omega_phi_08=dependence_values[(.8, 500)],
        maximum_absolute_replay_differences=maxima_error,
        checks=CHECKS, verifier_sha256=sha(__file__), producer_sha256=sha(ROOT/'research/r8_optimism_ar/run.py'),
        input_lock_sha256=sha(OUT/'lock.json'),
        output_sha256={p.name: sha(p) for p in sorted(PRIMARY.glob('*.csv'))},
        started_ns=started, finished_ns=time.time_ns(), new_histories=0,
        scope='Reused archived paths; independent implementation, not independent new-data validation',
        statistical_admission='FAIL_UNCHANGED', financial_panel='NOT_RUN')


if __name__ == '__main__':
    try:
        result = run()
    except Exception as exc:
        result = dict(status='FAIL', error=repr(exc), checks=CHECKS,
                      verifier_sha256=sha(__file__), finished_ns=time.time_ns())
        REPORT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        raise
    REPORT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('checks', 'output_sha256')}))
