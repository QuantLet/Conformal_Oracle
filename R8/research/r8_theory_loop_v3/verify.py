"""Independent frozen-array verification; no producer imports or random draws."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import integrate, stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/theory_loop_v3'
SYN = ROOT / 'results/theory_loop/synthetic'
DIAG = OUT / 'diagnostic'
REPORT = OUT / 'independent_verification.json'
SIZES = (250, 500, 700, 1000, 2000)
BIAS = .25 * np.sqrt(1e-5 / (1 - .10 - .85))
CHECKS = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def close(a, b, mixture=False):
    return np.allclose(a, b, atol=4e-11 if mixture else 1e-12,
                       rtol=4e-11 if mixture else 1e-10)


def check(name, defective, valid, predicate, defect):
    def accept(value):
        try:
            return bool(predicate(value))
        except (ValueError, AssertionError, KeyError, IndexError, TypeError):
            return False
    rejected = not accept(defective)
    accepted = accept(valid) if rejected else False
    CHECKS.append(dict(name=name, defect=defect,
                       defective_evaluated_first=True,
                       negative_control_rejected=rejected,
                       valid_case_accepted=accepted))
    if not rejected or not accepted:
        raise AssertionError(CHECKS[-1])


def binding_valid(records):
    for record in records:
        p = Path(record['path'])
        if (sha(p) != record['sha256'] or p.stat().st_mtime_ns != record['mtime_ns']
                or p.stat().st_size != record['size']):
            return False
    return True


def loss_change(c, scores):
    # Hinge-loss identity, independent of the producer's np.where pinball code.
    c = np.asarray(c)
    return .01*c + np.maximum(scores-c, 0) - np.maximum(scores, 0)


def mixture_reference(displacement, scales, law):
    dist = stats.norm() if law == 'normal' else stats.t(5, scale=np.sqrt(3/5))
    q = scales*dist.ppf(.01) + displacement
    if law == 'normal':
        z = q/scales
        F = stats.norm.cdf(z)
        partial = scales*stats.norm.pdf(z)
    else:
        scale = scales*np.sqrt(3/5)
        z = q/scale
        F = stats.t.cdf(z, 5)
        partial = scale*(5+z*z)/4*stats.t.pdf(z, 5)
    # Separate lower and upper partial moments rather than the producer formula.
    return float(np.mean(.99*(q*F+partial) + .01*(partial-q*(1-F))))


def barycentric(points, nodes, values):
    points = np.atleast_1d(points)
    theta = np.pi*(np.arange(len(nodes))+.5)/len(nodes)
    weights = (-1.)**np.arange(len(nodes))*np.sin(theta)
    difference = points[:, None]-nodes[None, :]
    answer = np.empty(len(points))
    exact = difference == 0
    for i in range(len(points)):
        if exact[i].any():
            answer[i] = values[np.flatnonzero(exact[i])[0]]
        else:
            quotients = weights/difference[i]
            answer[i] = np.dot(quotients, values)/quotients.sum()
    return answer


def bias_tag(x):
    if abs(float(x)) < 1e-15:
        return 0
    if abs(float(x)-BIAS) < 1e-15:
        return 1
    return -1


def history_keys(frame):
    return [(r.law, int(r.n), r.evaluation, bias_tag(r.bias), int(r.rep))
            for r in frame.itertuples()]


def family_keys(frame):
    return [(r.law, int(r.n), r.evaluation) for r in frame.itertuples()]


def exact_family(keys):
    return (len(keys) == 18 and len(set(keys)) == 18 and set(keys) == {
        (law, n, mode) for law in ('normal', 't5') for n in SIZES
        for mode in ('independent', 'contiguous') if mode == 'independent' or n < 2000})


def run():
    started = time.time_ns()
    lock = json.loads((OUT/'lock.json').read_text())
    corrupt = [dict(r) for r in lock['files']]
    corrupt[0]['sha256'] = '0'*64
    check('actual_stale_input_binding', corrupt, lock['files'], binding_valid,
          'Complete lock with first input SHA-256 replaced by zeroes')
    h = pd.read_csv(DIAG/'histories.csv')
    summary = pd.read_csv(DIAG/'summary.csv')
    bands = pd.read_csv(DIAG/'simultaneous_bands.csv')
    ranks = pd.read_csv(DIAG/'rank_diagnostics.csv')
    maxima_saved = pd.read_csv(DIAG/'bootstrap_maxima.csv')
    unavailable = pd.read_csv(DIAG/'unavailable.csv')
    old = pd.read_csv(SYN/'estimators.csv')
    old_losses = pd.read_csv(SYN/'loss_histories.csv')
    truth = pd.read_csv(SYN/'truth.csv').set_index('law')
    expected = {(law, n, mode, b, rep) for law in ('normal', 't5') for n in SIZES
                for mode in ('independent', 'contiguous') if mode == 'independent' or n < 2000
                for b in (0, 1) for rep in range(500)}
    design_ok = lambda keys: len(keys) == 18000 and len(set(keys)) == 18000 and set(keys) == expected
    keys = history_keys(h)
    check('complete_18000_history_keys', keys[:-1], keys, design_ok, 'Remove final history row')
    check('unique_18000_history_keys', keys[:-1]+keys[:1], keys, design_ok,
          'Duplicate first key in place of final distinct key')
    check('complete_18_band_family', family_keys(bands)[:-1], family_keys(bands), exact_family,
          'Remove final family cell')
    check('unique_18_band_family', family_keys(bands)[:-1]+family_keys(bands)[:1],
          family_keys(bands), exact_family, 'Duplicate a family key')
    rank_keys = [(r.law, r.n, r.rep) for r in ranks.itertuples()]
    expected_rank = {(law, n, rep) for law in ('normal', 't5') for n in SIZES for rep in range(500)}
    rank_design = lambda k: len(k) == 5000 and set(k) == expected_rank
    check('complete_rank_rows', rank_keys[:-1], rank_keys, rank_design, 'Remove a rank row')
    unavailable_ok = lambda f: len(f) == 2 and set(zip(f.law, f.n, f.evaluation, f.status)) == {
        ('normal', 2000, 'contiguous', 'NOT_AVAILABLE'), ('t5', 2000, 'contiguous', 'NOT_AVAILABLE')}
    check('unavailable_contiguous_2000', unavailable.iloc[:1], unavailable, unavailable_ok,
          'Drop one unavailable law')
    reference_columns = {}
    numerical_max = dict(training=0., contiguous=0., mixture=0., nuisance=0.)
    for law in ('normal', 't5'):
        path = np.load(SYN/f'calibration_{law}.npz')
        scores = path['scores']
        dist = stats.norm() if law == 'normal' else stats.t(5, scale=np.sqrt(3/5))
        correct_scores = dist.ppf(.01)*path['sigma']-path['returns']
        check('score_identity_'+law, -scores, scores, lambda s: close(s, correct_scores),
              'Reverse the score sign')
        long = np.load(SYN/f'truth_{law}.npz')['sigma'].ravel()
        fref = float(dist.pdf(dist.ppf(.01))*np.mean(1/long))
        check('density_reference_'+law, fref*2, fref,
              lambda f: close(f, float(truth.loc[law, 'f_true'])), 'Double reference density')
        grid = np.load(SYN/f'mixture_integration_{law}.npz')
        nodes, values = grid['nodes127'], grid['values127']
        domain = grid['domain']
        expected_nodes = (domain[1]+domain[0])/2 - (domain[1]-domain[0])/2*np.cos(
            np.pi*(np.arange(128)+.5)/128)
        check('barycentric_node_support_'+law, nodes[::-1], nodes,
              lambda z: close(z, expected_nodes), 'Reverse node order without values')
        # Independent direct marginal law at all nine locked integration probes.
        direct = np.array([mixture_reference(s, long, law) for s in grid['probe']])
        interpolated = barycentric(grid['probe'], nodes, values)
        check('independent_million_scale_mixture_'+law, interpolated+.01, interpolated,
              lambda z: close(z, direct, mixture=True), 'Offset interpolated risk by .01')
        check('saved_probe_values_'+law, grid['probe_exact']+.01, grid['probe_exact'],
              lambda z: close(z, direct, mixture=True), 'Corrupt saved risk probes')
        numerical_max['mixture'] = max(numerical_max['mixture'], float(np.max(abs(direct-interpolated))))
        oracle = mixture_reference(0., long, law)
        biased_raw = mixture_reference(BIAS, long, law)
        # Direct innovation quadrature checks the partial-moment formula separately.
        sigma = float(long[0]);q = sigma*dist.ppf(.01)+float(grid['probe'][3])
        integral = integrate.quad(lambda z: .99*(q-sigma*z)*dist.pdf(z), -np.inf,
            q/sigma, epsabs=1e-12)[0] + integrate.quad(
            lambda z: .01*(sigma*z-q)*dist.pdf(z), q/sigma, np.inf, epsabs=1e-12)[0]
        analytic = mixture_reference(float(grid['probe'][3]), np.array([sigma]), law)
        check('direct_innovation_quadrature_'+law, -analytic, analytic,
              lambda z: close(z, integral, mixture=True), 'Reverse expected-risk sign')
        for n in SIZES:
            sample = scores[:, :n]
            # Decimal-free independent rank: floor(p(n+1)) plus nonzero rational remainder.
            product = 99*(n+1)
            k = product//100 + int(product % 100 != 0)
            ordered = np.sort(sample, axis=1)
            c = ordered[:, k-1]
            wrong_c = ordered[:, math.ceil(.99*n)-1]
            saved_ranks = ranks[(ranks.law == law) & (ranks.n == n)].sort_values('rep')
            check(f'actual_conformal_rank_{law}_{n}', wrong_c, c,
                  lambda z: close(z, saved_ranks.C), 'Use empirical rather than conformal rank')
            expected_rank_data = np.column_stack([np.full(500, k), np.full(500, k-.99*n),
                (sample <= 0).sum(1), (sample <= c[:, None]).sum(1)])
            recorded_rank_data = saved_ranks[['k', 'k_minus_np', 'true_quantile_hits',
                                             'estimated_quantile_hits']].to_numpy()
            mutant = recorded_rank_data.copy();mutant[0, 2] += 1
            check(f'actual_hit_counts_{law}_{n}', mutant, recorded_rank_data,
                  lambda z: close(z, expected_rank_data), 'Add one population-cutoff hit')
            J = loss_change(c[:, None], sample).mean(1)
            o = old[(old.law == law) & (old.n == n) & old.bias.eq(0)].sort_values('rep')
            A0 = .0099/(2*n*fref)
            Ahat = o.omega.to_numpy()/(2*n*o.f_sj.to_numpy())
            Ao = .0099/(2*n*o.f_sj.to_numpy())
            Af = o.omega.to_numpy()/(2*n*fref)
            inverse_f = fref/o.f_sj.to_numpy()
            omega_ratio = o.omega.to_numpy()/.0099
            nuisance = (omega_ratio-1)+(inverse_f-1)+(omega_ratio-1)*(inverse_f-1)
            check(f'nuisance_interaction_{law}_{n}', (omega_ratio-1)+(inverse_f-1), nuisance,
                  lambda z: close(z, Ahat/A0-1), 'Omit the nuisance interaction term')
            wrong_density_power = o.omega.to_numpy()/(2*n*o.f_sj.to_numpy()**2)
            check(f'nuisance_first_density_power_{law}_{n}', wrong_density_power, Ahat,
                  lambda z: close(z, o.A_hat), 'Use quantile-variance density power in expected-loss cost')
            static = barycentric(-c, nodes, values)
            saved_base = old_losses[(old_losses.law == law) & (old_losses.n == n)
                                   & old_losses.bias.eq(0)].sort_values('rep')
            check(f'barycentric_static_loss_{law}_{n}', static+.01, static,
                  lambda z: close(z, saved_base.static_loss, mixture=True), 'Corrupt all mixture losses')
            independent_V = static-oracle
            modes = [('independent', independent_V, 0, None)]
            if n < 2000:
                H = math.floor(3*n/7)
                future = scores[:, n:n+H]
                V = loss_change(c[:, None], future).mean(1)
                bad_future = scores[:, n-1:n+H-1]
                wrong_V = loss_change(c[:, None], bad_future).mean(1)
                saved_V = h[(h.law == law) & (h.n == n) & h.evaluation.eq('contiguous')
                            & h.bias.eq(0)].sort_values('rep').V.to_numpy()
                check(f'actual_future_off_by_one_{law}_{n}', wrong_V, V,
                      lambda z: close(z, saved_V), 'Shift contiguous slice back one observation')
                modes.append(('contiguous', V, H, future))
            for mode, V, H, future in modes:
                O = V-J
                reference_columns[(law, n, mode)] = O/(2*A0)
                for tag, bias in enumerate((0., BIAS)):
                    row = h[(h.law == law) & (h.n == n) & h.evaluation.eq(mode)
                            & h.bias.map(bias_tag).eq(tag)].sort_values('rep')
                    I = loss_change((c+bias)[:, None], sample+bias).mean(1)
                    Kcal = loss_change(bias, sample+bias).mean(1)
                    if mode == 'independent':
                        raw = oracle if tag == 0 else biased_raw
                        delta = static-raw
                        Ktest = np.full(500, oracle-raw)
                        tol_mixture = True
                    else:
                        delta = loss_change((c+bias)[:, None], future+bias).mean(1)
                        Ktest = loss_change(bias, future+bias).mean(1)
                        tol_mixture = False
                    computed = np.column_stack([c+bias, J, V, I, delta, Kcal, Ktest,
                        O, delta-I, np.full(500, A0), Ahat, Ao, Af,
                        I+A0, I+2*A0, I+2*Ahat])
                    names = ['C', 'J', 'V', 'I', 'delta', 'Kcal', 'Ktest', 'optimism_centred',
                        'optimism_uncentred', 'A0_ref', 'Ahat', 'A_exact_omega',
                        'A_reference_density', 'prediction_one_A0', 'prediction_two_A0',
                        'prediction_two_Ahat']
                    actual = row[names].to_numpy()
                    mutant = actual.copy();mutant[:, names.index('I')] *= -1
                    # Train/threshold/nuisance retain the stricter replay budget.
                    strict_names = ['C', 'J', 'I', 'Kcal', 'A0_ref', 'Ahat', 'A_exact_omega',
                        'A_reference_density', 'prediction_one_A0', 'prediction_two_A0', 'prediction_two_Ahat']
                    strict = [names.index(name) for name in strict_names]
                    check(f'full_array_train_nuisance_{law}_{n}_{mode}_{tag}', mutant[:, strict], actual[:, strict],
                          lambda z: close(z, computed[:, strict]), 'Reverse actual training-loss column')
                    rest = [j for j in range(len(names)) if j not in strict]
                    bad = actual[:, rest].copy();bad[:, 0] += .01
                    check(f'full_array_test_optimism_{law}_{n}_{mode}_{tag}', bad, actual[:, rest],
                          lambda z: close(z, computed[:, rest], mixture=tol_mixture),
                          'Offset test regret while retaining all other fields')
                    check(f'centred_uncentred_identity_{law}_{n}_{mode}_{tag}', O+Ktest-Kcal+.01,
                          O+Ktest-Kcal, lambda z: close(z, delta-I, mixture=tol_mixture),
                          'Corrupt baseline-centering identity')
                    check(f'optimism_factor_two_{law}_{n}_{mode}_{tag}', I+A0, I+2*A0,
                          lambda z: close(z, row.prediction_two_A0), 'Use one cost instead of two')
                    numerical_max['training'] = max(numerical_max['training'], float(np.max(abs(J-row.J))),
                                                     float(np.max(abs(I-row.I))))
                    if mode == 'contiguous':
                        numerical_max['contiguous'] = max(numerical_max['contiguous'], float(np.max(abs(V-row.V))))
                    numerical_max['nuisance'] = max(numerical_max['nuisance'], float(np.max(abs(Ahat-row.Ahat))))
                    s = summary[(summary.law == law) & (summary.n == n) & summary.evaluation.eq(mode)
                                & summary.bias.map(bias_tag).eq(tag)]
                    assert len(s) == 1
                    se = dict(histories=500, A0_ref=A0, mean_J_over_A0=J.mean()/A0,
                        mean_V_over_A0=V.mean()/A0, mean_I=I.mean(), mean_delta=delta.mean(),
                        mean_optimism_centred=O.mean(), mean_optimism_uncentred=(delta-I).mean(),
                        one_A0_error=(I+A0-delta).mean(), two_A0_error=(I+2*A0-delta).mean(),
                        two_Ahat_error=(I+2*Ahat-delta).mean(), centred_two_A0_error=(J+2*A0-V).mean(),
                        centred_two_Ahat_error=(J+2*Ahat-V).mean(), mean_Ahat_over_A0=Ahat.mean()/A0,
                        mean_exact_omega_cost_over_A0=Ao.mean()/A0,
                        mean_reference_density_cost_over_A0=Af.mean()/A0,
                        median_Ahat_relative_error=float(np.median(abs(Ahat/A0-1))),
                        mean_omega_relative_error=(omega_ratio-1).mean(),
                        mean_inverse_density_relative_error=(inverse_f-1).mean(),
                        mean_nuisance_interaction=((omega_ratio-1)*(inverse_f-1)).mean(),
                        mean_C=c.mean(), squared_bias_C=c.mean()**2,
                        variance_C=np.var(c, ddof=1), mse_C=np.mean(c*c))
                    recorded = s[list(se)].iloc[0].to_numpy(float)
                    expected_s = np.array(list(se.values()))
                    bad_s = recorded.copy();bad_s[0] -= 1
                    # Ratios amplify interpolation discrepancies; use original static risks
                    # only after their independent mixture replay passed above.
                    check(f'all_summary_fields_{law}_{n}_{mode}_{tag}', bad_s, recorded,
                          lambda z: close(z, expected_s), 'Omit one history in summary count')
    idx = np.load(SYN/'history_bootstrap_indices.npy')
    idx_bad = idx.copy();idx_bad[0, 0] = (idx_bad[0, 0]+1) % 500
    check('actual_altered_bootstrap_indices', idx_bad, idx,
          lambda z: sha_array(z) == sha_array(idx), 'Change one legal resampling index')
    order = sorted(reference_columns)
    check('independent_complete_family', order[:-1], order, exact_family, 'Remove reconstructed cell')
    matrix = np.column_stack([reference_columns[key] for key in order])
    means = matrix.sum(0)/500
    fixed_se = np.sqrt(((matrix-means)**2).sum(0)/(499*500))
    if not np.isfinite(fixed_se).all() or (fixed_se <= 0).any():
        raise AssertionError('Nonpositive/nonfinite fixed standard error')
    counts = np.array([np.bincount(row, minlength=500) for row in idx])
    bootstrap_means = counts@matrix/500
    maxima = np.max(abs(bootstrap_means-means)/fixed_se, axis=1)
    wrong_maxima = np.max(abs(bootstrap_means-1)/fixed_se, axis=1)
    check('actual_null_vs_mean_bootstrap_centring', wrong_maxima, maxima,
          lambda z: close(z, maxima_saved.max_t), 'Centre each bootstrap at theoretical one')
    critical = np.sort(maxima)[math.ceil(.95*(len(maxima)-1))]
    actual_bands = bands.set_index(['law', 'n', 'evaluation']).loc[order]
    actual = actual_bands[['mean_optimism_over_2A0', 'standard_error', 'lower', 'upper',
                           'relative_discrepancy', 'critical_value']].to_numpy()
    reference = np.column_stack([means, fixed_se, means-critical*fixed_se,
                                means+critical*fixed_se, means-1, np.full(18, critical)])
    defect = actual.copy();defect[:, 5] = 1.959963984540054
    check('full_18_simultaneous_bands', defect, actual, lambda z: close(z, reference),
          'Replace simultaneous critical value by pointwise Normal critical value')
    inclusion = (reference[:, 2] <= 1) & (reference[:, 3] >= 1)
    check('reference_inclusion_flags', ~inclusion, inclusion,
          lambda z: np.array_equal(z, actual_bands.includes_first_order_reference.to_numpy(bool)),
          'Reverse all inclusion flags')
    check('completion_bindings', corrupt, lock['files'], binding_valid,
          'Recheck stale complete lock after all calculations')
    return dict(status='PASS', protocol_commit=lock['protocol_commit'],
        full_history_rows=18000, distinct_history_design_cells=36, primary_family=18,
        histories_per_cell=500, bootstrap_draws=999, critical_value=float(critical),
        bands_including_first_order_reference=int(inclusion.sum()),
        maximum_absolute_replay_differences=numerical_max,
        independent_truth_basis='Barycentric interpolation of saved node values plus direct million-scale risks at all 9 probes per law and innovation quadrature',
        input_lock_sha256=sha(OUT/'lock.json'), verifier_sha256=sha(__file__),
        producer_sha256=sha(ROOT/'research/r8_theory_loop_v3/engine.py'),
        output_sha256={p.name: sha(p) for p in sorted(DIAG.glob('*.csv'))},
        started_ns=started, finished_ns=time.time_ns(), checks=CHECKS,
        scope='Independent implementation check on reused histories, not new-data validation',
        statistical_admission='FAIL_UNCHANGED', financial_panel='NOT_RUN', new_random_histories=0)


def sha_array(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


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
