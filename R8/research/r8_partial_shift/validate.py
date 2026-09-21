"""Independent selection, loss-integral and archive checks for the fixed study."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from fractions import Fraction
import numpy as np
import pandas as pd
from scipy import integrate, stats
import engine as e
from run import ROOT, OUT, KEYS, configurations, forecasts, sha


def quadrature_checks():
    error = 0.
    for kind in ('normal', 't5'):
        density = stats.norm.pdf if kind == 'normal' else lambda x: stats.t.pdf(x / np.sqrt(3 / 5), 5) / np.sqrt(3 / 5)
        for sigma in (e.V0, 3 * e.V0):
            for alpha in (.01, .05):
                for z in (-4., -2., 0., 2.):
                    q = z * sigma
                    left = integrate.quad(lambda x: (1 - alpha) * (q - sigma * x) * density(x), -np.inf, z, epsabs=1e-13)[0]
                    right = integrate.quad(lambda x: alpha * (sigma * x - q) * density(x), z, np.inf, epsabs=1e-13)[0]
                    error = max(error, abs(left + right - float(e.expected_loss(kind, q, sigma, alpha))))
    assert error < 1e-11
    return error


def leakage_checks(paths):
    for n in (125, 1000):
        y = paths['ar_normal_0.8'][0, -n:]
        q = np.full_like(y, e.quantile('normal', e.V0, .01) + .25 * e.V0)
        a = np.concatenate([y, np.full(2000, 1.)])
        b = np.concatenate([y, np.full(2000, -1.)])
        future_q = np.concatenate([q, np.full(2000, q[0])])
        honest_a, honest_b = e.select(a[:n], future_q[:n], .01), e.select(b[:n], future_q[:n], .01)
        for key in honest_a:
            np.testing.assert_array_equal(honest_a[key], honest_b[key])
        leaking_a, leaking_b = e.select(a, future_q, .01), e.select(b, future_q, .01)
        assert not np.array_equal(leaking_a['inner'], leaking_b['inner']), 'Negative control failed'
    tied = e.select(np.zeros(125), np.zeros(125), .01)
    assert tied['selected_fraction'][0] == 0 and tied['rank_inner'] == 100
    for alpha, hits in ((.01, 3), (.05, 15)):
        scores = np.r_[np.ones(700), np.zeros(300 - hits), np.ones(hits)]
        flat = e.select(-scores, np.zeros(1000), alpha)
        assert flat['flat_minimum'][0] and flat['selected_fraction'][0] == 0
    try:
        e.select(np.zeros(100), np.zeros(100), .01)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing holdout was accepted')


def main():
    directory = OUT / 'run'
    receipt = json.loads((directory / 'validation.json').read_text())
    replay = json.loads((OUT / 'replay/validation.json').read_text())
    assert receipt == replay
    assert all(sha(ROOT / p) == h for p, h in receipt['inputs'].items())
    for folder in (directory, OUT / 'replay'):
        assert all(sha(folder / p) == h for p, h in receipt['outputs'].items())
    before = json.loads((OUT / 'before.json').read_text())
    assert all(sha(ROOT / p) == h for p, h in before['canonical'].items())
    paths = np.load(ROOT / 'artifacts/r8_mechanism/paths.npz')
    dec = pd.read_parquet(directory / 'decisions.parquet')
    results = pd.read_parquet(directory / 'replications.parquet')
    groups = {key: value.sort_values('replication') for key, value in dec.groupby(KEYS)}
    validation_error = gradient_error = 0.
    rational_comparisons = 0
    for cfg in configurations():
        d = groups[tuple(cfg.values())]
        y, raw, test_raw, sigma = forecasts(paths, cfg)
        n, alpha = cfg['n_cal'], cfg['alpha']
        m = max(100, int(.7 * n))
        k1, k2 = int(np.ceil((m + 1) * (1 - alpha))), int(np.ceil((n + 1) * (1 - alpha)))
        inner = np.sort(raw[:, :m] - y[:, :m], axis=1)[:, k1 - 1]
        full = np.sort(raw - y, axis=1)[:, k2 - 1]
        np.testing.assert_array_equal(inner, d.inner_shift.to_numpy())
        np.testing.assert_array_equal(full, d.full_shift.to_numpy())
        # Absolute-value representation gives a separate loss implementation.
        value = []
        for fraction in (0., .25, .5, .75, 1.):
            residual = y[:, m:] - (raw[:, m:] - fraction * inner[:, None])
            value.append((.5 * np.abs(residual) + (alpha - .5) * residual).mean(axis=1))
        independent = np.column_stack(value)
        stored = d[[f'validation_loss_{j}' for j in range(5)]].to_numpy()
        validation_error = max(validation_error, float(np.max(np.abs(independent - stored))))
        assert validation_error < 1e-14
        selected = np.argmin(independent, axis=1)
        different = np.flatnonzero(e.FRACTIONS[selected] != d.selected_fraction.to_numpy())
        for row in different:
            # Resolve disagreements without the producer's flat-interval logic.
            a = Fraction(str(alpha))
            scores = [Fraction(float(v)) for v in raw[row, m:] - y[row, m:]]
            values = []
            for fraction in (0., .25, .5, .75, 1.):
                residuals = [Fraction(float(fraction * inner[row])) - score for score in scores]
                values.append(sum((a - (r < 0)) * r for r in residuals))
            exact_index = min(range(5), key=lambda j: (values[j], j))
            assert e.FRACTIONS[exact_index] == d.selected_fraction.iloc[row]
            rational_comparisons += 1
        # Direct CDF gradient verifies the best constant without calling the solver.
        q = test_raw - d.best_constant_shift.iloc[0]
        probability = stats.norm.cdf(q / sigma) if cfg['innovation'] == 'normal' else stats.t.cdf(q, df=5, scale=sigma * np.sqrt(3 / 5))
        gradient_error = max(gradient_error, abs(float(probability.mean()) - alpha))
    assert gradient_error < 1e-11
    keys = KEYS + ['replication']
    wide = results.pivot(index=keys, columns='method', values='expected_loss')
    accounting = dec.set_index(keys).reindex(wide.index)
    total = (-accounting.removable_loss + accounting.inner_estimation_cost
             - accounting.oracle_shrinkage_gain + accounting.selection_regret)
    assert np.max(np.abs(total - (wide['Selected-Inner'] - wide.Raw))) < 1e-14
    assert np.min(wide['Selected-Inner'] - wide['Oracle-Grid']) > -1e-14
    assert np.min(wide['Oracle-Grid'] - wide['Oracle-Continuous']) > -1e-13
    assert np.max(np.abs(accounting.oracle_shrinkage_gain - (wide['Inner-CP'] - wide['Oracle-Grid']))) < 1e-14
    assert np.max(np.abs(accounting.selection_regret - (wide['Selected-Inner'] - wide['Oracle-Grid']))) < 1e-14
    cont = pd.read_parquet(directory / 'contiguous.parquet')
    cont_wide = cont.pivot(index=KEYS + ['replication', 'horizon'], columns='method', values='expected_loss')
    new = (cont_wide['Full-CP'] - cont_wide.Raw).rename('new_difference').reset_index()
    horizon_path = ROOT / 'artifacts/r8_horizon_bridge/horizon/replications.csv'
    previous = pd.read_csv(horizon_path)
    previous = previous[previous.H == 3 * previous.n_cal // 7]
    matched = new.merge(previous, left_on=['phi', 'n_cal', 'alpha', 'truth', 'replication', 'horizon'],
                        right_on=['phi', 'n_cal', 'alpha', 'truth', 'replication', 'H'], validate='one_to_one')
    assert len(matched) == 24000
    horizon_error = float(np.max(np.abs(matched.new_difference - matched.contiguous_change)))
    assert horizon_error < 1e-14
    integral_error = quadrature_checks()
    leakage_checks(paths)
    files = sorted((ROOT / 'research/r8_partial_shift').glob('*.py'))
    files += sorted((ROOT / 'research/r8_partial_shift').glob('*.md'))
    with tempfile.TemporaryDirectory(prefix='irfa-partial-whitespace-') as tmp:
        empty = Path(tmp) / 'empty'; empty.write_text('')
        for file in files:
            check = subprocess.run(['git', 'diff', '--no-index', '--check', str(empty), str(file)], capture_output=True, text=True)
            assert check.returncode in (0, 1) and not check.stdout and not check.stderr
    result = dict(status='passed',complete_fresh_process_replay=True,
                  independent_rank_and_selection_checks=72000,independent_validation_loss_error=validation_error,
                  exact_rational_tie_comparisons=rational_comparisons,
                  nonzero_flat_loss_controls=True,
                  scalar_optimum_gradient_error=gradient_error,quadrature_error=integral_error,
                  old_contiguous_reference_rows=24000,old_contiguous_reference_error=horizon_error,
                  exact_accounting=True,oracle_ordering=True,future_invariance=True,
                  leaking_negative_control=True,zero_shift_tie_control=True,infeasible_split_rejected=True,
                  protected_files=len(before['canonical']),canonical_unchanged=True,
                  producer_sha256=sha(__file__),additional_inputs={str(horizon_path.relative_to(ROOT)):sha(horizon_path)},
                  study_validation_sha256=sha(directory / 'validation.json'),
                  whitespace_files=len(files),git_metadata_present=(ROOT / '.git').exists())
    (OUT / 'independent_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
