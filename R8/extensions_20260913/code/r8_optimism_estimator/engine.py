"""Nuisance-free optimism estimators E1/E2 and feasible shrinkage on stored synthetic scores.

Protocol: research/r8_optimism_estimator/PROTOCOL.md (fixed 13 September 2026).
Inputs are read-only: results/theory_loop/synthetic/*. Outputs go to
artifacts/r8_optimism_estimator/. The loss, rank, empirical integral and the
simultaneous-band rule are imported from research/r8_theory_loop_v3/engine.py
(read-only import by path; no v3 producer is executed).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'research/r8_optimism_estimator'
OUT = ROOT / 'artifacts/r8_optimism_estimator'
SYN = ROOT / 'results/theory_loop/synthetic'
V3_ENGINE = ROOT / 'research/r8_theory_loop_v3/engine.py'
V3_LOCK = ROOT / 'results/theory_loop_v3/lock.json'
V3_BANDS = ROOT / 'results/theory_loop_v3/diagnostic/simultaneous_bands.csv'

LAWS = ('normal', 't5')
SIZES = (250, 500, 700, 1000, 2000)
BIASES = (0., .25 * np.sqrt(1e-5 / (1 - .10 - .85)))
OMEGA = .0099
P = .99
K_FOLDS = 5
R_BOOT = 200
SEED = 20260913
HISTORIES = 500
TOL = dict(atol=1e-12, rtol=1e-10)
MIXTURE_TOL = dict(atol=4e-11, rtol=4e-11)
ESTIMATORS = ('E1_blocked_cv', 'E2_block_bootstrap')
INPUT_FILES = ['calibration_normal.npz', 'calibration_t5.npz', 'truth.csv', 'mixture_integration_normal.npz',
               'mixture_integration_t5.npz', 'history_bootstrap_indices.npy', 'estimators.csv', 'loss_histories.csv']


def v3():
    spec = importlib.util.spec_from_file_location('r8_theory_loop_v3_engine', V3_ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rank(n):
    """k_n = min(n, ceil((n+1)p)) with p=0.99, in exact integer arithmetic."""
    return min(n, -(-(99 * (n + 1)) // 100))


def block_length(n):
    """ceil(n^(1/3)) as the smallest integer m with m^3 >= n."""
    m = int(round(n ** (1 / 3)))
    while m ** 3 < n:
        m += 1
    while m > 1 and (m - 1) ** 3 >= n:
        m -= 1
    return m


def conformal(x):
    """Conformal shift S_(k) along the last axis."""
    k = rank(x.shape[-1])
    return np.partition(x, k - 1, axis=-1)[..., k - 1]


def loss_change(rho, c, s):
    """D(c,s) = rho(c-s) - rho(-s), averaged along the last axis; c broadcast."""
    return (rho(c[..., None] - s) - rho(-s)).mean(axis=-1)


def train_loss(rho, c, s):
    """I_n(c) on a (histories, n) array of shifted scores."""
    return loss_change(rho, c, s)


def e1_blocked_cv(rho, shifted, cs, I):
    """Blocked cross-validation optimism, K contiguous folds, for all histories at once."""
    n = shifted.shape[1]
    assert n % K_FOLDS == 0
    size = n // K_FOLDS
    fold_losses = []
    fold_shifts = []
    for j in range(K_FOLDS):
        lo, hi = j * size, (j + 1) * size
        rest = np.concatenate([shifted[:, :lo], shifted[:, hi:]], axis=1)
        assert rest.shape[1] == n - size
        c_minus = conformal(rest)
        fold_shifts.append(c_minus)
        fold_losses.append(loss_change(rho, c_minus, shifted[:, lo:hi]))
    L_cv = np.mean(fold_losses, axis=0)
    O = (K_FOLDS - 1) / K_FOLDS * (L_cv - I)
    return O, L_cv, np.column_stack(fold_shifts)


def bootstrap_indices(rng, n, blen):
    """Circular block bootstrap indices, shape (R_BOOT, n)."""
    L = -(-n // blen)
    starts = rng.integers(0, n, size=(R_BOOT, L))
    idx = ((starts[:, :, None] + np.arange(blen)) % n).reshape(R_BOOT, -1)[:, :n]
    return idx


def e2_one_history(rho, x, biases, law_index, n, h):
    """Block-bootstrap optimism for one history and every bias translation.

    x: unshifted scores of length n. Returns (O per bias, c_star mean, index checksum).
    """
    blen = block_length(n)
    rng = np.random.default_rng([SEED, law_index, n, h])
    idx = bootstrap_indices(rng, n, blen)
    xs = x[idx]
    c_star = conformal(xs)
    fit_on_original = rho(c_star[:, None] - x[None, :]).mean(axis=1)
    fit_on_sample = rho(c_star[:, None] - xs).mean(axis=1)
    O = []
    for b in biases:
        base_original = rho(-(x + b)).mean()
        base_sample = rho(-(xs + b)).mean(axis=1)
        full_sample_loss = fit_on_original - base_original
        train_sample_loss = fit_on_sample - base_sample
        O.append(float(np.mean(full_sample_loss - train_sample_loss)))
    return np.array(O), float(c_star.mean()), int(idx.sum())


def e2_chunk(args):
    law_index, law, n, h_lo, h_hi = args
    module = v3()
    scores = np.load(SYN / f'calibration_{law}.npz')['scores'][:, :n]
    rows = []
    for h in range(h_lo, h_hi):
        O, cstar_mean, checksum = e2_one_history(module.rho, scores[h], BIASES, law_index, n, h)
        rows.append((law, n, h, O[0], O[1], cstar_mean, checksum))
    return rows


def shrinkage(O, I):
    """lambda_hat = clip(B_hat/(B_hat+A_hat),0,1); zero when B_hat+A_hat<=0."""
    A = O / 2
    B = -I - A
    denominator = B + A
    lam = np.where(denominator > 0, np.clip(B / np.where(denominator > 0, denominator, 1.), 0, 1), 0.)
    return A, B, lam


def reference_risk(law):
    m = np.load(SYN / f'mixture_integration_{law}.npz')
    poly = np.polynomial.Chebyshev(m['coef127'], domain=m['domain'])
    return poly, np.asarray(m['domain'], dtype=float)


def expected_loss(poly, domain, b, cs, lam):
    """R(lam*C_s) - R(0) on shifted scores: poly(b - lam*cs) - poly(b); asserts domain."""
    arg = b - lam * cs
    assert np.all(arg >= domain[0] - 1e-15) and np.all(arg <= domain[1] + 1e-15), 'Reference-risk domain'
    return poly(arg) - poly(b)


def check(rows, name, bad, good, predicate):
    """Negative control must be rejected and the valid case accepted (v3 convention)."""
    def accepts(value):
        try:
            return bool(predicate(value))
        except (ValueError, AssertionError, KeyError, IndexError):
            return False
    rejected = not accepts(bad)
    passed = accepts(good) if rejected else False
    rows.append(dict(name=name, negative_control_rejected=rejected, valid_case_accepted=passed))
    if not rejected or not passed:
        raise AssertionError(rows[-1])


def close(x, y, **tol):
    tol = tol or TOL
    return np.allclose(x, y, **tol)


def preflight(module):
    rows = []
    x = np.array([-2., -.5, 0., .5, 2.])
    wanted = np.array([1.98, .495, 0., .005, .02])
    check(rows, 'wrong_tail_loss', np.where(x >= 0, .99 * x, -.01 * x), module.rho(x), lambda z: close(z, wanted))
    for n in SIZES:
        for m in (n, n - n // K_FOLDS):
            check(rows, f'rank_rule_matches_v3_{m}', module.rank(m) + 1, rank(m), lambda z, m=m: z == module.rank(m) == int(np.ceil((m + 1) * P)) and z <= m)
    check(rows, 'block_length_is_ceil_cube_root', [7, 8, 9, 10, 12], [block_length(n) for n in SIZES],
          lambda z: z == [7, 8, 9, 10, 13] and all(v ** 3 >= n > (v - 1) ** 3 for v, n in zip(z, SIZES)))
    grid = np.linspace(-1, 1, 250)[None, :]
    c = conformal(grid)
    check(rows, 'empirical_instead_of_conformal_rank', grid[0, int(np.ceil(.99 * 250)) - 1], c[0], lambda z: z == grid[0, 248])
    I = train_loss(module.rho, c, grid)
    check(rows, 'integral_matches_v3_path', -I, I, lambda z: close(z, module.integral(grid, c[:, None])))
    O_full, _, _ = e1_blocked_cv(module.rho, grid, c, I)
    L_cv = O_full / ((K_FOLDS - 1) / K_FOLDS) + I
    check(rows, 'dropped_cv_rescaling_factor', L_cv - I, O_full, lambda z: close(z, .8 * (L_cv - I)))
    rng = np.random.default_rng(0)
    idx = bootstrap_indices(rng, 250, 7)
    check(rows, 'circular_block_indices', np.clip(idx + 250, 0, 249), idx,
          lambda z: z.shape == (R_BOOT, 250) and z.min() >= 0 and z.max() < 250
          and all(((z[r, t + 1] - z[r, t]) % 250 == 1) for r in range(3) for t in range(6)))
    A, B, lam = shrinkage(np.array([2e-6, 2e-6, 2e-6, -1e-6]), np.array([-5e-6, -1e-6, 1e-6, -1e-6]))
    check(rows, 'shrinkage_clip_and_zero_rule', np.array([.8, 0., 0., 2.]), lam,
          lambda z: close(z, [.8, 0., 0., 1.]) and close(A, [1e-6, 1e-6, 1e-6, -5e-7]))
    xx = np.array([[1., 3.], [2., 7.], [3., 5.], [6., 9.]])
    ii = np.array([[0, 0, 1, 2], [1, 2, 3, 3], [0, 1, 2, 3]])
    q, mean, se, maxima = module.critical_value(xx, ii)
    expected = np.max(np.abs(np.array([xx[r].mean(0) - mean for r in ii])) / se, axis=1)
    check(rows, 'band_rule_centres_on_original_mean', np.max(np.abs(xx[ii].mean(1) - 1) / se, axis=1), maxima,
          lambda z: close(z, expected))
    return rows


def record_inputs(module):
    lock = json.loads(V3_LOCK.read_text())
    locked = {Path(r['path']).name: r for r in lock['files']}
    records = []
    for name in INPUT_FILES:
        p = SYN / name
        digest = sha(p)
        rec = dict(path=str(p.relative_to(ROOT)), sha256=digest, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
        if name in locked:
            rec['v3_lock_sha256_match'] = bool(locked[name]['sha256'] == digest)
            rec['v3_lock_mtime_match'] = bool(locked[name]['mtime_ns'] == rec['mtime_ns'])
        records.append(rec)
    for p in (BASE / 'PROTOCOL.md', Path(__file__), V3_ENGINE, V3_LOCK):
        records.append(dict(path=str(p.relative_to(ROOT)), sha256=sha(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns))
    stale = []
    for r in lock['files']:
        p = Path(r['path'])
        if p.exists():
            same = sha(p) == r['sha256']
            if not same:
                stale.append(str(p.relative_to(ROOT)))
        else:
            stale.append(str(p) + ' (missing)')
    return records, stale, lock['protocol_commit']


def compute(workers):
    module = v3()
    checks = preflight(module)
    truth = pd.read_csv(SYN / 'truth.csv').set_index('law')
    indices = np.load(SYN / 'history_bootstrap_indices.npy')
    check(checks, 'history_bootstrap_rows', indices[:-1], indices,
          lambda z: z.shape == (999, 500) and z.min() >= 0 and z.max() < 500 and np.issubdtype(z.dtype, np.integer))
    estimates = pd.read_csv(SYN / 'estimators.csv')
    stored = pd.read_csv(SYN / 'loss_histories.csv')
    timing = {}
    t0 = time.monotonic()
    tasks = [(li, law, n, lo, min(lo + 50, HISTORIES)) for li, law in enumerate(LAWS) for n in SIZES for lo in range(0, HISTORIES, 50)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        e2_rows = [row for rows in pool.map(e2_chunk, tasks) for row in rows]
    timing['E2_seconds'] = time.monotonic() - t0
    e2 = pd.DataFrame(e2_rows, columns=['law', 'n', 'rep', 'O_b0', 'O_b1', 'c_star_mean', 'index_checksum'])
    t1 = time.monotonic()
    rows = []
    for li, law in enumerate(LAWS):
        scores = np.load(SYN / f'calibration_{law}.npz')['scores']
        check(checks, f'finite_saved_scores_{law}', scores[:499], scores, lambda z: z.shape == (500, 2000) and np.isfinite(z).all())
        f_ref = float(truth.loc[law, 'f_true'])
        poly, domain = reference_risk(law)
        for n in SIZES:
            x = scores[:, :n]
            c = conformal(x)
            two_A0 = OMEGA / (n * f_ref)
            blen = block_length(n)
            sub = e2[(e2.law == law) & (e2.n == n)].sort_values('rep')
            assert np.array_equal(sub.rep.to_numpy(), np.arange(HISTORIES))
            base0 = stored[(stored.law == law) & (stored.n == n) & stored.bias.eq(0)].sort_values('rep')
            check(checks, f'mixture_replay_{law}_{n}', poly(-c) + .01, poly(-c),
                  lambda z: np.allclose(z, base0.static_loss, **MIXTURE_TOL))
            for bi, b in enumerate(BIASES):
                shifted = x + b
                cs = c + b
                I = train_loss(module.rho, cs, shifted)
                ss = estimates[(estimates.law == law) & (estimates.n == n) & np.isclose(estimates.bias, b, atol=1e-15, rtol=0)].sort_values('rep')
                assert len(ss) == HISTORIES
                check(checks, f'shift_replay_{law}_{n}_{bi}', cs + .01, cs, lambda z: close(z, ss.C))
                check(checks, f'training_loss_replay_{law}_{n}_{bi}', I + .01, I, lambda z: close(z, ss.signed_ecdf_loss))
                st = stored[(stored.law == law) & (stored.n == n) & np.isclose(stored.bias, b, atol=1e-15, rtol=0)].sort_values('rep')
                loss_full = expected_loss(poly, domain, b, cs, 1.)
                loss_raw = expected_loss(poly, domain, b, cs, 0.)
                check(checks, f'full_correction_loss_replay_{law}_{n}_{bi}', loss_full + .01, loss_full,
                      lambda z: np.allclose(z, st.delta, **MIXTURE_TOL))
                check(checks, f'raw_loss_is_zero_{law}_{n}_{bi}', loss_raw + 1., loss_raw, lambda z: np.all(z == 0.))
                O1, L_cv, fold_shifts = e1_blocked_cv(module.rho, shifted, cs, I)
                O2 = sub['O_b1' if bi else 'O_b0'].to_numpy()
                for name, O in ((ESTIMATORS[0], O1), (ESTIMATORS[1], O2)):
                    A, B, lam = shrinkage(O, I)
                    loss_shrunk = expected_loss(poly, domain, b, cs, lam)
                    for h in range(HISTORIES):
                        rows.append(dict(estimator=name, law=law, n=n, bias=b, rep=h, C_shifted=cs[h], I=I[h],
                                         O_hat=O[h], ratio_to_2A0=O[h] / two_A0, two_A0=two_A0, A_hat=A[h], B_hat=B[h],
                                         lambda_hat=lam[h], loss_raw=loss_raw[h], loss_full=loss_full[h], loss_shrunk=loss_shrunk[h],
                                         cv_loss=L_cv[h] if name == ESTIMATORS[0] else np.nan,
                                         block_length=blen if name == ESTIMATORS[1] else 0,
                                         bootstrap_replicates=R_BOOT if name == ESTIMATORS[1] else 0))
    timing['E1_and_evaluation_seconds'] = time.monotonic() - t1
    histories = pd.DataFrame(rows)
    # Simultaneous families, one per estimator, over the 20 (law, n, bias) cells: ratio O_hat/(2A0).
    bands = []
    maxima = {}
    supplementary = []
    for name in ESTIMATORS:
        g = histories[histories.estimator == name]
        cells = sorted({(law, n, b) for law, n, b in zip(g.law, g.n, g.bias)})
        assert len(cells) == 20
        matrix = np.column_stack([g[(g.law == law) & (g.n == n) & (g.bias == b)].sort_values('rep').ratio_to_2A0.to_numpy() for law, n, b in cells])
        crit, means, se, mx = module.critical_value(matrix, indices)
        maxima[name] = mx
        for j, (law, n, b) in enumerate(cells):
            bands.append(dict(estimator=name, law=law, n=n, bias=b, mean_ratio=means[j], standard_error=se[j],
                              lower=means[j] - crit * se[j], upper=means[j] + crit * se[j],
                              includes_one=bool(means[j] - crit * se[j] <= 1 <= means[j] + crit * se[j]),
                              critical_value=crit, family_size=20, bootstrap_replicates=999))
        # Supplementary family (not a criterion): expected-loss differences shrunken-full (20 cells) and shrunken-raw (b>0, 10 cells).
        columns = []
        keys = []
        for law, n, b in cells:
            cell = g[(g.law == law) & (g.n == n) & (g.bias == b)].sort_values('rep')
            columns.append((cell.loss_shrunk - cell.loss_full).to_numpy())
            keys.append((law, n, b, 'shrunk_minus_full'))
            if b > 0:
                columns.append((cell.loss_shrunk - cell.loss_raw).to_numpy())
                keys.append((law, n, b, 'shrunk_minus_raw'))
        sup = np.column_stack(columns)
        se_sup = sup.std(axis=0, ddof=1) / np.sqrt(HISTORIES)
        if np.isfinite(se_sup).all() and (se_sup > 0).all():
            crit_sup, means_sup, se_sup, _ = module.critical_value(sup, indices)
            status = 'VALID'
        else:
            crit_sup, means_sup, status = np.nan, sup.mean(axis=0), 'INVALID_ZERO_SE'
        for j, (law, n, b, contrast) in enumerate(keys):
            supplementary.append(dict(estimator=name, law=law, n=n, bias=b, contrast=contrast, mean=means_sup[j],
                                      standard_error=se_sup[j], lower=means_sup[j] - crit_sup * se_sup[j],
                                      upper=means_sup[j] + crit_sup * se_sup[j], critical_value=crit_sup,
                                      family_size=len(keys), status=status))
    bands = pd.DataFrame(bands)
    supplementary = pd.DataFrame(supplementary)
    # Summary per (estimator, law, n, bias).
    oracle = pd.read_csv(V3_BANDS) if V3_BANDS.exists() else None
    summaries = []
    for (name, law, n, b), g in histories.groupby(['estimator', 'law', 'n', 'bias'], sort=True):
        band = bands[(bands.estimator == name) & (bands.law == law) & (bands.n == n) & (bands.bias == b)].iloc[0]
        d_full = g.loss_shrunk - g.loss_full
        d_raw = g.loss_shrunk - g.loss_raw
        rec = dict(estimator=name, law=law, n=n, bias=b, histories=len(g), two_A0=g.two_A0.iloc[0],
                   mean_O_hat=g.O_hat.mean(), mcse_O_hat=g.O_hat.std(ddof=1) / np.sqrt(len(g)),
                   ratio_to_2A0=g.O_hat.mean() / g.two_A0.iloc[0], ratio_mcse=band.standard_error,
                   band_lower=band.lower, band_upper=band.upper, band_includes_one=band.includes_one,
                   critical_value=band.critical_value, mean_I=g.I.mean(),
                   mean_lambda_hat=g.lambda_hat.mean(), sd_lambda_hat=g.lambda_hat.std(ddof=1),
                   share_lambda_zero=float((g.lambda_hat == 0).mean()), share_lambda_one=float((g.lambda_hat == 1).mean()),
                   loss_raw=g.loss_raw.mean(), loss_full=g.loss_full.mean(), loss_shrunk=g.loss_shrunk.mean(),
                   shrunk_minus_full=d_full.mean(), shrunk_minus_full_mcse=d_full.std(ddof=1) / np.sqrt(len(g)),
                   shrunk_minus_raw=d_raw.mean(), shrunk_minus_raw_mcse=d_raw.std(ddof=1) / np.sqrt(len(g)),
                   block_length=int(g.block_length.iloc[0]))
        if oracle is not None:
            o = oracle[(oracle.law == law) & (oracle.n == n) & (oracle.evaluation == 'independent')]
            rec['v3_true_optimism_ratio'] = float(o.mean_optimism_over_2A0.iloc[0]) if len(o) else np.nan
        summaries.append(rec)
    summary = pd.DataFrame(summaries)
    return dict(histories=histories, summary=summary, bands=bands, supplementary=supplementary,
                maxima=pd.DataFrame({'bootstrap': np.arange(999), **{f'max_t_{k}': v for k, v in maxima.items()}}),
                checks=checks, timing=timing, e2=e2)


def admission(summary, bands):
    result = dict(protocol=str((BASE / 'PROTOCOL.md').relative_to(ROOT)), estimators={})
    passing = []
    for name in ESTIMATORS:
        s = summary[summary.estimator == name]
        c1 = []
        for law in LAWS:
            for n in SIZES:
                r = float(s[(s.law == law) & (s.n == n) & (s.bias == 0)].ratio_to_2A0.iloc[0])
                r1 = float(s[(s.law == law) & (s.n == n) & (s.bias > 0)].ratio_to_2A0.iloc[0])
                c1.append(dict(law=law, n=n, ratio_bias0=r, ratio_bias1=r1, abs_error_bias0=abs(r - 1), abs_error_bias1=abs(r1 - 1),
                               applies=n >= 700, passes=bool(abs(r - 1) <= .15 and abs(r1 - 1) <= .15) if n >= 700 else None))
        c1_pass = all(c['passes'] for c in c1 if c['applies'])
        c2 = []
        for law in LAWS:
            for n in SIZES:
                for b in BIASES:
                    row = s[(s.law == law) & (s.n == n) & np.isclose(s.bias, b, atol=1e-15, rtol=0)].iloc[0]
                    below_full = bool(row.loss_shrunk < row.loss_full)
                    below_raw = bool(row.loss_shrunk < row.loss_raw)
                    required = below_full and below_raw if b > 0 else below_full
                    c2.append(dict(law=law, n=n, bias=float(b), loss_raw=float(row.loss_raw), loss_full=float(row.loss_full),
                                   loss_shrunk=float(row.loss_shrunk), below_full=below_full, below_raw=below_raw,
                                   applies=n >= 700, passes=bool(required) if n >= 700 else None))
        c2_pass = all(c['passes'] for c in c2 if c['applies'])
        bb = bands[bands.estimator == name]
        c3 = dict(family_size=int(len(bb)), critical_value=float(bb.critical_value.iloc[0]),
                  standard_errors_finite_positive=bool(np.isfinite(bb.standard_error).all() and (bb.standard_error > 0).all()),
                  cells_including_one=int(bb.includes_one.sum()),
                  cells=[dict(law=r.law, n=int(r.n), bias=float(r.bias), mean_ratio=float(r.mean_ratio), lower=float(r.lower),
                              upper=float(r.upper), includes_one=bool(r.includes_one)) for r in bb.itertuples()])
        c3['status'] = 'VALID' if c3['standard_errors_finite_positive'] and c3['family_size'] == 20 else 'INVALID'
        passes = bool(c1_pass and c2_pass)
        result['estimators'][name] = dict(criterion_1_penalty_accuracy=dict(threshold=.15, applies_to_n=[700, 1000, 2000], cells=c1, passes=c1_pass),
                                          criterion_2_rule_value=dict(applies_to_n=[700, 1000, 2000], cells=c2, passes=c2_pass),
                                          criterion_3_uncertainty=c3, passes=passes)
        if passes:
            passing.append(name)
    result['passing_estimators'] = passing
    result['financial_application'] = 'RUN' if passing else 'NOT_RUN'
    return result


def write(output, res):
    output.mkdir(parents=True, exist_ok=True)
    res['histories'].to_csv(output / 'histories.csv', index=False)
    res['summary'].to_csv(output / 'summary.csv', index=False)
    res['bands'].to_csv(output / 'simultaneous_bands.csv', index=False)
    res['supplementary'].to_csv(output / 'supplementary_loss_bands.csv', index=False)
    res['maxima'].to_csv(output / 'bootstrap_maxima.csv', index=False)
    res['e2'].to_csv(output / 'e2_bootstrap_diagnostics.csv', index=False)


def environment():
    import scipy
    return dict(python=sys.version, executable=sys.executable, numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                platform=platform.platform(), machine=platform.machine(), cpu_count=os.cpu_count())


def run(output, workers):
    assert not (output / 'histories.csv').exists(), 'Refuse to overwrite a completed calculation'
    module = v3()
    begin = time.monotonic()
    inputs, stale, commit = record_inputs(module)
    res = compute(workers)
    res['timing']['total_seconds'] = time.monotonic() - begin
    write(output, res)
    adm = admission(res['summary'], res['bands'])
    (output / 'admission.json').write_text(json.dumps(adm, indent=2, allow_nan=False) + '\n')
    meta = dict(seeds=dict(E2_block_bootstrap=SEED, E2_stream='numpy default_rng([20260913, law_index, n, history])',
                           history_bootstrap='saved results/theory_loop/synthetic/history_bootstrap_indices.npy (999 x 500)'),
                constants=dict(K=K_FOLDS, R=R_BOOT, block_lengths={n: block_length(n) for n in SIZES}, p=P, omega=OMEGA,
                               biases=list(map(float, BIASES)), sizes=list(SIZES), histories=HISTORIES),
                environment=environment(), timing=res['timing'], workers=workers, inputs=inputs,
                v3_lock_protocol_commit=commit, v3_lock_entries_changed_since_v3=stale,
                checks=res['checks'], output_sha256={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'run.json'})
    (output / 'run.json').write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(passing=adm['passing_estimators'], financial=adm['financial_application'], timing=res['timing']), indent=1))


def replay(output, workers):
    res = compute(workers)
    report = {}
    for name, frame in (('histories.csv', res['histories']), ('summary.csv', res['summary']), ('simultaneous_bands.csv', res['bands']),
                        ('supplementary_loss_bands.csv', res['supplementary']), ('bootstrap_maxima.csv', res['maxima']),
                        ('e2_bootstrap_diagnostics.csv', res['e2'])):
        saved = pd.read_csv(output / name)
        assert list(saved.columns) == list(frame.columns), name
        worst = 0.
        ok = True
        for col in frame.columns:
            a, b = frame[col].to_numpy(), saved[col].to_numpy()
            if np.issubdtype(frame[col].dtype, np.number):
                both_nan = pd.isna(a) & pd.isna(b)
                a = np.where(both_nan, 0., a).astype(float)
                b = np.where(both_nan, 0., b).astype(float)
                same = np.allclose(a, b, **TOL)
                worst = max(worst, float(np.nanmax(np.abs(a - b))) if len(a) else 0.)
            else:
                same = np.array_equal(a.astype(str), b.astype(str))
            ok = ok and bool(same)
        report[name] = dict(match=ok, rows=int(len(frame)), max_abs_difference=worst)
    saved_adm = json.loads((output / 'admission.json').read_text())
    new_adm = admission(res['summary'], res['bands'])
    report['admission.json'] = dict(match=saved_adm['passing_estimators'] == new_adm['passing_estimators']
                                    and saved_adm['financial_application'] == new_adm['financial_application']
                                    and all(saved_adm['estimators'][k]['passes'] == new_adm['estimators'][k]['passes'] for k in ESTIMATORS))
    status = 'PASS' if all(v['match'] for v in report.values()) else 'FAIL'
    (output / 'check.json').write_text(json.dumps(dict(status=status, tolerance=TOL, files=report, environment=environment()), indent=2) + '\n')
    print('REPLAY', status, json.dumps(report, indent=1))
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if args.check:
        sys.exit(0 if replay(args.output, args.workers) == 'PASS' else 1)
    run(args.output, args.workers)
