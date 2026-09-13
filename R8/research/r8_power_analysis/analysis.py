"""Prespecified exploratory decomposition of the static-minus-raw contrast (PROTOCOL.md).

Independent reimplementation of the common-calendar circular block bootstrap of
research/r8_ten_comparators/aggregate.py (999 draws, blocks of 20 and 60 calendar days,
seed = seed_for('panel-calendar', block)). Reads only artifacts/r8_ten_comparators/pairs.
Writes only artifacts/r8_power_analysis. `--check` recomputes everything and compares with
the saved outputs and with the published contrast 1.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '2'
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
PAIRS = PROJECT / 'artifacts/r8_ten_comparators/pairs'
PUBLISHED = PROJECT / 'artifacts/r8_ten_comparators/results'
OUT = PROJECT / 'artifacts/r8_power_analysis'
ALPHA = .01
DRAWS = 999
CHUNK = 25
BLOCKS = (20, 60)
SENSITIVITY_BLOCKS = (5, 10)
MODELS = ['PatchTST-FM', 'Chronos-2', 'TS-ICL', 'Moirai-1.1', 'Lag-Llama',
          'GJR-GARCH', 'GJR-GARCH-t', 'GARCH-N', 'Hist-Sim', 'EWMA']
FOUNDATION = MODELS[:5]
CLASSICAL = MODELS[5:]
# Classes copied from source/sections_r8/tab_assets.tex (ICLN is listed there as Equity).
CLASS = {'ASX200': 'Equity', 'BOVESPA': 'Equity', 'FCHI': 'Equity', 'FTSE100': 'Equity', 'GDAXI': 'Equity',
         'HSI': 'Equity', 'ICLN': 'Equity', 'NIFTY': 'Equity', 'NIKKEI': 'Equity', 'SP500': 'Equity',
         'STOXX': 'Equity', 'AUDUSD': 'FX', 'EURUSD': 'FX', 'GBPUSD': 'FX', 'USDJPY': 'FX',
         'BTC': 'Crypto', 'ETH': 'Crypto', 'CBU0': 'Bond ETF', 'IBGL': 'Bond ETF', 'TLT': 'Bond ETF',
         'DJCI': 'Commodity', 'GLD': 'Commodity', 'UNG': 'Commodity', 'USO': 'Commodity'}
ASSETS = sorted(CLASS)
STATIC, RAW = 'Shift-CP', 'Raw'
CONTRASTS = ['1_all_pairs', '2_classical_only', '3_foundation_only', '4_excluding_lag_llama',
             '5_equities_only', '6_fx_bonds_commodities', '7_crypto_only', '8_pair_normalised', '9_date_weighted']
TOP_DATES = 10


def seed_for(key, replicate=0):
    """Same convention as research/r8_decision/methods.py (reimplemented, not imported)."""
    return int.from_bytes(hashlib.sha256(f'20260909/{key}/{replicate}'.encode()).digest()[:4], 'little')


def pinball(y, q, alpha=ALPHA):
    e = np.asarray(y) - np.asarray(q)
    return np.where(e >= 0, alpha * e, (alpha - 1) * e)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    """Common-calendar panel of d_it = loss(static) - loss(raw), pair order MODELS x ASSETS."""
    frames, meta = [], []
    for model in MODELS:
        for asset in ASSETS:
            folder = PAIRS / f'{model}__{asset}'
            f = pd.read_parquet(folder / 'daily.parquet')
            metric = pd.read_csv(folder / 'metrics.csv').set_index('method')
            scale = float(metric.calibration_scale.iloc[0])
            assert (metric.calibration_scale == scale).all()
            d = pd.Series(pinball(f.r, f[STATIC]) - pinball(f.r, f[RAW]), index=f.index)
            raw_loss = pd.Series(pinball(f.r, f[RAW]), index=f.index)
            # Per-pair QS in metrics.csv must equal the mean of the recomputed daily loss.
            assert abs(raw_loss.mean() - metric.loc[RAW, 'QS']) < 1e-15
            assert abs(pinball(f.r, f[STATIC]).mean() - metric.loc[STATIC, 'QS']) < 1e-15
            frames.append(d)
            meta.append({'model': model, 'asset': asset, 'class': CLASS[asset], 'scale': scale,
                         'n_test': len(f), 'first': f.index[0], 'last': f.index[-1],
                         'raw_QS': metric.loc[RAW, 'QS'], 'static_QS': metric.loc[STATIC, 'QS']})
    assert len(frames) == 240
    meta = pd.DataFrame(meta)
    dates = pd.date_range(min(f.index[0] for f in frames), max(f.index[-1] for f in frames), freq='D')
    T, P = len(dates), len(frames)
    valid = np.zeros((T, P))
    values = np.zeros((T, P))
    for j, f in enumerate(frames):
        ix = dates.get_indexer(f.index)
        assert (ix >= 0).all()
        valid[ix, j] = 1
        values[ix, j] = f.to_numpy()
    return dates, valid, values, meta, frames


def block_counts(T, block, draws=DRAWS, chunk=CHUNK):
    """Circular calendar-block resampling weights, drawn exactly as in aggregate.py."""
    rng = np.random.default_rng(seed_for('panel-calendar', block))
    counts = np.empty((draws, T))
    b = 0
    for start in range(0, draws, chunk):
        B = min(chunk, draws - start)
        for _ in range(B):
            begins = rng.integers(0, T, size=int(np.ceil(T / block)))
            idx = ((begins[:, None] + np.arange(block)) % T).ravel()[:T]
            counts[b] = np.bincount(idx, minlength=T)
            b += 1
    assert b == draws
    return counts


def subsets(meta):
    all_pairs = np.ones(len(meta), bool)
    return {'1_all_pairs': all_pairs,
            '2_classical_only': meta.model.isin(CLASSICAL).to_numpy(),
            '3_foundation_only': meta.model.isin(FOUNDATION).to_numpy(),
            '4_excluding_lag_llama': (meta.model != 'Lag-Llama').to_numpy(),
            '5_equities_only': (meta['class'] == 'Equity').to_numpy(),
            '6_fx_bonds_commodities': meta['class'].isin(['FX', 'Bond ETF', 'Commodity']).to_numpy(),
            '7_crypto_only': (meta['class'] == 'Crypto').to_numpy(),
            '8_pair_normalised': all_pairs,
            '9_date_weighted': all_pairs}


def statistics(counts, valid, values, meta, masks):
    """Nine contrast statistics for each row of `counts` (rows of ones give the point estimate).

    Contrasts 1-7: unweighted mean over the pairs in the subset of the pair mean of d_it.
    Contrast 8: the same with each pair mean divided by its calibration-return SD.
    Contrast 9: mean over calendar dates of the cross-sectional mean of d_it over pairs present.
    """
    denom = counts @ valid                              # (B, P) resampled observation counts per pair
    assert (denom > 0).all()
    pair_means = (counts @ values) / denom              # (B, P)
    n_t = valid.sum(axis=1)                             # pairs present on each calendar date
    present = n_t > 0
    date_means = np.zeros(len(n_t))
    date_means[present] = values[present].sum(axis=1) / n_t[present]
    out = {}
    for name in CONTRASTS:
        m = masks[name]
        if name == '8_pair_normalised':
            out[name] = (pair_means[:, m] / meta.scale.to_numpy()[m]).mean(axis=1)
        elif name == '9_date_weighted':
            out[name] = (counts @ date_means) / (counts @ present.astype(float))
        else:
            out[name] = pair_means[:, m].mean(axis=1)
    return pd.DataFrame(out), pair_means


def bands(point, draws, family):
    """Pointwise percentile 95% and max-standardised-deviation simultaneous 95% over `family`."""
    center = point[family].to_numpy()
    dc = draws[family].to_numpy()
    sd = dc.std(axis=0, ddof=1)
    assert (sd > 0).all()
    crit = float(np.quantile(np.max(np.abs((dc - center) / sd), axis=1), .95))
    rows = []
    for k, name in enumerate(family):
        rows.append({'contrast': name, 'point': center[k], 'bootstrap_sd': sd[k],
                     'pointwise_lower': float(np.quantile(dc[:, k], .025)),
                     'pointwise_upper': float(np.quantile(dc[:, k], .975)),
                     'simultaneous_lower': center[k] - crit * sd[k],
                     'simultaneous_upper': center[k] + crit * sd[k]})
    return pd.DataFrame(rows), crit


def scale_rows(df, unit_col='contrast'):
    """Report in units of 1e-4 loss, except the normalised contrast (unitless)."""
    df = df.copy()
    factor = np.where(df[unit_col] == '8_pair_normalised', 1., 1e4)
    for c in ['point', 'bootstrap_sd', 'pointwise_lower', 'pointwise_upper', 'simultaneous_lower', 'simultaneous_upper']:
        df[c] = df[c] * factor
    df['units'] = np.where(df[unit_col] == '8_pair_normalised', 'loss / calibration SD', 'loss x 1e4')
    return df


def autocorrelation(frames, max_lag=20):
    acf = np.zeros((len(frames), max_lag))
    for j, f in enumerate(frames):
        x = f.to_numpy()
        x = x - x.mean()
        v = (x * x).sum()
        for lag in range(1, max_lag + 1):
            acf[j, lag - 1] = (x[lag:] * x[:-lag]).sum() / v
    return acf


def decomposition(dates, valid, values, meta, frames, masks, counts_by_block):
    rows = []
    obs = values[valid == 1]
    N = obs.size
    pooled = obs.mean()
    pair_means = np.array([f.mean() for f in frames])
    rows.append(('pooled_mean_all_observations_x1e4', pooled * 1e4, f'mean of d_it over all {N} pair-date observations'))
    rows.append(('pair_weighted_mean_x1e4', pair_means.mean() * 1e4, 'contrast 1: mean over 240 pairs of the pair mean'))
    rows.append(('n_observations', N, 'pair-date observations'))
    rows.append(('n_pairs', len(frames), ''))
    rows.append(('between_pair_sd_of_pair_means_x1e4', pair_means.std(ddof=1) * 1e4, 'SD (ddof=1) across 240 pair means'))
    within = np.array([f.std(ddof=1) for f in frames])
    rows.append(('average_within_pair_sd_x1e4', within.mean() * 1e4, 'mean over pairs of the within-pair SD (ddof=1) of d_it'))
    rows.append(('median_within_pair_sd_x1e4', np.median(within) * 1e4, ''))
    acf = autocorrelation(frames)
    for lag in range(1, 21):
        rows.append((f'average_lag{lag}_autocorrelation', acf[:, lag - 1].mean(), 'mean over pairs of the within-pair sample autocorrelation'))
    rows.append(('average_lag1_to_lag20_autocorrelation', acf.mean(), 'mean over pairs and lags 1..20'))
    n_t = valid.sum(axis=1)
    rows.append(('n_distinct_calendar_dates', int((n_t > 0).sum()), 'calendar dates with at least one pair present'))
    rows.append(('calendar_span_days', len(dates), f'{dates[0].date()} to {dates[-1].date()} inclusive'))
    rows.append(('first_test_date', str(dates[n_t > 0][0].date()), ''))
    rows.append(('last_test_date', str(dates[n_t > 0][-1].date()), ''))
    rows.append(('min_pairs_present_on_a_trading_date', int(n_t[n_t > 0].min()), ''))
    rows.append(('max_pairs_present_on_a_trading_date', int(n_t.max()), ''))
    # Shares of the gain, by model and by asset class, under pair weighting (contrast 1) and observation pooling.
    total_pair = pair_means.sum()
    total_obs = obs.sum()
    for model in MODELS:
        m = (meta.model == model).to_numpy()
        rows.append((f'share_pair_weighted_model_{model}', pair_means[m].sum() / total_pair, 'sum of pair means in group / sum over all pairs'))
    for cls in ['Equity', 'FX', 'Bond ETF', 'Commodity', 'Crypto']:
        m = (meta['class'] == cls).to_numpy()
        rows.append((f'share_pair_weighted_class_{cls}', pair_means[m].sum() / total_pair, 'sum of pair means in group / sum over all pairs'))
    for model in MODELS:
        m = (meta.model == model).to_numpy()
        rows.append((f'share_observation_pooled_model_{model}', values[:, m][valid[:, m] == 1].sum() / total_obs, 'sum of d_it in group / sum over all observations'))
    for cls in ['Equity', 'FX', 'Bond ETF', 'Commodity', 'Crypto']:
        m = (meta['class'] == cls).to_numpy()
        rows.append((f'share_observation_pooled_class_{cls}', values[:, m][valid[:, m] == 1].sum() / total_obs, 'sum of d_it in group / sum over all observations'))
    # Ten largest dates. Primary: calendar dates ranked by |sum over pairs present of d_it| (the date's
    # contribution to the pooled numerator). Alternative: ranked by max over pairs of |d_it|.
    date_sum = values.sum(axis=1)
    date_max = np.abs(values).max(axis=1)
    rankings = {'abs_date_sum': np.argsort(-np.abs(date_sum))[:TOP_DATES],
                'max_abs_dit': np.argsort(-date_max)[:TOP_DATES]}
    for label, top in rankings.items():
        rows.append((f'top{TOP_DATES}_dates_{label}', ';'.join(str(dates[i].date()) for i in top), 'ten calendar dates, descending'))
        rows.append((f'top{TOP_DATES}_dates_{label}_share_of_pooled_sum', date_sum[top].sum() / total_obs, 'sum of d_it on those dates / sum over all observations'))
        keep = np.ones(len(dates), bool)
        keep[top] = False
        valid_lo = valid * keep[:, None]
        values_lo = values * keep[:, None]
        for block, counts in counts_by_block.items():
            full = ((counts @ values) / (counts @ valid)).mean(axis=1)
            leave = ((counts @ values_lo) / (counts @ valid_lo)).mean(axis=1)
            v_full = full.var(ddof=1)
            v_leave = leave.var(ddof=1)
            rows.append((f'bootstrap_variance_contrast1_block{block}_x1e8', v_full * 1e8, 'variance (ddof=1) of the 999 contrast-1 draws'))
            rows.append((f'bootstrap_variance_contrast1_block{block}_without_top{TOP_DATES}_{label}_x1e8', v_leave * 1e8, 'same draws with those ten dates removed from every pair'))
            rows.append((f'fraction_of_bootstrap_variance_from_top{TOP_DATES}_{label}_block{block}', 1 - v_leave / v_full, '1 - leave-out variance / full variance'))
            rows.append((f'point_contrast1_block{block}_without_top{TOP_DATES}_{label}_x1e4',
                         ((valid_lo * values_lo).sum(axis=0) / valid_lo.sum(axis=0)).mean() * 1e4, 'contrast 1 point with those ten dates removed'))
    return pd.DataFrame(rows, columns=['item', 'value', 'definition'])


def run(out_dir):
    t0 = time.monotonic()
    dates, valid, values, meta, frames = load()
    masks = subsets(meta)
    T = len(dates)
    ones = np.ones((1, T))
    point, point_pair_means = statistics(ones, valid, values, meta, masks)
    point = point.iloc[0]
    seeds = {}
    counts_by_block = {}
    contrast_rows, sens_rows, sign_rows = [], [], []
    sign_point = float((point_pair_means[0] < 0).mean())
    n_improved = int((point_pair_means[0] < 0).sum())
    # Consistency with metrics.csv: pair improved iff static QS < raw QS.
    assert n_improved == int((meta.static_QS < meta.raw_QS).sum())
    for block in BLOCKS + SENSITIVITY_BLOCKS:
        seeds[block] = seed_for('panel-calendar', block)
        counts = block_counts(T, block)
        draws, pm = statistics(counts, valid, values, meta, masks)
        if block in BLOCKS:
            counts_by_block[block] = counts
            table, crit = bands(point, draws, CONTRASTS)
            table['block_calendar_days'] = block
            table['draws'] = DRAWS
            table['pairs'] = [int(masks[c].sum()) for c in CONTRASTS]
            table['simultaneous_family_size'] = len(CONTRASTS)
            table['simultaneous_critical_value'] = crit
            contrast_rows.append(table)
            frac = (pm < 0).mean(axis=1)
            sign_rows.append({'block_calendar_days': block, 'draws': DRAWS, 'pairs_improved': n_improved, 'pairs': 240,
                              'fraction_improved': sign_point,
                              'bootstrap_lower': float(np.quantile(frac, .025)), 'bootstrap_upper': float(np.quantile(frac, .975)),
                              'bootstrap_sd': float(frac.std(ddof=1))})
        else:
            table, _ = bands(point, draws, ['1_all_pairs'])
            table['block_calendar_days'] = block
            table['draws'] = DRAWS
            table['pairs'] = 240
            table['simultaneous_family_size'] = 1
            table['note'] = 'sensitivity only; not in the nine-contrast family; simultaneous band equals the max-|z| band of a single contrast'
            sens_rows.append(table)
    contrasts = scale_rows(pd.concat(contrast_rows, ignore_index=True))
    sensitivity = scale_rows(pd.concat(sens_rows, ignore_index=True))
    sign = pd.DataFrame(sign_rows)
    decomp = decomposition(dates, valid, values, meta, frames, masks, counts_by_block)
    elapsed = time.monotonic() - t0
    return {'contrasts': contrasts, 'sensitivity': sensitivity, 'sign': sign, 'decomposition': decomp,
            'seeds': seeds, 'elapsed': elapsed, 'meta': meta, 'T': T, 'point': point}


def _binom_cdf(k, n, p=.5):
    from math import comb
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def _binom_upper_tail(k, n, p=.5):
    """P(X >= k), summed directly so that a tail near 1e-19 does not vanish in 1 - cdf."""
    from math import comb
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def check_against_published(result):
    """Contrast 1 must reproduce the published static-minus-raw point to 4 decimals, and the
    reimplemented draws must reproduce the published bootstrap draws of Shift-CP minus Raw."""
    published = pd.read_csv(PUBLISHED / 'intervals.csv')
    row = published[(published.method == RAW) & (published.reference == STATIC) & (published.block_calendar_days == 20)].iloc[0]
    mine = result['contrasts']
    c1 = mine[(mine.contrast == '1_all_pairs') & (mine.block_calendar_days == 20)].iloc[0]
    static_minus_raw_published = -float(row.difference)
    print(f'published static-minus-raw x1e4: {static_minus_raw_published:.6f}; reimplemented: {c1.point:.6f}')
    assert round(static_minus_raw_published, 4) == round(float(c1.point), 4), 'contrast 1 point does not match published'
    assert abs(static_minus_raw_published - float(c1.point)) < 1e-9
    # Published pointwise band of Raw minus Shift-CP, negated, must equal my pointwise band of contrast 1.
    for block in BLOCKS:
        row = published[(published.method == RAW) & (published.reference == STATIC) & (published.block_calendar_days == block)].iloc[0]
        c1 = mine[(mine.contrast == '1_all_pairs') & (mine.block_calendar_days == block)].iloc[0]
        assert abs(-float(row.upper) - c1.pointwise_lower) < 1e-9 and abs(-float(row.lower) - c1.pointwise_upper) < 1e-9, block
        z = np.load(PUBLISHED / f'bootstrap_{block}.npz', allow_pickle=True)
        methods = list(z['methods'])
        pub = z['draws'][:, methods.index(STATIC)] - z['draws'][:, methods.index(RAW)]
        print(f'block {block}: published pointwise band of static minus raw x1e4 [{-row.upper:.4f}, {-row.lower:.4f}] reproduced')
        yield block, pub


def write(result):
    OUT.mkdir(parents=True, exist_ok=True)
    result['contrasts'].to_csv(OUT / 'contrasts.csv', index=False)
    result['sensitivity'].to_csv(OUT / 'sensitivity.csv', index=False)
    result['sign'].to_csv(OUT / 'sign.csv', index=False)
    result['decomposition'].to_csv(OUT / 'decomposition.csv', index=False)
    result['meta'].assign(pair_mean_x1e4=lambda m: (m.static_QS - m.raw_QS) * 1e4).to_csv(OUT / 'pair_means.csv', index=False)
    (OUT / 'run.json').write_text(json.dumps({
        'protocol_sha256': sha(Path(__file__).with_name('PROTOCOL.md')),
        'producer_sha256': sha(__file__),
        'seeds': {str(k): v for k, v in result['seeds'].items()},
        'seed_convention': "int.from_bytes(sha256('20260909/panel-calendar/{block}')[:4], 'little')",
        'draws': DRAWS, 'calendar_days': result['T'], 'elapsed_seconds': result['elapsed'],
        'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
        'inputs': {'pairs_dir': str(PAIRS.relative_to(PROJECT)), 'n_pairs': 240},
    }, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='recompute and compare with saved outputs and the published contrast 1')
    args = ap.parse_args()
    result = run(OUT)
    draws_ok = True
    for block, pub in check_against_published(result):
        dates, valid, values, meta, frames = load()
        counts = block_counts(len(dates), block)
        mine = ((counts @ values) / (counts @ valid)).mean(axis=1)
        err = np.abs(mine - pub).max()
        print(f'block {block}: max |draw difference| vs published bootstrap_{block}.npz = {err:.3e}')
        draws_ok &= err < 1e-12
    assert draws_ok, 'reimplemented draws differ from the published draws'
    print(f'elapsed {result["elapsed"]:.1f}s; seeds {result["seeds"]}')
    if args.check:
        for name in ['contrasts', 'sensitivity', 'sign', 'decomposition']:
            saved = pd.read_csv(OUT / f'{name}.csv')
            fresh = result[name]
            assert list(saved.columns) == list(fresh.columns), name
            for c in saved.columns:
                if saved[c].dtype.kind in 'fi':
                    assert np.allclose(saved[c].to_numpy(dtype=float), fresh[c].to_numpy(dtype=float), rtol=0, atol=1e-10, equal_nan=True), (name, c)
                else:
                    a = saved[c].fillna('').astype(str).to_numpy()
                    b = fresh[c].fillna('').astype(str).to_numpy()
                    for x, y in zip(a, b):
                        try:
                            assert abs(float(x) - float(y)) <= 1e-10, (name, c, x, y)
                        except ValueError:
                            assert x == y, (name, c, x, y)
        print('CHECK PASSED: saved outputs reproduced; published contrast 1 reproduced')
    else:
        write(result)
        print('written', OUT)


if __name__ == '__main__':
    main()
