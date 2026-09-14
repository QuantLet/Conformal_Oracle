"""Static-minus-raw contrast on the window common to all 24 assets (PROTOCOL.md)."""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ[_k] = '2'
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
PROJECT = Path(__file__).resolve().parents[2]
PAIRS = PROJECT / 'artifacts/r8_ten_comparators/pairs'
BOUND = PROJECT / 'artifacts/r8_model_extension/ten_common_evaluation/boundaries.csv'
OUT = PROJECT / 'artifacts/r8_common_window'
ALPHA = .01; DRAWS = 999; CHUNK = 25; BLOCKS = (20, 60)
MODELS = ['PatchTST-FM', 'Chronos-2', 'TS-ICL', 'Moirai-1.1', 'Lag-Llama', 'GJR-GARCH', 'GJR-GARCH-t', 'GARCH-N', 'Hist-Sim', 'EWMA']
METHODS = ['Raw', 'Shift-CP', 'Rolling250']

def seed_for(key, replicate=0):
    return int.from_bytes(hashlib.sha256(f'20260909/{key}/{replicate}'.encode()).digest()[:4], 'little')

def pinball(y, q, alpha=ALPHA):
    e = np.asarray(y) - np.asarray(q); return np.where(e >= 0, alpha * e, (alpha - 1) * e)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def block_counts(T, block, draws=DRAWS, chunk=CHUNK):
    rng = np.random.default_rng(seed_for('panel-calendar', block)); counts = np.empty((draws, T)); b = 0
    for start in range(0, draws, chunk):
        for _ in range(min(chunk, draws - start)):
            begins = rng.integers(0, T, size=int(np.ceil(T / block)))
            idx = ((begins[:, None] + np.arange(block)) % T).ravel()[:T]
            counts[b] = np.bincount(idx, minlength=T); b += 1
    return counts

def load():
    b = pd.read_csv(BOUND); starts = b.groupby('asset').common_test_first.agg(['min', 'max'])
    assert (starts['min'] == starts['max']).all(); start = pd.Timestamp(starts['max'].max())
    frames, rows = [], []
    for model in MODELS:
        for asset in sorted(starts.index):
            f = pd.read_parquet(PAIRS / f'{model}__{asset}' / 'daily.parquet'); g = f[f.index >= start]
            loss = {m: pinball(g.r, g[m]) for m in METHODS}
            d = pd.Series(loss['Shift-CP'] - loss['Raw'], index=g.index); frames.append(d)
            rows.append({'model': model, 'asset': asset, 'n_common': len(g), 'n_full': len(f), 'first': str(g.index[0].date()), 'last': str(g.index[-1].date()),
                         'raw_QS': loss['Raw'].mean(), 'static_QS': loss['Shift-CP'].mean(), 'rolling_QS': loss['Rolling250'].mean(),
                         'raw_pi': float((g.r < g['Raw']).mean()), 'static_pi': float((g.r < g['Shift-CP']).mean()), 'rolling_pi': float((g.r < g['Rolling250']).mean()),
                         'full_raw_QS': pinball(f.r, f['Raw']).mean(), 'full_static_QS': pinball(f.r, f['Shift-CP']).mean()})
    meta = pd.DataFrame(rows); assert len(frames) == 240
    dates = pd.date_range(start, max(f.index[-1] for f in frames), freq='D'); T = len(dates)
    valid = np.zeros((T, 240)); values = np.zeros((T, 240))
    for j, f in enumerate(frames):
        ix = dates.get_indexer(f.index); assert (ix >= 0).all(); valid[ix, j] = 1; values[ix, j] = f.to_numpy()
    return start, dates, valid, values, meta, b

def run():
    t0 = time.monotonic(); start, dates, valid, values, meta, b = load(); T = len(dates)
    point = float(((valid * values).sum(axis=0) / valid.sum(axis=0)).mean())
    rows = []
    for block in BLOCKS:
        counts = block_counts(T, block); denom = counts @ valid; assert (denom > 0).all()
        draws = ((counts @ values) / denom).mean(axis=1); sd = draws.std(ddof=1)
        crit = float(np.quantile(np.abs((draws - point) / sd), .95))
        rows.append({'block_calendar_days': block, 'draws': DRAWS, 'pairs': 240, 'point_x1e4': point * 1e4, 'bootstrap_sd_x1e4': sd * 1e4,
                     'pointwise_lower_x1e4': float(np.quantile(draws, .025)) * 1e4, 'pointwise_upper_x1e4': float(np.quantile(draws, .975)) * 1e4,
                     'studentised_critical_value': crit, 'studentised_lower_x1e4': (point - crit * sd) * 1e4, 'studentised_upper_x1e4': (point + crit * sd) * 1e4,
                     'seed': seed_for('panel-calendar', block), 'calendar_days': T})
    contrast = pd.DataFrame(rows)
    n_t = valid.sum(axis=1); present = n_t > 0
    summary = {'common_start': str(start.date()), 'common_end': str(dates[-1].date()), 'calendar_days': T, 'trading_dates': int(present.sum()),
               'min_pairs_present': int(n_t[present].min()), 'max_pairs_present': int(n_t.max()), 'pair_days': int(valid.sum()),
               'point_x1e4': point * 1e4, 'full_window_point_x1e4': float((meta.full_static_QS - meta.full_raw_QS).mean()) * 1e4,
               'raw_QS_x1e4': meta.raw_QS.mean() * 1e4, 'static_QS_x1e4': meta.static_QS.mean() * 1e4, 'rolling_QS_x1e4': meta.rolling_QS.mean() * 1e4,
               'raw_pi_pct': 100 * meta.raw_pi.mean(), 'static_pi_pct': 100 * meta.static_pi.mean(), 'rolling_pi_pct': 100 * meta.rolling_pi.mean(),
               'pairs_improved': int((meta.static_QS < meta.raw_QS).sum()), 'min_n_common': int(meta.n_common.min()), 'max_n_common': int(meta.n_common.max()),
               'elapsed_seconds': time.monotonic() - t0}
    split = b[b.model == MODELS[0]][['asset', 'common_first', 'common_test_first', 'last', 'common_n_cal', 'common_n']].copy()
    split['n_test'] = split.common_n - split.common_n_cal
    split = split.sort_values('common_test_first').rename(columns={'common_first': 'first_date', 'common_test_first': 'test_start', 'last': 'last_date', 'common_n_cal': 'n_calibration', 'common_n': 'n_total'}).reset_index(drop=True)
    return contrast, meta, split, summary

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
    contrast, meta, split, summary = run()
    files = {'contrast.csv': contrast, 'pairs.csv': meta, 'split_dates.csv': split}
    if a.check:
        for name, df in files.items():
            saved = pd.read_csv(OUT / name); assert list(saved.columns) == list(df.columns), name
            for c in df.columns:
                if df[c].dtype.kind in 'fi': assert np.allclose(saved[c].to_numpy(float), df[c].to_numpy(float), atol=1e-10), (name, c)
                else: assert saved[c].astype(str).tolist() == df[c].astype(str).tolist(), (name, c)
        old = json.loads((OUT / 'run.json').read_text())['summary']
        for k, v in summary.items():
            if k != 'elapsed_seconds': assert (abs(old[k] - v) < 1e-10) if isinstance(v, float) else old[k] == v, k
        print('CHECK PASSED: common-window contrast, pair table and split dates reproduced'); return
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in files.items(): df.to_csv(OUT / name, index=False)
    (OUT / 'run.json').write_text(json.dumps({'protocol_sha256': sha(Path(__file__).with_name('PROTOCOL.md')), 'producer_sha256': sha(__file__),
        'inputs': {'boundaries.csv': sha(BOUND), 'pairs_dir': str(PAIRS.relative_to(PROJECT))}, 'summary': summary, 'python': sys.version, 'numpy': np.__version__}, indent=2) + '\n')
    (OUT / 'RESULTS.md').write_text('# Common-window contrast (measured)\n\n' + json.dumps(summary, indent=2) + '\n\n' + contrast.to_string() + '\n')
    print((OUT / 'RESULTS.md').read_text())

if __name__ == '__main__': main()
