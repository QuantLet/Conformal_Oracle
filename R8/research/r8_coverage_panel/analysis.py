"""Panel-level violation rate with common-calendar block-bootstrap intervals (PROTOCOL.md).

Reuses the resampling convention of research/r8_power_analysis/analysis.py by importing its
seed_for and block_counts functions. Reads only artifacts/r8_ten_comparators; writes only
artifacts/r8_coverage_panel. `--check` recomputes and compares with the saved outputs.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '2'
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
PAIRS = PROJECT / 'artifacts/r8_ten_comparators/pairs'
RESULTS = PROJECT / 'artifacts/r8_ten_comparators/results'
OUT = PROJECT / 'artifacts/r8_coverage_panel'
METHODS = ['Raw', 'Shift-CP', 'Rolling250', 'Rolling500', 'Selected-rolling',
           'Gate-selected-rolling', 'Loss-gate', 'Past-minimum']
BLOCKS = (20, 60)
LEVEL = .05

_spec = importlib.util.spec_from_file_location('power_analysis', PROJECT / 'research/r8_power_analysis/analysis.py')
_power = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_power)
seed_for, block_counts, MODELS, ASSETS, DRAWS = _power.seed_for, _power.block_counts, _power.MODELS, _power.ASSETS, _power.DRAWS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    """Common-calendar violation indicators per method, pair order MODELS x ASSETS."""
    frames = {m: [] for m in METHODS}
    pihat = {m: [] for m in METHODS}
    for model in MODELS:
        for asset in ASSETS:
            folder = PAIRS / f'{model}__{asset}'
            f = pd.read_parquet(folder / 'daily.parquet')
            metric = pd.read_csv(folder / 'metrics.csv').set_index('method')
            for m in METHODS:
                v = (f.r < f[m]).astype(float)
                assert abs(v.mean() - metric.loc[m, 'pihat']) < 1e-12, (model, asset, m)
                frames[m].append(pd.Series(v.to_numpy(), index=f.index))
                pihat[m].append(float(metric.loc[m, 'pihat']))
    first = frames[METHODS[0]]
    assert len(first) == 240
    dates = pd.date_range(min(f.index[0] for f in first), max(f.index[-1] for f in first), freq='D')
    T, P = len(dates), 240
    valid = np.zeros((T, P))
    values = {m: np.zeros((T, P)) for m in METHODS}
    for j in range(P):
        ix = dates.get_indexer(first[j].index)
        assert (ix >= 0).all()
        valid[ix, j] = 1
        for m in METHODS:
            assert frames[m][j].index.equals(first[j].index)
            values[m][ix, j] = frames[m][j].to_numpy()
    return dates, valid, values, {m: np.array(v) for m, v in pihat.items()}


def counts_table():
    pairs = pd.read_csv(RESULTS / 'pairs.csv')
    rows = {}
    for m in METHODS:
        q = pairs[pairs.method == m]
        assert len(q) == 240
        rows[m] = dict(kupiec_rejections=int((q.p_kup < LEVEL).sum()), kupiec_undefined=int(q.p_kup.isna().sum()),
                       independence_rejections=int((q.p_ind < LEVEL).sum()), independence_undefined=int(q.p_ind.isna().sum()),
                       joint_rejections=int((q.p_cc < LEVEL).sum()), joint_undefined=int(q.p_cc.isna().sum()),
                       pihat_mean_pairs_csv=float(q.pihat.mean()))
    return rows


def run():
    t0 = time.monotonic()
    dates, valid, values, pihat = load()
    T = len(dates)
    counts = {b: block_counts(T, b) for b in BLOCKS}
    backtests = counts_table()
    rows = []
    for m in METHODS:
        point = float(((valid * values[m]).sum(axis=0) / valid.sum(axis=0)).mean())
        assert abs(point - pihat[m].mean()) < 1e-12 and abs(point - backtests[m]['pihat_mean_pairs_csv']) < 1e-12
        for b in BLOCKS:
            denom = counts[b] @ valid
            assert (denom > 0).all()
            draws = ((counts[b] @ values[m]) / denom).mean(axis=1)
            rows.append(dict(method=m, block_calendar_days=b, draws=DRAWS, pairs=240,
                             violation_rate_pct=100 * point, bootstrap_sd_pct=100 * float(draws.std(ddof=1)),
                             lower_pct=100 * float(np.quantile(draws, .025)), upper_pct=100 * float(np.quantile(draws, .975)),
                             excludes_nominal=bool(np.quantile(draws, .975) < .01 or np.quantile(draws, .025) > .01),
                             **{k: v for k, v in backtests[m].items() if k != 'pihat_mean_pairs_csv'}))
    table = pd.DataFrame(rows)
    return dict(table=table, elapsed=time.monotonic() - t0, T=T,
                seeds={str(b): seed_for('panel-calendar', b) for b in BLOCKS})


def write(result):
    OUT.mkdir(parents=True, exist_ok=True)
    t = result['table']
    t.to_csv(OUT / 'coverage.csv', index=False)
    (OUT / 'run.json').write_text(json.dumps(dict(
        protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')), producer_sha256=sha(__file__),
        power_analysis_sha256=sha(PROJECT / 'research/r8_power_analysis/analysis.py'),
        seeds=result['seeds'], draws=DRAWS, calendar_days=result['T'], elapsed_seconds=result['elapsed'],
        python=sys.version, numpy=np.__version__, pandas=pd.__version__,
        inputs=dict(pairs_dir='artifacts/r8_ten_comparators/pairs', pairs_csv='artifacts/r8_ten_comparators/results/pairs.csv', n_pairs=240)), indent=2) + '\n')
    lines = ['# Panel-level violation rates: results', '',
             'Descriptive; protocol fixed before computation (`PROTOCOL.md`). Pair-equal mean violation',
             f'rate over 240 pairs, percentile 95% intervals from 999 common-calendar circular block draws',
             f'(seeds {result["seeds"]}, calendar of {result["T"]} days). Backtest rejections at 5% from `pairs.csv`.', '',
             '| method | rate % | 20-day 95% | 60-day 95% | 20-day excludes 1% | 60-day excludes 1% | Kupiec | Indep. | Joint | undefined indep. |', '|---|---|---|---|---|---|---|---|---|---|']
    for m in METHODS:
        a = t[(t.method == m) & (t.block_calendar_days == 20)].iloc[0]; b = t[(t.method == m) & (t.block_calendar_days == 60)].iloc[0]
        lines.append(f'| {m} | {a.violation_rate_pct:.4f} | [{a.lower_pct:.4f}, {a.upper_pct:.4f}] | [{b.lower_pct:.4f}, {b.upper_pct:.4f}] | {a.excludes_nominal} | {b.excludes_nominal} | {a.kupiec_rejections} | {a.independence_rejections} | {a.joint_rejections} | {a.independence_undefined} |')
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
    result = run()
    if a.check:
        saved = pd.read_csv(OUT / 'coverage.csv'); fresh = result['table']
        assert list(saved.columns) == list(fresh.columns)
        for c in saved.columns:
            if saved[c].dtype.kind in 'fi':
                assert np.allclose(saved[c].to_numpy(float), fresh[c].to_numpy(float), rtol=0, atol=1e-10), c
            else:
                assert (saved[c].astype(str).to_numpy() == fresh[c].astype(str).to_numpy()).all(), c
        print(f'CHECK PASSED: coverage.csv reproduced ({len(saved)} rows); elapsed {result["elapsed"]:.1f}s')
    else:
        write(result); print('written', OUT, f'elapsed {result["elapsed"]:.1f}s')


if __name__ == '__main__':
    main()
