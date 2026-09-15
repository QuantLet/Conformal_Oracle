"""Expected Shortfall and the FZ0 joint score on the ten-model support (PROTOCOL.md)."""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ[_k] = '2'
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm, t as student
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'source/scripts/extension_20260831'))
from panel_statistics import scores, qshift
BASE = PROJECT / 'artifacts/r8_commodity_etp/panel/base'
SUPPORT = PROJECT / 'artifacts/r8_model_extension/full_preflight/support.csv'
NATIVE = PROJECT / 'artifacts/r8_model_extension/ten_common_evaluation'
EXT = PROJECT / 'artifacts/extension_20260831'; FUND = PROJECT / 'artifacts/r8_commodity_etp'; FUNDS = {'GLD', 'USO', 'UNG'}
OUT = PROJECT / 'artifacts/r8_es_fz0'
ALPHA = .01; WARMUP = 512; FRACTION = .70; WINDOW = 250; DRAWS = 1000; B = 999; CHUNK = 25; BLOCKS = (20, 60)
MODELS = {'Moirai-1.1': ('moirai', None), 'Lag-Llama': ('lagllama', None), 'GJR-GARCH': ('benchmarks', 'gjr_garch'),
          'GJR-GARCH-t': ('benchmarks', 'gjr_t'), 'GARCH-N': ('benchmarks', 'garch_n'), 'Hist-Sim': ('benchmarks', 'hs'), 'EWMA': ('benchmarks', 'ewma')}
Z = norm.ppf(ALPHA); NORMAL_ES = -norm.pdf(Z) / ALPHA  # ES of a standard Normal at alpha, negative

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed_for(key): return int.from_bytes(hashlib.sha256(f'20260915/es-fz0/{key}'.encode()).digest()[:4], 'little')

def fz0(y, v, e):
    return -(y <= v).astype(float) * (v - y) / (ALPHA * e) + v / e + np.log(-e) - 1

def student_es(nu):
    """ES at ALPHA of the unit-variance Student-t (arch convention); negative."""
    s = np.sqrt((nu - 2) / nu); ta = student.ppf(ALPHA, nu)
    return s * (-student.pdf(ta, nu) * (nu + ta ** 2) / ((nu - 1) * ALPHA)), s * ta

def load_draws(root, model, asset, dates, binding):
    """Native draws behind the stored VaR: the provenance record names the directory and array when the
    promoted forecast came from a different run (Lag-Llama with actual calendar timestamps)."""
    prov = root / 'provenance' / model / f'{asset}.json'; key, chunks = 'native', None
    if prov.exists():
        record = json.loads(prov.read_text()); binding[str(prov.relative_to(PROJECT))] = sha(prov)
        if 'native_directory' in record:
            folder = PROJECT / record['native_directory']; key = record['native_array']; chunks = [dict(file=f, sha256=h) for f, h in record['native_chunks'].items()]
    if chunks is None:
        folder = root / 'native' / model / asset; complete = json.loads((folder / 'complete.json').read_text())
        binding[str((folder / 'complete.json').relative_to(PROJECT))] = sha(folder / 'complete.json')
        if isinstance(complete['chunks'], int):  # August-vintage layout: one digest record beside each chunk
            chunks = [dict(file=f.name, sha256=json.loads(f.with_suffix('.json').read_text())['sha256']) for f in sorted(folder.glob('*.npz'))]
            assert len(chunks) == complete['chunks']
        else: chunks = complete['chunks']
        assert complete['rows'] == len(dates)
    blocks, seen, pos = [], [], []
    for chunk in chunks:
        f = folder / chunk['file']; digest = sha(f); assert digest == chunk['sha256'], f
        binding[str(f.relative_to(PROJECT))] = digest
        with np.load(f, allow_pickle=False) as z:
            blocks.append(z[key].astype(np.float64)); seen.extend(z['date'] if 'date' in z.files else z['dates']); pos.extend(z['positions'])
    draws = np.concatenate(blocks); assert draws.shape == (len(dates), DRAWS) and pd.DatetimeIndex(seen).equals(dates) and np.array_equal(pos, np.arange(WARMUP, WARMUP + len(dates)))
    return draws

def block_counts(T, block):
    rng = np.random.default_rng(seed_for(block)); counts = np.empty((B, T)); b = 0
    for start in range(0, B, CHUNK):
        for _ in range(min(CHUNK, B - start)):
            begins = rng.integers(0, T, size=int(np.ceil(T / block)))
            counts[b] = np.bincount(((begins[:, None] + np.arange(block)) % T).ravel()[:T], minlength=T); b += 1
    return counts

def run():
    t0 = time.monotonic(); support = pd.read_csv(SUPPORT).set_index('asset')
    calib = pd.read_csv(NATIVE / 'calibration.csv').set_index(['model', 'asset']); tenm = pd.read_csv(NATIVE / 'metrics.csv').set_index(['model', 'asset', 'method'])
    binding = {str(p.relative_to(PROJECT)): sha(p) for p in [SUPPORT, NATIVE / 'calibration.csv', NATIVE / 'metrics.csv']}
    metrics, undefined, frames = [], [], []
    for asset, row in support.iterrows():
        root = FUND if asset in FUNDS else EXT
        rp = BASE / 'data/returns' / f'{asset}.csv'; assert sha(rp) == row.input_sha256; binding[str(rp.relative_to(PROJECT))] = row.input_sha256
        ret = pd.read_csv(rp, index_col='date', parse_dates=True).log_return; full = ret.to_numpy()
        dates = ret.index[WARMUP:]; y = full[WARMUP:]; nc = int(FRACTION * len(y)); assert nc == row.n_cal and len(y) - nc == row.n_test
        daily = pd.read_parquet(NATIVE / 'daily' / f'{asset}.parquet'); assert daily.index.equals(dates[nc:])
        for model, (folder, suffix) in MODELS.items():
            fp = BASE / 'data' / folder / (f'{asset}_{suffix}.parquet' if suffix else f'{asset}.parquet'); binding[str(fp.relative_to(PROJECT))] = sha(fp)
            pred = pd.read_parquet(fp).loc[dates]; v = pred['VaR_0.01'].to_numpy(dtype=float); mu = pred['mean'].to_numpy(dtype=float); sd = pred['std'].to_numpy(dtype=float)
            if folder != 'benchmarks':
                draws = load_draws(root, folder, asset, dates, binding)
                assert np.allclose(np.percentile(draws, 100 * ALPHA, axis=1), v, rtol=1e-6, atol=1e-12), (model, asset, 'VaR guard')
                k = int(np.floor(DRAWS * ALPHA)); e = np.sort(draws, axis=1)[:, :k].mean(axis=1); law = 'draws'
            elif suffix == 'hs':
                windows = np.lib.stride_tricks.sliding_window_view(full, WINDOW)[WARMUP - WINDOW:len(full) - WINDOW]; assert len(windows) == len(dates)
                assert np.allclose(np.percentile(windows, 100 * ALPHA, axis=1), v, rtol=1e-9, atol=1e-15), (model, asset, 'VaR guard')
                k = int(np.floor(WINDOW * ALPHA)); e = np.sort(windows, axis=1)[:, :k].mean(axis=1); law = 'window'
            elif suffix == 'gjr_t':
                pp = root / 'parameters/gjr_t' / f'{asset}.parquet'; binding[str(pp.relative_to(PROJECT))] = sha(pp)
                nu = pd.read_parquet(pp).loc[dates, 'nu_used'].to_numpy(dtype=float); fin = np.isfinite(nu)
                es_t, q_t = np.full(len(nu), NORMAL_ES), np.full(len(nu), Z)
                es_t[fin], q_t[fin] = student_es(nu[fin])
                assert np.allclose(mu + sd * q_t, v, rtol=1e-9, atol=1e-15), (model, asset, 'VaR guard'); e = mu + sd * es_t; law = 'student'
            else:
                assert np.allclose(mu + sd * Z, v, rtol=1e-9, atol=1e-15), (model, asset, 'VaR guard'); e = mu + sd * NORMAL_ES; law = 'normal'
            assert np.isfinite(e).all() and (e <= v).all(), (model, asset)
            shift = qshift(v[:nc] - y[:nc]); assert abs(shift - calib.loc[(model, asset), 'qV']) < 1e-12
            yt = y[nc:]; vr, er = v[nc:], e[nc:]; vs, es = vr - shift, er - shift
            for method, (vv, ee) in {'Raw': (vr, er), 'Static': (vs, es)}.items():
                assert abs(scores(yt, vv)['QS'] - tenm.loc[(model, asset, method), 'QS']) < 1e-15
            ok = (er < 0) & (es < 0)
            undefined.append(dict(model=model, asset=asset, n_test=len(yt), es_nonneg_raw=int((er >= 0).sum()), es_nonneg_static=int((es >= 0).sum()),
                                  var_nonneg_raw=int((vr >= 0).sum()), var_nonneg_static=int((vs >= 0).sum()), scored_days=int(ok.sum())))
            lr, ls = fz0(yt[ok], vr[ok], er[ok]), fz0(yt[ok], vs[ok], es[ok])
            frames.append(pd.Series(ls - lr, index=dates[nc:][ok], name=f'{model}__{asset}'))
            for method, vv, ee, ll in [('Raw', vr, er, lr), ('Static', vs, es, ls)]:
                hit = yt <= vv
                metrics.append(dict(model=model, asset=asset, method=method, law=law, n_test=len(yt), scored_days=int(ok.sum()), qV=shift,
                                    FZ0=float(ll.mean()), QS=scores(yt, vv)['QS'], VaR_mean=float(vv.mean()), ES_mean=float(ee.mean()),
                                    violations=int(hit.sum()), pihat=float(hit.mean()), shortfall_gap=float((yt[hit] - ee[hit]).mean()) if hit.any() else np.nan,
                                    es_excess_days=int((yt < ee).sum())))
    table = pd.DataFrame(metrics); undefined = pd.DataFrame(undefined); assert len(table) == 7 * 24 * 2 and len(frames) == 168
    summary = []
    for (model, method), g in table.groupby(['model', 'method'], sort=False):
        summary.append(dict(model=model, method=method, law=g.law.iloc[0], assets=len(g), observations=int(g.n_test.sum()), scored_days=int(g.scored_days.sum()),
                            FZ0=g.FZ0.mean(), QS_x10000=g.QS.mean() * 1e4, VaR_mean=g.VaR_mean.mean(), ES_mean=g.ES_mean.mean(), pi_mean=g.pihat.mean(),
                            shortfall_gap=g.shortfall_gap.mean(), es_excess_days=int(g.es_excess_days.sum())))
    summary = pd.DataFrame(summary)
    # Bootstrap of the pair-equal static-minus-raw FZ0 contrast, per model and over all pairs.
    start = min(f.index[0] for f in frames); cal = pd.date_range(start, max(f.index[-1] for f in frames), freq='D'); T = len(cal)
    valid = np.zeros((T, 168)); values = np.zeros((T, 168))
    for j, f in enumerate(frames):
        ix = cal.get_indexer(f.index); assert (ix >= 0).all(); valid[ix, j] = 1; values[ix, j] = f.to_numpy()
    groups = {m: np.array([j for j, f in enumerate(frames) if f.name.startswith(m + '__')]) for m in MODELS}; groups['All'] = np.arange(168)
    pair_point = (valid * values).sum(axis=0) / valid.sum(axis=0); rows = []
    for block in BLOCKS:
        counts = block_counts(T, block); denom = counts @ valid; assert (denom > 0).all(); pair_draws = (counts @ values) / denom
        est = {m: pair_point[idx].mean() for m, idx in groups.items()}; draws = {m: pair_draws[:, idx].mean(axis=1) for m, idx in groups.items()}
        sd = {m: draws[m].std(ddof=1) for m in groups}; crit = float(np.quantile(np.max(np.abs(np.column_stack([(draws[m] - est[m]) / sd[m] for m in groups])), axis=1), .95))
        for m in groups:
            rows.append(dict(comparison=m, block_calendar_days=block, pairs=len(groups[m]), estimate=est[m], bootstrap_sd=sd[m],
                             pointwise_lower=float(np.quantile(draws[m], .025)), pointwise_upper=float(np.quantile(draws[m], .975)),
                             simultaneous_lower=est[m] - crit * sd[m], simultaneous_upper=est[m] + crit * sd[m], critical=crit, draws=B, seed=seed_for(block)))
    contrasts = pd.DataFrame(rows)
    meta = dict(models=list(MODELS), pairs=168, assets=24, test_observations=int(support.n_test.sum()), alpha=ALPHA, draws_per_date=DRAWS, es_draws=int(np.floor(DRAWS * ALPHA)),
                es_window_points=int(np.floor(WINDOW * ALPHA)), es_nonneg_days=int(undefined.es_nonneg_raw.sum() + undefined.es_nonneg_static.sum()),
                var_nonneg_raw_days=int(undefined.var_nonneg_raw.sum()), var_nonneg_static_days=int(undefined.var_nonneg_static.sum()),
                scored_days=int(undefined.scored_days.sum()), calendar_days=T, elapsed_seconds=time.monotonic() - t0)
    return table, summary, contrasts, undefined, meta, binding

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
    table, summary, contrasts, undefined, meta, binding = run()
    files = {'metrics.csv': table, 'summary.csv': summary, 'contrasts.csv': contrasts, 'undefined.csv': undefined}
    if a.check:
        for name, df in files.items():
            saved = pd.read_csv(OUT / name); assert list(saved.columns) == list(df.columns), name
            for c in df.columns:
                if df[c].dtype.kind in 'fi': assert np.allclose(saved[c].to_numpy(float), df[c].to_numpy(float), atol=1e-10, equal_nan=True), (name, c)
                else: assert saved[c].astype(str).tolist() == df[c].astype(str).tolist(), (name, c)
        old = json.loads((OUT / 'run.json').read_text()); assert old['inputs'] == binding, 'input digests changed'
        for k, v in meta.items():
            if k != 'elapsed_seconds': assert (abs(old['summary'][k] - v) < 1e-10) if isinstance(v, float) else old['summary'][k] == v, k
        print('CHECK PASSED: ES forecasts, FZ0 metrics, summary and contrasts reproduced'); return
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in files.items(): df.to_csv(OUT / name, index=False)
    (OUT / 'run.json').write_text(json.dumps({'protocol_sha256': sha(Path(__file__).with_name('PROTOCOL.md')), 'producer_sha256': sha(__file__),
        'statistics_sha256': sha(PROJECT / 'source/scripts/extension_20260831/panel_statistics.py'), 'inputs': binding, 'summary': meta,
        'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__}, indent=2) + '\n')
    show = summary[['model', 'method', 'FZ0', 'QS_x10000', 'VaR_mean', 'ES_mean', 'pi_mean', 'shortfall_gap', 'es_excess_days', 'scored_days']]
    (OUT / 'RESULTS.md').write_text('# ES and FZ0 on the ten-model support (measured)\n\n' + json.dumps(meta, indent=2) + '\n\n' + show.to_string(index=False) + '\n\n' + contrasts.to_string(index=False) + '\n')
    print((OUT / 'RESULTS.md').read_text())

if __name__ == '__main__': main()
