"""97.5% ES and the multiplied capital component on the ten-model support (PROTOCOL.md)."""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ[_k] = '2'
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm, t as student
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'source/scripts/extension_20260831')); sys.path.insert(0, str(PROJECT / 'research/r8_es_fz0'))
from panel_statistics import qshift
import analysis as es1  # the 1% ES study: same roots, loaders and guards
BASE, SUPPORT, NATIVE, EXT, FUND, FUNDS = es1.BASE, es1.SUPPORT, es1.NATIVE, es1.EXT, es1.FUND, es1.FUNDS
ZONES = PROJECT / 'artifacts/r8_basel_binomial/zones.csv'
OUT = PROJECT / 'artifacts/r9_es975'
ALPHA = .025; WARMUP = 512; FRACTION = .70; WINDOW = 250; DRAWS = 1000
MODELS = es1.MODELS; MULT = {'Green': 1.5, 'Yellow': 1.70, 'Red': 2.0}; MULT_HI = {'Green': 1.5, 'Yellow': 1.92, 'Red': 2.0}
Z = norm.ppf(ALPHA); NORMAL_ES = -norm.pdf(Z) / ALPHA
sha = es1.sha

def student_es(nu):
    s = np.sqrt((nu - 2) / nu); ta = student.ppf(ALPHA, nu)
    return s * (-student.pdf(ta, nu) * (nu + ta ** 2) / ((nu - 1) * ALPHA)), s * ta

def run():
    t0 = time.monotonic(); support = pd.read_csv(SUPPORT).set_index('asset')
    calib = pd.read_csv(NATIVE / 'calibration.csv').set_index(['model', 'asset']); zones = pd.read_csv(ZONES).set_index(['model', 'asset', 'method'])
    binding = {str(p.relative_to(PROJECT)): sha(p) for p in [SUPPORT, NATIVE / 'calibration.csv', ZONES]}
    rows = []
    for asset, row in support.iterrows():
        root = FUND if asset in FUNDS else EXT
        rp = BASE / 'data/returns' / f'{asset}.csv'; assert sha(rp) == row.input_sha256; binding[str(rp.relative_to(PROJECT))] = row.input_sha256
        ret = pd.read_csv(rp, index_col='date', parse_dates=True).log_return; full = ret.to_numpy()
        dates = ret.index[WARMUP:]; y = full[WARMUP:]; nc = int(FRACTION * len(y)); assert nc == row.n_cal and len(y) - nc == row.n_test
        for model, (folder, suffix) in MODELS.items():
            fp = BASE / 'data' / folder / (f'{asset}_{suffix}.parquet' if suffix else f'{asset}.parquet'); binding[str(fp.relative_to(PROJECT))] = sha(fp)
            pred = pd.read_parquet(fp).loc[dates]; v1 = pred['VaR_0.01'].to_numpy(dtype=float); v = pred['VaR_0.025'].to_numpy(dtype=float)
            mu = pred['mean'].to_numpy(dtype=float); sd = pred['std'].to_numpy(dtype=float)
            if folder != 'benchmarks':
                draws = es1.load_draws(root, folder, asset, dates, binding)
                assert np.allclose(np.percentile(draws, 100 * ALPHA, axis=1), v, rtol=1e-6, atol=1e-12), (model, asset, 'VaR guard')
                e = np.sort(draws, axis=1)[:, :int(np.floor(DRAWS * ALPHA))].mean(axis=1); law = 'draws'
            elif suffix == 'hs':
                windows = np.lib.stride_tricks.sliding_window_view(full, WINDOW)[WARMUP - WINDOW:len(full) - WINDOW]
                assert np.allclose(np.percentile(windows, 100 * ALPHA, axis=1), v, rtol=1e-9, atol=1e-15), (model, asset, 'VaR guard')
                e = np.sort(windows, axis=1)[:, :int(np.floor(WINDOW * ALPHA))].mean(axis=1); law = 'window'
            elif suffix == 'gjr_t':
                pp = root / 'parameters/gjr_t' / f'{asset}.parquet'; binding[str(pp.relative_to(PROJECT))] = sha(pp)
                nu = pd.read_parquet(pp).loc[dates, 'nu_used'].to_numpy(dtype=float); fin = np.isfinite(nu)
                es_t, q_t = np.full(len(nu), NORMAL_ES), np.full(len(nu), Z); es_t[fin], q_t[fin] = student_es(nu[fin])
                assert np.allclose(mu + sd * q_t, v, rtol=1e-9, atol=1e-15), (model, asset, 'VaR guard'); e = mu + sd * es_t; law = 'student'
            else:
                assert np.allclose(mu + sd * Z, v, rtol=1e-9, atol=1e-15), (model, asset, 'VaR guard'); e = mu + sd * NORMAL_ES; law = 'normal'
            assert np.isfinite(e).all() and (e <= v).all()
            shift = qshift(v1[:nc] - y[:nc]); assert abs(shift - calib.loc[(model, asset), 'qV']) < 1e-12
            for method, zmethod, vv, ee in [('Raw', 'Raw', v1[nc:], e[nc:]), ('Static', 'Shift-CP', v1[nc:] - shift, e[nc:] - shift)]:
                zone = zones.loc[(model, asset, zmethod), 'binomial_zone']; mean_es = float(-ee.mean())
                rows.append(dict(model=model, asset=asset, method=method, law=law, n_test=len(vv), qV=shift, VaR1_mean=float(-vv.mean()), ES975_mean=mean_es,
                                 zone=zone, multiplier=MULT[zone], component=MULT[zone] * mean_es, multiplier_hi=MULT_HI[zone], component_hi=MULT_HI[zone] * mean_es))
    table = pd.DataFrame(rows); assert len(table) == 7 * 24 * 2
    piv = table.pivot_table(index=['model', 'asset'], columns='method', values=['ES975_mean', 'component', 'component_hi', 'VaR1_mean'])
    summary = []
    for model, g in table.groupby('model', sort=False):
        r, s = g[g.method == 'Raw'].set_index('asset'), g[g.method == 'Static'].set_index('asset')
        summary.append(dict(model=model, law=r.law.iloc[0], assets=len(r), VaR1_raw=r.VaR1_mean.mean(), VaR1_static=s.VaR1_mean.mean(),
                            ES975_raw=r.ES975_mean.mean(), ES975_static=s.ES975_mean.mean(), ES_ratio=(s.ES975_mean / r.ES975_mean).mean(), VaR_ratio=(s.VaR1_mean / r.VaR1_mean).mean(),
                            component_raw=r.component.mean(), component_static=s.component.mean(), component_ratio=(s.component / r.component).mean(),
                            component_raw_hi=r.component_hi.mean(), component_static_hi=s.component_hi.mean(), component_ratio_hi=(s.component_hi / r.component_hi).mean(),
                            pairs_component_falls=int((s.component < r.component).sum()), pairs_component_falls_hi=int((s.component_hi < r.component_hi).sum()),
                            zones_raw=r.zone.value_counts().to_dict(), zones_static=s.zone.value_counts().to_dict()))
    summary = pd.DataFrame(summary)
    r, s = table[table.method == 'Raw'], table[table.method == 'Static']
    meta = dict(pairs=168, assets=24, alpha=ALPHA, es_draws=int(np.floor(DRAWS * ALPHA)), es_window_points=int(np.floor(WINDOW * ALPHA)),
                ES975_raw_pair_equal=float(r.ES975_mean.mean()), ES975_static_pair_equal=float(s.ES975_mean.mean()), ES_ratio_pair_equal=float((s.ES975_mean.to_numpy() / r.ES975_mean.to_numpy()).mean()),
                VaR1_ratio_pair_equal=float((s.VaR1_mean.to_numpy() / r.VaR1_mean.to_numpy()).mean()),
                component_raw=float(r.component.mean()), component_static=float(s.component.mean()), component_ratio=float(s.component.mean() / r.component.mean()),
                component_raw_hi=float(r.component_hi.mean()), component_static_hi=float(s.component_hi.mean()), component_ratio_hi=float(s.component_hi.mean() / r.component_hi.mean()),
                pairs_component_falls=int((s.component.to_numpy() < r.component.to_numpy()).sum()), pairs_component_falls_hi=int((s.component_hi.to_numpy() < r.component_hi.to_numpy()).sum()),
                zones_raw=r.zone.value_counts().to_dict(), zones_static=s.zone.value_counts().to_dict(), elapsed_seconds=time.monotonic() - t0)
    return table, summary, meta, binding

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
    table, summary, meta, binding = run(); files = {'metrics.csv': table, 'summary.csv': summary}
    if a.check:
        for name, df in files.items():
            saved = pd.read_csv(OUT / name); assert list(saved.columns) == list(df.columns), name
            for c in df.columns:
                if df[c].dtype.kind in 'fi': assert np.allclose(saved[c].to_numpy(float), df[c].to_numpy(float), atol=1e-10, equal_nan=True), (name, c)
                else: assert saved[c].astype(str).tolist() == df[c].astype(str).tolist(), (name, c)
        old = json.loads((OUT / 'run.json').read_text()); assert old['inputs'] == binding
        for k, v in meta.items():
            if k != 'elapsed_seconds': assert (abs(old['summary'][k] - v) < 1e-10) if isinstance(v, float) else old['summary'][k] == v, k
        print('CHECK PASSED: 97.5% ES, zones and capital components reproduced'); return
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in files.items(): df.to_csv(OUT / name, index=False)
    (OUT / 'run.json').write_text(json.dumps({'protocol_sha256': sha(Path(__file__).with_name('PROTOCOL.md')), 'producer_sha256': sha(__file__), 'es1_producer_sha256': sha(es1.__file__),
        'inputs': binding, 'summary': meta, 'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__}, indent=2) + '\n')
    show = summary.drop(columns=['zones_raw', 'zones_static'])
    (OUT / 'RESULTS.md').write_text('# 97.5% ES and capital component (measured)\n\n' + json.dumps(meta, indent=2) + '\n\n' + show.to_string(index=False) + '\n')
    print((OUT / 'RESULTS.md').read_text())

if __name__ == '__main__': main()
