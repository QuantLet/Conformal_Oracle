"""CAViaR, GAS and dedicated tail benchmarks on the ten-model support (PROTOCOL.md)."""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ[_k] = '2'
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'source/scripts/extension_20260831'))
from panel_statistics import scores, qshift
BASE = PROJECT / 'artifacts/r8_commodity_etp/panel/base'
SUPPORT = PROJECT / 'artifacts/r8_model_extension/full_preflight/support.csv'
NATIVE = PROJECT / 'artifacts/r8_model_extension/ten_common_evaluation'
REFERENCE = PROJECT / 'artifacts/extension_20260831/results/common_support.csv'
OUT = PROJECT / 'artifacts/r8_ten_benchmarks'
ALPHA = .01; WARMUP = 512; FRACTION = .70; WINDOW = 250
DYNAMIC = ['CAViaR-SAV', 'CAViaR-AS', 'GAS-t']; DEDICATED = {'EVT-POT': 'EVT_POT', 'FHS': 'FHS'}
METHODS = ['Raw', 'Static', 'Rolling250']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run():
    t0 = time.monotonic(); support = pd.read_csv(SUPPORT).set_index('asset'); binding = {}
    metrics, overlap = [], []
    for asset, row in support.iterrows():
        rp = BASE / 'data/returns' / f'{asset}.csv'; assert sha(rp) == row.input_sha256, asset
        binding[str(rp.relative_to(PROJECT))] = row.input_sha256
        ret = pd.read_csv(rp, index_col='date', parse_dates=True).log_return
        assert ret.index.is_unique and ret.index.is_monotonic_increasing
        dates = ret.index[WARMUP:]; y = ret.iloc[WARMUP:].to_numpy(); nc = int(FRACTION * len(y))
        assert nc == row.n_cal and len(y) - nc == row.n_test and str(dates[nc].date()) == row.first_test and str(dates[-1].date()) == row.last_date
        daily = pd.read_parquet(NATIVE / 'daily' / f'{asset}.parquet'); assert daily.index.equals(dates[nc:]), asset
        assert np.array_equal(daily.r.to_numpy(), y[nc:])
        fit_end = ret.index[int(FRACTION * len(ret)) - 1]
        rec = dict(asset=asset, n_returns=len(ret), n_eligible=len(y), n_cal=nc, n_test=len(y) - nc, first_test=str(dates[nc].date()),
                   dynamic_fit_last=str(fit_end.date()), calibration_dates_inside_fit=int((dates[:nc] <= fit_end).sum()),
                   test_after_fit=bool((dates[nc:] > fit_end).all()))
        for model in DYNAMIC:
            fp = BASE / 'data/dynamic' / f'{asset}_{model}.parquet'; binding[str(fp.relative_to(PROJECT))] = sha(fp)
            frame = pd.read_parquet(fp); assert frame.index.equals(ret.index), (asset, model)
            q = frame.loc[dates, 'VaR_0.01'].to_numpy(dtype=float); assert np.isfinite(q).all()
            residual = q - y; shift = qshift(residual[:nc])
            windows = np.lib.stride_tricks.sliding_window_view(residual, WINDOW)
            rolling = np.partition(windows[nc - WINDOW:len(y) - WINDOW], WINDOW - 2, axis=1)[:, WINDOW - 2]; assert len(rolling) == len(y) - nc
            cal = scores(y[:nc], q[:nc])
            for method, target in zip(METHODS, [q[nc:], q[nc:] - shift, q[nc:] - rolling]):
                metrics.append(dict(model=model, asset=asset, method=method, n_cal=nc, qV=shift,
                                    indicated=bool(cal['p_kup'] < .05 or cal['TL'] != 'Green'), **scores(y[nc:], target)))
        ep = BASE / 'evt_fhs' / f'{asset}.parquet'; binding[str(ep.relative_to(PROJECT))] = sha(ep)
        ev = pd.read_parquet(ep); assert ev.index[0] == ret.index[int(FRACTION * len(ret))] and dates[nc:].isin(ev.index).all(), asset
        assert np.array_equal(ev.loc[dates[nc:], 'r'].to_numpy(), y[nc:])
        rec['dedicated_pre_test_forecasts'] = int((ev.index < dates[nc]).sum())
        for model, col in DEDICATED.items():
            target = ev.loc[dates[nc:], col].to_numpy(dtype=float); assert np.isfinite(target).all()
            metrics.append(dict(model=model, asset=asset, method='Raw', n_cal=np.nan, qV=np.nan, indicated=np.nan, **scores(y[nc:], target)))
        overlap.append(rec)
    table = pd.DataFrame(metrics); overlap = pd.DataFrame(overlap)
    ref = pd.read_csv(REFERENCE); ref = ref[(ref.model == 'CAViaR-SAV') & (ref.method == 'Raw')].set_index('asset'); binding[str(REFERENCE.relative_to(PROJECT))] = sha(REFERENCE)
    shared = [a for a in overlap.asset if a in ref.index]
    overlap['reference_support_identical'] = [bool(a in ref.index and ref.loc[a, 'first'] == overlap.set_index('asset').loc[a, 'first_test'] and int(ref.loc[a, 'n_test']) == int(overlap.set_index('asset').loc[a, 'n_test'])) for a in overlap.asset]
    assert len(table) == 24 * (3 * 3 + 2) and table.groupby(['model', 'method']).n_test.sum().eq(int(support.n_test.sum())).all()
    summaries = []
    for (model, method), g in table.groupby(['model', 'method'], sort=False):
        summaries.append(dict(model=model, method=method, assets=len(g), observations=int(g.n_test.sum()), violations=int(g.viol.sum()),
            pi_mean=g.pihat.mean(), pi_pooled=g.viol.sum() / g.n_test.sum(), QS=g.QS.mean(), QS_x10000=g.QS.mean() * 1e4, width=g.width.mean(),
            kupiec_rejections=int((g.p_kup < .05).sum()), independence_rejections=int((g.p_ind < .05).sum()), independence_available=int(g.p_ind.notna().sum()),
            conditional_rejections=int((g.p_cc < .05).sum()), conditional_available=int(g.p_cc.notna().sum()),
            scaled_green=int((g.TL == 'Green').sum()), scaled_yellow=int((g.TL == 'Yellow').sum()), scaled_red=int((g.TL == 'Red').sum())))
    summary = pd.DataFrame(summaries)
    meta = dict(assets=24, test_observations=int(support.n_test.sum()), alpha=ALPHA, warmup=WARMUP, calibration_fraction=FRACTION, rolling_window=WINDOW,
                calibration_inside_fit_min=int(overlap.calibration_dates_inside_fit.min()), calibration_inside_fit_max=int(overlap.calibration_dates_inside_fit.max()),
                calibration_inside_fit_share_min=float((overlap.calibration_dates_inside_fit / overlap.n_cal).min()),
                calibration_inside_fit_share_max=float((overlap.calibration_dates_inside_fit / overlap.n_cal).max()),
                test_after_fit_all=bool(overlap.test_after_fit.all()), shared_series=len(shared), shared_support_identical=int(overlap.reference_support_identical.sum()),
                replaced_series=sorted(set(overlap.asset) - set(ref.index)), pre_test_min=int(overlap.dedicated_pre_test_forecasts.min()),
                pre_test_max=int(overlap.dedicated_pre_test_forecasts.max()), elapsed_seconds=time.monotonic() - t0)
    return table, summary, overlap, meta, binding

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
    table, summary, overlap, meta, binding = run()
    files = {'metrics.csv': table, 'summary.csv': summary, 'overlap.csv': overlap}
    if a.check:
        for name, df in files.items():
            saved = pd.read_csv(OUT / name); assert list(saved.columns) == list(df.columns), name
            for c in df.columns:
                if df[c].dtype.kind in 'fi': assert np.allclose(saved[c].to_numpy(float), df[c].to_numpy(float), atol=1e-12, equal_nan=True), (name, c)
                else: assert saved[c].astype(str).tolist() == df[c].astype(str).tolist(), (name, c)
        old = json.loads((OUT / 'run.json').read_text())
        assert old['inputs'] == binding, 'input digests changed'
        for k, v in meta.items():
            if k != 'elapsed_seconds': assert (abs(old['summary'][k] - v) < 1e-12) if isinstance(v, float) else old['summary'][k] == v, k
        print('CHECK PASSED: ten-support benchmark metrics, summary and overlap reproduced'); return
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in files.items(): df.to_csv(OUT / name, index=False)
    (OUT / 'run.json').write_text(json.dumps({'protocol_sha256': sha(Path(__file__).with_name('PROTOCOL.md')), 'producer_sha256': sha(__file__),
        'statistics_sha256': sha(PROJECT / 'source/scripts/extension_20260831/panel_statistics.py'), 'support_sha256': sha(SUPPORT),
        'inputs': binding, 'summary': meta, 'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__}, indent=2) + '\n')
    show = summary[['model', 'method', 'QS_x10000', 'pi_mean', 'width', 'kupiec_rejections', 'independence_rejections', 'conditional_rejections', 'scaled_green']]
    (OUT / 'RESULTS.md').write_text('# Benchmarks on the ten-model support (measured)\n\n' + json.dumps(meta, indent=2) + '\n\n' + show.to_string(index=False) + '\n\n' + overlap.to_string(index=False) + '\n')
    print((OUT / 'RESULTS.md').read_text())

if __name__ == '__main__': main()
