"""Fixed direct-quantile base forecaster; all fits and outputs archived."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/irfa-gbm-mpl')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT/'artifacts/r8_referee_revision/gbm'
INPUT = PROJECT/'artifacts/r8_commodity_etp/panel/base/data/returns'
SUPPORT = PROJECT/'artifacts/r8_model_extension/full_preflight/support.csv'
CONFIG = dict(objective='quantile', alpha=.01, n_estimators=200,
    learning_rate=.05, num_leaves=15, max_depth=4, min_child_samples=20,
    min_child_weight=.001, max_bin=255, subsample=1., colsample_bytree=1.,
    subsample_freq=0, reg_alpha=0., reg_lambda=0., random_state=20260910,
    n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, data):
    p.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def features(y):
    x = pd.DataFrame({f'lag_{i}': y.shift(i) for i in range(1, 21)})
    for w in (5, 20, 60, 250):
        x[f'mean_{w}'] = y.shift(1).rolling(w).mean()
        x[f'std_{w}'] = y.shift(1).rolling(w).std(ddof=1)
    return x


def run(out):
    assert lgb.__version__ == '4.6.0', lgb.__version__
    assert not (out/'complete.json').exists(), 'Completed run is immutable'
    out.mkdir(parents=True, exist_ok=True)
    bindings = {str(p.relative_to(PROJECT)): sha(p) for p in
                [Path(__file__), Path(__file__).with_name('GBM_PROTOCOL.md'), SUPPORT]}
    support = pd.read_csv(SUPPORT).set_index('asset')
    fits, total = [], 0
    for asset, row in support.iterrows():
        path = INPUT/f'{asset}.csv'
        assert sha(path) == row.input_sha256
        bindings[str(path.relative_to(PROJECT))] = sha(path)
        y = pd.read_csv(path, index_col='date', parse_dates=True).log_return
        x = features(y); dates = y.index[512:]
        assert x.iloc[250:].notna().all().all()
        starts = np.r_[512, np.flatnonzero(np.diff(y.index.year) != 0)+1]
        starts = np.unique(starts[starts >= 512])
        stops = np.r_[starts[1:], len(y)]
        q = np.full(len(y)-512, np.nan)
        folder = out/asset; folder.mkdir(exist_ok=True)
        for start, stop in zip(starts, stops):
            first = max(250, int(start)-1250)
            model = lgb.LGBMRegressor(**CONFIG)
            model.fit(x.iloc[first:start], y.iloc[first:start])
            pred = model.predict(x.iloc[start:stop])
            assert np.isfinite(pred).all()
            q[start-512:stop-512] = pred
            tag = str(y.index[start].date())
            model.booster_.save_model(str(folder/f'{tag}.txt'))
            dump(folder/f'{tag}.json', model.booster_.dump_model())
            fits.append(dict(asset=asset, tag=tag, train_start=first,
                train_stop=int(start), predict_stop=int(stop), train_rows=int(start)-first,
                train_first=str(y.index[first].date()), train_last=str(y.index[start-1].date()),
                predict_first=tag, predict_last=str(y.index[stop-1].date()),
                trees=model.booster_.num_trees()))
        assert np.isfinite(q).all()
        nc = int(.7*len(q))
        assert nc == row.n_cal and len(q)-nc == row.n_test
        pd.DataFrame({'r': y.iloc[512:].to_numpy(), 'q': q}, index=dates).to_parquet(folder/'forecast.parquet')
        total += len(q)
        print(asset, len(q), 'forecasts;', len(starts), 'fits', flush=True)
    pd.DataFrame(fits).to_csv(out/'fits.csv', index=False)
    dump(out/'configuration.json', dict(config=CONFIG, feature_names=list(x.columns),
        lightgbm=lgb.__version__, numpy=np.__version__, pandas=pd.__version__,
        warmup=512, train_window=1250, annual_refit=True, alpha=.01))
    for p, h in bindings.items():
        assert sha(PROJECT/p) == h
    outputs = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    dump(out/'complete.json', dict(status='complete', bindings=bindings, outputs=outputs,
        assets=len(support), forecasts=total, fits=len(fits)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'run')
    run(parser.parse_args().output)
