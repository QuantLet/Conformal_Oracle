"""Read-only replay of current EWMA forecasts; no refitting or new inference."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    producer = ROOT / 'source/scripts/extension_20260831/classical.py'
    rows = []
    alphas = [.01, .025, .05, .1]
    for archive in [ROOT / 'artifacts/extension_20260831', ROOT / 'artifacts/r8_commodity_etp']:
        for meta in sorted((archive / 'provenance/ewma').glob('*.json')):
            asset = meta.stem
            if archive.name == 'extension_20260831' and asset in ['WTI', 'GOLD', 'NATGAS']:
                continue
            provenance = json.loads(meta.read_text())
            returns = archive / 'data/returns' / (asset + '.csv')
            forecasts = archive / 'data/benchmarks' / (asset + '_ewma.parquet')
            assert provenance['binding']['producer_sha256'] == sha(producer)
            assert provenance['binding']['input_sha256'] == sha(returns)
            assert provenance['forecast_sha256'] == sha(forecasts)
            r = pd.read_csv(returns, index_col='date', parse_dates=True).log_return
            q = pd.read_parquet(forecasts)
            values = r.to_numpy()
            variance = np.empty(len(values))
            variance[0] = values[:250].var(ddof=1)
            for t in range(1, len(values)):
                variance[t] = .94 * variance[t - 1] + .06 * values[t - 1] ** 2
            expected = np.sqrt(variance[250:, None]) * norm.ppf(alphas)
            actual = q[[f'VaR_{a:g}' for a in alphas]].to_numpy()
            assert q.index.equals(r.index[250:])
            assert np.array_equal(expected, actual), asset
            rows.append(dict(asset=asset, dates=len(q), max_abs_difference=0.,
                             inputs={str(p.relative_to(ROOT)): sha(p) for p in [meta, returns, forecasts]}))
    assert len(rows) == len({r['asset'] for r in rows}) == 24
    return dict(status='passed', producer=str(producer.relative_to(ROOT)), producer_sha256=sha(producer),
                audit_producer_sha256=sha(Path(__file__)), arrays_bitwise_equal=True,
                scope='Current 24-asset EWMA forecasts, excluding replaced futures quotations.',
                daily_forecast_rows=sum(r['dates'] for r in rows), rows=rows,
                new_inference_or_fitting=False)


if __name__ == '__main__':
    report = check()
    path = ROOT / 'artifacts/r8_team_review/ewma_current_replay.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS:', len(report['rows']), 'assets;', report['daily_forecast_rows'], 'daily forecasts')
