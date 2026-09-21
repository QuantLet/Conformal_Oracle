"""Calendar facts of the evaluated series and lag-1 autocorrelation of external returns (PROTOCOL.md)."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
PROJECT = Path(__file__).resolve().parents[2]
RET = PROJECT / 'artifacts/r8_commodity_etp/panel/base/data/returns'
PAIRS = PROJECT / 'artifacts/r8_ten_comparators/pairs'
DECOMP = PROJECT / 'artifacts/r8_power_analysis/decomposition.csv'
EXT2 = PROJECT / 'artifacts/r8_external2/devexus/pairs'; EXT1 = PROJECT / 'artifacts/r8_external/july2026/pairs'
OUT = PROJECT / 'artifacts/r8_calendar'
SERIES = ['BTC', 'ETH', 'SP500', 'GDAXI']
DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def acf1(x):
    x = np.asarray(x, float); x = x - x.mean(); return float((x[1:] * x[:-1]).sum() / (x * x).sum())

def run():
    cal = []
    for s in SERIES:
        r = pd.read_csv(RET / f'{s}.csv', index_col=0, parse_dates=True).iloc[:, 0]
        wd = r.index.dayofweek; gaps = np.diff(r.index.values).astype('timedelta64[D]').astype(int)
        row = {'series': s, 'observations': len(r), 'first': str(r.index[0].date()), 'last': str(r.index[-1].date())}
        for i, d in enumerate(DAYS): row[d] = int((wd == i).sum())
        row['weekend_observations'] = int(((wd == 5) | (wd == 6)).sum())
        row['gaps_1_day'] = int((gaps == 1).sum()); row['gaps_3_day'] = int((gaps == 3).sum()); row['gaps_over_3_days'] = int((gaps > 3).sum())
        row['gaps_total'] = int(len(gaps)); cal.append(row)
    cal = pd.DataFrame(cal)
    d = pd.read_csv(DECOMP).set_index('item')['value']
    dates = [pd.Timestamp(x) for x in d['top10_dates_abs_date_sum'].split(';')]
    frames = {f.name: pd.read_parquet(f / 'daily.parquet').index for f in sorted(PAIRS.iterdir()) if (f / 'daily.parquet').exists()}
    assert len(frames) == 240
    counts = pd.Series(0, index=pd.date_range(min(i[0] for i in frames.values()), max(i[-1] for i in frames.values())))
    for ix in frames.values(): counts.loc[ix] += 1
    stress = pd.DataFrame([{'date': str(t.date()), 'weekday': DAYS[t.dayofweek], 'pairs_present': int(counts.loc[t]),
                            'crypto_only': bool(counts.loc[t] <= 20 and t.dayofweek >= 5)} for t in dates])
    present = counts[counts > 0]
    ext = []
    for label, root in [('devexus_25', EXT2), ('french_12', EXT1)]:
        for f in sorted(root.glob('HS__*')):
            r = pd.read_parquet(f / 'daily.parquet')
            wd = r.index.dayofweek
            ext.append({'universe': label, 'portfolio': f.name.split('__')[1], 'n': len(r), 'first': str(r.index[0].date()),
                        'last': str(r.index[-1].date()), 'lag1_autocorrelation': acf1(r.r), 'weekend_observations': int(((wd == 5) | (wd == 6)).sum()),
                        'weekday_count_min': int(min((wd == i).sum() for i in range(5))), 'weekday_count_max': int(max((wd == i).sum() for i in range(5)))})
    ext = pd.DataFrame(ext)
    assert (ext.universe == 'devexus_25').sum() == 25 and (ext.universe == 'french_12').sum() == 12
    summary = {'min_pairs_present': int(present.min()), 'max_pairs_present': int(present.max()), 'dates_with_pairs': int(len(present))}
    return cal, stress, ext, summary

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
    cal, stress, ext, summary = run()
    files = {'calendar.csv': cal, 'stress_dates.csv': stress, 'external_acf.csv': ext}
    if a.check:
        for name, df in files.items():
            saved = pd.read_csv(OUT / name); assert list(saved.columns) == list(df.columns), name
            for c in df.columns:
                if df[c].dtype.kind in 'fi': assert np.allclose(saved[c].to_numpy(float), df[c].to_numpy(float), atol=1e-12), (name, c)
                else: assert saved[c].astype(str).tolist() == df[c].astype(str).tolist(), (name, c)
        assert json.loads((OUT / 'run.json').read_text())['summary'] == summary
        print('CHECK PASSED: calendar, stress-date and autocorrelation outputs reproduced'); return
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in files.items(): df.to_csv(OUT / name, index=False)
    (OUT / 'run.json').write_text(json.dumps({'protocol_sha256': sha(Path(__file__).with_name('PROTOCOL.md')), 'producer_sha256': sha(__file__),
        'inputs': {f'returns/{s}.csv': sha(RET / f'{s}.csv') for s in SERIES} | {'decomposition.csv': sha(DECOMP)}, 'summary': summary, 'python': sys.version}, indent=2) + '\n')
    e = ext.groupby('universe').lag1_autocorrelation.agg(['mean', 'min', 'max'])
    (OUT / 'RESULTS.md').write_text('# Calendar facts and external autocorrelation (measured)\n\n' + cal.to_string() + '\n\n' + stress.to_string() + f"\n\n{summary}\n\n" + e.to_string() + '\n')
    print((OUT / 'RESULTS.md').read_text())

if __name__ == '__main__': main()
