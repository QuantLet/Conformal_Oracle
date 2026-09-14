"""Traffic-light zones under the scaled rule and the 1996 binomial rule (PROTOCOL.md).

Reads artifacts/r8_ten_comparators/results/pairs.csv only. --check recomputes and compares.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
PROJECT = Path(__file__).resolve().parents[2]
PAIRS = PROJECT / 'artifacts/r8_ten_comparators/results/pairs.csv'
OUT = PROJECT / 'artifacts/r8_basel_binomial'
P = .01; YELLOW_Q = .95; RED_Q = .9999

def binom_cdf(n, p=P):
    """Cumulative probabilities P(X<=x), x=0..n, from the log probability mass (Fraction-free, double precision)."""
    from math import lgamma, log, exp
    logpmf = np.array([lgamma(n + 1) - lgamma(x + 1) - lgamma(n - x + 1) + x * log(p) + (n - x) * log(1 - p) for x in range(n + 1)])
    return np.cumsum(np.exp(logpmf))

def starts(n):
    cdf = binom_cdf(n)
    return int(np.argmax(cdf >= YELLOW_Q)), int(np.argmax(cdf >= RED_Q))

def scaled_zone(v, n):
    s = 250 * v / n
    return 'Green' if s <= 4 else ('Yellow' if s <= 9 else 'Red')

def binomial_zone(v, y, r):
    return 'Green' if v < y else ('Yellow' if v < r else 'Red')

def run():
    p = pd.read_csv(PAIRS)
    assert starts(250) == (5, 10), starts(250)
    with_zone = p[p.TL.notna()]
    rep = [scaled_zone(v, n) for v, n in zip(with_zone.viol, with_zone.n_test)]
    assert rep == with_zone.TL.tolist(), 'scaled rule does not reproduce the stored TL column'
    table = {}
    for n in sorted(p.n_test.unique()):
        y, r = starts(int(n)); table[int(n)] = (y, r)
    ceilings = pd.DataFrame([{'n_test': n, 'yellow_from': y, 'red_from': r,
                              'green_ceiling_rate': (y - 1) / n, 'red_boundary_rate': r / n,
                              'scaled_green_ceiling_rate': 4 / 250, 'scaled_red_boundary_rate': 9 / 250,
                              'assets': ';'.join(sorted(p[p.n_test == n].asset.unique()))}
                             for n, (y, r) in table.items()])
    z = p[['model', 'asset', 'method', 'n_test', 'viol', 'pihat', 'TL']].copy()
    z['scaled_zone'] = [scaled_zone(v, n) if isinstance(t, str) else np.nan for v, n, t in zip(z.viol, z.n_test, z.TL)]
    z['binomial_zone'] = [binomial_zone(v, *table[int(n)]) if isinstance(t, str) else np.nan for v, n, t in zip(z.viol, z.n_test, z.TL)]
    rows = []
    for method, g in z[z.TL.notna()].groupby('method', sort=False):
        row = {'method': method, 'pairs': len(g)}
        for rule in ['scaled', 'binomial']:
            for zone in ['Green', 'Yellow', 'Red']:
                row[f'{rule}_{zone}'] = int((g[f'{rule}_zone'] == zone).sum())
        row['moved_down'] = int(((g.scaled_zone == 'Green') & (g.binomial_zone != 'Green')).sum() + ((g.scaled_zone == 'Yellow') & (g.binomial_zone == 'Red')).sum())
        rows.append(row)
    counts = pd.DataFrame(rows)
    return z, counts, ceilings

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
    z, counts, ceilings = run()
    files = {'zones.csv': z, 'zone_counts.csv': counts, 'ceilings.csv': ceilings}
    if a.check:
        for name, df in files.items():
            saved = pd.read_csv(OUT / name)
            assert list(saved.columns) == list(df.columns), name
            for c in df.columns:
                if df[c].dtype.kind in 'fi':
                    assert np.allclose(saved[c].to_numpy(float), df[c].to_numpy(float), atol=1e-12, equal_nan=True), (name, c)
                else:
                    assert saved[c].fillna('').astype(str).tolist() == df[c].fillna('').astype(str).tolist(), (name, c)
        print('CHECK PASSED: zones, counts and ceilings reproduced'); return
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in files.items(): df.to_csv(OUT / name, index=False)
    raw = counts.set_index('method').loc['Raw']; st = counts.set_index('method').loc['Shift-CP']
    (OUT / 'run.json').write_text(json.dumps({'protocol_sha256': sha(Path(__file__).with_name('PROTOCOL.md')),
        'producer_sha256': sha(__file__), 'inputs': {'pairs.csv': sha(PAIRS)}, 'anchor_250': starts(250),
        'yellow_quantile': YELLOW_Q, 'red_quantile': RED_Q, 'coverage': 1 - P, 'python': sys.version}, indent=2) + '\n')
    (OUT / 'RESULTS.md').write_text(f"""# Traffic-light zones: scaled rule versus 1996 binomial rule

Measured on the 240 pairs of the main panel. The scaled rule reproduces the stored `TL` column.
Anchor at 250 observations: yellow from {starts(250)[0]}, red from {starts(250)[1]}.

| Method | Scaled G/Y/R | Binomial G/Y/R |
|---|---|---|
| Raw | {raw.scaled_Green}/{raw.scaled_Yellow}/{raw.scaled_Red} | {raw.binomial_Green}/{raw.binomial_Yellow}/{raw.binomial_Red} |
| Shift-CP | {st.scaled_Green}/{st.scaled_Yellow}/{st.scaled_Red} | {st.binomial_Green}/{st.binomial_Yellow}/{st.binomial_Red} |

Green ceiling (violation rate) under the binomial rule across the {len(ceilings)} window lengths:
{100*ceilings.green_ceiling_rate.min():.2f}% to {100*ceilings.green_ceiling_rate.max():.2f}%; scaled rule: {100*4/250:.2f}%.
""")
    print(counts.to_string()); print('written', OUT)

if __name__ == '__main__': main()
