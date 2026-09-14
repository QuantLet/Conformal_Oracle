"""VaR level, width ratio and breach severity of the 240-pair panel (PROTOCOL.md).

Reads artifacts/r8_ten_comparators/pairs/*/daily.parquet only. --check recomputes and compares.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
PROJECT = Path(__file__).resolve().parents[2]
PAIRS = PROJECT / 'artifacts/r8_ten_comparators/pairs'
SUMMARY = PROJECT / 'artifacts/r8_model_extension/ten_common_evaluation/summary.csv'
OUT = PROJECT / 'artifacts/r8_economics'
MODELS = ['PatchTST-FM', 'Chronos-2', 'TS-ICL', 'Moirai-1.1', 'Lag-Llama', 'GJR-GARCH', 'GJR-GARCH-t', 'GARCH-N', 'Hist-Sim', 'EWMA']
METHODS = ['Raw', 'Shift-CP', 'Shift-ERM', 'Vol-CP', 'Vol-ERM', 'State2-ERM', 'State4-ERM', 'State-L1', 'POT-Shift', 'POT-Vol',
           'Rolling250', 'Rolling500', 'Selected-rolling', 'Gate-selected-rolling', 'Loss-gate', 'Past-minimum']
SEV = ['Raw', 'Shift-CP', 'Rolling250']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run():
    level_rows, ratio_rows, sev_rows, year_rows = [], [], [], []
    sums = {m: [0., 0] for m in METHODS}
    sev_pool = {m: [0., 0., 0] for m in SEV}
    year_pool = {}
    hashes = {}
    for model in MODELS:
        for folder in sorted(PAIRS.glob(f'{model}__*')):
            asset = folder.name.split('__')[1]
            f = pd.read_parquet(folder / 'daily.parquet'); hashes[folder.name] = sha(folder / 'daily.parquet')
            r = f.r.to_numpy()
            raw_level = float((-f['Raw']).mean())
            for m in METHODS:
                q = f[m].to_numpy(); lvl = float((-q).mean())
                sums[m][0] += float((-q).sum()); sums[m][1] += len(q)
                level_rows.append({'model': model, 'asset': asset, 'method': m, 'n_test': len(q), 'mean_var_pct': 100 * lvl,
                                   'ratio_to_raw': lvl / raw_level})
            for m in SEV:
                q = f[m].to_numpy(); hit = r < q
                sev_rows.append({'model': model, 'asset': asset, 'method': m, 'breaches': int(hit.sum()),
                                 'mean_exceedance_pct': 100 * float((q[hit] - r[hit]).mean()) if hit.any() else np.nan,
                                 'mean_loss_on_breach_pct': 100 * float((-r[hit]).mean()) if hit.any() else np.nan})
                sev_pool[m][0] += float((q[hit] - r[hit]).sum()); sev_pool[m][1] += float((-r[hit]).sum()); sev_pool[m][2] += int(hit.sum())
            for year, g in f.groupby(f.index.year):
                y = year_pool.setdefault(int(year), [0., 0., 0])
                y[0] += float((-g['Raw']).sum()); y[1] += float((-g['Shift-CP']).sum()); y[2] += len(g)
    levels = pd.DataFrame(level_rows)
    widths = []
    for m in METHODS:
        g = levels[levels.method == m]
        widths.append({'method': m, 'pairs': len(g), 'pair_equal_mean_var_pct': g.mean_var_pct.mean(),
                       'pooled_mean_var_pct': 100 * sums[m][0] / sums[m][1], 'pair_days': sums[m][1],
                       'ratio_mean': g.ratio_to_raw.mean(), 'ratio_median': g.ratio_to_raw.median(),
                       'ratio_min': g.ratio_to_raw.min(), 'ratio_max': g.ratio_to_raw.max(),
                       'pairs_wider_than_raw': int((g.ratio_to_raw > 1).sum()),
                       'pair_equal_pct_change_vs_raw': 100 * (g.mean_var_pct.mean() / levels[levels.method == 'Raw'].mean_var_pct.mean() - 1),
                       'pooled_pct_change_vs_raw': 100 * ((sums[m][0] / sums[m][1]) / (sums['Raw'][0] / sums['Raw'][1]) - 1)})
    widths = pd.DataFrame(widths)
    sev = pd.DataFrame(sev_rows)
    severity = pd.DataFrame([{'method': m, 'breaches': sev_pool[m][2],
                              'pooled_mean_exceedance_pct': 100 * sev_pool[m][0] / sev_pool[m][2],
                              'pooled_mean_loss_on_breach_pct': 100 * sev_pool[m][1] / sev_pool[m][2],
                              'pair_equal_mean_exceedance_pct': sev[sev.method == m].mean_exceedance_pct.mean(),
                              'pair_equal_mean_loss_on_breach_pct': sev[sev.method == m].mean_loss_on_breach_pct.mean(),
                              'pairs_with_breaches': int((sev[sev.method == m].breaches > 0).sum())} for m in SEV])
    by_year = pd.DataFrame([{'year': y, 'pair_days': v[2], 'raw_mean_var_pct': 100 * v[0] / v[2], 'static_mean_var_pct': 100 * v[1] / v[2],
                             'pct_change': 100 * (v[1] / v[0] - 1)} for y, v in sorted(year_pool.items())])
    # Cross-check against the ten-model summary width column (mean over models of per-model mean over 24 assets).
    s = pd.read_csv(SUMMARY)
    for m, key in [('Raw', 'Raw'), ('Shift-CP', 'Static'), ('Rolling250', 'Rolling250')]:
        ref = float(s[s.method == key].width.mean()) * 100
        mine = float(widths.set_index('method').loc[m, 'pair_equal_mean_var_pct'])
        assert abs(ref - mine) < 1e-9, (m, ref, mine)
    return widths, levels, severity, by_year, hashes

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
    widths, levels, severity, by_year, hashes = run()
    files = {'widths.csv': widths, 'pair_ratios.csv': levels, 'severity.csv': severity, 'by_year.csv': by_year}
    if a.check:
        for name, df in files.items():
            saved = pd.read_csv(OUT / name); assert list(saved.columns) == list(df.columns), name
            for c in df.columns:
                if df[c].dtype.kind in 'fi': assert np.allclose(saved[c].to_numpy(float), df[c].to_numpy(float), atol=1e-10, equal_nan=True), (name, c)
                else: assert saved[c].astype(str).tolist() == df[c].astype(str).tolist(), (name, c)
        print('CHECK PASSED: widths, ratios, severity and yearly widening reproduced; summary.csv cross-check passed'); return
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in files.items(): df.to_csv(OUT / name, index=False)
    (OUT / 'run.json').write_text(json.dumps({'protocol_sha256': sha(Path(__file__).with_name('PROTOCOL.md')), 'producer_sha256': sha(__file__),
        'inputs': {'daily_parquet_sha256': hashes, 'summary.csv': sha(SUMMARY)}, 'python': sys.version, 'pandas': pd.__version__}, indent=2) + '\n')
    w = widths.set_index('method'); sv = severity.set_index('method')
    (OUT / 'RESULTS.md').write_text(f"""# Width, severity and yearly widening (measured, 240 pairs)

Pooled mean VaR level (% of notional): Raw {w.loc['Raw','pooled_mean_var_pct']:.4f}, Shift-CP {w.loc['Shift-CP','pooled_mean_var_pct']:.4f}
({w.loc['Shift-CP','pooled_pct_change_vs_raw']:+.2f}%), Rolling250 {w.loc['Rolling250','pooled_mean_var_pct']:.4f}.
Pair-equal: Raw {w.loc['Raw','pair_equal_mean_var_pct']:.4f}, Shift-CP {w.loc['Shift-CP','pair_equal_mean_var_pct']:.4f} ({w.loc['Shift-CP','pair_equal_pct_change_vs_raw']:+.2f}%).
Shift-CP/Raw ratio: mean {w.loc['Shift-CP','ratio_mean']:.3f}, median {w.loc['Shift-CP','ratio_median']:.3f}, max {w.loc['Shift-CP','ratio_max']:.3f}, min {w.loc['Shift-CP','ratio_min']:.3f}; {int(w.loc['Shift-CP','pairs_wider_than_raw'])} of 240 pairs wider.
Breach severity (pooled): exceedance Raw {sv.loc['Raw','pooled_mean_exceedance_pct']:.3f}% -> Shift-CP {sv.loc['Shift-CP','pooled_mean_exceedance_pct']:.3f}%;
loss on breach Raw {sv.loc['Raw','pooled_mean_loss_on_breach_pct']:.3f}% -> Shift-CP {sv.loc['Shift-CP','pooled_mean_loss_on_breach_pct']:.3f}%.
Yearly pooled widening: {', '.join(f"{int(r.year)}: {r.pct_change:+.1f}%" for r in by_year.itertuples())}.
""")
    print((OUT / 'RESULTS.md').read_text())

if __name__ == '__main__': main()
