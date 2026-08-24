"""How far the two divergent conventions move published numbers.

LEVEL_K_OVER_N: np.quantile(s, ceil((n+1)(1-a))/n) -- interpolates ABOVE S_(k).
PLAIN_QUANTILE: np.quantile(s, 1-a)                -- interpolates BELOW S_(k).
Measured on the real panel scores the published tables are computed from.
"""
import numpy as np, pandas as pd, json, math, pathlib
ALPHA, FC = 0.01, 0.70
SERIES = {"Chronos-Small": ("dir","chronos_small"), "Chronos-Mini": ("dir","chronos_mini"),
          "Chronos-Small-A": ("dir","chronos_small_analytic"), "Chronos-Mini-A": ("dir","chronos_mini_analytic"),
          "TimesFM-2.5": ("dir","timesfm25"), "Moirai-2.0": ("dir","moirai2"), "Moirai-1.1": ("dir","moirai"),
          "Lag-Llama": ("dir","lagllama"), "GJR-GARCH": ("bm","gjr_garch"), "GJR-GARCH-t": ("bm","gjr_t"),
          "GARCH-N": ("bm","garch_n"), "Hist-Sim": ("bm","hs"), "EWMA": ("bm","ewma")}
assets = sorted(p.stem for p in pathlib.Path("cfp_ijf_data/returns").glob("*.csv"))

def conv(s, kind):
    x = np.sort(np.asarray(s)); n = len(x); k = math.ceil((n+1)*(1-ALPHA))
    if kind == "order": return float(x[min(k, n)-1])
    if kind == "level": return float(np.quantile(x, min(k/n, 1.0)))
    if kind == "plain": return float(np.quantile(x, 1-ALPHA))

rows = []
for m,(kind,tag) in SERIES.items():
    for a in assets:
        r = pd.read_csv(f"cfp_ijf_data/returns/{a}.csv", parse_dates=["date"]).set_index("date")["log_return"]
        p = f"cfp_ijf_data/{tag}/{a}.parquet" if kind=="dir" else f"cfp_ijf_data/benchmarks/{a}_{tag}.parquet"
        j = pd.read_parquet(p)[[f"VaR_{ALPHA}"]].join(r, how="inner").dropna()
        n = len(j); nc = int(math.floor(n*FC))
        cal, test = j.iloc[:nc], j.iloc[nc:]
        sc = (cal[f"VaR_{ALPHA}"] - cal["log_return"]).to_numpy()
        q = {c: conv(sc, c) for c in ("order","level","plain")}
        rec = {"model":m,"symbol":a,"n_cal":nc,"n_test":len(test), **{f"q_{c}":v for c,v in q.items()}}
        for c in ("order","level","plain"):
            v = (test["log_return"] < test[f"VaR_{ALPHA}"] - q[c]).sum()
            rec[f"viol_{c}"] = int(v); rec[f"pi_{c}"] = v/len(test)
        rows.append(rec)
d = pd.DataFrame(rows); d.to_csv("analysis/convention/impact_panel.csv", index=False)

print(f"312 pairs, calibration samples {d.n_cal.min()}-{d.n_cal.max()}\n")
for c,label in (("level","LEVEL_K_OVER_N (interpolates above)"), ("plain","PLAIN_QUANTILE (interpolates below)")):
    rel = ((d[f"q_{c}"] - d.q_order).abs() / d.q_order.abs().replace(0, np.nan))
    dv = (d[f"viol_{c}"] - d.viol_order)
    print(f"{label}")
    print(f"   |q - q_eq8| relative:  median {rel.median():.2e}   90th pct {rel.quantile(0.9):.2e}   max {rel.max():.2e}")
    print(f"   violation-count change: pairs affected {int((dv!=0).sum())}/312   max |change| {int(dv.abs().max())}")
    print(f"   cell-mean corrected pi: eq.(8) {d.pi_order.mean():.6f}  vs  {c} {d[f'pi_{c}'].mean():.6f}"
          f"   -> {'IDENTICAL at 4 dp' if round(d.pi_order.mean(),4)==round(d[f'pi_{c}'].mean(),4) else 'DIFFERS at 4 dp'}")
    print()
json.dump({c: {"median_rel": float(((d[f"q_{c}"]-d.q_order).abs()/d.q_order.abs().replace(0,np.nan)).median()),
               "max_rel": float(((d[f"q_{c}"]-d.q_order).abs()/d.q_order.abs().replace(0,np.nan)).max()),
               "pairs_violation_count_changed": int(((d[f"viol_{c}"]-d.viol_order)!=0).sum()),
               "cell_mean_pi": float(d[f"pi_{c}"].mean())} for c in ("order","level","plain")},
          open("analysis/convention/impact_panel.json","w"), indent=2)
