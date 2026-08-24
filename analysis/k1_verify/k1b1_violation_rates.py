"""K1b1. The fraction of realised returns below the stored threshold, for
TimesFM-2.5 and Moirai-2.0, recomputed from returns + stored series alone.

Protocol read from Section 3.3: chronological 70/30 split of the stored forecast
series; the test window is the last 30%. Both the cell reading (mean over 24
assets) and the panel reading (pooled) are reported, since the manuscript prints
one number and does not say which.
"""
import numpy as np, pandas as pd, json, math

STORES = {"TimesFM-2.5": "timesfm25", "Moirai-2.0": "moirai2",
          "Chronos-Small-A": "chronos_small_analytic", "Chronos-Mini-A": "chronos_mini_analytic",
          "Moirai-1.1": "moirai", "Lag-Llama": "lagllama",
          "Chronos-Small": "chronos_small", "Chronos-Mini": "chronos_mini"}
ALPHA, FC = 0.01, 0.70
assets = sorted(p.stem for p in __import__("pathlib").Path("cfp_ijf_data/returns").glob("*.csv"))
ref = pd.read_csv("cfp_ijf_data/paper_outputs/tables/all_results.csv")
ref = ref[np.isclose(ref.alpha, ALPHA)]

def rates(store, reverse=False, full_sample=False):
    rows = []
    for a in assets:
        r = pd.read_csv(f"cfp_ijf_data/returns/{a}.csv", parse_dates=["date"]).set_index("date")["log_return"]
        f = pd.read_parquet(f"cfp_ijf_data/{store}/{a}.csv".replace(".csv", ".parquet"))
        j = f.join(r, how="inner").dropna(subset=["log_return", f"VaR_{ALPHA}"])
        n = len(j); ncal = int(math.floor(n * FC))
        t = j if full_sample else j.iloc[ncal:]
        v = (t["log_return"] > t[f"VaR_{ALPHA}"]) if reverse else (t["log_return"] < t[f"VaR_{ALPHA}"])
        rows.append({"symbol": a, "n": n, "n_cal": ncal, "n_test": len(t),
                     "viol": int(v.sum()), "pihat": float(v.mean())})
    return pd.DataFrame(rows)

out = {}
print(f"{'model':16s} {'cell mean':>10s} {'panel pooled':>13s} {'stored cell':>12s} {'stored panel':>13s}  printed")
printed = {"TimesFM-2.5": 0.0143, "Moirai-2.0": 0.0178, "Chronos-Small-A": 0.0175, "Chronos-Mini-A": 0.0178,
           "Moirai-1.1": 0.0154, "Lag-Llama": 0.0294}
for model, store in STORES.items():
    d = rates(store)
    cell, panel = d.pihat.mean(), d.viol.sum() / d.n_test.sum()
    s = ref[ref.model == model]
    scell, spanel = s.pihat_raw.mean(), s.viol_raw.sum() / s.n_test.sum()
    p = printed.get(model)
    print(f"{model:16s} {cell:10.6f} {panel:13.6f} {scell:12.6f} {spanel:13.6f}  {p if p else '-'}")
    d.to_csv(f"analysis/k1_verify/k1b1_{store}.csv", index=False)
    out[model] = {"cell_mean": float(cell), "panel_pooled": float(panel),
                  "stored_cell_mean": float(scell), "stored_panel_pooled": float(spanel),
                  "printed": p,
                  "n_test_total": int(d.n_test.sum()), "n_test_total_stored": int(s.n_test.sum())}

print("\n--- NEGATIVE CONTROLS (each must fail to reproduce the printed number) ---")
for tag, kw in [("inequality reversed", {"reverse": True}), ("full sample not test window", {"full_sample": True})]:
    d = rates("timesfm25", **kw)
    print(f"TimesFM-2.5, {tag:32s} cell {d.pihat.mean():.6f}  panel {d.viol.sum()/d.n_test.sum():.6f}  "
          f"(printed 0.0143) -> {'FAILS as required' if abs(d.pihat.mean()-0.0143)>5e-4 else '!! AGREES'}")
    out[f"neg_{tag}"] = {"cell_mean": float(d.pihat.mean()), "panel_pooled": float(d.viol.sum()/d.n_test.sum())}
json.dump(out, open("analysis/k1_verify/k1b1_result.json", "w"), indent=2)
