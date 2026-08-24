"""K1b2. One thousand draws under the packaged default contain exactly 50 distinct
values. Pre-registered as a 40-cell subsample of the manuscript's 1600; an exact
equality is refuted by one counterexample, which is what a subsample can do.
"""
import numpy as np, pandas as pd, torch, json
from chronos import ChronosPipeline

CONTEXT = 512
rows = []
for ckpt in ["amazon/chronos-t5-small", "amazon/chronos-t5-mini"]:
    pipe = ChronosPipeline.from_pretrained(ckpt, device_map="cpu", torch_dtype=torch.float32)
    print(f"{ckpt}: packaged defaults top_k={pipe.model.config.top_k}, top_p={pipe.model.config.top_p}, "
          f"temperature={pipe.model.config.temperature}, num_samples={pipe.model.config.num_samples}", flush=True)
    for asset in ["SP500", "EURUSD"]:
        r = pd.read_csv(f"cfp_ijf_data/returns/{asset}.csv", parse_dates=["date"]).set_index("date")["log_return"]
        idx = pd.read_parquet(f"cfp_ijf_data/chronos_small_analytic/{asset}.parquet").index
        dates = idx[np.linspace(1000, len(idx) - 10, 10).astype(int)]
        ctxs = [torch.tensor(r.iloc[r.index.get_loc(d) - CONTEXT:r.index.get_loc(d)].to_numpy(),
                             dtype=torch.float32) for d in dates]
        for label, kw in [("default", {}), ("top_k=4094", {"top_k": 4094}), ("top_k=10", {"top_k": 10})]:
            for j, d in enumerate(dates):
                torch.manual_seed(0)
                with torch.no_grad():
                    fc = pipe.predict(ctxs[j], prediction_length=1, num_samples=1000,
                                      limit_prediction_length=False, **kw)
                rows.append({"ckpt": ckpt.split("-")[-1], "asset": asset, "date": str(d.date()),
                             "config": label, "n_distinct": int(len(np.unique(fc[0, :, 0].numpy())))})
            n = [x["n_distinct"] for x in rows if x["config"] == label and x["asset"] == asset
                 and x["ckpt"] == ckpt.split("-")[-1]]
            print(f"   {asset:7s} {label:11s} distinct values in 1000 draws: "
                  f"min {min(n)} max {max(n)} -- {sorted(set(n))}", flush=True)
d = pd.DataFrame(rows); d.to_csv("analysis/k1_verify/k1b2_distinct.csv", index=False)
dflt = d[d.config == "default"]
ok = bool((dflt.n_distinct == 50).all())
print(f"\ndefault configuration: all {len(dflt)} cells return exactly 50 distinct values: {ok}")
print("NEGATIVE CONTROLS")
c10 = d[d.config == "top_k=10"]; cfull = d[d.config == "top_k=4094"]
print(f"  top_k=10   returns exactly 10 in {int((c10.n_distinct==10).sum())}/{len(c10)} cells "
      f"-> {'control fires' if (c10.n_distinct==10).all() else 'CONTROL FAILED'}")
print(f"  top_k=4094 returns far more than 50: min {int(cfull.n_distinct.min())} "
      f"-> {'control fires' if cfull.n_distinct.min() > 50 else 'CONTROL FAILED'}")
json.dump({"n_cells": len(dflt), "all_exactly_50": ok,
           "default_counts": sorted(dflt.n_distinct.unique().tolist()),
           "top_k_10_counts": sorted(c10.n_distinct.unique().tolist()),
           "top_k_full_min": int(cfull.n_distinct.min()), "top_k_full_max": int(cfull.n_distinct.max())},
          open("analysis/k1_verify/k1b2_result.json", "w"), indent=2)
