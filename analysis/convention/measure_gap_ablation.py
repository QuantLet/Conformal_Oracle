"""Does the gap ablation's conclusion depend on which quantile convention it uses?

scripts/gap_ablation.py computes the shift as np.quantile(cal_scores, 1-alpha),
the plain empirical quantile. Section 3.2.1 and the Data and Code statement both
say every static and rolling result uses equation (8), the conformal order
statistic. Section 4.4 of the rebuilt manuscript cites this ablation as the
empirical evidence for the separation condition, so the gap between the two
conventions is measured here rather than assumed negligible.
"""
import numpy as np, pandas as pd, json
from math import ceil, floor
from pathlib import Path

BASE = Path("cfp_ijf_data"); ALPHA, FC = 0.01, 0.70
PAIRS = [("Chronos-Small-A", "chronos_small_analytic", "SP500"),
         ("Lag-Llama", "lagllama", "BTC"),
         ("GJR-GARCH", "benchmarks", "WTI"),
         ("Moirai-2.0", "moirai2", "NATGAS")]
BENCH = {"GJR-GARCH": "gjr_garch"}
PERIODS = {"Full": (None, None), "GFC": ("2008-09-01", "2009-03-31"),
           "COVID": ("2020-02-01", "2020-06-30"), "SVB": ("2023-03-01", "2023-06-30")}

def q_plain(s, a=ALPHA):     return float(np.quantile(s, 1 - a))
def q_conformal(s, a=ALPHA):
    x = np.sort(np.asarray(s)); n = len(x)
    return float(x[min(ceil((n + 1) * (1 - a)), n) - 1])

def load(model, sub, sym):
    p = BASE/sub/f"{sym}_{BENCH.get(model, model.lower().replace('-','_'))}.parquet" if sub == "benchmarks" \
        else BASE/sub/f"{sym}.parquet"
    pq = pd.read_parquet(p); pq.index = pd.to_datetime(pq.index)
    ret = pd.read_csv(BASE/"returns"/f"{sym}.csv", parse_dates=["date"]).set_index("date")
    m = pq[["VaR_0.01"]].join(ret["log_return"], how="inner").dropna()
    return (m["VaR_0.01"] - m["log_return"]).to_numpy(), m.index

rows = []
for model, sub, sym in PAIRS:
    s_all, dates = load(model, sub, sym)
    for per, (a, b) in PERIODS.items():
        s = s_all if a is None else s_all[(dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))]
        if len(s) < 100: continue
        n = len(s); n_cal = int(FC*n)
        rho = pd.Series(s).autocorr(lag=1)
        g_log = max(5, int((1.0/abs(np.log(rho)))*np.log(n))) if rho and 0 < rho < 0.999 else max(5, int(np.log(n)))
        rec = {"model": model, "symbol": sym, "period": per, "n": n, "n_cal": n_cal, "rho": rho}
        for name, qf in (("plain", q_plain), ("conformal", q_conformal)):
            qv = qf(s[:n_cal])
            rec[f"qV_{name}"] = qv
            for gap, tag in ((0, "g0"), (g_log, "glog")):
                t = s[n_cal+gap:]
                rec[f"pi_{name}_{tag}"] = float((t > qv).mean()) if len(t) else np.nan
            rec[f"absdiff_{name}"] = abs(rec[f"pi_{name}_g0"] - rec[f"pi_{name}_glog"])
        rows.append(rec)
d = pd.DataFrame(rows)
d.to_csv("analysis/convention/gap_ablation_conventions.csv", index=False)

print(f"{'pair':28s} {'period':7s} {'qV plain':>10s} {'qV conf':>10s} {'rel':>8s} "
      f"| {'|dpi| plain':>11s} {'|dpi| conf':>11s}")
for _, r in d.iterrows():
    rel = abs(r.qV_conformal - r.qV_plain)/max(abs(r.qV_plain), 1e-12)
    print(f"{r.model+'/'+r.symbol:28s} {r.period:7s} {r.qV_plain:10.6f} {r.qV_conformal:10.6f} "
          f"{rel:7.2%} | {r.absdiff_plain:11.6f} {r.absdiff_conformal:11.6f}")

print(f"\nthe manuscript's two claims, under each convention:")
full = d[d.period == "Full"]; cov = d[d.period == "COVID"]
for name in ("plain", "conformal"):
    print(f"  {name:10s}  max |dpi| over the full sample: {full[f'absdiff_{name}'].max():.4f}"
          f"   within COVID: {cov[f'absdiff_{name}'].max():.4f}")
print(f"\n  printed in the manuscript: 0.0005 full sample, 0.0058 COVID")
json.dump({"full_plain": float(full.absdiff_plain.max()), "full_conformal": float(full.absdiff_conformal.max()),
           "covid_plain": float(cov.absdiff_plain.max()), "covid_conformal": float(cov.absdiff_conformal.max()),
           "max_qV_rel_gap": float(((d.qV_conformal-d.qV_plain).abs()/d.qV_plain.abs()).max())},
          open("analysis/convention/gap_ablation_conventions.json","w"), indent=2)
