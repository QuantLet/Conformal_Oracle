"""K1b3. Pairs with n11 = n10 = 0, against what the code records as a
Christoffersen pass.

Violations are recomputed from returns + stored forecast series, with the
conformal shift rebuilt from equation (8) of the manuscript, rather than read from
the stored violation sequences. The three degeneracy conditions of the
pre-registration are counted separately; only afterwards is the producing code's
rule read off and matched to them.
"""
import numpy as np, pandas as pd, json, math, pathlib

ALPHA, FC = 0.01, 0.70
SERIES = {"Chronos-Small": ("dir", "chronos_small"), "Chronos-Mini": ("dir", "chronos_mini"),
          "Chronos-Small-A": ("dir", "chronos_small_analytic"), "Chronos-Mini-A": ("dir", "chronos_mini_analytic"),
          "TimesFM-2.5": ("dir", "timesfm25"), "Moirai-2.0": ("dir", "moirai2"), "Moirai-1.1": ("dir", "moirai"),
          "Lag-Llama": ("dir", "lagllama"),
          "GJR-GARCH": ("bm", "gjr_garch"), "GJR-GARCH-t": ("bm", "gjr_t"), "GARCH-N": ("bm", "garch_n"),
          "Hist-Sim": ("bm", "hs"), "EWMA": ("bm", "ewma")}
assets = sorted(p.stem for p in pathlib.Path("cfp_ijf_data/returns").glob("*.csv"))

def load(kind, tag, asset):
    p = f"cfp_ijf_data/{tag}/{asset}.parquet" if kind == "dir" else f"cfp_ijf_data/benchmarks/{asset}_{tag}.parquet"
    return pd.read_parquet(p)

def conformal_q(scores, alpha):
    """Equation (8): the ceil((n+1)(1-alpha))-th order statistic, S_(n) if k > n."""
    s = np.sort(np.asarray(scores)); n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return float(s[min(k, n) - 1])

def transitions(v):
    v = np.asarray(v).astype(int)
    return dict(n00=int(((v[:-1]==0)&(v[1:]==0)).sum()), n01=int(((v[:-1]==0)&(v[1:]==1)).sum()),
                n10=int(((v[:-1]==1)&(v[1:]==0)).sum()), n11=int(((v[:-1]==1)&(v[1:]==1)).sum()))

rows = []
for model, (kind, tag) in SERIES.items():
    for a in assets:
        r = pd.read_csv(f"cfp_ijf_data/returns/{a}.csv", parse_dates=["date"]).set_index("date")["log_return"]
        f = load(kind, tag, a)
        j = f[[f"VaR_{ALPHA}"]].join(r, how="inner").dropna()
        n = len(j); ncal = int(math.floor(n * FC))
        cal, test = j.iloc[:ncal], j.iloc[ncal:]
        qv = conformal_q(cal[f"VaR_{ALPHA}"] - cal["log_return"], ALPHA)
        vr = (test["log_return"] < test[f"VaR_{ALPHA}"]).to_numpy()
        vc = (test["log_return"] < test[f"VaR_{ALPHA}"] - qv).to_numpy()
        for lab, v in (("raw", vr), ("cor", vc)):
            t = transitions(v)
            rows.append({"model": model, "symbol": a, "which": lab, "n_test": len(test), "qV": qv,
                         "n_viol": int(v.sum()), **t,
                         "A_n11_n10_zero": t["n11"] == 0 and t["n10"] == 0,
                         "B_n11_zero_n10_pos": t["n11"] == 0 and t["n10"] > 0,
                         "C_n01_n11_zero": t["n01"] == 0 and t["n11"] == 0,
                         "code_nan": (t["n00"]+t["n01"])==0 or (t["n10"]+t["n11"])==0 or (t["n01"]+t["n11"])==0
                                     or t["n11"] == 0 or t["n01"] == 0})
d = pd.DataFrame(rows)
d.to_csv("analysis/k1_verify/k1b3_transitions.csv", index=False)

ref = pd.read_csv("cfp_ijf_data/paper_outputs/tables/all_results.csv")
ref = ref[np.isclose(ref.alpha, ALPHA)].set_index(["model", "symbol"])

print(f"pairs: {len(d)//2} per reading, {d.model.nunique()} forecasters x {d.symbol.nunique()} assets\n")
out = {}
for lab, printed in (("raw", 34.6), ("cor", 53.5)):
    s = d[d.which == lab]
    nA, nB, nC = int(s.A_n11_n10_zero.sum()), int(s.B_n11_zero_n10_pos.sum()), int(s.C_n01_n11_zero.sum())
    nAB = int((s.n11 == 0).sum())
    stored_nan = int(ref[f"p_cc_{ 'raw' if lab=='raw' else 'cp'}"].isna().sum())
    print(f"[{lab}]  (A) n11=n10=0: {nA:3d} ({100*nA/len(s):.1f}%)   "
          f"(B) n11=0<n10: {nB:3d}   (A|B) n11=0: {nAB:3d} ({100*nAB/len(s):.1f}%)   "
          f"(C) n01=n11=0: {nC:3d}")
    print(f"       my reimplementation of the code's rule: {int(s.code_nan.sum()):3d} "
          f"({100*s.code_nan.mean():.1f}%)   stored NaN in all_results: {stored_nan:3d} "
          f"({100*stored_nan/len(s):.1f}%)   manuscript prints {printed}%")
    out[lab] = {"A": nA, "B": nB, "C": nC, "n11_zero": nAB, "code_rule": int(s.code_nan.sum()),
                "stored_nan": stored_nan, "printed_pct": printed, "n_pairs": len(s)}
    # which condition matches the code
    for cond, name in ((s.A_n11_n10_zero, "A"), (s.n11 == 0, "A|B"), (s.C_n01_n11_zero, "C")):
        if int(cond.sum()) == stored_nan:
            print(f"       -> stored NaN count equals condition {name}")
    print()

print("--- NEGATIVE CONTROLS ---")
v_ok = np.zeros(1000, int); v_ok[[10, 11, 300, 301, 700]] = 1     # has n11 > 0
v_bad = np.zeros(1000, int)                                       # no exceedance at all
v_spread = np.zeros(1000, int); v_spread[[100, 300, 500, 700]] = 1 # exceedances, never consecutive
for name, v in (("n11>0 (must NOT be flagged)", v_ok), ("all zeros (must be flagged)", v_bad),
                ("spread out, n11=0<n10 (condition B)", v_spread)):
    t = transitions(v)
    flag = (t["n00"]+t["n01"])==0 or (t["n10"]+t["n11"])==0 or (t["n01"]+t["n11"])==0 or t["n11"]==0 or t["n01"]==0
    print(f"  {name:38s} n00={t['n00']:4d} n01={t['n01']:2d} n10={t['n10']:2d} n11={t['n11']:2d} -> "
          f"{'FLAGGED undefined' if flag else 'informative'}")
    out[f"neg_{name}"] = {**t, "flagged": bool(flag)}
json.dump(out, open("analysis/k1_verify/k1b3_result.json", "w"), indent=2)
