"""K1a, part 4. How far the one-bin offset moves the published numbers.

The offset is deterministic: stored = correct + binwidth * scale_t, where scale_t
is the tokenizer's scale, the mean absolute value of the 512-observation context.
scale_t needs no model, so the whole panel can be corrected exactly from the
returns. The reconstruction rule is checked against the 200 dates on which the
model was actually re-run before it is applied anywhere else.
"""
import numpy as np, pandas as pd, json, math, pathlib

CONTEXT, LEVELS, FC = 512, [0.01, 0.025, 0.05, 0.1], 0.70
BINW = 30.0 / 4092.0                       # linspace(-15, 15, 4093)
assets = sorted(p.stem for p in pathlib.Path("cfp_ijf_data/returns").glob("*.csv"))

def scales(asset, index):
    r = pd.read_csv(f"cfp_ijf_data/returns/{asset}.csv", parse_dates=["date"]).set_index("date")["log_return"]
    pos = r.index.get_indexer(index)
    return pd.Series([np.abs(r.iloc[p-CONTEXT:p].to_numpy()).mean() for p in pos], index=index), r

# --- check the reconstruction rule against the 200 re-run dates -----------------
mine = pd.read_csv("analysis/k1_verify/k1a_reimplementation.csv", parse_dates=["date"]).set_index("date")
stored = pd.read_parquet("cfp_ijf_data/chronos_small_analytic/SP500.parquet").loc[mine.index]
sc, _ = scales("SP500", mine.index)
print(f"tokenizer scale, reconstructed from returns vs returned by the model: "
      f"max |rel| {float(np.max(np.abs(sc - mine['scale'])/mine['scale'])):.3e}")
rebuilt = stored["VaR_0.01"] - BINW * sc
err = np.abs(rebuilt - mine["VaR_0.01_renorm"]) / np.abs(mine["VaR_0.01_renorm"])
print(f"stored - binwidth*scale vs the re-run estimator: median |rel| {float(np.median(err)):.3e}  "
      f"max |rel| {float(np.max(err)):.3e}   -> {'reconstruction rule holds' if np.max(err) < 1e-4 else 'RULE FAILS'}")

# --- apply it to both analytic panels ------------------------------------------
def kupiec_p(n_viol, n, alpha):
    if n == 0: return np.nan
    pi = n_viol / n
    if n_viol == 0:
        lr = -2 * (n * np.log(1 - alpha))
    else:
        lr = -2 * ((n - n_viol) * np.log(1 - alpha) + n_viol * np.log(alpha)
                   - (n - n_viol) * np.log(1 - pi) - n_viol * np.log(pi))
    from scipy import stats
    return float(stats.chi2.sf(max(lr, 0.0), 1))

def qs(r, q, alpha):
    return float(np.mean((alpha - (r < q).astype(float)) * (r - q)))

out = {}
for tag, store, label in [("small", "chronos_small_analytic", "Chronos-Small-A"),
                          ("mini", "chronos_mini_analytic", "Chronos-Mini-A")]:
    rows = []
    for a in assets:
        f = pd.read_parquet(f"cfp_ijf_data/{store}/{a}.parquet")
        sc, r = scales(a, f.index)
        j = f.join(r, how="inner").dropna(subset=["log_return"])
        sc = sc.loc[j.index]
        n = len(j); ncal = int(math.floor(n * FC)); test = j.iloc[ncal:]; sct = sc.iloc[ncal:]
        rec = {"symbol": a, "n_test": len(test)}
        for al in LEVELS:
            v_st = test[f"VaR_{al}"]
            v_fx = v_st - BINW * sct
            for nm, v in (("stored", v_st), ("fixed", v_fx)):
                viol = int((test["log_return"] < v).sum())
                rec[f"pi_{nm}_{al}"] = viol / len(test)
                rec[f"viol_{nm}_{al}"] = viol
                rec[f"kup_{nm}_{al}"] = kupiec_p(viol, len(test), al)
                if al == 0.01:
                    rec[f"qs_{nm}"] = qs(test["log_return"].to_numpy(), v.to_numpy(), al) * 1e4
        rows.append(rec)
    d = pd.DataFrame(rows); d.to_csv(f"analysis/k1_verify/k1a_impact_{tag}.csv", index=False)
    print(f"\n=== {label} ===")
    o = {}
    for al in LEVELS:
        ps, pf = d[f"pi_stored_{al}"].mean(), d[f"pi_fixed_{al}"].mean()
        ks, kf = int((d[f"kup_stored_{al}"] > 0.05).sum()), int((d[f"kup_fixed_{al}"] > 0.05).sum())
        print(f"  a={al:<6} pi(cell mean) published {ps:.4f} -> corrected {pf:.4f}   "
              f"ratio {ps/al:.3f} -> {pf/al:.3f}   Kupiec pass {ks}/24 -> {kf}/24")
        o[str(al)] = {"pi_published": round(float(ps),6), "pi_corrected": round(float(pf),6),
                      "ratio_published": round(float(ps/al),4), "ratio_corrected": round(float(pf/al),4),
                      "kupiec_published": ks, "kupiec_corrected": kf}
    print(f"  QS at a=0.01 (x1e4): published {d.qs_stored.mean():.4f} -> corrected {d.qs_fixed.mean():.4f}")
    o["QS_1e4"] = {"published": round(float(d.qs_stored.mean()),4), "corrected": round(float(d.qs_fixed.mean()),4)}
    out[label] = o
json.dump(out, open("analysis/k1_verify/k1a_impact.json","w"), indent=2)
