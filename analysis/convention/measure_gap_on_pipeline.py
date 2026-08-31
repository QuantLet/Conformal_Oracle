"""The gap's effect on the PIPELINE's own numbers. Writes nothing."""
import sys, importlib.util
from math import ceil, log
from pathlib import Path
import numpy as np, pandas as pd
BASE = Path("/Users/danpele/dev/cfp-llm-var")
sys.path.insert(0, str(BASE / "Quantlets"))
spec = importlib.util.spec_from_file_location(
    "rfe", BASE / "Quantlets" / "CO_full_evaluation" / "run_full_evaluation.py")
rfe = importlib.util.module_from_spec(spec); spec.loader.exec_module(rfe)

def run(with_gap):
    rows = []
    for model in rfe.MODELS:
        for sym in sorted(rfe.SYMBOLS):
            for a in rfe.ALPHAS:
                try:
                    got = rfe.load_pair(model, sym, a)
                except Exception:
                    got = None
                if got is None: continue
                r, v = got[0], got[1]
                r = np.asarray(r); v = np.asarray(v)
                n_cal = int(len(r) * rfe.F_CAL)
                g = 0
                if with_gap:
                    s = v[:n_cal] - r[:n_cal]
                    rho = pd.Series(s).autocorr(lag=1)
                    g = max(5, int(ceil((1.0/abs(log(rho))) * log(n_cal)))) \
                        if rho and 0 < rho < 0.999 else max(5, int(ceil(log(n_cal))))
                    r = np.concatenate([r[:n_cal], r[n_cal+g:]])
                    v = np.concatenate([v[:n_cal], v[n_cal+g:]])
                res = rfe.conformal_backtest(r, v, a, f_cal=n_cal/len(r))
                rows.append({"model": model, "symbol": sym, "alpha": a, "gap": g, **res})
    return pd.DataFrame(rows)

a0, a1 = run(False), run(True)
key = ["model","symbol","alpha"]
m = a0.merge(a1, on=key, suffixes=("_0","_g"))
one = m[m.alpha == 0.01]
print(f"cells at alpha=0.01: {len(one)}   gap {int(one.gap_g.min())}-{int(one.gap_g.max())}, median {int(one.gap_g.median())}")
print(f"  max |d pihat_cp|      {float((one.pihat_cp_g-one.pihat_cp_0).abs().max()):.6f}")
print(f"  Basel zone changes    {int((one.TL_cp_0 != one.TL_cp_g).sum())} of {len(one)}")
print(f"  Kupiec flips at 5%    {int(((one.p_kup_cp_0>0.05)!=(one.p_kup_cp_g>0.05)).sum())}")
print(f"  Green before/after    {int((one.TL_cp_0=='Green').sum())} -> {int((one.TL_cp_g=='Green').sum())}")
print(f"  Kupiec passes         {int((one.p_kup_cp_0>0.05).sum())} -> {int((one.p_kup_cp_g>0.05).sum())}")
print(f"  CC pass changes       {int(((one.p_cc_cp_0>0.05)!=(one.p_cc_cp_g>0.05)).sum())}")
print(f"\nall {len(m)} cells over four levels:")
print(f"  max |d pihat_cp|      {float((m.pihat_cp_g-m.pihat_cp_0).abs().max()):.6f}")
print(f"  zone changes          {int((m.TL_cp_0 != m.TL_cp_g).sum())}")
print(f"  Kupiec flips          {int(((m.p_kup_cp_0>0.05)!=(m.p_kup_cp_g>0.05)).sum())}")
