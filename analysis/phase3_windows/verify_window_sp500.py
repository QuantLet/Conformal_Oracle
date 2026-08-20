#!/usr/bin/env python3
"""Reproduce the AE-7 window sweep on SP500 only and diff against the stored CSV.

The re-run memo marked the sweep MUST-BE-RE-RUN on the grounds that its GJR rows
used the defective raw t5 quantile map. The script on disk uses norm.ppf and
re-estimates GARCH from returns, so the claim is testable rather than a matter of
judgement: if SP500 reproduces to floating point, the sweep was never
contaminated and the other 23 assets need no refit.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("/Users/danielpele/Documents/2026 CFP LLM VaR")
sys.path.insert(0, str(BASE / "analysis" / "phase3_windows"))
from run_window_sensitivity import (  # noqa: E402
    MODELS, WINDOWS, ALPHA, DATA, var_path, evaluate,
)

SYM = "SP500"
ret = pd.read_csv(DATA / "returns" / f"{SYM}.csv", index_col=0, parse_dates=True)
r = ret.iloc[:, 0].values.astype(float)

rows = []
for label, key in MODELS.items():
    for w in WINDOWS:
        var, fails = var_path(r, w, key, ALPHA)
        res = evaluate(r, var, ALPHA)
        if not res:
            continue
        res.update({"model": label, "asset": SYM, "w": w,
                    "convergence_failures": fails, "n_fits": max(len(r) - w, 0)})
        rows.append(res)
        print(f"  {label} w={w}: pihat_raw={res['pihat_raw']:.6f} qV={res['qV']:.8f}",
              flush=True)

new = pd.DataFrame(rows).set_index(["model", "w"]).sort_index()
old = pd.read_csv(BASE / "analysis/phase3_windows/window_sensitivity.csv")
old = old[old["asset"] == SYM].set_index(["model", "w"]).sort_index()

cols = ["qV", "pihat_raw", "pihat_cp", "QS_raw", "QS_cp", "p_kup_raw", "p_kup_cp"]
print("\n--- max abs difference, stored vs reproduced ---")
worst = 0.0
for c in cols:
    d = (new[c] - old[c]).abs().max()
    worst = max(worst, float(d))
    print(f"  {c:12s} {d:.3e}")
print(f"\nzones identical: TL_raw {bool((new['TL_raw']==old['TL_raw']).all())} "
      f"TL_cp {bool((new['TL_cp']==old['TL_cp']).all())}")
print(f"WORST {worst:.3e} -> {'REPRODUCES' if worst < 1e-9 else 'DIFFERS'}")
