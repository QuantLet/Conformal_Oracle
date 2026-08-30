#!/usr/bin/env python3
"""The separation gap of Theorem 4.5, applied to all 312 cells.

Section 4.4 reports that the protocol runs g_n = 0, so the theorem does not
cover the estimator as run. The four-pair ablation says the gap is cheap. This
extends it to the panel so the concession can be closed or its size stated.

Pre-registration: analysis/convention/PREREG_GAP_RUN.md
Unit: one cell is one forecaster x one asset at alpha = 0.01. 312 expected.

The gap comes off the START OF THE TEST BLOCK, matching
scripts/gap_ablation.py: test_start = n_cal + gap. The shift is therefore
unchanged and only the evaluation window moves.

    python analysis/convention/run_gap_panel.py
    python analysis/convention/run_gap_panel.py --controls
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
DATA = BASE / "cfp_ijf_data"
sys.path.insert(0, str(BASE / "Quantlets"))
from cfp_config import MODELS, SYMBOLS, conformal_quantile  # noqa: E402

ALPHA, FC = 0.01, 0.70
RED, GRN, YEL = "\033[31m", "\033[32m", "\033[33m"; OFF = "\033[0m"


def gap_for(scores: np.ndarray) -> int:
    """g_n = ceil(c log n), c = 1/|log rho|, floored at 5 as in gap_ablation.py."""
    rho = pd.Series(scores).autocorr(lag=1)
    n = len(scores)
    if rho and 0 < rho < 0.999:
        return max(5, int(np.ceil((1.0 / abs(np.log(rho))) * np.log(n))))
    return max(5, int(np.ceil(np.log(n))))


def evaluate(scores: np.ndarray, gap: int):
    n = len(scores)
    n_cal = int(FC * n)
    qV = conformal_quantile(scores[:n_cal], ALPHA)
    start = n_cal + gap
    if start >= n:
        return None
    test = scores[start:]
    v = int((test > qV).sum()); n_test = len(test)
    pi = v / n_test
    if pi in (0.0, 1.0):
        p = 0.0
    else:
        lr = -2 * (v * np.log(ALPHA / pi)
                   + (n_test - v) * np.log((1 - ALPHA) / (1 - pi)))
        p = float(1 - chi2.cdf(abs(lr), 1))
    rate = v / (n_test / 250)
    tl = "Green" if rate < 5 else ("Yellow" if rate < 10 else "Red")
    return {"qV": float(qV), "n_test": n_test, "viol": v, "pi_hat": pi,
            "p_kupiec": p, "TL": tl}


def load(model: str, asset: str):
    sub, suf = MODELS[model]
    f = DATA / sub / (f"{asset}_{suf}.parquet" if suf else f"{asset}.parquet")
    if not f.is_file():
        return None
    fc = pd.read_parquet(f)
    col = f"VaR_{ALPHA}"
    if col not in fc.columns:
        return None
    ret = pd.read_csv(DATA / "returns" / f"{asset}.csv", index_col=0, parse_dates=True)
    idx = ret.index.intersection(fc.index).sort_values()
    return (fc.loc[idx, col] - ret.loc[idx, "log_return"]).to_numpy()


def control_gap_shrinks_test() -> bool:
    """Inserting a gap must remove exactly g_n test observations and nothing else."""
    s = np.random.default_rng(0).standard_normal(1000)
    a, b = evaluate(s, 0), evaluate(s, 7)
    return a["n_test"] - b["n_test"] == 7 and np.isclose(a["qV"], b["qV"])


def control_gap_is_positive() -> bool:
    """g_n must be at least the floor and must grow with n."""
    rng = np.random.default_rng(1)
    small = gap_for(rng.standard_normal(500))
    large = gap_for(rng.standard_normal(50000))
    return small >= 5 and large >= small


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--controls", action="store_true")
    a = ap.parse_args()
    ok = True
    for name, fn in (("a gap removes exactly g_n test points, shift unchanged",
                      control_gap_shrinks_test),
                     ("g_n respects its floor and grows with n", control_gap_is_positive)):
        if fn(): print(f"  {YEL}ctrl{OFF}   {name}")
        else: print(f"  {RED}FAIL{OFF}   control does not fire: {name}"); ok = False
    if not ok: return 2
    if a.controls: return 0

    rows, blocked = [], []
    for model in MODELS:
        for asset in sorted(SYMBOLS):
            sc = load(model, asset)
            if sc is None:
                continue
            g = gap_for(sc[:int(FC * len(sc))])
            z, w = evaluate(sc, 0), evaluate(sc, g)
            if z is None or w is None:
                blocked.append((model, asset)); continue
            rows.append({"model": model, "asset": asset, "gap": g,
                         **{f"g0_{k}": v for k, v in z.items()},
                         **{f"gn_{k}": v for k, v in w.items()}})
    d = pd.DataFrame(rows)
    print(f"\n  cells: {len(d)}   blocked: {len(blocked)}")
    if blocked:
        print(f"  {RED}BLOCKED{OFF}: {blocked[:5]}")
    d["dpi"] = (d["gn_pi_hat"] - d["g0_pi_hat"]).abs()
    d.to_csv(HERE / "gap_panel.csv", index=False)
    print(f"  gap g_n: {d.gap.min()}-{d.gap.max()}, median {int(d.gap.median())}"
          f"  ({100*d.gap.median()/d.g0_n_test.median():.2f}% of the test window)")
    print(f"  |dpi|: median {d.dpi.median():.6f}  max {d.dpi.max():.6f}")
    zone = (d.g0_TL != d.gn_TL).sum()
    kup = ((d.g0_p_kupiec > 0.05) != (d.gn_p_kupiec > 0.05)).sum()
    print(f"  Basel zone changes: {zone} of {len(d)}")
    print(f"  Kupiec verdict flips at 5%: {kup} of {len(d)}")
    print(f"  green before {int((d.g0_TL=='Green').sum())}  after {int((d.gn_TL=='Green').sum())}")
    print(f"  kupiec passes before {int((d.g0_p_kupiec>0.05).sum())}"
          f"  after {int((d.gn_p_kupiec>0.05).sum())}")
    print(f"\n  written: {HERE / 'gap_panel.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
