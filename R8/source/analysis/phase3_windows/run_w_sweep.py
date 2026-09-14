#!/usr/bin/env python3
"""K4b: the rolling calibration window at w in {125, 250, 500}.

Section 3.2.1 promises Section 7 reports what w = 125 does. The analytic half is
exact -- k = ceil((w+1)(1-alpha)) >= w whenever w < 2/alpha - 1, so at
alpha = 0.01 the shift at w = 125 IS the window maximum -- and this measures
whether the change of estimator shows up in the panel.

Pre-registration: analysis/phase3_windows/PREREG_W_SWEEP.md

Unit: one cell is one forecaster x one asset x one window, alpha = 0.01.
Cells are compared on the intersection of dates defined at all three windows, so
the longest window sets the common sample.

    python analysis/phase3_windows/run_w_sweep.py
    python analysis/phase3_windows/run_w_sweep.py --controls
"""
from __future__ import annotations

import argparse
import sys
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
DATA = BASE / "cfp_ijf_data"
sys.path.insert(0, str(BASE / "Quantlets"))
from cfp_config import MODELS, SYMBOLS  # noqa: E402

ALPHA = 0.01
WINDOWS = (125, 250, 500)
RED, GRN, YEL = "\033[31m", "\033[32m", "\033[33m"; OFF = "\033[0m"


def k_index(w: int, alpha: float = ALPHA) -> int:
    """Equation (8): the k-th smallest of w scores, 1-based."""
    return ceil((w + 1) * (1 - alpha))


def rolling_shift(scores: np.ndarray, w: int) -> np.ndarray:
    """S_(k) of each trailing window of length w, NaN before the window fills.

    k is clipped at w: when ceil((w+1)(1-alpha)) exceeds w the estimator is the
    sample maximum, which is the regime this script exists to measure. Clipping
    is the estimator's definition here, not a numerical guard.
    """
    k = min(k_index(w), w)
    out = np.full(len(scores), np.nan)
    for t in range(w, len(scores)):
        out[t] = np.sort(scores[t - w:t])[k - 1]
    return out


def load(model: str, asset: str):
    subdir, suffix = MODELS[model]
    f = DATA / subdir / (f"{asset}_{suffix}.parquet" if suffix else f"{asset}.parquet")
    if not f.is_file():
        return None
    fc = pd.read_parquet(f)
    col = f"VaR_{ALPHA}"
    if col not in fc.columns:
        return None
    ret = pd.read_csv(DATA / "returns" / f"{asset}.csv", index_col=0, parse_dates=True)
    idx = ret.index.intersection(fc.index).sort_values()
    return fc.loc[idx, col].to_numpy(), ret.loc[idx, "log_return"].to_numpy()


# --------------------------------------------------------------- controls ----
def control_k_at_125() -> bool:
    """The arithmetic the whole exercise rests on, checked rather than asserted."""
    return (k_index(125) >= 125 and k_index(250) == 249 and k_index(500) == 496
            and k_index(199) < 199 and k_index(198) >= 198)


def control_max_is_max() -> bool:
    """At w = 125 the estimator must return the window maximum, not near it."""
    rng = np.random.default_rng(0)
    s = rng.standard_normal(400)
    r = rolling_shift(s, 125)
    t = 200
    return np.isclose(r[t], s[t - 125:t].max())


def control_interior_is_not_max() -> bool:
    """At w = 250 it must NOT be the maximum, or the comparison is vacuous."""
    rng = np.random.default_rng(1)
    s = rng.standard_normal(800)
    r = rolling_shift(s, 250)
    t = 600
    return not np.isclose(r[t], s[t - 250:t].max())


CONTROLS = [("k >= w exactly at w <= 198", control_k_at_125),
            ("w = 125 returns the window maximum", control_max_is_max),
            ("w = 250 does not", control_interior_is_not_max)]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--controls", action="store_true")
    a = ap.parse_args()
    ok = True
    for name, fn in CONTROLS:
        if fn(): print(f"  {YEL}ctrl{OFF}   {name}")
        else: print(f"  {RED}FAIL{OFF}   control does not fire: {name}"); ok = False
    if not ok: return 2
    if a.controls: return 0

    rows = []
    for model in MODELS:
        for asset in sorted(SYMBOLS):
            got = load(model, asset)
            if got is None:
                continue
            var, r = got
            scores = var - r                      # VaR is the lower quantile
            shifts = {w: rolling_shift(scores, w) for w in WINDOWS}
            common = np.all([np.isfinite(shifts[w]) for w in WINDOWS], axis=0)
            if common.sum() < 250:
                continue
            for w in WINDOWS:
                q = shifts[w][common]
                # Subset both series to the common dates FIRST; masking the
                # shift and then the threshold compares two different spans.
                corrected = var[common] - q       # shift widens the threshold
                hit = r[common] < corrected
                rows.append({"model": model, "asset": asset, "w": w,
                             "k": min(k_index(w), w),
                             "is_max": int(k_index(w) >= w),
                             "n": int(common.sum()),
                             "mean_shift": float(q.mean()),
                             "sd_shift": float(q.std(ddof=1)),
                             "pi_hat": float(hit.mean())})
    d = pd.DataFrame(rows)
    n_cells = len(d) // len(WINDOWS)
    expected = len([m for m in MODELS]) * len(SYMBOLS)
    print(f"\n  cells per window: {n_cells} of an expected {expected}")
    if n_cells < 0.9 * expected:
        print(f"  {RED}BLOCKED{OFF}: fewer than 90% of cells defined on all windows")
        return 2
    d.to_csv(HERE / "w_sweep.csv", index=False)

    piv = d.pivot_table(index=["model", "asset"], columns="w",
                        values=["mean_shift", "sd_shift", "pi_hat"])
    print(f"\n  {'w':>5} {'k':>5} {'max?':>5} {'mean shift':>12} "
          f"{'sd of shift':>12} {'pi-hat':>9}")
    for w in WINDOWS:
        g = d[d.w == w]
        print(f"  {w:5d} {int(g.k.iloc[0]):5d} {'yes' if g.is_max.iloc[0] else 'no':>5} "
              f"{g.mean_shift.median():12.6f} {g.sd_shift.median():12.6f} "
              f"{g.pi_hat.median():9.5f}")
    print("\n  paired against w = 250, over cells:")
    for w in (125, 500):
        a_, b_ = piv[("sd_shift", w)], piv[("sd_shift", 250)]
        s_, t_ = piv[("mean_shift", w)], piv[("mean_shift", 250)]
        p_, q_ = piv[("pi_hat", w)], piv[("pi_hat", 250)]
        print(f"    w={w:3d}: sd ratio median {float((a_/b_).median()):.3f} "
              f"| shift larger on {int((s_ > t_).sum())}/{len(s_)} cells "
              f"| pi-hat lower on {int((p_ < q_).sum())}/{len(p_)}")
    print(f"\n  sqrt(250/125) = {np.sqrt(2):.3f} is what halving the window alone gives")
    print(f"  written: {HERE / 'w_sweep.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
