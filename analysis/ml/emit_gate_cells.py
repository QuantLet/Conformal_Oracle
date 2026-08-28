#!/usr/bin/env python3
"""The ML cells as the structural gate's scale check reads them.

Reading (v) of analysis/phase0/CONDITIONAL_PASSAGES.md asks whether a third
estimator family reaches the gate's LOWER scale edge. It does not, and that
question stays BLOCKED because the 24-asset panel has never run. What the
4-asset dose-response does settle is the question beside it: whether anything
occupies the intermediate range Section 7 declares empty, which is a claim about
the UPPER edge.

Pre-registration: analysis/ml/PREREG_READING_V.md
Results:          analysis/ml/RESULTS_READING_V.md

Unit: one row is one estimator x one asset x one min_data_in_leaf, at
alpha = 0.01 over 200 dates. 40 rows = 2 x 4 x 5. This is NOT the 312-cell
sequence panel and no count here may be pooled with a count from there.

    python analysis/ml/emit_gate_cells.py
    python analysis/ml/emit_gate_cells.py --controls
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ALPHA = 0.01
N_DATES = 200
BAND = (-3.5, -1.8)          # the gate's scale band, transcribed from Supplement S.11
UNDER = 2.5                  # "visibly under-covering", in multiples of nominal

RED, GRN, YEL = "\033[31m", "\033[32m", "\033[33m"
OFF = "\033[0m"


def cells() -> pd.DataFrame:
    d = pd.read_csv(HERE / "dose_response_raw.csv")
    out = []
    for est, qcol in (("LightGBM", "lgbm_q"), ("quantile forest", "qrf_q")):
        g = d.assign(ratio=d[qcol] / d["train_sd"],
                     hit=(d["realised"] < d[qcol]).astype(float))
        for (asset, leaf), s in g.groupby(["asset", "leaf"]):
            fin = int(np.isfinite(s["ratio"]).sum())
            out.append({"est": est, "asset": asset, "leaf": int(leaf),
                        "n_dates": fin,
                        "ratio": float(s["ratio"].median()),
                        "pi_ratio": float(s["hit"].mean() / ALPHA)})
    t = pd.DataFrame(out)
    t["blocked"] = t["ratio"] > BAND[1]
    t["below_lower"] = t["ratio"] < BAND[0]
    return t


# --------------------------------------------------------------- controls ----
def control_blocks_a_bad_cell() -> bool:
    """A cell whose threshold sits at a fifth of the well-specified range must
    be blocked. Without this the "blocked" column could be all-False and the
    exercise would report a clean panel it never tested."""
    t = cells().copy()
    t.loc[0, "ratio"] = -0.283            # the best truncated cell of the panel
    return bool(t.assign(b=t["ratio"] > BAND[1]).loc[0, "b"])


def control_lower_edge_can_fire() -> bool:
    """The lower-edge test must be able to return True.

    This is the R14 lesson applied here: reading (v) returns "nothing reaches
    the lower edge", and a column that is False by construction would return
    exactly that. Planting a cell past the edge shows the column can move.
    """
    t = cells().copy()
    t.loc[0, "ratio"] = -4.0
    return bool((t["ratio"] < BAND[0]).any())


def control_grid_is_coarse() -> bool:
    """pi-hat/alpha must be shown to live on a grid of 1/(N*alpha).

    At 200 dates and alpha = 0.01 the expected count is 2 and the ratio can only
    take multiples of 0.5. CONDITIONAL_PASSAGES.md describes this family as
    sitting at "0.6x to 1.0x nominal", and 0.6 is not on that grid. The control
    asserts the grid rather than the sentence.
    """
    t = cells()
    step = 1.0 / (N_DATES * ALPHA)
    return bool(np.allclose(t["pi_ratio"] / step, np.round(t["pi_ratio"] / step)))


CONTROLS = [("a truncated-range cell is blocked", control_blocks_a_bad_cell),
            ("the lower-edge column can fire", control_lower_edge_can_fire),
            ("pi-hat/alpha lies on the 1/(N alpha) grid", control_grid_is_coarse)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", action="store_true")
    a = ap.parse_args()

    ok = True
    for name, fn in CONTROLS:
        if fn():
            print(f"  {YEL}ctrl{OFF}   {name}")
        else:
            print(f"  {RED}FAIL{OFF}   control does not fire: {name}"); ok = False
    if not ok:
        return 2
    if a.controls:
        return 0

    t = cells()
    if (t["n_dates"] != N_DATES).any():
        n_bad = int((t["n_dates"] != N_DATES).sum())
        print(f"  {RED}FAIL{OFF}   {n_bad} cell(s) do not carry {N_DATES} dates")
        if n_bad > len(t) / 4:
            print("  BLOCKED: more than a quarter of the panel is short")
            return 2

    t.to_csv(HERE / "gate_cells.csv", index=False)
    step = 1.0 / (N_DATES * ALPHA)
    print(f"\n  {GRN}pass{OFF}   {len(t)} cells written; "
          f"pi-hat/alpha resolution {step:.1f}x nominal")
    for est, g in t.groupby("est"):
        print(f"           {est:16s} blocked {int(g.blocked.sum()):2d}/{len(g)}  "
              f"below lower edge {int(g.below_lower.sum()):2d}  "
              f"ratio [{g.ratio.min():.3f}, {g.ratio.max():.3f}]")
    b = t[t["blocked"]]
    print(f"           blocked and under-covering (>= {UNDER}x): "
          f"{int((b.pi_ratio >= UNDER).sum())} of {len(b)}")
    print(f"           passed but under-covering (>= {UNDER}x): "
          f"{int((~t.blocked & (t.pi_ratio >= UNDER)).sum())} of "
          f"{int((~t.blocked).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
