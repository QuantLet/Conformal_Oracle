#!/usr/bin/env python3
"""Rebuild the part of `phase2_numbers.json` that has an object behind it.

`paper_numbers.py` reads this file for the Table 2 figures and says it is
"emitted by analysis/phase2/{construct_pair,delta_by_class,band_sweep}.py". Two
of those three write nothing of the kind and the third does not exist: no script
in this repository writes `phase2_numbers.json`. It is the fifth artefact here to
carry published numbers with no producer, and the first whose docstring asserts
one.

Most of it turns out to be reconstructible. This script rebuilds every entry that
an artefact in `analysis/phase2/` determines, checks each against the frozen file,
and prints the entries that remain unbacked rather than quietly omitting them.

    python analysis/phase2/emit_phase2_numbers.py --verify
    python analysis/phase2/emit_phase2_numbers.py --write
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "phase2_numbers.json"


def rebuild() -> dict:
    dbc = {r["cls"]: r for r in json.loads((HERE / "delta_by_class.json").read_text())}
    bs = pd.read_csv(HERE / "band_sweep.csv")
    pr = json.loads((HERE / "pair_report.json").read_text())
    cells = pd.read_csv(HERE / "panel_scale_ratios_by_asset.csv")

    band_now, band_lower = -1.8, -3.5
    row = bs[np.isclose(bs["edge"], band_now)].iloc[0]
    trunc = cells[cells["series"].str.contains(r"\(default\)")]
    good = cells[~cells["series"].str.contains(r"\(default\)")]

    out = {
        "DeltaFree": dbc["no shape restriction"]["delta"],
        "UndFree": dbc["no shape restriction"]["understatement"],
        "DeltaUni": dbc["unimodal"]["delta"],
        "UndUni": dbc["unimodal"]["understatement"],
        # The gate's restriction is the band's UPPER edge alone. The lower edge
        # does not enter it: band_sweep.csv sweeps the upper edge, and the lower
        # one blocks no cell in the panel, which CellsLowerBlocks records.
        "DeltaGateNow": float(row["delta"]),
        "UndGateNow": float(row["und"]),
        "BandNow": band_now,
        "BandLower": band_lower,
        "QTrueStd": pr["q_true_P"],
        "CellsTotal": int(len(cells)),
        "CellsTrunc": int(len(trunc)),
        "CellGoodWorst": round(float(good["ratio"].max()), 3),
        "CellGoodBest": round(float(good["ratio"].min()), 3),
        "CellTruncBest": round(float(trunc["ratio"].min()), 3),
        "CellTruncWorst": round(float(trunc["ratio"].max()), 3),
        "GapWidth": round(float(trunc["ratio"].min() - good["ratio"].max()), 3),
        "MarginUpper": round(float(abs(good["ratio"].max() - band_now)), 3),
        "MarginLower": round(float(abs(good["ratio"].min() - band_lower)), 3),
        "CellsLowerBlocks": int((cells["ratio"] < band_lower).sum()),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    new = rebuild()
    frozen = json.loads(FROZEN.read_text()) if FROZEN.exists() else {}

    bad = []
    for k, v in new.items():
        f = frozen.get(k)
        if f is None:
            bad.append(f"{k}: absent from the frozen file")
        elif abs(float(f) - float(v)) > 5e-4:
            bad.append(f"{k}: frozen {f} vs rebuilt {v}")
    print(f"{len(new)} of {len(frozen)} entries have an object behind them")
    for b in bad:
        print(f"  DIFFERS  {b}")
    if not bad:
        print("  every rebuilt entry reproduces the frozen file")

    unbacked = sorted(set(frozen) - set(new))
    print(f"\n{len(unbacked)} entries remain unbacked by anything in analysis/phase2/:")
    print("  " + ", ".join(unbacked))

    if a.write:
        merged = {**frozen, **new}
        FROZEN.write_text(json.dumps(merged, indent=2) + "\n")
        print(f"\nwrote {FROZEN}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
