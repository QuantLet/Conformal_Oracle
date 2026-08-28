#!/usr/bin/env python3
"""Rebuild `band_sweep.csv`, which had no producer and feeds Section 8's headline.

The residual understatement at a band edge is a closed form, not a solver output.
The gate's scale band admits a series whose reported quantile, in units of the
true conditional scale, is at least the edge; the deepest truncation such a
series can hide is the one whose reported quantile sits exactly on the edge. So

    und(edge) = 100 * (1 - |edge| / |q_true|),
    delta(edge) solves  G_ppf(delta + alpha (1 - 2 delta)) = edge,

with G the standardised Student-t_5 the construction uses. Every row of the
shipped file reproduces from these two lines to six decimals, which is worth
recording: the file was consumed by `paper_numbers.py` for two years' worth of
macros -- including the 30.9% of Table 2 -- while nothing in the repository wrote
it, and it turns out to have been right all along.

    python analysis/phase2/emit_band_sweep.py --verify
    python analysis/phase2/emit_band_sweep.py --write
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "band_sweep.csv"

ALPHA, NU = 0.01, 5
_sc = np.sqrt(NU / (NU - 2))
G_ppf = lambda p: stats.t.ppf(p, NU) / _sc          # noqa: E731
Q_TRUE = G_ppf(ALPHA)

EDGES = [-1.400, -1.600, -1.800, -1.900, -2.000, -2.050,
         -2.085, -2.100, -2.150, -2.200, -2.300, -2.400]


def understatement(edge: float) -> float:
    return 100.0 * (1.0 - abs(edge) / abs(Q_TRUE))


def critical_delta(edge: float) -> float:
    return optimize.brentq(
        lambda d: G_ppf(d + ALPHA * (1 - 2 * d)) - edge, 1e-9, 0.49)


def rebuild(edges=EDGES) -> pd.DataFrame:
    return pd.DataFrame({
        "edge": edges,
        "delta": [critical_delta(e) for e in edges],
        "q": edges,
        "und": [understatement(e) for e in edges],
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--edge", type=float, default=None,
                    help="report one edge and exit, e.g. the tightened -1.94")
    a = ap.parse_args()

    if a.edge is not None:
        print(f"edge {a.edge}: delta {critical_delta(a.edge):.6f}  "
              f"understatement {understatement(a.edge):.4f}%")
        return 0

    new = rebuild()
    if OUT.exists():
        old = pd.read_csv(OUT)
        j = old.merge(new, on="edge", suffixes=("_old", "_new"))
        assert len(j) == len(old), "edge grid differs"
        for c in ("delta", "und"):
            d = (j[f"{c}_old"] - j[f"{c}_new"]).abs().max()
            print(f"  {c:6s} max |difference| against the shipped file: {d:.3e}")
        print("  reproduces" if all((j[f"{c}_old"] - j[f"{c}_new"]).abs().max() < 1e-6
                                    for c in ("delta", "und")) else "  DIFFERS")
    if a.write:
        cols = pd.read_csv(OUT).columns if OUT.exists() else new.columns
        out = new.reindex(columns=[c for c in cols if c in new.columns])
        for c in cols:
            if c not in out.columns:
                out[c] = pd.read_csv(OUT)[c]
        out[list(cols)].to_csv(OUT, index=False)
        print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
