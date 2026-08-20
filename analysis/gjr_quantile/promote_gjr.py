#!/usr/bin/env python3
"""Promote the corrected GJR-GARCH series, but only through the gate.

Decision (2026-08-17): the `normal` candidate. GJR-GARCH is fitted with
`dist='normal'` and its VaR taken from the Gaussian quantile, which is what
Sec. 3.3 and Appendix E have described all along. The alternative -- keeping the
skewed-t fit and taking the quantile from the fitted distribution -- is the
better econometrics but redefines the benchmark relative to every prior version
of the paper, and was declined for that reason.

What the shipped series did instead:

    (VaR_alpha - mean)/std = stats.t.ppf(alpha, 5), one distinct value across
    all 25 files and every date -- df hard-coded at 5, and the RAW t quantile
    rather than the standardised one, so too wide by a further sqrt(5/3) = 1.29
    even read as a Student-t model.

No committed version of `pipeline/CFP_Parametric_Benchmarks.ipynb` produces
that; every version computes `mu + sigma * norm.ppf(alpha)` = -2.32635. The
notebook is a post-hoc reconstruction (parquets 2026-03-22, earliest notebook
commit 2026-04-12) that does not reconstruct.

Choosing `normal` has a second payoff beyond matching the text: once the
notebook's `dist` is changed from `'skewt'` to `'normal'`, the notebook's own
code reproduces the promoted series exactly, so GJR moves from "no producer
exists in the repository" to verified by `scripts/verify_producers.py`.

This script refuses to promote a series that does not pass
`scripts/promotion_gate.py`. The superseded files are preserved, and the
promotion is recorded with a before/after coverage table.

Usage:
    python analysis/gjr_quantile/promote_gjr.py            # dry run: gate only
    python analysis/gjr_quantile/promote_gjr.py --promote
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "cfp_ijf_data"
BENCH = DATA / "benchmarks"
OUT = Path(__file__).resolve().parent
CAND = OUT / "candidate_normal"
STAGE = OUT / "_stage"
ALPHAS = [0.01, 0.025, 0.05, 0.10]

sys.path.insert(0, str(BASE / "Quantlets"))
from cfp_config import SYMBOLS  # noqa: E402

sys.path.insert(0, str(BASE / "analysis" / "ae_point4"))
from run_ae_point4 import kupiec_p, traffic_light  # noqa: E402


def stage() -> Path:
    """Lay the candidate out the way the gate expects to find a series tree."""
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "benchmarks").mkdir(parents=True)
    (STAGE / "returns").symlink_to(DATA / "returns")
    n = 0
    for sym in SYMBOLS:
        f = CAND / f"{sym}.parquet"
        if f.exists():
            shutil.copy2(f, STAGE / "benchmarks" / f"{sym}_gjr_garch.parquet")
            n += 1
    return n


def score(fp: Path, sym: str) -> dict | None:
    if not fp.exists():
        return None
    fc = pd.read_parquet(fp)
    ret = pd.read_csv(DATA / "returns" / f"{sym}.csv", index_col=0, parse_dates=True)
    ret.columns = ["r"]
    i = ret.index.intersection(fc.index)
    if not len(i):
        return None
    r = ret.loc[i, "r"].values
    sd = ret.loc[i, "r"].rolling(250).std().values
    q = fc.loc[i, "VaR_0.01"].values
    m = np.isfinite(q)
    v, n = int(np.sum(r[m] < q[m])), int(m.sum())
    k = m & np.isfinite(sd) & (sd > 0)
    return {"asset": sym, "n": n, "pihat": v / n, "p_kupiec": kupiec_p(v, n, 0.01),
            "TL": traffic_light(v, n), "width001": float(np.mean(np.abs(q[m]))),
            "implied_z": float(np.median((q[m] - fc.loc[i, "mean"].values[m])
                                         / fc.loc[i, "std"].values[m])),
            "scale": float(np.median(q[k] / sd[k]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true")
    a = ap.parse_args()

    n = stage()
    print(f"staged {n} candidate files", file=sys.stderr)

    r = subprocess.run(
        [sys.executable, str(BASE / "scripts" / "promotion_gate.py"),
         "--dir", str(STAGE), "--out", str(OUT / "GATE_candidate_normal.md")],
        capture_output=True, text=True)
    gate_txt = r.stdout
    line = [l for l in gate_txt.splitlines() if l.startswith("| GJR-GARCH")]
    print("\n".join(line) or gate_txt[-1500:])
    passed = bool(line) and "BLOCK" not in line[0]
    print(f"\ngate verdict for GJR-GARCH: {'PASS' if passed else 'BLOCK'}")

    # before/after, computed from the same scorer on both series
    rows = []
    for sym in SYMBOLS:
        for lab, fp in (("superseded", OUT / "superseded" / f"{sym}.parquet"),
                        ("corrected", CAND / f"{sym}.parquet")):
            s = score(fp, sym)
            if s:
                rows.append({"series": lab, **s})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "promotion_before_after.csv", index=False)
    g = df.groupby("series").agg(
        assets=("asset", "size"), pihat=("pihat", "mean"),
        implied_z=("implied_z", "median"), scale=("scale", "median"),
        width=("width001", "mean"),
        kupiec_pass=("p_kupiec", lambda x: int((x > 0.05).sum())),
        green=("TL", lambda x: int((x == "Green").sum())))
    print("\n", g.to_string(), sep="")

    if not a.promote:
        print("\ndry run — nothing written to cfp_ijf_data/. Re-run with --promote.")
        return 0
    if not passed:
        print("\nBLOCKED — not promoting. A tolerance is never widened to admit "
              "a series.", file=sys.stderr)
        return 1

    keep = OUT / "superseded"
    keep.mkdir(exist_ok=True)
    moved = 0
    for sym in SYMBOLS:
        src, dst = CAND / f"{sym}.parquet", BENCH / f"{sym}_gjr_garch.parquet"
        if not src.exists():
            continue
        if dst.exists() and not (keep / f"{sym}.parquet").exists():
            shutil.copy2(dst, keep / f"{sym}.parquet")
        shutil.copy2(src, dst)
        moved += 1
    print(f"\npromoted {moved} files into {BENCH}")
    print(f"superseded originals preserved in {keep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
