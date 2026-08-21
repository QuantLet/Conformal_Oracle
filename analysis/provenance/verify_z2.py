#!/usr/bin/env python3
"""Independent check of the Z_2 result before it is promoted into the body.

The claim under test: the Acerbi-Szekely Z_2 test passes on a series whose 1%
violation rate is 0.39 -- i.e. an ES backtest accepts a forecast that is wrong by
a factor of thirty-nine. If true it is the sharpest instance of the paper's
argument; it is not going into Section 6 on the strength of one implementation.

Two differences from the repository's routine, both deliberate:

  1. CANONICAL DENOMINATOR. Acerbi and Szekely (2014, eq. 8) put ES_t inside the
     sum: Z_2 = (1/(N*alpha)) * sum_t [ I_t * X_t / ES_t ] + 1. The repository
     uses a time-averaged ES_bar outside it. They coincide only if ES_t is
     constant, which it is not.
  2. SEPARATE CODE PATH. Written from the paper's definition, sharing nothing
     with CFP_ES_Correction_Z2.py beyond the data files.

SIGN CONVENTION, which is the whole point of running this separately. Acerbi and
Szekely write Z_2 with ES_t as a POSITIVE magnitude and X_t as the P&L, so on a
violation day X_t/ES_t < 0, the sum is negative, and Z_2 = (1/(N*alpha)) * sum +
1 sits near zero when the forecast is calibrated. The stored columns hold
lower-tail quantities and are NEGATIVE. Feeding them in unchanged flips the sign
of every term and produces a large positive statistic that can never reach a
one-sided lower critical value -- an artefact, not a pass. This script negates
them explicitly.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "cfp_ijf_data"
sys.path.insert(0, str(BASE / "Quantlets"))
from cfp_config import MODELS, SYMBOLS  # noqa: E402

ALPHA = 0.025          # the ES level the manuscript reports, per FRTB
CRIT = -1.96


def load(model, asset):
    subdir, suffix = MODELS[model]
    fname = f"{asset}_{suffix}.parquet" if suffix else f"{asset}.parquet"
    fc = pd.read_parquet(DATA / subdir / fname)
    ret = pd.read_csv(DATA / "returns" / f"{asset}.csv", index_col=0,
                      parse_dates=True)
    common = ret.index.intersection(fc.index).sort_values()
    es_col = [c for c in fc.columns if c.lower().startswith("es")]
    if not es_col:
        return None
    return (ret.loc[common, "log_return"].values,
            fc.loc[common, f"VaR_{ALPHA}"].values,
            fc.loc[common, es_col[0]].values)


def z2_canonical(r, var, es):
    """Acerbi-Szekely test 2 with the per-date ES denominator."""
    ok = np.isfinite(r) & np.isfinite(var) & np.isfinite(es) & (es < 0)
    r, var, es = r[ok], var[ok], es[ok]
    n = len(r)
    if n == 0:
        return np.nan, np.nan
    es_pos = -es                      # ES as a positive magnitude, as in the paper
    ind = (r < var).astype(float)
    z2 = np.sum(ind * r / es_pos) / (n * ALPHA) + 1.0
    return z2, float(ind.mean())


def z2_timeaveraged(r, var, es):
    """The repository's variant, for comparison only."""
    ok = np.isfinite(r) & np.isfinite(var) & np.isfinite(es)
    r, var, es = r[ok], var[ok], es[ok]
    n = len(r)
    esbar = np.mean(-es)              # same convention, averaged denominator
    if n == 0 or esbar == 0:
        return np.nan
    return np.sum((r < var).astype(float) * r) / (n * ALPHA * esbar) + 1.0


def main() -> int:
    rows = []
    for model in ("Chronos-Small", "Chronos-Small-A", "Chronos-Mini",
                  "Chronos-Mini-A", "GJR-GARCH-t", "Moirai-1.1"):
        passes_c = passes_t = n_assets = 0
        zs, pis = [], []
        for asset in sorted(SYMBOLS):
            got = load(model, asset)
            if got is None:
                continue
            zc, pi = z2_canonical(*got)
            zt = z2_timeaveraged(*got)
            if not np.isfinite(zc):
                continue
            n_assets += 1
            passes_c += int(zc >= CRIT)
            passes_t += int(np.isfinite(zt) and zt >= CRIT)
            zs.append(zc); pis.append(pi)
        if n_assets:
            rows.append({"model": model, "n": n_assets,
                         "pihat_at_ES_level": float(np.mean(pis)),
                         "Z2_canonical_median": float(np.median(zs)),
                         "pass_canonical": passes_c,
                         "pass_time_averaged": passes_t})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(Path(__file__).parent / "z2_verification.csv", index=False)
    print(f"\nES level alpha = {ALPHA}; a pass is Z_2 >= {CRIT}.")
    print("pihat_at_ES_level is the VaR violation rate at that same level.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
