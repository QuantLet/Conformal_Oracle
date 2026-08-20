#!/usr/bin/env python3
"""Rebuild GJR-GARCH under both readings of the fit/quantile mismatch.

The shipped pipeline (pipeline/CFP_Parametric_Benchmarks.ipynb) does:

    for method, vol, dist in [('gjr_garch','GARCH','skewt'), ('garch_n','GARCH','normal')]:
        am = arch_model(r_win, vol='GARCH', p=1, o=1, q=1, dist=dist)
        ...
        row[f'VaR_{alpha}'] = mu + sigma * stats.norm.ppf(alpha)

GJR is fitted with a skewed-t innovation and its quantile is then taken from the
*normal*. arch scales the conditional variance to the fitted distribution, so
sigma is the scale of a skewed-t; multiplying it by a Gaussian quantile is a
mismatch. GARCH-N shares the same line and is correct only because its `dist` is
'normal'. Diagnostic (analysis/provenance/gjr_diagnostic.csv): implied
z = VaR_0.01/sigma_pred is -3.365 for GJR against GARCH-N's exact -2.326, while
predicted/realised sigma is 0.937 vs 0.946 -- the variance dynamics are sound in
both, only the multiplier is wrong.

Two repairs exist and they are different models, not two spellings of one:

    normal   dist='normal', Gaussian quantile. Matches the manuscript's own
             description of GJR-GARCH in Sec. 3.3 / App. E. Smallest change; the
             leverage term becomes the only difference from GARCH-N.
    skewt    dist='skewt' retained, quantile taken from the fitted skewed-t via
             arch's own ppf. Better econometrics and the stronger benchmark, but
             it changes what "GJR-GARCH" denotes relative to every prior version.

Both are built here as *candidates*. Neither is promoted; the shipped series is
copied to superseded/ verbatim and left in place in cfp_ijf_data.

The fitted innovation parameters (eta, lambda) are stored this time, so the
quantile map can be changed later without another 140k fits.

Outputs (analysis/gjr_quantile/):
    candidate_normal/<asset>.parquet
    candidate_skewt/<asset>.parquet
    superseded/<asset>.parquet
    gjr_rebuild.csv, GJR_REBUILD.md
"""

from __future__ import annotations

import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "cfp_ijf_data"
OUT = Path(__file__).resolve().parent
ALPHAS = [0.01, 0.025, 0.05, 0.10]
WINDOW = 250

sys.path.insert(0, str(BASE / "Quantlets"))
from cfp_config import SYMBOLS  # noqa: E402

sys.path.insert(0, str(BASE / "analysis" / "ae_point4"))
from run_ae_point4 import kupiec_p, traffic_light  # noqa: E402


def load_returns(asset: str) -> pd.Series:
    """Exactly the pipeline's loader, so the rebuild is comparable date-for-date."""
    df = pd.read_csv(DATA / "returns" / f"{asset}.csv", parse_dates=["date"]) \
        .set_index("date").sort_index()
    df = df[df["log_return"].abs() <= 0.50]
    return df["log_return"]


def fit_one(am, dist_obj, alphas=ALPHAS):
    """Fitted (mu, sigma) in return units plus the standardised innovation quantiles."""
    res = am.fit(disp="off", show_warning=False)
    fc = res.forecast(horizon=1, reindex=False)
    mu = float(fc.mean.iloc[-1, 0]) / 100
    sigma = float(np.sqrt(fc.variance.iloc[-1, 0])) / 100
    # arch keeps the distribution's shape parameters at the tail of res.params
    npar = dist_obj.num_params
    shape = res.params.values[-npar:] if npar else np.array([])
    # ppf of the *standardised* (unit-variance) innovation
    z = np.asarray(dist_obj.ppf(alphas, shape), dtype=float)
    return mu, sigma, z, shape


def main() -> int:
    from arch import arch_model
    from arch.univariate import SkewStudent

    for d in ("candidate_normal", "candidate_skewt", "superseded"):
        (OUT / d).mkdir(parents=True, exist_ok=True)

    zn = stats.norm.ppf(ALPHAS)
    t0 = time.time()

    for ai, asset in enumerate(SYMBOLS):
        src = DATA / "benchmarks" / f"{asset}_gjr_garch.parquet"
        if src.exists():
            shutil.copy2(src, OUT / "superseded" / f"{asset}.parquet")

        ret = load_returns(asset)
        n = len(ret)
        rec_n, rec_s = [], []
        for t in range(WINDOW, n):
            r_win = ret.iloc[t - WINDOW:t] * 100
            date = ret.index[t]
            # --- candidate "normal": dist='normal', Gaussian quantile ----------
            try:
                am = arch_model(r_win, vol="GARCH", p=1, o=1, q=1, dist="normal")
                res = am.fit(disp="off", show_warning=False)
                fc = res.forecast(horizon=1, reindex=False)
                mu = float(fc.mean.iloc[-1, 0]) / 100
                sd = float(np.sqrt(fc.variance.iloc[-1, 0])) / 100
            except Exception:
                mu, sd = 0.0, float(r_win.std()) / 100
            rec_n.append({"date": date, "mean": mu, "std": sd,
                          **{f"VaR_{a:g}": mu + sd * z for a, z in zip(ALPHAS, zn)}})
            # --- candidate "skewt": dist='skewt', skewed-t quantile ------------
            try:
                am = arch_model(r_win, vol="GARCH", p=1, o=1, q=1, dist="skewt")
                mu, sd, z, shape = fit_one(am, SkewStudent())
                eta, lam = (float(shape[0]), float(shape[1])) if len(shape) == 2 \
                    else (np.nan, np.nan)
            except Exception:
                mu, sd = 0.0, float(r_win.std()) / 100
                z, eta, lam = zn, np.nan, np.nan
            rec_s.append({"date": date, "mean": mu, "std": sd,
                          "eta": eta, "lambda": lam,
                          **{f"VaR_{a:g}": mu + sd * zz for a, zz in zip(ALPHAS, z)}})

        for recs, sub in ((rec_n, "candidate_normal"), (rec_s, "candidate_skewt")):
            pd.DataFrame(recs).set_index("date").to_parquet(
                OUT / sub / f"{asset}.parquet")
        el = time.time() - t0
        print(f"  [{ai + 1:2d}/{len(SYMBOLS)}] {asset:8s} {n - WINDOW:5d} dates "
              f"| {el / 60:.1f} min elapsed", file=sys.stderr, flush=True)

    # ---------------------------------------------------------------- scoring
    rows = []
    for asset in SYMBOLS:
        rp = DATA / "returns" / f"{asset}.csv"
        if not rp.exists():
            continue
        ret = pd.read_csv(rp, index_col=0, parse_dates=True)
        ret.columns = ["r"]
        srcs = {"shipped": OUT / "superseded" / f"{asset}.parquet",
                "normal": OUT / "candidate_normal" / f"{asset}.parquet",
                "skewt": OUT / "candidate_skewt" / f"{asset}.parquet"}
        for label, fp in srcs.items():
            if not fp.exists():
                continue
            fc = pd.read_parquet(fp)
            i = ret.index.intersection(fc.index)
            r = ret.loc[i, "r"].values
            sd_real = ret.loc[i, "r"].rolling(250).std().values
            q = fc.loc[i, "VaR_0.01"].values
            m = np.isfinite(q)
            v, nn = int(np.sum(r[m] < q[m])), int(m.sum())
            A = np.vstack([fc.loc[i, f"VaR_{a:g}"].values for a in ALPHAS]).T
            g = np.all(np.isfinite(A), axis=1)
            ps = fc.loc[i, "std"].values
            k = m & np.isfinite(sd_real) & (sd_real > 0)
            rows.append({
                "asset": asset, "series": label, "n": nn, "pihat": v / nn,
                "p_kupiec": kupiec_p(v, nn, 0.01), "TL": traffic_light(v, nn),
                "implied_z": float(np.median(q[m] / ps[m])),
                "scale": float(np.median(q[k] / sd_real[k])),
                "width001": float(np.mean(np.abs(q[m]))),
                "frac_monotone": float(np.mean(np.all(np.diff(A[g], axis=1) > 0, axis=1))),
                **{f"pihat_{a:g}": float(np.mean(
                    r[np.isfinite(fc.loc[i, f'VaR_{a:g}'].values)]
                    < fc.loc[i, f"VaR_{a:g}"].values[
                        np.isfinite(fc.loc[i, f"VaR_{a:g}"].values)]))
                   for a in ALPHAS},
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gjr_rebuild.csv", index=False)

    g = df.groupby("series").agg(
        n_assets=("asset", "size"), implied_z=("implied_z", "median"),
        scale=("scale", "median"), width001=("width001", "mean"),
        monotone=("frac_monotone", "mean"),
        kup_pass=("p_kupiec", lambda x: int((x > 0.05).sum())),
        green=("TL", lambda x: int((x == "Green").sum())),
        **{f"pihat_{a:g}": (f"pihat_{a:g}", "mean") for a in ALPHAS}).reset_index()

    L = ["# GJR-GARCH: the fit/quantile mismatch, and both repairs", "",
         "`pipeline/CFP_Parametric_Benchmarks.ipynb` fits GJR with `dist='skewt'` "
         "and takes its quantile from `stats.norm.ppf`. GARCH-N shares that line "
         "and is correct only because its `dist` is `'normal'`. Neither candidate "
         "below is promoted.", "",
         "| series | assets | implied z(1%) | VaR₀.₀₁/σ | mean width | monotone | "
         + " | ".join(f"π̂({a:g})" for a in ALPHAS) + " | Kupiec | Green |",
         "|---|---|---|---|---|---|" + "---|" * (len(ALPHAS) + 2)]
    order = ["shipped", "normal", "skewt"]
    for lab in order:
        if lab not in set(g["series"]):
            continue
        r = g[g["series"] == lab].iloc[0]
        pis = " | ".join(f"**{r[f'pihat_{a:g}']:.4f}**" for a in ALPHAS)
        L.append(f"| {lab} | {int(r['n_assets'])} | {r['implied_z']:+.3f} | "
                 f"{r['scale']:+.3f} | {r['width001']:.5f} | "
                 f"{100 * r['monotone']:.1f}% | {pis} | "
                 f"{int(r['kup_pass'])}/{int(r['n_assets'])} | "
                 f"{int(r['green'])}/{int(r['n_assets'])} |")
    L += ["", "Nominal is the α in each π̂ column. GARCH-N's implied z is −2.326 "
          "by construction; the shipped GJR series sits at −3.365.", "",
          "`candidate_skewt` stores the fitted `eta` and `lambda`, so the quantile "
          "map can be revisited without refitting.", ""]
    (OUT / "GJR_REBUILD.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
